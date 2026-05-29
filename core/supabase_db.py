"""Supabase persistence layer (Phase 2, Mode 3 per-user history).

Backend module ONLY. Uses the service_role (secret) key, which BYPASSES RLS,
so every write must set user_id explicitly. Never import this anywhere a
browser can read — the frontend uses the anon/publishable key instead.

Provides:
    fetch_compliance_series  -> DataFrame[ds, y] for the Prophet forecast
    insert_inspection        -> create one inspection row, return its id
    insert_violations        -> bulk insert violation rows
    log_inspection           -> convenience: 1 inspection + N violations
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Protocol, cast

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

# Allowed enum values (mirror the SQL CHECK constraints in 0001_schema.sql)
SOURCE_TYPES = {"image", "video", "api"}
SOURCES = {"next_js", "hf_spaces", "api", "synthetic"}


@lru_cache(maxsize=1)
def _client() -> Client:
    """Service-role client (bypasses RLS). Cached for the process lifetime."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # secret key — backend only
    return create_client(url, key)


# ─── Read: forecast input ────────────────────────────────────────────────────
def fetch_compliance_series(
    violation_type: str, days: int = 30, user_id: str | None = None
) -> pd.DataFrame:
    """Return DataFrame[ds, y] where y = 1 - violations/inspections per day.

    Mirrors the SQLite LEFT-JOIN logic: every day that had inspections counts,
    so a day with inspections but zero violations of this type -> y = 1.0.
    user_id=None aggregates across all users (global/admin view).

    Note: PostgREST returns up to 1000 rows per request by default. Fine for
    free-tier / demo volumes; paginate if a single window ever exceeds that.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    cli = _client()

    iq = cli.table("inspections").select("uploaded_at").gte("uploaded_at", cutoff.isoformat())
    vq = (cli.table("violations").select("timestamp_ms")
          .eq("violation_type", violation_type).gte("timestamp_ms", cutoff_ms))
    if user_id:
        iq = iq.eq("user_id", user_id)
        vq = vq.eq("user_id", user_id)

    insp = pd.DataFrame(iq.execute().data or [])
    viol = pd.DataFrame(vq.execute().data or [])
    if insp.empty:
        return pd.DataFrame(columns=["ds", "y"])

    insp["d"] = pd.to_datetime(insp["uploaded_at"], utc=True, format="ISO8601").dt.normalize()
    daily = insp.groupby("d").size().rename("total").reset_index()

    if viol.empty:
        daily["v"] = 0
    else:
        viol["d"] = pd.to_datetime(viol["timestamp_ms"], unit="ms", utc=True).dt.normalize()
        dv = viol.groupby("d").size().rename("v").reset_index()
        daily = daily.merge(dv, on="d", how="left")
        daily["v"] = daily["v"].fillna(0)

    daily["y"] = (1.0 - daily["v"] / daily["total"]).clip(0.0, 1.0)
    daily["ds"] = daily["d"].dt.tz_localize(None)  # Prophet wants tz-naive
    return daily[["ds", "y"]].sort_values("ds").reset_index(drop=True)


# ─── Write: inspection + violations ──────────────────────────────────────────
def insert_inspection(
    user_id: str, source_type: str, image_url: str | None,
    total_detections: int, total_violations: int,
) -> str:
    inspection_id = str(uuid.uuid4())
    _client().table("inspections").insert({
        "inspection_id": inspection_id,
        "user_id": user_id,
        "source_type": source_type,
        "image_url": image_url,
        "total_detections": total_detections,
        "total_violations": total_violations,
    }).execute()
    return inspection_id


def insert_violations(rows: list[dict]) -> None:
    if rows:
        _client().table("violations").insert(rows).execute()


def log_inspection(
    user_id: str,
    items: list[tuple],          # list of (Violation, incident_report_dict)
    source: str = "next_js",
    source_type: str = "image",
    image_url: str | None = None,
    total_detections: int | None = None,
) -> tuple[str, list[str]]:
    """Create one inspection + its violations. Returns (inspection_id, violation_ids).

    `items` is the agent's per-image output. Wire this into the agent log step
    (agent/graph.py) when a user_id is present — see the next-step note.
    """
    n = len(items)
    inspection_id = insert_inspection(
        user_id, source_type, image_url,
        total_detections if total_detections is not None else n, n,
    )
    ts_ms = int(time.time() * 1000)
    rows, vids = [], []
    for violation, report in items:
        vid = str(uuid.uuid4())
        vids.append(vid)
        bbox = getattr(violation, "bbox", None)
        rows.append({
            "violation_id": vid,
            "inspection_id": inspection_id,
            "user_id": user_id,
            "timestamp_ms": ts_ms,
            "violation_type": violation.type,
            "risk_level": report.get("risk_level", violation.risk_level),
            "confidence": float(violation.confidence),
            "bbox_json": [round(float(v), 1) for v in bbox] if bbox is not None else None,
            "regulation_cited": report.get("regulation_cited"),
            "summary": report.get("summary"),
            "source": source,
        })
    insert_violations(rows)
    return inspection_id, vids

class _DetectionResult(Protocol):
    violations: list  # duck-typed; real type is core.detector's result


def persist_inspection_from_result(
    user_id: str,
    result: _DetectionResult,
    incident_report: dict | None = None,
    source: str = "api",
    image_url: str | None = None,
) -> tuple[str, list[str]]:
    """Adapt a detector result + incident report into one inspection + violations.

    Highest-confidence violation carries the report's citation/summary; the rest
    carry detection fields only. Zero-violation results STILL create an inspection
    row (a clean check is the forecast's compliance=1.0 denominator).
    Returns the persisted (inspection_id, [violation_id ...]).
    """
    violations = list(result.violations)
    report = incident_report or {}
    if violations:
        primary = max(violations, key=lambda v: v.confidence)
        items = [(v, report if v is primary else {}) for v in violations]
    else:
        items = []
    n_det = len(getattr(result, "detections", violations))
    return log_inspection(
        user_id, items, source=source, source_type="image",
        image_url=image_url, total_detections=n_det,
    )

def fetch_violations(user_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """Return a user's violation history, newest first, paginated.

    Service-role read scoped explicitly by user_id (RLS is bypassed here, so the
    user_id filter is mandatory). PostgREST range is inclusive.
    """
    resp = (
        _client()
        .table("violations")
        .select(
            "violation_id, inspection_id, timestamp_ms, violation_type, "
            "risk_level, confidence, regulation_cited, summary, "
            "pdf_report_url, source"
        )
        .eq("user_id", user_id)
        .order("timestamp_ms", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return cast("list[dict]", resp.data or [])

# ─── Write: report PDF storage ───────────────────────────────────────────────
REPORTS_BUCKET = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
# Long TTL so stored demo links don't rot; the production-correct pattern is
# on-read re-signing in Mode-3 (TODO when the frontend lands).
REPORT_URL_TTL = int(os.getenv("SUPABASE_REPORT_URL_TTL", str(365 * 24 * 3600)))


@lru_cache(maxsize=512)
def get_user_label(user_id: str) -> str:
    """Best-effort label for the PDF Subject field: email if we can fetch it
    from Supabase Auth, else a short UUID prefix. Cached per container.
    Never raises."""
    try:
        res = _client().auth.admin.get_user_by_id(user_id)
        email = getattr(getattr(res, "user", None), "email", None)
        if email:
            return str(email)
    except Exception:  # noqa: BLE001 -- best-effort cosmetic lookup
        pass
    return f"user-{user_id[:8]}"


def store_pdf_for_violation(
    user_id: str, violation_id: str, pdf_bytes: bytes, ttl: int | None = None,
) -> str | None:
    """Upload a report PDF to the private `reports` bucket, stamp the signed URL
    onto violations.pdf_report_url, and return the URL.

    Service-role (bypasses RLS); the object path is namespaced by user_id.
    """
    cli = _client()
    path = f"{user_id}/{violation_id}.pdf"
    cli.storage.from_(REPORTS_BUCKET).upload(
        path, pdf_bytes,
        {"content-type": "application/pdf", "upsert": "true"},
    )
    raw: object = cli.storage.from_(REPORTS_BUCKET).create_signed_url(
        path, ttl or REPORT_URL_TTL
    )
    url: str | None = None
    if isinstance(raw, str):
        url = raw
    elif isinstance(raw, dict):
        url = raw.get("signedURL") or raw.get("signedUrl") or raw.get("signed_url")
    if url:
        cli.table("violations").update({"pdf_report_url": url}).eq(
            "violation_id", violation_id
        ).execute()
    return url