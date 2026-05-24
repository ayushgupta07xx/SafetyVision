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

    insp["d"] = pd.to_datetime(insp["uploaded_at"], utc=True).dt.normalize()
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
