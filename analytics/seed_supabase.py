#!/usr/bin/env python3
"""Seed synthetic Supabase history for one user so Mode-3 /forecast has >=14
days of data for Prophet. Each violation type gets a distinct base rate + trend
(improving / declining / flat) so the dashboard shows variety. Inspections marked
image_url='synthetic://seed', violations source='synthetic' -> reruns idempotent.

  EMAIL=khsfkvbhjsd@gmail.com python analytics/seed_supabase.py
Needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in .env (service role bypasses RLS).
"""
import os
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from supabase import create_client  # noqa: E402

from core.detector import RISK_LEVELS  # noqa: E402  derive types+levels, don't hardcode

DAYS = int(os.environ.get("DAYS", "30"))
SENTINEL = "synthetic://seed"
random.seed(42)

# Per-type (base per-inspection violation prob, trend). Keyed by RISK_LEVELS names;
# unlisted types fall back to DEFAULT. violation_type written still comes from RISK_LEVELS.
DEFAULT = (0.20, "flat")
PROFILE = {
    "NO-Hardhat":     (0.35, "improving"),
    "No_Harness":     (0.15, "declining"),   # critical -> show it worsening
    "NO-Safety Vest": (0.30, "improving"),
    "NO-Goggles":     (0.20, "flat"),
    "NO-Mask":        (0.10, "flat"),
    "NO-Gloves":      (0.40, "improving"),
    "Fall-Detected":  (0.05, "flat"),
}


def prob(base: float, trend: str, d: int) -> float:
    f = d / max(DAYS - 1, 1)
    if trend == "improving":
        return base * (1.0 - 0.7 * f)
    if trend == "declining":
        return base * (1.0 + 1.2 * f)
    return base


sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

user_id = os.environ.get("USER_ID")
if not user_id:
    email = os.environ.get("EMAIL", "").lower()
    if not email:
        raise SystemExit("Set EMAIL or USER_ID env var")
    users = sb.auth.admin.list_users()
    user_id = next((u.id for u in users if (u.email or "").lower() == email), None)
    if not user_id:
        raise SystemExit(f"No user with email {email}")
print("Seeding user_id:", user_id)

TYPES = list(RISK_LEVELS.keys())

sb.table("violations").delete().eq("user_id", user_id).eq("source", "synthetic").execute()
sb.table("inspections").delete().eq("user_id", user_id).eq("image_url", SENTINEL).execute()
print("Cleared prior synthetic rows.")

now = datetime.now(UTC)
inspections, violations = [], []

for d in range(DAYS):
    day = now - timedelta(days=(DAYS - 1 - d))
    for _ in range(random.randint(3, 6)):
        insp_id = str(uuid.uuid4())
        ts = day.replace(hour=random.randint(7, 18), minute=random.randint(0, 59),
                         second=random.randint(0, 59), microsecond=random.randint(1, 999999))
        ts_ms = int(ts.timestamp() * 1000)
        hit = []
        for vt in TYPES:
            base, trend = PROFILE.get(vt, DEFAULT)
            if random.random() < prob(base, trend, d):
                hit.append(vt)
        for vt in hit:
            violations.append({
                "violation_id": str(uuid.uuid4()),
                "inspection_id": insp_id,
                "user_id": user_id,
                "timestamp_ms": ts_ms,
                "violation_type": vt,
                "risk_level": RISK_LEVELS[vt],
                "confidence": round(random.uniform(0.45, 0.95), 3),
                "source": "synthetic",
            })
        inspections.append({
            "inspection_id": insp_id,
            "user_id": user_id,
            "uploaded_at": ts.isoformat(),
            "source_type": "api",
            "image_url": SENTINEL,
            "total_detections": len(hit) + random.randint(1, 3),
            "total_violations": len(hit),
        })

sb.table("inspections").insert(inspections).execute()  # type: ignore[arg-type]
if violations:
    sb.table("violations").insert(violations).execute()  # type: ignore[arg-type]

print(f"Inserted {len(inspections)} inspections, {len(violations)} violations over {DAYS} days.")
print("Profiles:", {vt: PROFILE.get(vt, DEFAULT) for vt in TYPES})
