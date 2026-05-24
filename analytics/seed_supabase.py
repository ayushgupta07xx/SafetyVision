"""Seed synthetic inspection + violation history into Supabase for the forecast demo.

violations.user_id is a FK to auth.users, so you need a REAL auth user UUID first:
    Supabase dashboard -> Authentication -> Users -> Add user (email + password)
    -> copy the user's UID.

Usage (WSL, ~/safetyvision, venv active):
    python -m analytics.seed_supabase --user-id <auth-user-uuid> --days 35 --reset

Deterministic (seed=42). Compliance improves across the window so the 7-day
forecast shows a visible upward trend. Violation types and risk levels are pulled
from core.detector.RISK_LEVELS -- the single source of truth -- so they can never
drift from what the detector actually logs as violation.type.
"""
from __future__ import annotations

import argparse
import random
import uuid
from datetime import UTC, datetime, timedelta

from core import supabase_db
from core.detector import RISK_LEVELS

# All real v2 violation classes, straight from the detector. No local list to drift.
VIOLATION_TYPES: list[str] = list(RISK_LEVELS.keys())

# Per-class base appearance probability (demo tuning only -- not a detector concern).
# Unknown/new classes fall back to 0.15 so adding a detector class never KeyErrors.
BASE_RATE: dict[str, float] = {
    "Fall-Detected": 0.04,
    "No_Harness": 0.08,
    "NO-Hardhat": 0.35,
    "NO-Safety Vest": 0.25,
    "NO-Goggles": 0.18,
    "NO-Mask": 0.15,
    "NO-Gloves": 0.20,
}


def _reset(user_id: str) -> None:
    """Delete this user's existing rows (violations before inspections, FK order)."""
    cli = supabase_db._client()
    cli.table("violations").delete().eq("user_id", user_id).execute()
    cli.table("inspections").delete().eq("user_id", user_id).execute()
    print(f"Reset: cleared existing inspections + violations for user {user_id}")


def seed(user_id: str, days: int = 35, seed_val: int = 42) -> None:
    random.seed(seed_val)
    now = datetime.now(UTC)
    insp_rows: list[dict] = []
    viol_rows: list[dict] = []
    for d in range(days, 0, -1):
        day = now - timedelta(days=d)
        progress = (days - d) / days          # 0 -> 1 across the window
        for _ in range(random.randint(8, 16)):  # inspections that day
            insp_id = str(uuid.uuid4())
            stamp = day + timedelta(minutes=random.randint(0, 600))
            todays = [
                vt for vt in VIOLATION_TYPES
                if random.random() < BASE_RATE.get(vt, 0.15) * (1 - 0.5 * progress)
            ]
            insp_rows.append({
                "inspection_id": insp_id,
                "user_id": user_id,
                "uploaded_at": stamp.isoformat(),
                "source_type": "image",
                "total_detections": len(todays) + random.randint(1, 3),
                "total_violations": len(todays),
            })
            for vt in todays:
                viol_rows.append({
                    "violation_id": str(uuid.uuid4()),
                    "inspection_id": insp_id,
                    "user_id": user_id,
                    "timestamp_ms": int(stamp.timestamp() * 1000),
                    "violation_type": vt,
                    "risk_level": RISK_LEVELS[vt],
                    "confidence": round(random.uniform(0.5, 0.95), 3),
                    "source": "synthetic",
                })
    cli = supabase_db._client()
    for i in range(0, len(insp_rows), 200):
        cli.table("inspections").insert(insp_rows[i:i + 200]).execute()
    for i in range(0, len(viol_rows), 200):
        cli.table("violations").insert(viol_rows[i:i + 200]).execute()
    print(f"Seeded {len(insp_rows)} inspections + {len(viol_rows)} violations "
          f"across {days} days for user {user_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", required=True, help="auth.users UUID (from dashboard)")
    p.add_argument("--days", type=int, default=35)
    p.add_argument("--reset", action="store_true",
                   help="delete this user's existing rows before seeding (idempotent re-seed)")
    args = p.parse_args()
    if args.reset:
        _reset(args.user_id)
    seed(args.user_id, args.days)
