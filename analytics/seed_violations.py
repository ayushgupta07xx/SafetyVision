"""Generate synthetic 30-day violation history for forecasting demos.

Production reality: agent/tools.py writes one violation event per detection.
For Prophet/SARIMA we need ~30 days with realistic temporal structure
(weekly seasonality + slow trend + noise). This script seeds both
violation events and a daily_inspections aggregate table.

Compliance formula (matches brief):
    compliance_rate[type, day] = 1 - violations[type, day] / total_inspections[day]

Synthetic rows are tagged source='synthetic' so reruns don't pollute real
smoke-test data. Run:
    python -m analytics.seed_violations              # 30 days, seed=42
    python -m analytics.seed_violations --days 45    # longer history
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

DB_PATH = Path(os.getenv("SAFETYVISION_DB", "/tmp/violations.db"))

# Per-type base rates (violations per inspection on a normal weekday)
# Class names match what the YOLOv8 model outputs (chat 5 smoke-tested).
BASE_RATES = {
    "NO-Hardhat":     0.10,
    "NO-Safety Vest": 0.08,
    "NO-Mask":        0.05,
    "No_Harness":     0.03,
}

RISK_LEVELS = {
    "NO-Hardhat":     "HIGH",
    "NO-Safety Vest": "MEDIUM",
    "NO-Mask":        "MEDIUM",
    "No_Harness":     "CRITICAL",
}

REGULATIONS = {
    "NO-Hardhat":     "OSHA 29 CFR 1926.100(a)",
    "NO-Safety Vest": "OSHA 29 CFR 1926.201",
    "NO-Mask":        "OSHA 29 CFR 1910.134",
    "No_Harness":     "OSHA 29 CFR 1926.104",
}

# Day-of-week multiplier (0=Mon ... 6=Sun). More incidents Monday, fewer weekends.
DOW_FACTOR = {0: 1.3, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.85, 5: 0.6, 6: 0.5}


def _ensure_violations_schema(conn: sqlite3.Connection) -> None:
    """Mirror schema from agent/tools.py::_ensure_sqlite_schema (kept in sync)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            violation_id     TEXT PRIMARY KEY,
            timestamp_ms     INTEGER NOT NULL,
            violation_type   TEXT NOT NULL,
            risk_level       TEXT NOT NULL,
            confidence       REAL NOT NULL,
            regulation_cited TEXT,
            summary          TEXT,
            source           TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp_ms)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON violations(violation_type)")


def _ensure_inspections_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_inspections (
            inspection_date    TEXT PRIMARY KEY,    -- YYYY-MM-DD UTC
            total_inspections  INTEGER NOT NULL,
            source             TEXT
        )
        """
    )


def seed(days: int = 30, rng_seed: int = 42) -> dict:
    random.seed(rng_seed)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_violations_schema(conn)
        _ensure_inspections_schema(conn)

        # Wipe synthetic rows only (preserve real smoke-test data)
        conn.execute("DELETE FROM violations WHERE source = 'synthetic'")
        conn.execute("DELETE FROM daily_inspections WHERE source = 'synthetic'")

        today_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        total_v = 0
        total_i = 0

        for day_idx in range(days):
            day = today_utc - timedelta(days=days - 1 - day_idx)
            dow = day.weekday()
            day_start_ms = int(day.timestamp() * 1000)

            # Inspections per day: base 100, weekly pattern, ±15% noise
            inspections = int(100 * DOW_FACTOR[dow] * random.uniform(0.85, 1.15))
            inspections = max(20, inspections)

            conn.execute(
                "INSERT INTO daily_inspections VALUES (?, ?, ?)",
                (day.strftime("%Y-%m-%d"), inspections, "synthetic"),
            )
            total_i += inspections

            # Trend: gentle improvement over the window (1.1x -> 0.9x)
            trend = 1.1 - 0.2 * (day_idx / max(1, days - 1))

            for vtype, rate in BASE_RATES.items():
                effective = rate * DOW_FACTOR[dow] * trend * random.uniform(0.7, 1.3)
                count = max(0, int(round(inspections * effective)))

                for _ in range(count):
                    ts_ms = day_start_ms + random.randint(0, 86_399_000)
                    conn.execute(
                        "INSERT INTO violations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            ts_ms,
                            vtype,
                            RISK_LEVELS[vtype],
                            round(random.uniform(0.55, 0.95), 3),
                            REGULATIONS[vtype],
                            f"Synthetic {vtype} detection",
                            "synthetic",
                        ),
                    )
                    total_v += 1

        conn.commit()

    summary = {
        "days": days,
        "inspections": total_i,
        "violations": total_v,
        "db_path": str(DB_PATH),
    }
    logger.info(
        "Seeded %d days: %d inspections, %d violations -> %s",
        days, total_i, total_v, DB_PATH,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42, dest="rng_seed")
    args = p.parse_args()
    seed(days=args.days, rng_seed=args.rng_seed)
