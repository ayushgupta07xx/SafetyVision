"""Tests for analytics.forecast, analytics.forecast_baseline,
analytics.seed_violations, and agent.tools.log_violation.

These tests seed a small per-test SQLite DB and exercise the full
forecasting math. No network or LLM calls.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from agent.tools import log_violation
from analytics import forecast, forecast_baseline, seed_violations
from core.detector import Violation


# ─── seed_violations.seed() ─────────────────────────────────────────────────
class TestSeedViolations:
    def test_seed_writes_rows(self, tmp_db):
        summary = seed_violations.seed(days=20, rng_seed=42)
        assert summary["days"] == 20
        assert summary["violations"] > 0
        assert summary["inspections"] > 0

        with sqlite3.connect(tmp_db) as conn:
            n_v = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE source='synthetic'"
            ).fetchone()[0]
            n_i = conn.execute(
                "SELECT COUNT(*) FROM daily_inspections WHERE source='synthetic'"
            ).fetchone()[0]
        assert n_v == summary["violations"]
        assert n_i == summary["days"]

    def test_seed_is_deterministic_for_fixed_seed(self, tmp_db):
        s1 = seed_violations.seed(days=15, rng_seed=42)
        s2 = seed_violations.seed(days=15, rng_seed=42)
        assert s1["violations"] == s2["violations"]
        assert s1["inspections"] == s2["inspections"]

    def test_seed_preserves_real_rows(self, tmp_db):
        # Pre-insert a non-synthetic row; seeder should leave it alone.
        with sqlite3.connect(tmp_db) as conn:
            seed_violations._ensure_violations_schema(conn)
            conn.execute(
                "INSERT INTO violations VALUES (?,?,?,?,?,?,?,?)",
                ("real-1", 1, "NO-Hardhat", "HIGH", 0.9, "1926.100", "real row", "local"),
            )
            conn.commit()

        seed_violations.seed(days=15, rng_seed=42)

        with sqlite3.connect(tmp_db) as conn:
            real_count = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE source='local'"
            ).fetchone()[0]
        assert real_count == 1


# ─── load_compliance_series ─────────────────────────────────────────────────
class TestLoadComplianceSeries:
    def test_returns_dataframe_with_ds_and_y(self, tmp_db):
        seed_violations.seed(days=30, rng_seed=42)
        df = forecast.load_compliance_series("NO-Hardhat", db_path=tmp_db, days=30)
        assert isinstance(df, pd.DataFrame)
        assert {"ds", "y"} <= set(df.columns)
        assert len(df) >= 14

    def test_compliance_rate_in_unit_interval(self, tmp_db):
        seed_violations.seed(days=30, rng_seed=42)
        df = forecast.load_compliance_series("NO-Hardhat", db_path=tmp_db, days=30)
        assert df["y"].between(0.0, 1.0).all()


# ─── Prophet forecast ───────────────────────────────────────────────────────
class TestProphetForecast:
    def test_returns_dataframe_and_figure(self, tmp_db):
        seed_violations.seed(days=30, rng_seed=42)
        df, fig = forecast.forecast_compliance(
            "NO-Hardhat", db_path=tmp_db, history_days=30, horizon_days=7,
        )
        assert {"ds", "yhat", "yhat_lower", "yhat_upper"} <= set(df.columns)
        # History (≥14 rows from seed) + 7-day horizon
        assert len(df) >= 14
        assert hasattr(fig, "data")  # plotly.graph_objects.Figure

    def test_raises_on_insufficient_history(self, tmp_db):
        seed_violations.seed(days=10, rng_seed=42)
        with pytest.raises(ValueError, match="14 days"):
            forecast.forecast_compliance(
                "NO-Hardhat", db_path=tmp_db, history_days=10,
            )


# ─── SARIMA baseline ────────────────────────────────────────────────────────
class TestSarimaBaseline:
    def test_returns_horizon_rows(self, tmp_db):
        seed_violations.seed(days=30, rng_seed=42)
        fc_df, fig = forecast_baseline.forecast_compliance_sarima(
            "NO-Hardhat", db_path=tmp_db, history_days=30, horizon_days=7,
        )
        assert len(fc_df) == 7
        assert {"ds", "yhat", "yhat_lower", "yhat_upper"} <= set(fc_df.columns)
        assert hasattr(fig, "data")

    def test_yhat_clipped_to_unit_interval(self, tmp_db):
        seed_violations.seed(days=30, rng_seed=42)
        fc_df, _ = forecast_baseline.forecast_compliance_sarima(
            "NO-Hardhat", db_path=tmp_db, history_days=30, horizon_days=7,
        )
        assert fc_df["yhat"].between(0.0, 1.0).all()
        assert fc_df["yhat_lower"].between(0.0, 1.0).all()
        assert fc_df["yhat_upper"].between(0.0, 1.0).all()

    def test_sarima_raises_on_insufficient_history(self, tmp_db):
        seed_violations.seed(days=10, rng_seed=42)
        with pytest.raises(ValueError, match="14"):
            forecast_baseline.forecast_compliance_sarima(
                "NO-Hardhat", db_path=tmp_db, history_days=10,
            )


# ─── agent.tools.log_violation → SQLite ─────────────────────────────────────
class TestLogViolation:
    def test_writes_row(self, tmp_db):
        v = Violation(
            type="NO-Hardhat", risk_level="HIGH", confidence=0.92,
            bbox=(0, 0, 100, 100), person_bbox=(0, 0, 200, 200),
        )
        report = {
            "risk_level": "HIGH",
            "regulation_cited": "OSHA 29 CFR 1926.100(a)",
            "summary": "Worker missing hard hat",
        }
        vid = log_violation(v, report, source="unit-test")
        assert isinstance(vid, str) and len(vid) > 0

        with sqlite3.connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT violation_type, risk_level, regulation_cited, source "
                "FROM violations WHERE violation_id = ?",
                (vid,),
            ).fetchone()
        assert row == ("NO-Hardhat", "HIGH", "OSHA 29 CFR 1926.100(a)", "unit-test")

    def test_falls_back_to_violation_risk_when_report_lacks_it(self, tmp_db):
        v = Violation(
            type="NO-Mask", risk_level="MEDIUM", confidence=0.7,
            bbox=(0, 0, 10, 10), person_bbox=None,
        )
        # report missing risk_level → falls back to violation.risk_level
        vid = log_violation(v, {"summary": "no risk_level in report"}, source="fallback")
        with sqlite3.connect(tmp_db) as conn:
            risk = conn.execute(
                "SELECT risk_level FROM violations WHERE violation_id=?", (vid,)
            ).fetchone()[0]
        assert risk == "MEDIUM"
