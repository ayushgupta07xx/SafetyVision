"""Shared fixtures for the SafetyVision test suite.

Goals:
    - Never hit HuggingFace Hub, Qdrant Cloud, Gemini, or Groq in CI.
    - Give tests a per-test SQLite DB so writes don't bleed across tests.
    - Provide an "uninitialized" PPEDetector for testing pure logic without
      loading the ONNX model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ─── SQLite DB redirected to tmp ────────────────────────────────────────────
@pytest.fixture
def tmp_db(tmp_path, monkeypatch) -> Path:
    """Point every module's SQLite path at a per-test tmp file.

    agent.tools.SQLITE_DB_PATH and analytics.seed_violations.DB_PATH are read
    at module-import time, so we monkeypatch the module attribute (not just
    the env var). analytics.forecast / forecast_baseline accept db_path as an
    explicit arg in tests, so they don't need patching.
    """
    db = tmp_path / "violations.db"
    import agent.tools
    import analytics.seed_violations
    monkeypatch.setattr(agent.tools, "SQLITE_DB_PATH", db)
    monkeypatch.setattr(analytics.seed_violations, "DB_PATH", db)
    return db


# ─── Sample image ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_bgr() -> np.ndarray:
    """Small synthetic BGR image (100×200×3, dtype=uint8) for fast tests."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:80, 40:160] = (100, 150, 200)
    return img


# ─── Uninitialized PPEDetector ──────────────────────────────────────────────
@pytest.fixture
def detector_instance():
    """PPEDetector instance with __init__ skipped (no HF download, no ONNX).

    Sets just enough state for the pure-logic methods (_iou,
    _detect_violations, _preprocess, _letterbox) to work. predict() and
    _postprocess will not work without the session — tests that need them
    mock self.session directly.
    """
    from core.detector import PPEDetector

    det = PPEDetector.__new__(PPEDetector)
    det.conf_threshold = 0.40
    det.iou_threshold = 0.45
    # Class IDs roughly mirror the trained model layout
    det.class_names = {
        0: "Person",
        1: "Hardhat",
        2: "NO-Hardhat",
        3: "Safety Vest",
        4: "NO-Safety Vest",
        5: "Mask",
        6: "NO-Mask",
    }
    return det
