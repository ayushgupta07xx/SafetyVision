"""End-to-end smoke test: violation → RAG → Gemini → SQLite log.

Every external boundary is mocked so this passes offline in CI:
    - retrieve_osha_context → canned context string
    - genai.GenerativeModel → returns canned JSON
    - SQLite path → per-test tmp file

Detector internals are tested in test_detector.py / test_violation.py — here
we construct a Violation directly and assert the agent stack wires correctly.
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def fake_gemini_report() -> dict:
    return {
        "violation_type": "NO-Hardhat",
        "risk_level": "HIGH",
        "regulation_cited": "OSHA 29 CFR 1926.100(a)",
        "regulation_text": "Employees in areas with possible head-injury danger...",
        "summary": "Worker observed without hard hat in active construction zone.",
        "corrective_actions": [
            "Provide approved hard hat immediately",
            "Document verbal warning",
            "Re-train crew on PPE policy within 48h",
        ],
        "follow_up_timeline": "Immediate",
        "confidence": 0.92,
        "image_observations": "Worker visible in center frame, no head protection.",
    }


def test_end_to_end_pipeline(tmp_db, fake_gemini_report, monkeypatch):
    """Full agent run with mocked RAG + Gemini, asserts report schema + persistence."""
    from agent import graph as agent_graph
    from agent import tools as agent_tools
    from core.detector import Violation

    # 1) Mock RAG — patch both modules because graph.py imported the name directly
    fake_context = (
        "[Context 1 — 29 CFR 1926.100]\n"
        "Employers must require hard hats where head-injury danger exists."
    )
    monkeypatch.setattr(
        agent_tools, "retrieve_osha_context",
        lambda violation_type, top_k=3: fake_context,
    )
    monkeypatch.setattr(
        agent_graph, "retrieve_osha_context",
        lambda violation_type, top_k=3: fake_context,
    )

    # 2) Mock Gemini — return our canned JSON in the SDK's response shape
    fake_response = MagicMock(text=json.dumps(fake_gemini_report))
    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response
    monkeypatch.setattr(
        agent_tools.genai, "GenerativeModel",
        lambda *args, **kwargs: fake_model,
    )

    # 3) Build the violation that would have come out of the detector
    image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    violation = Violation(
        type="NO-Hardhat", risk_level="HIGH", confidence=0.91,
        bbox=(100.0, 50.0, 300.0, 400.0),
        person_bbox=(80.0, 40.0, 320.0, 460.0),
    )

    # 4) Run the agent
    out = agent_graph.run_agent(image_bgr, violation, source="smoke-test")

    # 5) Top-level shape
    assert {"incident_report", "violation_id", "osha_context"} <= set(out.keys())

    # 6) Report schema (matches RESPONSE_SCHEMA required fields)
    report = out["incident_report"]
    for key in (
        "violation_type", "risk_level", "regulation_cited",
        "summary", "corrective_actions", "image_observations",
    ):
        assert key in report, f"missing required key: {key}"
    assert report["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(report["corrective_actions"], list)
    assert len(report["corrective_actions"]) > 0

    # 7) Violation ID and context plumbed through
    assert isinstance(out["violation_id"], str)
    assert len(out["violation_id"]) > 0
    assert fake_context in out["osha_context"]

    # 8) SQLite persistence
    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT violation_type, risk_level, source "
            "FROM violations WHERE violation_id = ?",
            (out["violation_id"],),
        ).fetchone()
    assert row == ("NO-Hardhat", "HIGH", "smoke-test")


def test_malformed_gemini_response_falls_back_to_stub(tmp_db, monkeypatch):
    """If Gemini returns non-JSON, generate_incident_report returns an error stub
    instead of raising — keeps the agent pipeline robust."""
    from agent import tools as agent_tools
    from core.detector import Violation

    fake_response = MagicMock(text="this is not JSON {{{")
    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response
    monkeypatch.setattr(
        agent_tools.genai, "GenerativeModel",
        lambda *args, **kwargs: fake_model,
    )

    image_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    violation = Violation(
        type="NO-Hardhat", risk_level="HIGH", confidence=0.9,
        bbox=(0.0, 0.0, 50.0, 50.0), person_bbox=None,
    )

    result = agent_tools.generate_incident_report(image_bgr, violation, "ctx")

    # Fallback stub keeps the required schema fields populated
    assert result["violation_type"] == "NO-Hardhat"
    assert result["risk_level"] == "HIGH"
    assert "Error" in result["regulation_cited"]
    assert "Could not" in result["image_observations"]
    assert isinstance(result["corrective_actions"], list)


def test_graph_compiles_without_errors():
    """Sanity: build_graph() produces a runnable compiled graph object."""
    from agent.graph import build_graph
    compiled = build_graph()
    # langgraph compiled graphs expose .invoke
    assert hasattr(compiled, "invoke")
