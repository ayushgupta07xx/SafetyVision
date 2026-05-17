"""End-to-end smoke test with the REAL YOLO model from HF Hub.

Marked @pytest.mark.slow — excluded from CI by default. Run locally:
    pytest -m slow tests/test_smoke_real_model.py -v

Validates the brief-specified pipeline (Layer 1-7 wired together):
    image → detector → explainer → RAG → report → log

Detector and explainer use the real models from HF Hub
(ayushgupta7777/safetyvision-yolov8). RAG and Gemini are mocked because
they're external network deps already tested in their own files; the
value here is exercising the real ONNX inference + PyTorch grad / SHAP
paths that can't be unit-tested without a real model.

Cold run downloads ~15MB ONNX + 12MB PyTorch from HF Hub — first
execution takes ~60-90s; subsequent runs use HF's local cache and run
in ~20-30s. Reason this is marker-gated and not in CI: download time +
HF Hub availability would slow the gate from ~2min to ~7min and make
CI brittle to upstream HF outages.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_real_model_full_pipeline(tmp_db, monkeypatch):
    """Image → real detector → real explainer → mocked RAG/Gemini → SQLite log."""
    from agent import graph as agent_graph
    from agent import tools as agent_tools
    from core.detector import PPEDetector
    from core.explainer import explain_result

    # 1. Real detector — loads ONNX from HF Hub on first call (cached after)
    detector = PPEDetector()
    assert detector.session is not None
    assert detector.class_names, "ONNX metadata should expose class names"

    # 2. Synthetic-but-shaped input. We don't require any detection to land —
    # the goal is to exercise the full pipeline without exceptions.
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[100:540, 100:540] = (120, 140, 160)

    # 3. Real inference
    result = detector.predict(img)
    assert hasattr(result, "detections")
    assert hasattr(result, "violations")
    assert result.inference_ms > 0

    # 4. Real explainer (loads PyTorch + ultralytics on first call)
    # explain_result returns None if there are no detections; both branches
    # are valid for smoke purposes — we just need the call not to raise.
    explanation = explain_result(img, result)
    if explanation is not None:
        assert explanation.gradcam_b64
        assert explanation.shap_b64
        assert explanation.annotations_b64

    # 5. Agent run (only if we got a violation to feed it)
    if result.violations:
        monkeypatch.setattr(
            agent_tools, "retrieve_osha_context",
            lambda v, top_k=3: "Hard hats required (OSHA 29 CFR 1926.100).",
        )
        monkeypatch.setattr(
            agent_graph, "retrieve_osha_context",
            lambda v, top_k=3: "Hard hats required (OSHA 29 CFR 1926.100).",
        )
        fake_response = MagicMock(text=json.dumps({
            "violation_type": result.violations[0].type,
            "risk_level": "HIGH",
            "regulation_cited": "OSHA 29 CFR 1926.100",
            "summary": "Smoke test report",
            "corrective_actions": ["Issue hard hat immediately"],
            "image_observations": "Smoke test synthetic input.",
        }))
        fake_model = MagicMock()
        fake_model.generate_content.return_value = fake_response
        monkeypatch.setattr(
            agent_tools.genai, "GenerativeModel",
            lambda *a, **kw: fake_model,
        )

        out = agent_graph.run_agent(
            img, result.violations[0], source="smoke-real",
        )
        assert "incident_report" in out
        assert "violation_id" in out
        assert "osha_context" in out
