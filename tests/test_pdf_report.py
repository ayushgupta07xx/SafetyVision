"""Unit tests for core/pdf_report.build_incident_pdf (pure, no network)."""
from datetime import UTC, datetime

import cv2
import numpy as np

from core.pdf_report import build_incident_pdf


def _png() -> bytes:
    img = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (140, 100), (0, 0, 255), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


_REPORT = {
    "violation_type": "NO-Hardhat",
    "risk_level": "HIGH",
    "regulation_cited": "OSHA 29 CFR 1910.135(a)(1)",
    "regulation_text": "Each affected employee shall wear protective helmets.",
    "summary": "Worker without a hard hat in an active zone.",
    "corrective_actions": ["Stop work", "Issue a hard hat", "Re-brief crew"],
    "follow_up_timeline": "Immediate.",
    "confidence": 0.94,
    "image_observations": "Head uncovered, center-left of frame.",
}


def test_returns_valid_pdf_bytes():
    pdf = build_incident_pdf(
        _REPORT, _png(), report_id="vid-123",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC), subject="user-x",
    )
    assert isinstance(pdf, bytes)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1000


def test_tolerates_missing_fields():
    pdf = build_incident_pdf({"violation_type": "NO-Mask"}, _png(), report_id="v2")
    assert pdf[:5] == b"%PDF-"


def test_escapes_markup_from_llm_text():
    r = dict(_REPORT, summary="bad <script> & <b>x</b> input")
    pdf = build_incident_pdf(r, _png(), report_id="v3")
    assert pdf[:5] == b"%PDF-"
