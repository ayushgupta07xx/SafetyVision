"""Mocked unit tests for safetyvision_client (no network)."""
from __future__ import annotations

import safetyvision_client.client as client_mod
from safetyvision_client import AnalysisResult, SafetyVision, SafetyVisionError


class FakeResp:
    def __init__(self, status, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


RESP = {
    "inspection_id": "i1",
    "violations": [
        {
            "violation_id": "v1",
            "class": "NO-Hardhat",
            "confidence": 0.9,
            "bbox": [1, 2, 3, 4],
            "risk_level": "HIGH",
        }
    ],
    "annotated_image_b64": "",
    "gradcam_b64": None,
    "shap_chart_b64": None,
    "incident_report": {"regulation_cited": "OSHA 29 CFR 1910.135(a)(1)", "summary": "x"},
    "pdf_report_url": "https://x/report.pdf?token=t",
    "processing_time_ms": 12.3,
}


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("SAFETYVISION_API_KEY", raising=False)
    try:
        SafetyVision(base_url="https://api")
        raise AssertionError("expected SafetyVisionError")
    except SafetyVisionError:
        pass


def test_analyze_parses(monkeypatch):
    sv = SafetyVision(api_key="sv_x", base_url="https://api")
    monkeypatch.setattr(sv._session, "post", lambda *a, **k: FakeResp(200, RESP))
    r = sv.analyze(b"bytes", filename="t.jpg")
    assert isinstance(r, AnalysisResult)
    assert len(r.violations) == 1
    assert r.violations[0]["class"] == "NO-Hardhat"
    assert r.pdf_report_url.endswith("token=t")
    assert r.incident_report["regulation_cited"].startswith("OSHA")


def test_analyze_401_raises(monkeypatch):
    sv = SafetyVision(api_key="bad", base_url="https://api")
    monkeypatch.setattr(
        sv._session, "post",
        lambda *a, **k: FakeResp(401, {"detail": "Missing or invalid API key"}),
    )
    try:
        sv.analyze(b"bytes", filename="t.jpg")
        raise AssertionError("expected SafetyVisionError")
    except SafetyVisionError as e:
        assert "401" in str(e)


def test_save_pdf(tmp_path, monkeypatch):
    r = AnalysisResult(RESP)
    monkeypatch.setattr(
        client_mod.requests, "get",
        lambda url, timeout=None: FakeResp(200, None, content=b"%PDF-fake"),
    )
    out = r.save_pdf(str(tmp_path / "r.pdf"))
    assert out.read_bytes().startswith(b"%PDF-")


def test_save_pdf_missing_url():
    r = AnalysisResult({"violations": [], "pdf_report_url": None})
    try:
        r.save_pdf("/tmp/none.pdf")
        raise AssertionError("expected SafetyVisionError")
    except SafetyVisionError:
        pass
