"""SafetyVision API client.

Thin requests-based wrapper over the SafetyVision Lambda Function URL.
Auth is via an X-API-Key header (mint one from your account / the api-keys CLI).
"""
from __future__ import annotations

import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws"
DEFAULT_TIMEOUT = 130  # Lambda timeout is 120s; allow headroom for cold starts


class SafetyVisionError(Exception):
    """Raised on auth failures, HTTP errors, or malformed API responses."""


class AnalysisResult:
    """Wraps a single /analyze response."""

    def __init__(self, data: dict) -> None:
        self.raw = data

    @property
    def inspection_id(self) -> str | None:
        return self.raw.get("inspection_id")

    @property
    def violations(self) -> list:
        return self.raw.get("violations", [])

    @property
    def incident_report(self) -> dict | None:
        return self.raw.get("incident_report")

    @property
    def pdf_report_url(self) -> str | None:
        return self.raw.get("pdf_report_url")

    @property
    def annotated_image_b64(self) -> str | None:
        return self.raw.get("annotated_image_b64")

    @property
    def gradcam_b64(self) -> str | None:
        return self.raw.get("gradcam_b64")

    @property
    def shap_chart_b64(self) -> str | None:
        return self.raw.get("shap_chart_b64")

    @property
    def processing_time_ms(self) -> float | None:
        return self.raw.get("processing_time_ms")

    def save_pdf(self, path: str | Path, timeout: int = 60) -> Path:
        """Download the incident PDF (signed URL, no auth) to `path`."""
        url = self.pdf_report_url
        if not url:
            raise SafetyVisionError(
                "No pdf_report_url on this result "
                "(image had no violation, or report generation failed)."
            )
        resp = requests.get(url, timeout=timeout)
        if resp.status_code >= 400:
            raise SafetyVisionError(f"PDF download failed: HTTP {resp.status_code}")
        out = Path(path)
        out.write_bytes(resp.content)
        return out

    def save_annotated(self, path: str | Path) -> Path:
        """Decode the annotated PNG (bounding boxes) to `path`."""
        b64 = self.annotated_image_b64
        if not b64:
            raise SafetyVisionError("No annotated_image_b64 on this result.")
        out = Path(path)
        out.write_bytes(base64.b64decode(b64))
        return out

    def __repr__(self) -> str:
        has_pdf = "yes" if self.pdf_report_url else "no"
        return f"AnalysisResult(violations={len(self.violations)}, pdf={has_pdf})"


class SafetyVision:
    """Client for the SafetyVision PPE compliance API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key or os.getenv("SAFETYVISION_API_KEY")
        if not key:
            raise SafetyVisionError(
                "API key required: pass api_key=... or set SAFETYVISION_API_KEY."
            )
        self.base_url = (
            base_url or os.getenv("SAFETYVISION_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": key})

    @staticmethod
    def _check(resp) -> None:
        if resp.status_code < 400:
            return
        detail: object = None
        try:
            body = resp.json()
            detail = body.get("detail") if isinstance(body, dict) else body
        except Exception:  # noqa: BLE001
            detail = getattr(resp, "text", None)
        raise SafetyVisionError(f"API error {resp.status_code}: {detail}")

    def analyze(
        self, image: str | Path | bytes, *, filename: str | None = None
    ) -> AnalysisResult:
        """Analyze one image (path or raw bytes). Returns an AnalysisResult."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            data = p.read_bytes()
            name = filename or p.name
        else:
            data = image
            name = filename or "image.jpg"
        ctype = mimetypes.guess_type(name)[0] or "image/jpeg"
        resp = self._session.post(
            f"{self.base_url}/analyze",
            files={"image": (name, data, ctype)},
            timeout=self.timeout,
        )
        self._check(resp)
        return AnalysisResult(resp.json())

    def analyze_batch(self, images, *, max_workers: int = 4) -> list:
        """Analyze multiple images concurrently. Preserves input order."""
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self.analyze, list(images)))

    def history(self, *, limit: int = 50, offset: int = 0) -> list:
        """Paginated violation history for the authenticated user (newest first)."""
        resp = self._session.get(
            f"{self.base_url}/violations",
            params={"limit": limit, "offset": offset},
            timeout=self.timeout,
        )
        self._check(resp)
        return resp.json().get("violations", [])

    def forecast(self, violation_type: str, *, days: int = 30, horizon: int = 7) -> dict:
        """7-day Prophet compliance forecast for one violation type."""
        resp = self._session.get(
            f"{self.base_url}/forecast",
            params={"violation_type": violation_type, "days": days, "horizon": horizon},
            timeout=self.timeout,
        )
        self._check(resp)
        return resp.json()
