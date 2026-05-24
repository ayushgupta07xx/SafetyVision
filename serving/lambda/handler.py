"""SafetyVision Mode-2 Lambda handler — FastAPI + Mangum.

Full pipeline (option C, ADR-014): image -> YOLOv8s ONNX detection ->
GradCAM/SHAP explanation -> OSHA-grounded incident report (Gemini Flash
multimodal via single-node LangGraph) -> full JSON response.

Endpoints (Phase 1):
    POST /analyze   multipart image (<=6MB Function URL cap) -> full report JSON
    GET  /docs      Swagger UI (FastAPI auto)
    GET  /redoc     ReDoc (FastAPI auto)
    GET  /health    liveness probe

Image-only by design: Lambda Function URLs cap synchronous payloads at 6MB,
which rules out video. Video stays on Mode 1 (HF Spaces). See README.

Models are baked into the container image (HF cache pre-warmed at build,
HF_HUB_OFFLINE=1 at runtime) so cold start does no network for weights.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from mangum import Mangum

from agent.graph import run_agent
from core.detector import PPEDetector, draw_annotations
from core.explainer import explain_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safetyvision.lambda")

MAX_BYTES = 6 * 1024 * 1024  # Lambda Function URL synchronous payload cap

app = FastAPI(
    title="SafetyVision API",
    version="2.0",
    description=(
        "Open-source PPE compliance monitor. Upload a worksite image to get "
        "PPE violation detections with bounding boxes, a GradCAM heatmap, a "
        "SHAP attribution map, and an OSHA-grounded incident report written by "
        "Gemini Flash. Image-only (6MB cap); video analysis lives on the HF "
        "Spaces demo. AI-assisted — human safety-officer review required."
    ),
)


def _png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "service": "SafetyVision API",
        "version": "2.0",
        "endpoints": ["/analyze (POST)", "/docs", "/redoc", "/health"],
        "note": "Image-only (6MB cap). Video -> HF Spaces demo.",
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),  # noqa: B008
    user_id: str | None = Form(default=None),  # noqa: B008 — Layer 10: replace with X-API-Key resolution
) -> dict:
    """Run the full detection + explanation + incident-report pipeline.

    When user_id is supplied, the inspection + violations persist to Supabase
    (Mode 3 user history) and the response references the real rows. Layer 10
    swaps the user_id form field for X-API-Key resolution — this call is unchanged.
    """
    t0 = time.perf_counter()

    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 6MB Lambda Function URL limit. "
            "Use the HF Spaces demo for larger files / video.",
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # 1) Detection (singleton reused across warm invocations)
    detector = PPEDetector.get()
    result = detector.predict(bgr)

    annotated_b64 = _png_b64(draw_annotations(bgr, result))

    # 2) Explanation + 3) incident report — only when there's a violation
    gradcam_b64: str | None = None
    shap_b64: str | None = None
    incident_report: dict | None = None

    if result.violations:
        explanation = explain_result(bgr, result)
        if explanation is not None:
            gradcam_b64 = explanation.gradcam_b64
            shap_b64 = explanation.shap_b64

        primary = max(result.violations, key=lambda v: v.confidence)
        try:
            agent_out = run_agent(image_bgr=bgr, violation=primary, source="api")
            incident_report = agent_out["incident_report"]
        except Exception as exc:  # noqa: BLE001 — surface, don't 500 the request
            logger.exception("agent report failed")
            incident_report = {"error": str(exc)}

    # 4) Persist to Supabase user history when authenticated (Mode 3 / API key).
    if user_id:
        try:
            from core import supabase_db  # lazy — keeps cold start lean
            inspection_id, vids = supabase_db.persist_inspection_from_result(
                user_id, result, incident_report, source="api",
            )
        except Exception:  # noqa: BLE001 — a log-write must never fail the request
            logger.exception("supabase persist failed")
            inspection_id = str(uuid.uuid4())
            vids = [str(uuid.uuid4()) for _ in result.violations]
    else:
        inspection_id = str(uuid.uuid4())
        vids = [str(uuid.uuid4()) for _ in result.violations]

    violations_json = [
        {
            "violation_id": vid,
            "class": v.type,
            "confidence": round(v.confidence, 4),
            "bbox": [round(c, 1) for c in v.bbox],
            "risk_level": v.risk_level,
        }
        for v, vid in zip(result.violations, vids, strict=True)
    ]

    return {
        "inspection_id": inspection_id,
        "violations": violations_json,
        "annotated_image_b64": annotated_b64,
        "gradcam_b64": gradcam_b64,
        "shap_chart_b64": shap_b64,
        "incident_report": incident_report,
        "pdf_report_url": None,  # Phase 2 (Supabase Storage signed URL)
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# Mangum adapts the ASGI app to Lambda (Function URL / API GW v2 events).
handler = Mangum(app, lifespan="off")
