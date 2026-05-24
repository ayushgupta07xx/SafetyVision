"""SafetyVision Mode-2 Lambda handler -- FastAPI + Mangum.

Full pipeline (option C, ADR-014): image -> YOLOv8s ONNX detection ->
GradCAM/SHAP explanation -> OSHA-grounded incident report (Gemini Flash
multimodal via single-node LangGraph) -> full JSON response.

Endpoints:
    POST /analyze     multipart image (<=6MB Function URL cap) -> full report JSON
    GET  /violations  authenticated user's violation history (Supabase, paginated)
    GET  /forecast    7-day Prophet compliance forecast for a violation type
    GET  /docs        Swagger UI (FastAPI auto)
    GET  /redoc       ReDoc (FastAPI auto)
    GET  /health      liveness probe

Auth: /analyze, /violations, /forecast require an X-API-Key header. Keys are
provisioned via core.apikeys (Supabase api_keys table) and resolved to a user_id
per request. /health, /, /docs, /redoc are public.

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
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from mangum import Mangum

from agent.graph import run_agent
from core.apikeys import resolve_user_id
from core.detector import VIOLATION_CLASSES, PPEDetector, draw_annotations
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
        "Spaces demo. AI-assisted -- human safety-officer review required."
    ),
)


def require_user_id(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),  # noqa: B008
) -> str:
    """Resolve the X-API-Key header to a user_id, or 401 if missing/invalid."""
    uid = resolve_user_id(x_api_key)
    if uid is None:
        raise HTTPException(
            status_code=401, detail="Missing or invalid API key (X-API-Key)."
        )
    return uid


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
        "endpoints": [
            "/analyze (POST)",
            "/violations (GET)",
            "/forecast (GET)",
            "/docs",
            "/redoc",
            "/health",
        ],
        "auth": "X-API-Key header required for /analyze, /violations, /forecast",
        "note": "Image-only (6MB cap). Video -> HF Spaces demo.",
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),  # noqa: B008
    user_id: str = Depends(require_user_id),  # noqa: B008
) -> dict:
    """Run the full detection + explanation + incident-report pipeline.

    Authenticated via X-API-Key; the resolved user_id owns the persisted
    inspection + violations (Mode 3 user history) and the response references
    the real Supabase rows.
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

    # 2) Explanation + 3) incident report -- only when there's a violation
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
        except Exception as exc:  # noqa: BLE001 -- surface, don't 500 the request
            logger.exception("agent report failed")
            incident_report = {"error": str(exc)}

    # 4) Persist to Supabase user history (authenticated via API key).
    try:
        from core import supabase_db  # lazy -- keeps cold start lean
        inspection_id, vids = supabase_db.persist_inspection_from_result(
            user_id, result, incident_report, source="api",
        )
    except Exception:  # noqa: BLE001 -- a log-write must never fail the request
        logger.exception("supabase persist failed")
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


@app.get("/violations")
def violations_endpoint(
    user_id: str = Depends(require_user_id),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=200),  # noqa: B008
    offset: int = Query(default=0, ge=0),  # noqa: B008
) -> dict:
    """Paginated violation history for the authenticated user (newest first)."""
    from core import supabase_db  # lazy (already loaded via apikeys; cached)
    rows = supabase_db.fetch_violations(user_id, limit=limit, offset=offset)
    return {"count": len(rows), "limit": limit, "offset": offset, "violations": rows}


@app.get("/forecast")
def forecast_endpoint(
    user_id: str = Depends(require_user_id),  # noqa: B008
    violation_type: str = Query(...),  # noqa: B008
    days: int = Query(default=30, ge=14, le=90),  # noqa: B008
    horizon: int = Query(default=7, ge=1, le=30),  # noqa: B008
) -> dict:
    """7-day Prophet compliance forecast for one violation type (Supabase per-user)."""
    if violation_type not in VIOLATION_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown violation_type. Valid values: {sorted(VIOLATION_CLASSES)}",
        )
    from analytics.forecast import forecast_json  # lazy -- Prophet is heavy
    try:
        return forecast_json(
            violation_type,
            history_days=days,
            horizon_days=horizon,
            source="supabase",
            user_id=user_id,
        )
    except ValueError as exc:  # not enough history to fit
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Mangum adapts the ASGI app to Lambda (Function URL / API GW v2 events).
handler = Mangum(app, lifespan="off")
