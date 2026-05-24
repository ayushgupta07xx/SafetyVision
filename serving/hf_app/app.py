"""SafetyVision Gradio app — Mode 1 (HF Spaces).

Image or short video clip → PPE violation detection (YOLOv8s ONNX, v2) →
GradCAM + SHAP explanations → OSHA-grounded incident report (Gemini Flash
multimodal via single-node LangGraph) → 7-day Prophet compliance forecast.

Run locally:    python serving/hf_app/app.py
Deploy:         push contents of serving/hf_app/ (plus core/, agent/, analytics/)
                to the HF Space repo, with app.py at the Space root.
"""

from __future__ import annotations

import base64
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Make project root importable so `core`, `agent`, `analytics` resolve when this
# file is run as `python serving/hf_app/app.py` locally. On HF Spaces the repo
# is flat (app.py at root) so this no-ops there.
_HERE = Path(__file__).resolve()
if len(_HERE.parents) >= 3:
    _REPO_ROOT = _HERE.parents[2]
    if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from core.detector import PPEDetector, draw_annotations  # noqa: E402
from core.explainer import explain_result  # noqa: E402
from agent.graph import run_agent  # noqa: E402
from agent.tools import SQLITE_DB_PATH  # noqa: E402
from analytics.forecast import forecast_compliance  # noqa: E402
from analytics.seed_violations import seed  # noqa: E402

# ---------------------------------------------------------------------------
# Logging + constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("safetyvision.app")

MAX_VIDEO_SECONDS = 30
SEED_DAYS = 30
MIN_DB_ROWS_BEFORE_SEED = 50  # below this, re-seed synthetic baseline


# ---------------------------------------------------------------------------
# Bootstrap: seed synthetic violation history on cold start so the forecast
# tab renders immediately. /tmp on HF Spaces is ephemeral so this runs every
# container restart. Real uploads layer on top of the synthetic baseline.
# ---------------------------------------------------------------------------
def _bootstrap_db() -> list[str]:
    """Seed DB if sparse, return distinct violation_types for forecast dropdown."""
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM violations")
            row_count = int(cur.fetchone()[0])
        except sqlite3.OperationalError:
            row_count = 0
        conn.close()
    except sqlite3.OperationalError:
        row_count = 0

    if row_count < MIN_DB_ROWS_BEFORE_SEED:
        log.info(
            "DB sparse (%d rows) — seeding %d days of synthetic history",
            row_count,
            SEED_DAYS,
        )
        seed(days=SEED_DAYS, rng_seed=42)

    types: list[str] = []
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cur = conn.execute(
            "SELECT DISTINCT violation_type FROM violations ORDER BY violation_type"
        )
        types = [r[0] for r in cur.fetchall()]
        conn.close()
    except sqlite3.OperationalError as e:
        log.warning("Failed reading violation types from DB: %s", e)

    log.info("Bootstrap done. Forecast types available: %s", types)
    return types


VIOLATION_TYPES = _bootstrap_db()
DETECTOR = PPEDetector.get()  # warm-load ONNX at startup (slow cold start, fast inference)
log.info("Detector loaded; app ready")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b64_png_to_rgb(b64: str | None) -> np.ndarray | None:
    """Decode base64-encoded PNG bytes to an RGB ndarray (Gradio expects RGB)."""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        log.exception("Failed to decode base64 image")
        return None


def _format_report_md(
    n_violations: int,
    n_detections: int,
    agent_out: dict | None,
    worst_t: float | None,
) -> str:
    if n_detections == 0:
        return (
            "## ⚠️ Nothing detected in this frame\n\n"
            "The detector found no persons or PPE here. This is a known model "
            "limitation — best results come from clear worksite imagery. Very "
            "small/distant workers, low light, glare, and fast motion are still "
            "missed (v2 augmentation reduced but didn't eliminate this). "
            "See the [model card failure modes]"
            "(https://huggingface.co/ayushgupta7777/safetyvision-yolov8) for details."
        )

    if n_violations == 0:
        return (
            f"## ✅ No violations detected  ·  {n_detections} detection(s) — "
            "see explanations below for what the model attended to"
        )

    header = f"## ⚠️ {n_violations} violation(s) detected"
    if worst_t is not None:
        header += f"  ·  worst frame at t={worst_t:.1f}s"

    if agent_out is None:
        return header
    report = agent_out.get("incident_report")
    if report is None:
        return header
    if isinstance(report, dict):
        body = json.dumps(report, indent=2, ensure_ascii=False)
        return header + "\n\n### 📋 Incident Report\n```json\n" + body + "\n```"
    return header + f"\n\n### 📋 Incident Report\n\n{report}"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def _explain_and_report(
    image_bgr: np.ndarray,
    result,
    source: str,
    progress: gr.Progress,
    base: float,
):
    """Run explainer if any detections exist (shows what model attended to).
    Run agent only if there are violations (no incident to report otherwise).
    Returns (explanation, agent_out)."""
    explanation = None
    agent_out = None

    if not result.detections:
        return None, None

    progress(base + 0.10, desc="Generating GradCAM + SHAP explanations...")
    explanation = explain_result(image_bgr, result)

    if not result.violations:
        return explanation, None

    progress(base + 0.40, desc="Retrieving OSHA + writing incident report (Gemini)...")
    primary = result.violations[0]
    try:
        agent_out = run_agent(image_bgr=image_bgr, violation=primary, source=source)
    except Exception as e:  # noqa: BLE001
        log.exception("Agent run failed")
        agent_out = {
            "incident_report": {"error": str(e)},
            "violation_id": None,
            "osha_context": "",
        }
    return explanation, agent_out


def _analyze_image(image_rgb: np.ndarray, progress: gr.Progress):
    progress(0.05, desc="Detecting PPE violations...")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = DETECTOR.predict(image_bgr)
    explanation, agent_out = _explain_and_report(
        image_bgr, result, "hf_spaces", progress, base=0.10
    )
    return result, explanation, agent_out, image_bgr, None


def _analyze_video(video_path: str, progress: gr.Progress):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0.0

    if duration > MAX_VIDEO_SECONDS + 1:
        cap.release()
        raise gr.Error(f"Video is {duration:.0f}s; max allowed is {MAX_VIDEO_SECONDS}s.")

    progress(0.05, desc=f"Sampling video at 1 fps ({duration:.0f}s)...")
    step = max(int(round(fps)), 1)
    sampled: list[tuple[float, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            sampled.append((idx / fps if fps else float(idx), frame))
        idx += 1
    cap.release()

    if not sampled:
        raise gr.Error("Could not extract any frames from the video.")

    progress(0.15, desc=f"Running detector on {len(sampled)} sampled frames...")
    per_frame = []
    for t, frame in sampled:
        r = DETECTOR.predict(frame)
        per_frame.append((t, frame, r))

    # Pick frame with most violations; tiebreak on sum of detection confidence.
    worst_t, worst_frame, worst_result = max(
        per_frame,
        key=lambda x: (len(x[2].violations), sum(d.confidence for d in x[2].detections)),
    )

    # No early return on clean worst frame — explainer can still attribute on
    # any positive detections (vests, hardhats) found in the chosen frame.
    explanation, agent_out = _explain_and_report(
        worst_frame, worst_result, "hf_spaces_video", progress, base=0.35
    )
    return worst_result, explanation, agent_out, worst_frame, worst_t


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------
def analyze(image_rgb, video_path, progress=gr.Progress()):
    if image_rgb is None and not video_path:
        raise gr.Error("Upload an image or a short video clip first.")
    if image_rgb is not None and video_path:
        raise gr.Error("Pick one: image OR video, not both.")

    if image_rgb is not None:
        result, explanation, agent_out, frame_bgr, worst_t = _analyze_image(image_rgb, progress)
    else:
        result, explanation, agent_out, frame_bgr, worst_t = _analyze_video(video_path, progress)

    progress(0.95, desc="Rendering outputs...")
    annotated_bgr = draw_annotations(frame_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    gradcam_rgb = _b64_png_to_rgb(explanation.gradcam_b64) if explanation else None
    shap_rgb = _b64_png_to_rgb(explanation.shap_b64) if explanation else None

    report_md = _format_report_md(
        len(result.violations), len(result.detections), agent_out, worst_t
    )
    return annotated_rgb, gradcam_rgb, shap_rgb, report_md


def build_forecast(violation_type: str):
    if not violation_type:
        raise gr.Error("Pick a violation type.")
    try:
        _, fig, summary = forecast_compliance(violation_type)
        return summary, fig
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("Forecast failed")
        raise gr.Error(f"Forecast unavailable: {e}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_HEADER = """
# 🦺 SafetyVision
### AI-Powered PPE Compliance Monitor
Open-source · Free · No signup · Self-hostable

Upload a worksite photo or short clip → bounding boxes, GradCAM heatmap,
SHAP attribution, OSHA-grounded incident report (Gemini Flash multimodal),
and a 7-day compliance forecast (Prophet).
"""

_ABOUT = """
## Stack
- **Detection:** YOLOv8s (v2, ONNX, CPU) fine-tuned on 80k+ PPE images with Albumentations augmentation.
  [Model card →](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)
- **Explainability:** GradCAM on the SPPF layer + SHAP per-pixel attribution at 320×320.
- **RAG:** Qdrant Cloud + BAAI/bge-small-en-v1.5 over 15 OSHA standards (29 CFR 1910 + 29 CFR 1926).
- **Report generation:** Gemini Flash multimodal — receives annotated image + violation metadata + OSHA chunks, returns structured JSON.
- **Orchestration:** Single-node LangGraph (retrieve → report → log).
- **Forecasting:** Prophet (primary, ADR-008) with SARIMA(1,1,1)(1,1,1,7) baseline.

## Held-out test metrics (deployed config: ONNX @ 896)
- mAP@50: **0.763**  ·  mAP@50-95: **0.482**
- Strongest: Fall-Detected (0.956), Hardhat (0.936), Safety Vest (0.891).
- Weakest: NO-Safety Vest (0.382); Mask / NO-Mask (~0.57-0.59).
- A/B test 1 (RAG vs no-RAG report quality): RAG wins, Cohen's d=0.65, p=0.0197
- A/B test 2 (confidence threshold 0.40 vs 0.55): 0.40 wins, McNemar p=4×10⁻⁵

## Honest limitations
- **Pre-screening tool** — does NOT replace human safety judgment.
- v2 augmentation reduced the v1 frontal-pose bias, but small/distant workers, low light, glare, and fast motion still degrade accuracy.
- Free-tier Gemini quota is ~20 reports/day per model on this Space (resets midnight Pacific).
  One report per analysis to preserve quota.
- Video sampled at 1 fps; sub-second events can be missed.

## Mode 2 — AWS Lambda
Serverless image inference via Lambda Function URL. Image-only (6MB payload cap).
See the GitHub repo for the `terraform apply` walkthrough.

---
License: **AGPL-3.0** (inherited from Ultralytics YOLOv8).
"""

with gr.Blocks(title="SafetyVision — AI Workplace Safety Monitor") as demo:
    gr.Markdown(_HEADER)

    with gr.Tab("🔍 Analyze"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(
                    label="Image (JPG / PNG)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=320,
                )
                vid_in = gr.Video(
                    label=f"Video (MP4 / MOV, ≤{MAX_VIDEO_SECONDS}s)",
                    sources=["upload"],
                    height=240,
                )
                analyze_btn = gr.Button("Analyze for Violations", variant="primary", size="lg")
                gr.Markdown(
                    "_Free-tier Gemini quota is ~20 reports/day. If you see a quota error, try again later._"
                )

        report_out = gr.Markdown()
        with gr.Row():
            annotated_out = gr.Image(label="Annotated (bounding boxes)", height=320)
            gradcam_out = gr.Image(label="GradCAM Heatmap", height=320)
        shap_out = gr.Image(label="SHAP Per-pixel Attribution", height=320)

        analyze_btn.click(
            fn=analyze,
            inputs=[img_in, vid_in],
            outputs=[annotated_out, gradcam_out, shap_out, report_out],
        )

    with gr.Tab("📈 7-Day Compliance Forecast"):
        gr.Markdown(
            "**Prophet** 7-day forecast on the last 30 days of violation history. "
            "ADR-008 documents the choice (Prophet won on 2 of 3 violation types vs SARIMA baseline). "
            "On a fresh container the history is synthetic seed data; real uploads layer on top."
        )
        vtype_in = gr.Dropdown(
            choices=VIOLATION_TYPES if VIOLATION_TYPES else [],
            value=VIOLATION_TYPES[0] if VIOLATION_TYPES else None,
            label="Violation type",
        )
        forecast_btn = gr.Button("Render forecast", variant="primary")
        forecast_summary = gr.Markdown()
        forecast_out = gr.Plot()
        forecast_btn.click(
            fn=build_forecast, inputs=[vtype_in],
            outputs=[forecast_summary, forecast_out],
        )

    with gr.Tab("ℹ️ About"):
        gr.Markdown(_ABOUT)


if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", theme=gr.themes.Soft())
