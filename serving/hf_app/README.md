---
title: SafetyVision
emoji: 🦺
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: agpl-3.0
short_description: AI PPE compliance monitor with OSHA-grounded reports
---
# 🦺 SafetyVision
AI-powered PPE compliance monitor. Upload a worksite photo or short video clip →
bounding boxes, GradCAM heatmap, SHAP attribution, OSHA-grounded incident report
(Gemini Flash multimodal), and a 7-day compliance forecast (Prophet).

- **Model:** YOLOv8s (v2), 13-class, fine-tuned on 80k+ PPE images with Albumentations augmentation
- **Model card:** [ayushgupta7777/safetyvision-yolov8](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)
- **GitHub:** https://github.com/ayushgupta07xx/SafetyVision
- **License:** AGPL-3.0 (inherited from Ultralytics YOLOv8)

## Required Space secrets
Set these in **Settings → Variables and secrets**:

| Secret | Purpose |
|---|---|
| `GOOGLE_AI_STUDIO_KEY` | Gemini Flash multimodal — incident report generation |
| `QDRANT_URL` | OSHA RAG vector store |
| `QDRANT_API_KEY` | OSHA RAG vector store |

Set `SV_ONNX_FILENAME=v2/best_896.onnx` as a **Variable** to serve the 896 export.
`HF_TOKEN` is only required if the model repo is private (it isn't).

## Held-out test metrics (deployed config: ONNX @ 896)
- mAP@0.5: **0.763**  ·  mAP@0.5:0.95: **0.482**  (12,080 instances; full per-class table → model card)
- Strongest: Fall-Detected 0.956, Hardhat 0.936, Safety Vest 0.891.
- Weakest: NO-Safety Vest 0.382; Mask / NO-Mask ~0.57-0.59.

## Honest limitations
- Pre-screening tool — does NOT replace human safety judgment.
- v2 augmentation reduced the v1 frontal-pose bias but small/distant workers, low light, glare, and fast motion still degrade accuracy.
- Free-tier Gemini quota is ~20 reports/day per model (resets midnight Pacific).
- Video is sampled at 1 fps; sub-second events can be missed.
- Mode 2 (AWS Lambda) is image-only — 6MB payload cap.
