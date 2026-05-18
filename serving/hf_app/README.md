---
title: SafetyVision
emoji: 🦺
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: agpl-3.0
short_description: AI PPE compliance monitor with OSHA-grounded incident reports
---

# 🦺 SafetyVision

AI-powered PPE compliance monitor. Upload a worksite photo or short video clip →
bounding boxes, GradCAM heatmap, SHAP attribution, OSHA-grounded incident report
(Gemini Flash multimodal), and a 7-day compliance forecast (Prophet).

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

`HF_TOKEN` is only required if the model repo is private (it isn't).

## Honest limitations

- Pre-screening tool — does NOT replace human safety judgment.
- Misses side-view, back-view, partially-occluded workers (training data is heavily frontal).
- Free-tier Gemini quota is ~20 reports/day per model (resets midnight Pacific).
- Video is sampled at 1 fps; sub-second events can be missed.
- Mode 2 (AWS Lambda) is image-only — 6MB payload cap.
