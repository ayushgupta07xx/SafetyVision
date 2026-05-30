<div align="center">

# 🦺 SafetyVision

### Open-source AI workplace safety monitor — detect PPE violations, explain every decision, forecast compliance. Free, self-hostable, $0 to run.

<!-- Core ML -->
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-0B23A0)
![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-CC0066)

<!-- Explainability, LLM, RAG -->
![GradCAM](https://img.shields.io/badge/GradCAM-6E40C9)
![SHAP](https://img.shields.io/badge/SHAP-FF4B4B)
![Gemini](https://img.shields.io/badge/Gemini_Flash-8E75B2?logo=googlegemini&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?logo=qdrant&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-BGE_embeddings-2EA043)

<!-- Forecasting, experiment tracking -->
![Prophet](https://img.shields.io/badge/Prophet-005AC1)
![SARIMA](https://img.shields.io/badge/statsmodels_SARIMA-6C757D)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=black)

<!-- Frontend -->
![Next.js](https://img.shields.io/badge/Next.js_14-000000?logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Tailwind](https://img.shields.io/badge/TailwindCSS-06B6D4?logo=tailwindcss&logoColor=white)
![shadcn/ui](https://img.shields.io/badge/shadcn%2Fui-000000?logo=shadcnui&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?logo=vercel&logoColor=white)

<!-- Backend, cloud -->
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/AWS_Lambda-FF9900?logo=awslambda&logoColor=white)
![S3](https://img.shields.io/badge/Amazon_S3-569A31?logo=amazons3&logoColor=white)
![DynamoDB](https://img.shields.io/badge/DynamoDB-4053D6?logo=amazondynamodb&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-7B42BC?logo=terraform&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?logo=supabase&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)

<!-- Project -->
![PyPI](https://img.shields.io/badge/PyPI-safetyvision--client-3775A9?logo=pypi&logoColor=white)
![mAP](https://img.shields.io/badge/mAP@50-0.763-2EA043)
![Tests](https://img.shields.io/badge/tests-142_passing-2EA043)
![Runtime](https://img.shields.io/badge/runtime_cost-%240-2EA043)
![License](https://img.shields.io/badge/License-AGPL--3.0-A42E2B)

**[🌐 Live app](https://safetyvision.vercel.app)** · **[🤗 Open-source demo](https://huggingface.co/spaces/ayushgupta7777/safetyvision)** · **[🔌 API docs](https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws/docs)** · **[📄 Model card](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)**

</div>

---

Missing hard hats, no high-vis vest, no fall harness at height — the violations human eyes miss when scanning hundreds of frames. SafetyVision is an open-source pre-screening tool that surfaces them for human review, with **explainable** detections, **OSHA-grounded** incident reports, and forward-looking **compliance forecasts**. It replaces commercial tools that charge $500-$2,000/month — and the entire runtime is free-tier, $0.

<div align="center">

<img src="docs/demo.gif" alt="SafetyVision demo — upload, detection, GradCAM, OSHA report, PDF" width="820">

[![Watch the full 3-min walkthrough demo](https://img.shields.io/badge/%E2%96%B6%20Watch%20the%20full%20~3--min%20walkthrough%20demo-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://youtu.be/I9FxbBiZ18c)

</div>

---

## Try it — four ways

| | What | Link |
|---|---|---|
| 🌐 **Web app** | The full product — upload, history, dashboard, forecasts, PDF reports, account + API keys | **[safetyvision.vercel.app](https://safetyvision.vercel.app)** |
| 🤗 **Open-source demo** | No signup, image *or* short video — try it instantly in the browser | **[HF Spaces](https://huggingface.co/spaces/ayushgupta7777/safetyvision)** |
| 🔌 **Production API** | Documented REST API with Swagger, Python SDK + CLI on PyPI | **[`/docs`](https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws/docs)** |
| 🛠️ **Deploy your own** | One command — `cd infra/aws && terraform apply` | [GitHub](https://github.com/ayushgupta07xx/SafetyVision) |

---

## What it does

Upload a worksite photo (or short clip on the demo) and SafetyVision:

- **Detects PPE compliance** — finds each worker and checks for hard hats, safety vests, masks, gloves, and fall harnesses, flagging missing-PPE violations in red, ranked by risk level. Fine-tuned **YOLOv8s** exported to **ONNX**, running on **CPU** (no GPU needed).
- **Explains every decision** — a **GradCAM** heatmap shows where the model looked, plus **SHAP** per-pixel attribution. Auditable, not a black box.
- **Writes OSHA-grounded incident reports** — a **multimodal Gemini Flash** model reads the annotated image alongside the actual OSHA regulation (retrieved via **Qdrant + BGE** RAG over 29 CFR 1910 & 1926) and cites the real CFR number.
- **Exports an audit-ready PDF** — one click per violation: annotated image, citation, corrective actions, explainability section.
- **Forecasts 7-day compliance** — a **Prophet** model (benchmarked against a SARIMA baseline) projects the trend per violation type.
- **Remembers your history** — every inspection saved per user (Supabase Postgres + row-level security), surfaced in a history table and a roll-up dashboard.

## Headline metrics

YOLOv8s v2, held-out test set (deployed config: ONNX @ 896):

| Metric | Value |
|---|---|
| **mAP@50** | **0.763** |
| **mAP@50:95** | **0.482** |
| Warm CPU inference (AWS Lambda) | ~500-800 ms / image |

Strongest classes: Fall-Detected (0.956), Hardhat (0.936), Safety Vest (0.891). Weakest: NO-Safety Vest (0.382). The v2 retrain (YOLOv8s + Albumentations on 80k+ images) targeted **0.78** and landed at **0.763** — an honest near-miss, documented in full in the **[model card](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)**.

**Statistical validation (A/B tests):**
- Incident-report quality, RAG vs no-RAG: RAG wins, Cohen's d = 0.65, p = 0.0197 (paired t-test, N=16)
- Confidence threshold 0.40 vs 0.55: 0.40 wins, McNemar p = 4x10^-5 (N=200)

## Architecture

```mermaid
flowchart TD
    U([Image / short video clip]) --> DET[YOLOv8s ONNX detection<br/>boxes + risk-ranked violations]
    DET --> EXP[GradCAM + SHAP<br/>explainability]
    EXP --> LG{Single-node LangGraph}
    LG --> RAG[OSHA RAG<br/>Qdrant + BGE]
    RAG --> GEM[Gemini Flash multimodal<br/>incident report]
    GEM --> PDF[PDF report]
    GEM --> LOG[(Supabase history<br/>+ DynamoDB audit)]
    LOG --> FC[Prophet 7-day forecast<br/>SARIMA baseline]
    FC --> OUT([Vercel app / Gradio demo / Lambda API])

    classDef io fill:#1f9e84,stroke:#0f7a66,color:#04201a;
    classDef core fill:#1a1815,stroke:#2b2824,color:#f3f1ea;
    classDef store fill:#173029,stroke:#0f7a66,color:#7fe0cb;
    class U,OUT io;
    class DET,EXP,LG,RAG,GEM,PDF,FC core;
    class LOG store;
```

The same core (`core/`, `agent/`, `analytics/`) powers all three surfaces — the only thing that changes per surface is the entry point. Full diagram: [`docs/architecture.md`](docs/architecture.md).

## The three deployment surfaces

- **Mode 3 — Next.js + Vercel (primary):** the product. Next.js 14 + Tailwind + shadcn/ui, Supabase auth (email + Google OAuth), per-user history, forecast dashboard, PDF downloads, API-key management. → [safetyvision.vercel.app](https://safetyvision.vercel.app)
- **Mode 1 — Hugging Face Spaces (open-source demo):** a free Gradio app, no signup, image **or** short video (≤30s). → [HF Spaces](https://huggingface.co/spaces/ayushgupta7777/safetyvision)
- **Mode 2 — AWS Lambda Function URL (production API):** serverless ONNX inference behind a free-forever HTTPS endpoint, API-key auth, Swagger/Redoc docs. **Image-only** (Lambda's 6MB on-wire cap ≈ 4MB raw after base64; video stays on Modes 1 & 3). → [`/docs`](https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws/docs)

## Key design decisions & tradeoffs

Each decision lists the choice, the reason, and what was traded. ADRs in [`docs/decisions.md`](docs/decisions.md).

### Lambda Function URLs over API Gateway (ADR-006)
**Choice.** A Lambda Function URL, not API Gateway. **Why.** Function URLs are free *forever* (API Gateway's free tier expires after 12 months), with a single clean endpoint. API-key auth and rate-limiting are done at the handler level against Supabase. **Trade.** No built-in usage plans / request transformations — not needed for a single `/analyze` endpoint, and API Gateway stays an evaluated, documented alternative.

### YOLOv8s v2 — and an honest 0.763
**Choice.** Upgraded v1 (YOLOv8n, 0.701) to v2 (YOLOv8s + Albumentations on 80k+ images), targeting mAP@50 ≥ 0.78. It landed at **0.763**. **Why ship it anyway.** It's a real improvement with documented per-class gains, and the small variant still fits Lambda's CPU memory budget. **Trade.** A near-miss on the target — stated plainly in the model card rather than buried, because honest metrics are the point.

### Image-only Mode 2 (the 4MB reality)
**Choice.** The production API is image-only; video lives on Modes 1 & 3. **Why.** Lambda Function URLs cap payloads at 6MB on the wire ≈ **4MB raw** after base64 inflation. Responses (annotated + GradCAM + SHAP images) are JPEG q85 and resolution-capped to stay under the same ceiling. **Trade.** No video through the API — a clean documented constraint rather than a flaky 413.

### GradCAM **and** SHAP, not one
**Choice.** Both explainers on every detection. **Why.** GradCAM answers "where did it look" (spatial, intuitive); SHAP answers "which pixels moved the score" (attribution). Together they make a detection trustworthy to a safety officer *and* a reviewer. **Trade.** The SHAP pass is the slowest step — acceptable for a pre-screening tool, surfaced honestly in the latency notes.

### Reused, not rebuilt
**Choice.** Single-node LangGraph (not multi-node), minimal CI, no bespoke observability stack. **Why.** Those patterns already exist in a sibling project; SafetyVision spends its complexity budget on the *new* surface — CV, explainability, forecasting, serverless deploy — not on re-implementing orchestration. **Trade.** Less infra flourish, a tighter and more honest scope.

## API, SDK & CLI

```bash
pip install safetyvision-client
```

```python
from safetyvision_client import SafetyVision

sv = SafetyVision(api_key="sv_...")        # mint a key on the Account page
result = sv.analyze("worksite.jpg")
print(result.violations)
result.save_pdf("incident.pdf")
```

```bash
safetyvision analyze worksite.jpg --pdf report.pdf   # CLI ships with the SDK
```

Full reference: [`docs/api_usage.md`](docs/api_usage.md) · interactive Swagger at the Lambda URL's `/docs`.

## Deploy your own

```bash
git clone https://github.com/ayushgupta07xx/SafetyVision.git
cd SafetyVision

./bootstrap.sh && source .venv/bin/activate     # local dev
cd infra/aws && terraform apply                 # AWS stack (Lambda + S3 + DynamoDB + ECR, ap-south-1)
```

Walkthroughs: [`docs/aws_deploy.md`](docs/aws_deploy.md) · [`docs/supabase_setup.md`](docs/supabase_setup.md). Everything runs on always-free tiers — AWS Lambda/S3/DynamoDB/ECR, Supabase, Vercel, Hugging Face, Qdrant Cloud, Google AI Studio.

## Repository layout

```
safetyvision/
├── model/            # YOLOv8 training artifacts (results, curves, confusion matrix)
├── mlruns/           # MLflow run history (committed)
├── core/             # ONNX detector, GradCAM/SHAP explainer, RAG, Supabase + PDF adapters
├── agent/            # Single-node LangGraph (retrieve -> report -> PDF -> log)
├── analytics/        # Prophet + SARIMA forecasting, synthetic seeders
├── evaluation/       # A/B harness, golden set, committed eval results
├── serving/
│   ├── hf_app/       # Gradio app (Mode 1, HF Spaces)
│   └── lambda/       # FastAPI + Mangum handler, Dockerfile (Mode 2)
├── frontend/         # Next.js 14 + Tailwind + shadcn/ui (Mode 3, Vercel)
├── sdk/              # safetyvision-client — Python SDK + CLI (PyPI)
├── infra/
│   ├── aws/          # Terraform: Lambda + Function URL + S3 + DynamoDB + ECR
│   └── supabase/     # SQL migrations + RLS policies
├── rag_data/         # OSHA corpus scrape + Qdrant ingest
└── docs/             # Model card, ADRs, experiments, deploy + API guides
```

## Tech stack

**Computer vision:** YOLOv8s · OpenCV · ONNX · ONNX Runtime · Albumentations
**Explainability:** GradCAM · SHAP
**LLM / RAG:** Gemini Flash (multimodal) · Qdrant Cloud · BAAI/bge-small-en-v1.5 · bge-reranker-base · LangGraph
**Time series:** Prophet · statsmodels SARIMA
**Evaluation:** Groq llama-3.3-70b (LLM judge) · paired t-test · McNemar · Cohen's d
**Frontend:** Next.js 14 · TypeScript · TailwindCSS · shadcn/ui · Vercel
**Backend / auth:** Supabase (PostgreSQL · Auth · row-level security · OAuth · Storage) · FastAPI + Mangum
**API:** OpenAPI/Swagger · Python SDK + CLI (PyPI)
**ML infra:** MLflow · Weights & Biases
**Cloud:** AWS Lambda · Lambda Function URLs · S3 · DynamoDB · ECR · CloudWatch · Terraform

## Who it's for

Small factory owners and safety managers priced out of enterprise tools · construction-site supervisors · warehouse compliance officers · developers who want to self-host or integrate PPE detection via API/SDK · industrial-CV researchers. Target geography: India + Southeast Asia.

> **Intended use:** an AI-assisted pre-screening tool to support human safety officers. **Not** a replacement for human judgment. See the model card for failure modes and out-of-scope settings.

## License

**AGPL-3.0** for both the repository code and the model weights on Hugging Face Hub (inherited from Ultralytics YOLOv8). Strong copyleft — any derivative, including a network-accessible hosted service, must also be released under AGPL-3.0. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{safetyvision_2026,
  author = {Gupta, Ayush},
  title  = {SafetyVision: Open-Source AI Workplace Safety Monitor},
  year   = {2026},
  url    = {https://github.com/ayushgupta07xx/SafetyVision}
}
```

## Acknowledgements

[Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8) · [Roboflow Universe](https://universe.roboflow.com) (PPE datasets) · [OSHA](https://www.osha.gov) (public-domain regulations) · [Hugging Face](https://huggingface.co), [Vercel](https://vercel.com), [Supabase](https://supabase.com), [Qdrant](https://qdrant.tech) (free-tier hosting).
