# 🦺 SafetyVision

**Open-source AI workplace safety monitor — detects PPE violations, explains every decision, and forecasts compliance trends. Free, self-hostable, $0 to run.**

[Live app](https://safetyvision.vercel.app) · [Open-source demo](https://huggingface.co/spaces/ayushgupta7777/safetyvision) · [API docs](https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws/docs) · [Model card](https://huggingface.co/ayushgupta7777/safetyvision-yolov8) · License: AGPL-3.0

Missing hard hats, no high-vis vest, no fall harness at height — the violations human eyes miss when scanning hundreds of frames. SafetyVision is an open-source pre-screening tool that surfaces them for human review, with explainable detections, OSHA-grounded incident reports, and forward-looking compliance forecasts. It replaces commercial tools that charge $500–$2,000/month.

[![Watch the demo](https://img.youtube.com/vi/-LsOMLM9hkI/hqdefault.jpg)](https://youtu.be/-LsOMLM9hkI)

> ▶️ **[Watch the 3-minute walkthrough](https://youtu.be/-LsOMLM9hkI)**

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

- **Detects PPE compliance** — finds each worker and checks for hard hats, safety vests, masks, gloves, and fall harnesses, flagging the missing-PPE violations in red, ranked by risk level. Powered by a fine-tuned **YOLOv8s** model exported to ONNX, running on **CPU** (no GPU needed).
- **Explains every decision** — a **GradCAM** heatmap shows where the model looked, plus **SHAP** per-pixel attribution. Auditable, not a black box.
- **Writes OSHA-grounded incident reports** — a **multimodal Gemini Flash** model reads the annotated image alongside the actual OSHA regulation (retrieved via **Qdrant + BGE** RAG over 29 CFR 1910 and 1926) and cites the real CFR number.
- **Exports an audit-ready PDF** — one click per violation, with the annotated image, citation, corrective actions, and an explainability section.
- **Forecasts 7-day compliance** — a **Prophet** time-series model (benchmarked against a SARIMA baseline) projects the trend per violation type.
- **Remembers your history** — every inspection is saved per user (Supabase Postgres with row-level security), surfaced in a history table and a roll-up dashboard.

## Headline metrics

YOLOv8s v2, held-out test set (deployed config: ONNX @ 896):

| | Value |
|---|---|
| **mAP@50** | **0.763** |
| **mAP@50:95** | **0.482** |
| Warm CPU inference (AWS Lambda) | ~500–800 ms / image |

Strongest classes: Fall-Detected (0.956), Hardhat (0.936), Safety Vest (0.891). Weakest: NO-Safety Vest (0.382). The v2 retrain (YOLOv8s + Albumentations augmentation on 80k+ images) targeted **0.78** mAP@50 and landed at **0.763** — an honest near-miss, documented in full. Per-class breakdown, v1→v2 comparison, and failure modes: **[model card](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)**.

**Statistical validation (A/B tests):**
- Incident-report quality, RAG vs no-RAG: RAG wins, Cohen's d = 0.65, p = 0.0197 (paired t-test, N=16)
- Confidence threshold 0.40 vs 0.55: 0.40 wins, McNemar p = 4×10⁻⁵ (N=200)

## The three deployment surfaces

- **Mode 3 — Next.js + Vercel (primary):** the product. Next.js 14 + Tailwind + shadcn/ui, Supabase auth (email + Google OAuth), per-user history, forecast dashboard, PDF downloads, API-key management. → [safetyvision.vercel.app](https://safetyvision.vercel.app)
- **Mode 1 — Hugging Face Spaces (open-source demo):** a free Gradio app, no signup, accepts image **or** short video (≤30s). The community / quick-try surface. → [HF Spaces](https://huggingface.co/spaces/ayushgupta7777/safetyvision)
- **Mode 2 — AWS Lambda Function URL (production API):** serverless ONNX inference behind a free-forever HTTPS endpoint, API-key auth, Swagger/Redoc docs. **Image-only** (Lambda's 6MB on-wire payload cap ≈ 4MB raw after base64; video stays on Modes 1 & 3). → [`/docs`](https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws/docs)

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
# CLI (ships with the SDK)
safetyvision analyze worksite.jpg --pdf report.pdf
```

Full API + SDK + CLI reference: [`docs/api_usage.md`](docs/api_usage.md) · interactive Swagger at the Lambda URL's `/docs`.

## Deploy your own

```bash
git clone https://github.com/ayushgupta07xx/SafetyVision.git
cd SafetyVision

# Local dev
./bootstrap.sh && source .venv/bin/activate

# AWS production stack (Lambda + S3 + DynamoDB + ECR, ap-south-1)
cd infra/aws && terraform apply
```

Walkthroughs: [`docs/aws_deploy.md`](docs/aws_deploy.md) · [`docs/supabase_setup.md`](docs/supabase_setup.md). Everything runs on always-free tiers — AWS Lambda/S3/DynamoDB/ECR, Supabase, Vercel, Hugging Face, Qdrant Cloud, Google AI Studio.

## Architecture

```
Image / short video clip
       │
       ▼
[YOLOv8s ONNX inference]  ──►  bounding boxes + risk-ranked violations
       │
       ▼
[GradCAM + SHAP explainer]
       │
       ▼
[Single-node LangGraph]  ──►  OSHA RAG (Qdrant + BGE) ──► Gemini Flash report ──► PDF ──► log
       │
       ▼
[Persistence]  ──►  Supabase (per-user history)  +  DynamoDB (stateless audit)
       │
       ▼
[Prophet 7-day forecast]  (SARIMA baseline)
       │
       ▼
[Vercel UI (Mode 3) · Gradio (Mode 1) · Lambda API (Mode 2)]
```

Full Mermaid diagram: [`docs/architecture.md`](docs/architecture.md).

## Repository layout

```
safetyvision/
├── model/            # YOLOv8 training artifacts (results, curves, confusion matrix)
├── mlruns/           # MLflow run history (committed)
├── core/             # ONNX detector, GradCAM/SHAP explainer, RAG, Supabase + PDF adapters
├── agent/            # Single-node LangGraph orchestration (retrieve → report → PDF → log)
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
**Multimodal LLM:** Gemini Flash (multimodal)
**RAG:** Qdrant Cloud · BAAI/bge-small-en-v1.5 · BAAI/bge-reranker-base
**Orchestration:** LangGraph (single-node)
**Time series:** Prophet · statsmodels SARIMA (baseline)
**Evaluation:** Groq llama-3.3-70b-versatile (LLM judge) · paired t-test · McNemar · Cohen's d
**Frontend:** Next.js 14 · TypeScript · TailwindCSS · shadcn/ui · Vercel
**Backend / auth:** Supabase (PostgreSQL · Auth · row-level security · OAuth · Storage)
**API:** FastAPI + Mangum on Lambda · OpenAPI/Swagger · Python SDK + CLI (PyPI)
**ML infrastructure:** MLflow · Weights & Biases
**Cloud:** AWS Lambda · Lambda Function URLs · S3 · DynamoDB · ECR · CloudWatch · Terraform
**Training:** single cloud L4 GPU (see model card for the full procedure)

## Who it's for

Small factory owners and safety managers priced out of enterprise tools · construction-site supervisors · warehouse compliance officers · developers who want to self-host or integrate PPE detection via API/SDK · industrial-CV researchers. Target geography: India + Southeast Asia.

> **Intended use:** an AI-assisted pre-screening tool to support human safety officers. **Not** a replacement for human judgment. See the model card for failure modes and out-of-scope settings.

## License

**AGPL-3.0** for both the repository code and the model weights on Hugging Face Hub (inherited from Ultralytics YOLOv8). Strong copyleft — any derivative work, including a network-accessible hosted service, must also be released under AGPL-3.0. See [`LICENSE`](LICENSE).

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
