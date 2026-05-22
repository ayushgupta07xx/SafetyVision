# SafetyVision

> Open-source AI workplace safety monitor. Detects PPE violations, explains every decision, and forecasts compliance trends — for free, self-hostable.

Hard-hat-free zones, missing high-vis vests, exposed hands — safety violations that human eyes miss when scanning hundreds of cameras. SafetyVision is an open-source pre-screening tool that surfaces these for human review, replacing $500–$2,000/month enterprise tools like Protex AI and Intenseye.

[Model on Hugging Face](https://huggingface.co/ayushgupta7777/safetyvision-yolov8) · [Training run on W&B](https://wandb.ai/agcr7jw-vellore-institute-of-technology/Ultralytics/runs/yolov8s-ppe-v2_20260519_065053) · [Model card](docs/model_card.md) · [Architecture decisions](docs/decisions.md)

---

## Status — Week 1 of 3

| Component | Status |
|---|---|
| YOLOv8n detector (13-class PPE) | ✅ Trained, weights on HF Hub |
| Model card with honest metrics | ✅ Published |
| Architecture decisions (ADRs) | ✅ Documented |
| GradCAM + SHAP explainability | 🚧 Week 2 |
| OSHA RAG + Gemini multimodal incident reports | 🚧 Week 2 |
| Prophet 7-day compliance forecast (+ SARIMA baseline) | 🚧 Week 2 |
| A/B testing harness (paired t-test, McNemar) | 🚧 Week 2 |
| Hugging Face Spaces demo (Mode 1, free, no signup) | 🚧 Week 3 |
| AWS Lambda serverless API (Mode 2, Terraform IaC) | 🚧 Week 3 |

## Headline metrics

Held-out test split (5,148 images, never touched during training):

| | Value |
|---|---|
| **mAP@0.5** | **0.701** |
| **mAP@0.5:0.95** | **0.441** |
| Precision | 0.607 |
| Recall | 0.711 |
| Inference (T4 GPU) | 2.9 ms / image |

Per-class breakdown, failure modes, and bias notes: [model card](docs/model_card.md).

## What it does

- **Detects 13 PPE classes:** hard hat, safety vest, goggles, gloves, mask, fall harness — and their "missing" violation counterparts, plus Fall-Detected and Person.
- **Will generate OSHA-grounded incident reports** via Gemini Flash + RAG over OSHA 29 CFR 1926 and 1910 (Week 2).
- **Will explain every decision** with GradCAM heatmaps over the detection head and SHAP feature scores (Week 2).
- **Will forecast 7-day site compliance** using Prophet, benchmarked against a statsmodels SARIMA baseline (Week 2).
- **Will deploy two ways:** a hosted Gradio demo on Hugging Face Spaces (image + short video, free, no signup) and a serverless AWS Lambda API with Terraform-managed infrastructure (image-only, 6 MB cap) (Week 3).

## Who it's for

- Small factory owners and safety managers who can't afford enterprise compliance software
- Construction-site supervisors
- Warehouse safety compliance officers
- Researchers working on industrial computer vision
- Anyone who wants to self-host PPE monitoring

Target geography: India + Southeast Asia.

## Try it

**Use the published detector** (no clone needed, just `pip install ultralytics huggingface_hub`):

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

weights = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="best.pt")
model = YOLO(weights)
results = model("worksite_image.jpg")
results[0].show()
```

ONNX Runtime (CPU-friendly, what AWS Lambda will use in Mode 2):

```python
import cv2, numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# Both files must live in the same directory (ONNX external data)
hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="best.onnx.data")
onnx_path = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="best.onnx")

session = ort.InferenceSession(onnx_path)
img = cv2.imread("worksite_image.jpg")
img = cv2.resize(img, (640, 640))
inp = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
outputs = session.run(None, {"images": inp})
# outputs[0] shape: (1, 17, 8400) — apply NMS for final boxes
```

**Develop locally** (clone the repo):

```bash
git clone https://github.com/ayushgupta07xx/SafetyVision.git
cd SafetyVision
./bootstrap.sh
source .venv/bin/activate
```

## Architecture (preview)

```
Image / short video clip
       │
       ▼
[YOLOv8n ONNX inference]        ← Week 1 ✅
       │
       ▼
[GradCAM + SHAP explainer]      ← Week 2
       │
       ▼
[OSHA RAG → Gemini Flash report] ← Week 2
       │
       ▼
[Log violation → DynamoDB / SQLite]
       │
       ▼
[Prophet 7-day compliance forecast] ← Week 2
       │
       ▼
[Gradio response (Mode 1) / Lambda response (Mode 2)] ← Week 3
```

Full architecture diagram (Mermaid): [docs/architecture.md](docs/architecture.md) (Week 3).

## Training

- **Hardware:** Kaggle Notebooks, 2× Tesla T4 GPUs (free tier)
- **Framework:** Ultralytics 8.3.40, PyTorch 2.10.0 + CUDA 12.8
- **Dataset:** [PPE-Combined v1](https://universe.roboflow.com/mazz-maxx/ppe-combined-9bprl-mmcaf) on Roboflow Universe — 57,904 images, 13 classes
- **Schedule:** 100 epochs · batch=32 · imgsz=640 · SGD with default ultralytics LR schedule
- **Wall time:** ~15 hours across two Kaggle Save Versions (12-hour session cap forced a mid-run resume at epoch 82 — see [ADR-003](docs/decisions.md))
- **W&B run (public):** original v1 run `9nctv2ai` (expired); current run [`yolov8s-ppe-v2`](https://wandb.ai/agcr7jw-vellore-institute-of-technology/Ultralytics/runs/yolov8s-ppe-v2_20260519_065053) — epochs 1–82 in W&B charts; epochs 83–100 metrics in [`model/yolov8n-ppe-v1/results.csv`](model/yolov8n-ppe-v1/results.csv) (see [ADR-004](docs/decisions.md))
- **MLflow registry:** committed at [`mlruns/`](mlruns/), run ID `f1932e539038417dad6db757affd50e6`

Why Kaggle and not GCP? See [ADR-001](docs/decisions.md) — diagnosed an undocumented GCP N1/G2 family throttle on new paid accounts via systematic VM-class testing across 30+ zones.

## Repository layout

```
safetyvision/
├── model/                      # Training artifacts (results.csv, curves, confusion matrix)
├── mlruns/                     # MLflow run history (committed)
├── core/                       # YOLOv8 wrapper, GradCAM/SHAP, RAG client (Week 2)
├── agent/                      # Single-node LangGraph orchestration (Week 2)
├── analytics/                  # Prophet + SARIMA forecasting (Week 2)
├── evaluation/                 # A/B test harness, golden set, eval results (Week 2)
├── serving/
│   ├── hf_app/                 # Gradio app for HF Spaces (Week 3)
│   └── lambda/                 # AWS Lambda container handler (Week 3)
├── infra/aws/                  # Terraform: Lambda + Function URL + S3 + DynamoDB + ECR (Week 3)
├── rag_data/                   # OSHA PDF scrape + Qdrant ingest (Week 2)
└── docs/                       # Model card, ADRs, experiments, deploy guides
```

## Roadmap

**Week 2 — core pipeline & explainability**
- OSHA RAG ingestion (Qdrant Cloud + BGE-small embeddings, BGE-reranker-base)
- GradCAM + SHAP explainability on the detector head
- Gemini Flash multimodal incident reports with RAG-grounded regulation citations
- Prophet 7-day compliance forecast + SARIMA baseline (MAPE comparison)
- Single-node LangGraph orchestration
- A/B testing harness: prompt variants (paired t-test) and confidence thresholds (McNemar)

**Week 3 — ship**
- Gradio app deployed on Hugging Face Spaces (image + video ≤30s)
- AWS Lambda container image, deployed via Terraform to ap-south-1
- Lambda Function URL endpoint (chose this over API Gateway — [ADR-006](docs/decisions.md) to come)
- AWS Budgets cost safety nets ($1/$5/$10 alerts)
- Demo GIF in README, blog post, LinkedIn launch

## Tech stack

**Computer vision:** YOLOv8n · OpenCV · ONNX · ONNX Runtime  
**Explainability:** GradCAM · SHAP  
**Multimodal LLM:** Gemini 1.5 Flash  
**RAG:** Qdrant Cloud · BAAI/bge-small-en-v1.5 · BAAI/bge-reranker-base  
**Orchestration:** LangGraph (single-node)  
**Time series:** Prophet · statsmodels SARIMA (baseline)  
**Evaluation:** Groq llama-3.3-70b-versatile (LLM judge) · paired t-test · McNemar test  
**ML infrastructure:** MLflow · Weights & Biases  
**Serving:** Gradio (Hugging Face Spaces) · AWS Lambda container · Lambda Function URL  
**Cloud:** AWS S3 · DynamoDB · ECR · CloudWatch · Terraform  
**Training:** Kaggle Notebooks (2× T4 GPU) · ultralytics 8.3.40 · PyTorch 2.10.0

## License

**AGPL-3.0** for both the repository code and the model weights distributed on Hugging Face Hub. Strong copyleft: any derivative work (modifications, integrations, hosted services accessible over a network) must also be released under AGPL-3.0. See [`LICENSE`](LICENSE) for the full license text and the [model card](docs/model_card.md#license) for the inheritance from Ultralytics YOLOv8.

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

[Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 and the training framework · [Roboflow Universe](https://universe.roboflow.com) for the PPE-Combined dataset · [OSHA](https://www.osha.gov) for public-domain regulations · [Kaggle Notebooks](https://www.kaggle.com/code) for free 2× T4 GPU training
