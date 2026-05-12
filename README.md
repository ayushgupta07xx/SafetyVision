# 🦺 SafetyVision

> Open-source AI-powered PPE compliance monitor.

PPE violation detection with explainability (GradCAM + SHAP), OSHA-grounded incident reports (Gemini Flash + RAG), 7-day compliance forecasting (Prophet), and an A/B testing harness with paired statistical significance.

## Status

🚧 **Under active build** — Week 1 in progress. Demo links land Week 3.

## Quickstart (local)

```bash
git clone https://github.com/ayushgupta07xx/safetyvision.git
cd safetyvision
./bootstrap.sh
source .venv/bin/activate
```

## Deployment (coming Week 3)

- **Live demo:** Hugging Face Spaces
- **Production API:** AWS Lambda Function URL
- **Self-host:** `cd infra/aws && terraform apply`

## Stack

YOLOv8n · OpenCV · ONNX · GradCAM · SHAP · Qdrant · BGE · Gemini Flash (multimodal) · LangGraph · Prophet · SARIMA · MLflow · W&B · AWS Lambda · S3 · DynamoDB · ECR · Terraform · Gradio

## License

MIT
