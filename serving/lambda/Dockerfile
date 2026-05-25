# SafetyVision Mode-2 Lambda — full pipeline (detect + GradCAM/SHAP + agent report).
#
# Custom slim base: the AWS-provided python:3.11 base (Amazon Linux 2023) caps
# onnxruntime at 1.16.3 (max ONNX opset 19); our v2 ONNX is opset 20. Debian
# slim (glibc 2.36) resolves onnxruntime >=1.19, so the validated opset-20
# artifact ships untouched. awslambdaric supplies the Lambda runtime bootstrap
# the AWS base had built in. (ADR-014 / ADR-006 note.)
#
# Build from the REPO ROOT (context must see core/ agent/ analytics/).
# --provenance=false --sbom=false: Lambda rejects BuildKit's OCI manifest-list +
# attestation ("image manifest ... not supported"); these force a single Docker
# manifest. --platform pins amd64 (Lambda x86_64).
#     docker build --provenance=false --sbom=false --platform linux/amd64 \
#       -f serving/lambda/Dockerfile -t safetyvision-lambda .
# First build ~10-20 min, image ~3-4GB. Do NOT Ctrl+C on pip/layer progress.
FROM python:3.11-slim

# System libs the AWS base bundled but slim doesn't:
#   libgomp1     — OpenMP runtime (onnxruntime, opencv)
#   libglib2.0-0 — glib threads (cv2 import on slim)
#   libgl1       — libGL.so.1 (ultralytics pulls full opencv-python, needs it)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Lambda task root + HF cache (baked, read at runtime; offline flags set later).
ENV LAMBDA_TASK_ROOT=/var/task \
    PYTHONPATH=/var/task \
    HF_HOME=/opt/hf-cache \
    MPLCONFIGDIR=/tmp/mpl
WORKDIR ${LAMBDA_TASK_ROOT}

# 1) CPU-only torch first, pinned to the locally-validated versions (torch 2.12
#    is required by transformers 5.x). --extra-index-url keeps PyPI for
#    transitive deps; +cpu pins force the CPU wheels; numpy pinned first.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "numpy==1.26.4" \
 && pip install --no-cache-dir \
      torch==2.12.0+cpu torchvision==0.27.0+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu

# 2) Runtime deps + the Lambda Runtime Interface Client (the bootstrap entrypoint).
COPY serving/lambda/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir awslambdaric -r /tmp/requirements.txt

# 3) App code. handler.py at task root; core/ + agent/ + analytics/ importable
#    via PYTHONPATH. analytics/ added in Layer 10 for /forecast (lazy-imports
#    analytics.forecast -> Prophet); /analyze still never touches it.
COPY serving/lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY core/      ${LAMBDA_TASK_ROOT}/core/
COPY agent/     ${LAMBDA_TASK_ROOT}/agent/
COPY analytics/ ${LAMBDA_TASK_ROOT}/analytics/

# 4) Pre-warm HF cache (public repo, no token). Network at BUILD time only.
#    Bakes: v2 ONNX (detector), v2 .pt (GradCAM/SHAP), bge-small (RAG embed).
RUN python -c "from huggingface_hub import hf_hub_download as d; r='ayushgupta7777/safetyvision-yolov8'; d(r,'v2/best_640.onnx'); d(r,'v2/best.pt')" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 5) Lock runtime to the baked cache — no network at cold start.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# awslambdaric is the entrypoint; CMD is the handler target (module.attr).
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["handler.handler"]
