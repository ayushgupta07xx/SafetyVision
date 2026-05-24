variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "project" {
  type    = string
  default = "safetyvision"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

# Local path to the v2 ONNX, relative to infra/aws/. Uploaded to S3 as the
# canonical artifact. (Downloaded earlier to artifacts/onnx/dl/v2/best_640.onnx.)
variable "onnx_local_path" {
  type    = string
  default = "../../artifacts/onnx/dl/v2/best_640.onnx"
}

variable "lambda_memory" {
  type    = number
  default = 5120
}

variable "lambda_timeout" {
  type    = number
  default = 120
}

# null = no reserved concurrency (safe on fresh accounts). Set to 10 to cap
# runaway invocations once your account concurrency limit is >= 110.
variable "reserved_concurrency" {
  type    = number
  default = null
}

variable "gemini_model" {
  type    = string
  default = "gemini-flash-latest"
}

# ── Secrets — provide via gitignored terraform.tfvars (never commit/chat) ───────
variable "google_ai_studio_key" {
  type      = string
  sensitive = true
}

variable "qdrant_url" {
  type      = string
  sensitive = true
}

variable "qdrant_api_key" {
  type      = string
  sensitive = true
}

# ── Budget alert recipient ──────────────────────────────────────────────────────
variable "alert_email" {
  type = string
}
