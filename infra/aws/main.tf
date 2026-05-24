###############################################################################
# SafetyVision Mode-2 — AWS infrastructure (Phase 1)
# ECR + S3 (model artifact store) + DynamoDB (audit, provisioned-only) +
# IAM + Lambda (container image, full pipeline) + Function URL + CloudWatch +
# Budgets. Region ap-south-1. Free-tier-friendly; ECR lifecycle keeps storage low.
#
# Deploy is two-phase (a container Lambda needs its image in ECR first):
#   1) terraform apply -target=aws_ecr_repository.safetyvision
#   2) build --provenance=false, tag, push image to ECR
#   3) terraform apply           (creates the rest; Lambda pulls the image)
# See infra/aws/README.md for the full runbook.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  account_id     = data.aws_caller_identity.current.account_id
  function_name  = "${var.project}-inference"
  models_bucket  = "${var.project}-models-${local.account_id}"
  ddb_table_name = "${var.project}-violations"
}

# ─── ECR ──────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "safetyvision" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"
  force_delete         = true # lets `terraform destroy` clean images
  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the 2 most recent images so the ~4GB image doesn't accumulate
# past ECR's 500MB free tier across re-pushes.
resource "aws_ecr_lifecycle_policy" "safetyvision" {
  repository = aws_ecr_repository.safetyvision.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the last 2 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 2
      }
      action = { type = "expire" }
    }]
  })
}

# ─── S3 — model artifact store (option a) ──────────────────────────────────────
# Canonical weights store (mirrors HF Hub). The Lambda image BAKES the ONNX at
# build time for zero cold-start network; S3 is the versioned source of record.
resource "aws_s3_bucket" "models" {
  bucket        = local.models_bucket
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "onnx_weights" {
  bucket = aws_s3_bucket.models.id
  key    = "v2/best_640.onnx"
  source = var.onnx_local_path
  etag   = filemd5(var.onnx_local_path)
}

# ─── DynamoDB — audit table (provisioned; handler write is Phase 2) ─────────────
resource "aws_dynamodb_table" "violations" {
  name         = local.ddb_table_name
  billing_mode = "PAY_PER_REQUEST" # $0 at zero traffic
  hash_key     = "violation_id"
  range_key    = "timestamp"
  attribute {
    name = "violation_id"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "N"
  }
}

# ─── IAM — Lambda execution role ────────────────────────────────────────────────
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project}-lambda-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Data-plane access (DynamoDB audit table, S3 model read) — ready for Phase 2.
resource "aws_iam_role_policy" "lambda_data" {
  name = "${var.project}-lambda-data"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.violations.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.models.arn}/*"
      }
    ]
  })
}

# ─── CloudWatch log group (explicit, 7-day retention) ───────────────────────────
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 7
}

# ─── Lambda (container image, full pipeline) ────────────────────────────────────
resource "aws_lambda_function" "inference" {
  function_name = local.function_name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.safetyvision.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  memory_size   = var.lambda_memory
  timeout       = var.lambda_timeout

  # null by default — fresh accounts often can't reserve concurrency yet.
  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    variables = {
      GOOGLE_AI_STUDIO_KEY = var.google_ai_studio_key
      QDRANT_URL           = var.qdrant_url
      QDRANT_API_KEY       = var.qdrant_api_key
      GEMINI_MODEL         = var.gemini_model
      YOLO_CONFIG_DIR      = "/tmp/Ultralytics"
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda_logs,
    aws_iam_role_policy_attachment.lambda_logs,
  ]
}

resource "aws_lambda_function_url" "inference_url" {
  function_name      = aws_lambda_function.inference.function_name
  authorization_type = "NONE" # public; ADR-006. API-key auth is Phase 2 (handler-level).
}

# Public Function URL needs an explicit invoke permission for principal "*".
resource "aws_lambda_permission" "function_url" {
  statement_id           = "AllowPublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.inference.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

# ─── Budgets — $1 / $5 / $10 email alerts ───────────────────────────────────────
resource "aws_budgets_budget" "monthly" {
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = "10"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = [10, 50, 100] # % of $10 => $1 / $5 / $10
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
