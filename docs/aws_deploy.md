# AWS Deployment (Mode 2) — Lambda Function URL Runbook

Mode 2 serves the v2 ONNX as a public, **image-only** inference API on AWS Lambda
behind a Function URL. Every service here is always-free tier. See **ADR-006**
(Function URLs vs API Gateway) and **ADR-014** (container base) for the *why*.

## Prerequisites
- AWS CLI v2 + Terraform installed (HashiCorp apt repo); IAM user `safetyvision-cli`.
- Credentials via `aws configure` — never in `.env` or code. Account `290294660188`, region **`ap-south-1`**.
- Docker (WSL integration on). Build context = **repo root**.
- Confirm region before any create: `aws configure get region` → `ap-south-1`.

## What the stack creates (`infra/aws/main.tf`)
ECR repo · S3 model bucket + ONNX object · DynamoDB audit table (on-demand /
PAY_PER_REQUEST, $0 idle) · IAM exec role · the Lambda (default 2048MB / 30s;
Chat-12 proven at 3008MB via tfvars) · the Function URL (`auth_type = NONE`) ·
the public-invoke permissions · CloudWatch log group (7-day retention).

## Deploy
> The build→push→apply ordering below is reconstructed from the committed
> Dockerfile + main.tf resource list. Reconcile resource/output names against
> your actual `main.tf` / `outputs.tf` where flagged.

**1. Build the container** (from repo root — context must see `core/` + `agent/`; ADR-014):
```bash
This is **not** an account guardrail. Fix (also committed as an
`aws_lambda_permission` in main.tf so a clean apply reproduces it):
```bash
aws lambda add-permission --function-name safetyvision-inference \
  --region ap-south-1 --statement-id UrlPolicyInvokeFunction \
  --action lambda:InvokeFunction --principal "*" --invoked-via-function-url
```
Verify with `aws lambda get-policy --function-name safetyvision-inference
--region ap-south-1`: two `Allow` statements — `InvokeFunctionUrl`
(cond `FunctionUrlAuthType=NONE`) and `InvokeFunction` (cond `InvokedViaFunctionUrl=true`).

## Build gotchas (ADR-014)
- Custom `python:3.11-slim` base, not the AWS base: AWS base caps onnxruntime at
  1.16.3 (opset-19); our v2 ONNX is opset-20. Slim resolves onnxruntime ≥1.19.
- `awslambdaric` is the container entrypoint (the AWS base bundled this).
- apt libs slim needs: `libgomp1`, `libglib2.0-0`, `libgl1`.
- A `413` on a request = payload > 6MB. Expected, not a bug → route through Mode 1 (ADR-006).

## Cost controls
- AWS Budget `safetyvision-monthly`: $10 limit, alerts at 10/50/100% = $1/$5/$10 (ACTUAL spend).
- Lambda reserved concurrency = 10 — caps runaway invocations (matters now the URL
  is public + unauthenticated; per-key auth is Phase 2).
- DynamoDB on-demand → $0 idle.

## Teardown
End every AWS work session with teardown **unless** the demo is intentionally kept
live (then confirm the budget above):
```bash
cd infra/aws && terraform destroy
```
`destroy` removes the function and all its inline policy statements (including the
CLI-added one), so the next clean apply rebuilds everything from Terraform.
