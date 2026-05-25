# SafetyVision API Usage

The SafetyVision production API (Mode 2) runs on AWS Lambda behind a Function
URL. It is **image-only** — the Function URL caps request payloads at 6 MB, so
video analysis lives on the HF Spaces demo (Mode 1).

**Base URL:** `https://ssbjfzly4mljxkb45moiu2bb6a0nnrrb.lambda-url.ap-south-1.on.aws/`

All responses are JSON. AI-assisted analysis — a human safety officer must
review before any action is taken.

## Authentication

Every endpoint except `/health`, `/`, `/docs`, and `/redoc` requires an API key
in the `X-API-Key` header. Keys are SHA-256 hashed at rest; the raw key is shown
once at creation and is unrecoverable. A missing or invalid key returns `401`.

Provision a key (until the account page ships, via the CLI against a Supabase
auth user):

    python -m core.apikeys mint --user-id <auth-user-uuid> --label "my key"
    python -m core.apikeys revoke --key-id <key-uuid>

## Endpoints

### POST /analyze
Full pipeline on one image: detection, GradCAM + SHAP explainability, and an
OSHA-grounded incident report. The inspection and its violations persist to the
caller's history.

- Body: `multipart/form-data` with an `image` file (JPEG/PNG, <= 6 MB)
- `413` if the image exceeds 6 MB -> use the HF Spaces demo for larger files / video
Response (abridged): `inspection_id`, `violations[]` (`class`, `confidence`,
`bbox`, `risk_level`), `annotated_image_b64`, `gradcam_b64`, `shap_chart_b64`,
`incident_report`, `pdf_report_url`, `processing_time_ms`.

### GET /violations
Paginated violation history for the authenticated user, newest first.

- Query: `limit` (1-200, default 50), `offset` (default 0)
### GET /forecast
7-day Prophet compliance forecast for one violation type, from the caller's
Supabase history.

- Query: `violation_type` (required, see below), `days` (14-90, default 30),
  `horizon` (1-30, default 7)
- `400` for an unknown `violation_type`; `422` if there is not enough history
  (>= 14 days needed for weekly seasonality)
Returns `recent_compliance`, a plain-language `summary`, the `history` series,
and `forecast` points (`yhat` plus the 80% interval `yhat_lower`/`yhat_upper`).

### Valid violation_type values
These match exactly what the detector logs; any other string returns `400`:

    Fall-Detected, No_Harness, NO-Hardhat, NO-Safety Vest,
    NO-Goggles, NO-Mask, NO-Gloves

### GET /docs and GET /redoc
Auto-generated OpenAPI documentation (Swagger UI and ReDoc).

### GET /health
Liveness probe -> `{"status": "ok"}`. No auth.

## Notes
- Cold start ~10s (model + Prophet load); warm requests are sub-second.
- Rate limiting: the function caps at 10 reserved concurrent invocations.
- Region: `ap-south-1` (Mumbai). Image-only; video -> HF Spaces (Mode 1).
- Python SDK (`safetyvision-client`) and CLI ship in a later release.
