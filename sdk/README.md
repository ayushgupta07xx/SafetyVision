# safetyvision-client

Python client + CLI for the [SafetyVision](https://github.com/ayushgupta07xx/SafetyVision)
PPE compliance API — upload a worksite image, get PPE violation detections, an
OSHA-grounded incident report, and a downloadable PDF.

## Install
```bash
pip install safetyvision-client
```

## Quickstart
```python
from safetyvision_client import SafetyVision

sv = SafetyVision(api_key="sv_...")          # or set SAFETYVISION_API_KEY
result = sv.analyze("worksite.jpg")
print(result.violations)
result.save_pdf("incident.pdf")             # downloads the signed PDF URL

results = sv.analyze_batch(["a.jpg", "b.jpg"])   # threaded
```

## CLI
```bash
export SAFETYVISION_API_KEY=sv_...
safetyvision analyze worksite.jpg --pdf report.pdf
safetyvision history --limit 10
safetyvision forecast "NO-Safety Vest"
```

## Config
- `SAFETYVISION_API_KEY` — your API key (or `--api-key`)
- `SAFETYVISION_BASE_URL` — override the API endpoint (or `--base-url`)

Image-only, 6 MB max (Lambda Function URL cap); video → the HF Spaces demo.
