"""Build a Lambda Function-URL event for POST /analyze with a real image,
so we can direct-invoke /analyze (bypassing the 403'd public URL) and read
Max Memory Used from CloudWatch. Stdlib only — no boto3 needed.

Usage (from repo root, venv active):
    python build_analyze_event.py [image_path]
Writes /tmp/analyze_event.json, then:
    aws lambda invoke --function-name safetyvision-inference --region ap-south-1 \\
      --cli-read-timeout 180 --cli-binary-format raw-in-base64-out \\
      --payload file:///tmp/analyze_event.json /tmp/analyze_resp.json --query StatusCode
"""

import base64
import json
import sys

img_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "evaluation/golden_set/images/unsplash/construction_site_no_hardhat.jpg"
)
with open(img_path, "rb") as f:
    img = f.read()

boundary = "----svboundary"
body = (
    (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    + img
    + f"\r\n--{boundary}--\r\n".encode()
)

event = {
    "version": "2.0",
    "routeKey": "$default",
    "rawPath": "/analyze",
    "rawQueryString": "",
    "headers": {
        "host": "local",
        "content-type": f"multipart/form-data; boundary={boundary}",
    },
    "requestContext": {
        "http": {
            "method": "POST",
            "path": "/analyze",
            "protocol": "HTTP/1.1",
            "sourceIp": "127.0.0.1",
            "userAgent": "cli",
        },
        "requestId": "t",
        "stage": "$default",
    },
    "body": base64.b64encode(body).decode(),
    "isBase64Encoded": True,
}

with open("/tmp/analyze_event.json", "w") as f:
    json.dump(event, f)

print(f"wrote /tmp/analyze_event.json  (image {len(img)} bytes, body_b64 {len(event['body'])} chars)")
