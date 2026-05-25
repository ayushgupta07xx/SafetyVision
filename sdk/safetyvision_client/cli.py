"""Command-line interface for the SafetyVision SDK.

    safetyvision analyze worksite.jpg --pdf report.pdf
    safetyvision history --limit 10
    safetyvision forecast NO-Hardhat
"""
from __future__ import annotations

import argparse
import json
import sys

from safetyvision_client import __version__
from safetyvision_client.client import SafetyVision, SafetyVisionError


def _client(args: argparse.Namespace) -> SafetyVision:
    return SafetyVision(api_key=args.api_key, base_url=args.base_url)


def _cmd_analyze(args: argparse.Namespace) -> int:
    result = _client(args).analyze(args.image)
    if args.json:
        print(json.dumps(result.raw, indent=2))
        return 0
    n = len(result.violations)
    if n == 0:
        print("No violations detected.")
    else:
        print(f"[{n} violation(s) detected]")
        for v in result.violations:
            conf = v.get("confidence", 0.0)
            print(f"  {v.get('class')} ({v.get('risk_level')}, {conf:.2f} confidence)")
        citation = (result.incident_report or {}).get("regulation_cited")
        if citation:
            print(f"  -> {citation}")
    if args.pdf:
        print(f"PDF saved to {result.save_pdf(args.pdf)}")
    if args.annotated:
        print(f"Annotated image saved to {result.save_annotated(args.annotated)}")
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    rows = _client(args).history(limit=args.limit, offset=args.offset)
    if not rows:
        print("No violations in history.")
        return 0
    for r in rows:
        print(
            f"{str(r.get('violation_type')):<16} {str(r.get('risk_level')):<9} "
            f"conf={float(r.get('confidence', 0.0)):.2f}  {r.get('regulation_cited') or ''}"
        )
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    data = _client(args).forecast(
        args.violation_type, days=args.days, horizon=args.horizon
    )
    print(json.dumps(data, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="safetyvision", description="SafetyVision PPE compliance CLI."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--api-key", default=None, help="API key (or set SAFETYVISION_API_KEY).")
    p.add_argument("--base-url", default=None, help="API base URL (or SAFETYVISION_BASE_URL).")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Analyze an image for PPE violations.")
    a.add_argument("image", help="Path to a JPEG/PNG image (<=6MB).")
    a.add_argument("--pdf", default=None, help="Save the incident PDF to this path.")
    a.add_argument("--annotated", default=None, help="Save the annotated image to this path.")
    a.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    a.set_defaults(func=_cmd_analyze)

    h = sub.add_parser("history", help="List the user's violation history.")
    h.add_argument("--limit", type=int, default=50)
    h.add_argument("--offset", type=int, default=0)
    h.set_defaults(func=_cmd_history)

    f = sub.add_parser("forecast", help="7-day compliance forecast for a violation type.")
    f.add_argument("violation_type", help="e.g. NO-Hardhat, 'NO-Safety Vest'")
    f.add_argument("--days", type=int, default=30)
    f.add_argument("--horizon", type=int, default=7)
    f.set_defaults(func=_cmd_forecast)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SafetyVisionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
