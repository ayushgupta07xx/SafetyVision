#!/usr/bin/env bash
# SafetyVision local setup — one-command bootstrap
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "🦺 SafetyVision bootstrap"
echo "Repo: $REPO_ROOT"

if ! command -v python3.11 >/dev/null 2>&1; then
    echo "❌ python3.11 not found. Install Python 3.11 first."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "→ Creating .venv"
    python3.11 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "→ Upgrading pip / wheel / setuptools"
pip install --upgrade pip wheel setuptools

echo "→ Installing safetyvision (editable, dev extras) — this takes 5–10 min on first run"
pip install -e ".[dev]"

if [ ! -f ".env" ]; then
    echo "→ Creating .env from .env.example"
    cp .env.example .env
    echo "⚠️  Edit .env and fill in keys before running anything that hits an API."
fi

echo ""
echo "✅ Bootstrap complete."
echo "Activate venv with:  source .venv/bin/activate"
