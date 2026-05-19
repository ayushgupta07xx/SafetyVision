#!/usr/bin/env bash
# Launches v2 training in a detached tmux session.
# SSH disconnects do NOT kill the training.
#
# Usage:
#   bash scripts/launch_training.sh --dry-run    # DO THIS FIRST (~5 min)
#   bash scripts/launch_training.sh              # full training (~57 hrs)
#
# Monitor:
#   tmux attach -t v2train         (Ctrl+B then D to detach)
#   tail -f logs/v2_train_*.log
#
# Stop:
#   tmux kill-session -t v2train

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

LOG_FILE="logs/v2_train_$(date +%Y%m%d_%H%M%S).log"

if [[ "${1:-}" == "--dry-run" ]]; then
  SESSION="v2train-dry"
  TRAIN_ARGS="--dry-run"
else
  SESSION="v2train"
  TRAIN_ARGS=""
fi

INNER_CMD="cd $REPO_ROOT && source .venv/bin/activate && python -m model.train_v2 $TRAIN_ARGS 2>&1 | tee $LOG_FILE"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$INNER_CMD"

echo ""
echo "Launched: session=$SESSION"
echo "Log:      $LOG_FILE"
echo ""
echo "Attach:   tmux attach -t $SESSION"
echo "Detach:   Ctrl+B then D"
echo "Tail log: tail -f $LOG_FILE"
echo "Stop:     tmux kill-session -t $SESSION"
