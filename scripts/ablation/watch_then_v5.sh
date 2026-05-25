#!/usr/bin/env bash
# Watcher: blocks until the v4 wrapper PID exits, sanity-checks v4 summary,
# then launches v5 in the same shell (so its full stdout lands in v5.runner.log).
# Usage: nohup ./watch_then_v5.sh <V4_PID> &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

V4_PID="${1:-4068402}"
WATCH_LOG="logs/FixedStageFv5/watcher.log"
mkdir -p "$(dirname "$WATCH_LOG")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

log "watching V4_PID=$V4_PID"
if ! kill -0 "$V4_PID" 2>/dev/null; then
  log "V4_PID $V4_PID is not alive at startup; will still verify v4 summary and proceed"
else
  # Block until PID exits. tail --pid works for non-child PIDs.
  tail --pid="$V4_PID" -f /dev/null
  log "V4_PID $V4_PID exited"
fi

# Verify v4 produced at least one post,* row in summary.csv before launching v5.
V4_SUM="logs/FixedStageFv4/summary.csv"
if [ ! -f "$V4_SUM" ]; then
  log "ERROR: $V4_SUM not found; aborting v5 launch"
  exit 1
fi
POST_ROWS=$(grep -c "^post," "$V4_SUM" 2>/dev/null || echo 0)
log "v4 summary post rows = $POST_ROWS"
if [ "$POST_ROWS" -lt 1 ]; then
  log "ERROR: no post,* eval rows in v4 summary; aborting v5 launch (inspect manually)"
  log "tail of v4 summary:"
  tail -10 "$V4_SUM" | tee -a "$WATCH_LOG"
  exit 2
fi

# Pick the best post-eval ckpt by ep_len for v5 warm-load (overrides default
# which is just newest model_*.pt).
BEST_LINE=$(grep "^post," "$V4_SUM" | sort -t, -k6 -g -r | head -1)
BEST_ITER=$(echo "$BEST_LINE" | cut -d, -f2)
BEST_EPLEN=$(echo "$BEST_LINE" | cut -d, -f6)
BEST_FALLRATE=$(echo "$BEST_LINE" | cut -d, -f5)
log "best v4 ckpt: iter=$BEST_ITER ep_len=$BEST_EPLEN fall_rate=$BEST_FALLRATE"
BEST_CKPT=$(ls logs/FixedStageFv4/*fixedFv4-locomotion-hunter/model_${BEST_ITER}.pt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
  log "WARN: model_${BEST_ITER}.pt not found; v5 script will auto-pick newest model_*.pt"
  unset BEST_CKPT
else
  log "warm-loading v5 from $BEST_CKPT"
  export WARM_CKPT="$BEST_CKPT"
fi

log "launching v5 (scripts/ablation/fixed_stageF_v5.sh)"
nohup bash scripts/ablation/fixed_stageF_v5.sh > logs/FixedStageFv5/runner.log 2>&1 &
V5_PID=$!
log "v5 launched, PID=$V5_PID, log=logs/FixedStageFv5/runner.log"
echo "$V5_PID" > logs/FixedStageFv5/v5.pid
log "watcher exiting cleanly"
