#!/usr/bin/env bash
# Watchdog (per Codex thread 019e454a, option D execution):
# Detects when the running warm-start pilot finishes A8 and enters its A9_full
# Stage P, then KILLS the pilot before A9 wastes GPU on a confounded run
# (history-arm Stage P at 500 iters is insufficient -> A9 would launch from a
# near-degenerate policy). Immediately launches A9_longP with PLANE_ITERS=2000
# so A9 starts from a verified-walking plane policy.
#
# Leaves A0 / A3 / A8 results from the current pilot intact (A0 + A8 are
# clean; A3 is acknowledged-confounded and will be excluded from the verdict).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

WAVE_LOG="logs/WarmstartPilot_wave.log"
A9_LONGP_LOG="logs/WarmstartPilot_A9longP.log"
TRIGGER="STAGE P (plane,NO_DR,500 iters): A9_full_P"

echo "[watchdog] $(date +%H:%M:%S) waiting for A8 -> A9_full_P transition..."
while true; do
  if grep -Fq "$TRIGGER" "$WAVE_LOG" 2>/dev/null; then
    echo "[watchdog] $(date +%H:%M:%S) detected A9 start - killing pilot"
    pkill -f "warmstart_pilot.sh" 2>/dev/null
    sleep 3
    pkill -9 -f "experiment_name=A9_full_P" 2>/dev/null
    sleep 5
    rm -rf logs/WarmstartPilot/A9_full logs/WarmstartPilot/2026*A9_full_*-locomotion-hunter 2>/dev/null
    sleep 2
    echo "[watchdog] $(date +%H:%M:%S) launching A9_longP (PLANE_ITERS=2000)"
    PLANE_ITERS=2000 FURROW_ITERS=1500 SAVE_EVERY=250 \
      NUM_ENVS=2048 EVAL_ENVS=256 EVAL_EPS=64 \
      ARMS="A9_full" \
      nohup bash scripts/ablation/warmstart_pilot.sh > "$A9_LONGP_LOG" 2>&1 &
    echo "[watchdog] $(date +%H:%M:%S) A9_longP launched (PID $!)"
    exit 0
  fi
  sleep 30
done
