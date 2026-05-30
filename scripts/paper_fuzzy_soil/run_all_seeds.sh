#!/usr/bin/env bash
# Sequential queue for the fuzzy-soil-on-furrows A/B experiment.
#
# Assumes Arm A seed 1 may already be running. Each step:
#   1. Waits for the previous PID to exit (or skips if no previous PID).
#   2. Skips its own work if the expected output checkpoint already exists.
#   3. Launches the next training run in the foreground (one GPU).
# After all 6 training runs, runs eval_arms.sh.
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/run_all_seeds.sh > logs/FuzzySoilFurrows/queue.log 2>&1 &
#   disown
#
# To skip the wait for an existing seed-1 PID, leave logs/FuzzySoilFurrows/ArmA_v7_baseline_seed1/pid absent.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ROOT="logs/FuzzySoilFurrows"
mkdir -p "$ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [queue] $*"; }

wait_for_pid() {
  # Block until $1 is no longer a running PID (or never was).
  local pid="${1:-}"
  [ -z "$pid" ] && return 0
  if ! ps -p "$pid" > /dev/null 2>&1; then
    log "PID $pid already exited"
    return 0
  fi
  log "waiting for PID $pid"
  while ps -p "$pid" > /dev/null 2>&1; do
    sleep 30
  done
  log "PID $pid done"
}

run_step() {
  # $1 = label, $2 = script-path, $3 = SEED, $4 = tracking-dir (also creates it),
  # $5 = sibling-dir glob to detect existing checkpoints. Hydra writes the actual
  # checkpoints to a TIMESTAMPED SIBLING of the tracking dir (e.g.
  # logs/FuzzySoilFurrows/20260528_HHMMSS-armA_seed1-locomotion-hunter/), so the
  # skip-check must search the sibling glob, not the tracking dir itself.
  local label="$1" script="$2" seed="$3" outdir="$4" ckpt_glob="$5"
  mkdir -p "$outdir"
  # shellcheck disable=SC2086
  if ls $ckpt_glob 2>/dev/null | grep -q .; then
    log "$label SKIP (checkpoint already exists matching $ckpt_glob)"
    return 0
  fi
  log "$label START"
  SEED="$seed" "$script" > "$outdir/nohup.out" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "$label FAIL exit=$rc — aborting queue"
    return $rc
  fi
  log "$label DONE"
  return 0
}

# ---- 0. wait for any currently-running Arm A seed 1 ----
PRE_PID_FILE="$ROOT/ArmA_v7_baseline_seed1/pid"
PRE_PID=""
[ -f "$PRE_PID_FILE" ] && PRE_PID="$(cat "$PRE_PID_FILE" 2>/dev/null)"
wait_for_pid "$PRE_PID"

# ---- 1-3. Arm A seeds 1, 2, 3 (skip if a sibling timestamped dir already has model_5000.pt) ----
run_step "armA s1" "$SCRIPT_DIR/train_armA_v7_baseline.sh" 1 \
  "$ROOT/ArmA_v7_baseline_seed1" \
  "$ROOT/*-armA_seed1-locomotion-*/model_5000.pt" || exit 1
run_step "armA s2" "$SCRIPT_DIR/train_armA_v7_baseline.sh" 2 \
  "$ROOT/ArmA_v7_baseline_seed2" \
  "$ROOT/*-armA_seed2-locomotion-*/model_5000.pt" || exit 1
run_step "armA s3" "$SCRIPT_DIR/train_armA_v7_baseline.sh" 3 \
  "$ROOT/ArmA_v7_baseline_seed3" \
  "$ROOT/*-armA_seed3-locomotion-*/model_5000.pt" || exit 1

# ---- 4-6. Arm B seeds 1, 2, 3 (each a 3-stage chain; skip if final S3 ckpt exists) ----
run_step "armB s1" "$SCRIPT_DIR/train_armB_paper_chain.sh" 1 \
  "$ROOT/ArmB_paper_seed1" \
  "$ROOT/*-armB_S3_seed1-locomotion-*/model_*.pt" || exit 1
run_step "armB s2" "$SCRIPT_DIR/train_armB_paper_chain.sh" 2 \
  "$ROOT/ArmB_paper_seed2" \
  "$ROOT/*-armB_S3_seed2-locomotion-*/model_*.pt" || exit 1
run_step "armB s3" "$SCRIPT_DIR/train_armB_paper_chain.sh" 3 \
  "$ROOT/ArmB_paper_seed3" \
  "$ROOT/*-armB_S3_seed3-locomotion-*/model_*.pt" || exit 1

# ---- 7. Joint evaluation ----
log "eval START"
"$SCRIPT_DIR/eval_arms.sh" > "$ROOT/eval.log" 2>&1
log "eval DONE — results in $ROOT/results.csv"
log "QUEUE COMPLETE"
