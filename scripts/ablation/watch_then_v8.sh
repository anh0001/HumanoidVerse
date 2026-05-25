#!/usr/bin/env bash
# Watcher: blocks until the v7 m4250 192-ep validation eval finishes, then
# applies Codex's gate (thread 019e54fe) and auto-launches v8 stage2 if it
# passes.
#
# Gate (Codex):
#   - >= 850 ep_len           -> stage1 SOLVED, auto-launch v8 stage2
#   - 700-850 ep_len           -> ambiguous, run another seed validation
#   - <700  ep_len             -> noise spike, re-eval m3800/m3900/m4250
#                                 same-seed and pick best lower-bound ckpt
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VAL_LOG="logs/diagnostics/v7_m4250_extended_eval_192.log"
WATCH_LOG="logs/FixedStageFv8_s2/watcher.log"
mkdir -p "$(dirname "$WATCH_LOG")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

log "watching for validation eval to complete: $VAL_LOG"

# Wait for sample_eps process to exit (validation eval).
while pgrep -f "humanoidverse/sample_eps" > /dev/null 2>&1; do sleep 20; done
log "no sample_eps procs alive"

if [ ! -f "$VAL_LOG" ]; then
  log "ERROR: $VAL_LOG missing; aborting"
  exit 1
fi

VAL_EPLEN=$(grep "Average episode length:" "$VAL_LOG" | tail -1 | sed -E 's/.*: //;s/ steps.*//')
VAL_EPS=$(grep "Episodes completed:" "$VAL_LOG" | tail -1 | sed -E 's/.*: //;s/[^0-9]//g')
log "validation result: $VAL_EPS episodes, ep_len = $VAL_EPLEN"

# Apply Codex gate.
VAL_INT=$(python3 -c "print(int(float('$VAL_EPLEN')))")
if [ "$VAL_INT" -ge 850 ]; then
  log "GATE PASS (>=850): stage1 SOLVED. Auto-launching v8 stage2."
  nohup bash scripts/ablation/fixed_stageF_v8_stage2.sh \
        > logs/FixedStageFv8_s2/runner.log 2>&1 &
  V8_PID=$!
  disown
  log "v8 stage2 launched, PID=$V8_PID, log=logs/FixedStageFv8_s2/runner.log"
  echo "$V8_PID" > logs/FixedStageFv8_s2/v8.pid
elif [ "$VAL_INT" -ge 700 ]; then
  log "GATE AMBIGUOUS (700-850): running second-seed validation"
  CKPT=logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt
  OUT=logs/diagnostics/v7_m4250_validation_seed2.log
  export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
  export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
  export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
  export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
  # shellcheck disable=SC1091
  source "$ISAAC_PATH/setup_python_env.sh"
  PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$CKPT" \
    +terrain=terrain_furrows_stage1_easy +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs=256 ++simulator.config.scene.num_envs=256 \
    ++env.config.env_spacing=2.5 +num_episodes=192 headless=True \
    ++env.config.termination.terminate_by_contact=True seed=7 \
    > "$OUT" 2>&1
  SEED2_EPLEN=$(grep "Average episode length:" "$OUT" | tail -1 | sed -E 's/.*: //;s/ steps.*//')
  log "second-seed validation: ep_len = $SEED2_EPLEN"
  SEED2_INT=$(python3 -c "print(int(float('$SEED2_EPLEN')))")
  if [ "$SEED2_INT" -ge 850 ] || [ "$VAL_INT" -ge 850 ]; then
    log "promoted: at least one seed >=850, auto-launching v8 stage2"
    nohup bash scripts/ablation/fixed_stageF_v8_stage2.sh \
          > logs/FixedStageFv8_s2/runner.log 2>&1 &
    V8_PID=$!
    disown
    log "v8 stage2 launched, PID=$V8_PID"
    echo "$V8_PID" > logs/FixedStageFv8_s2/v8.pid
  else
    log "both seeds <850; NOT auto-launching. Manual review needed."
    log "next step: same-seed eval of m3800/m3900/m4250, pick best lower-bound"
  fi
else
  log "GATE FAIL (<700): 1029 was eval noise. NOT launching v8."
  log "next step: same-seed 192-ep eval of m3800/m3900/m4250"
fi

log "watcher exiting"
