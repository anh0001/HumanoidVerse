#!/usr/bin/env bash
# Overnight driver: run the walk-repair training, then auto-eval the result for
# vel_err (the metric that matters now) so the morning report says whether
# forward walking was restored on gentle soil.
#
# Stage 1: train_dfh_walk_repair_v1.sh (1500 iters, v7 warm-start, gentle soil,
#          v7 walking objective).
# Stage 2: eval the final ckpt with sample_eps at cmd [0.3,0,0], eval_match_train
#          (default True), logging Velocity error + ep_len + falls.
#
# Decision left for morning:
#   vel_err <= 0.13  -> walking restored on gentle soil; re-run dose ladder WITH
#                       a vel_err gate per rung.
#   vel_err  > 0.20  -> repair insufficient; needs stronger tracking / reward
#                       redesign (escalate to Codex).

set -uo pipefail
cd "$(dirname "$0")/../../.."

SEED="${SEED:-1}"
SCRIPTS="extensions/dfh/scripts"
PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"
LOGROOT="logs/DFH_Hunter_ROA"
AP_LOG="${LOGROOT}/DFH_walk_repair_overnight.log"

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${AP_LOG}"; }

log "=========================================================="
log "  WALK-REPAIR OVERNIGHT START (seed=${SEED})"
log "=========================================================="

# Stage 1: train
log "Stage 1: launching walk-repair training (1500 iters)..."
SEED="${SEED}" ITERS=1500 ENVS=2048 \
  bash "${SCRIPTS}/train_dfh_walk_repair_v1.sh" \
  > "${LOGROOT}/DFH_walk_repair_v1_train.log" 2>&1
log "Stage 1: training process returned."

# Find the result run dir + highest ckpt.
RUN_DIR=$(ls -dt ${LOGROOT}/*-DFH_walk_repair_v1_from_v7_seed${SEED}-locomotion-hunter/ 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
  log "ERROR: no walk-repair run dir found. Aborting eval."
  exit 1
fi
CKPT=$(ls "${RUN_DIR}"model_*.pt 2>/dev/null \
  | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}')
log "Stage 1 result: ${CKPT}"

# Stage 2: eval the result for vel_err (the metric that matters).
log "Stage 2: evaluating restored walking (cmd 0.3, train-matched)..."
EVAL_LOG="${LOGROOT}/DFH_walk_repair_v1_eval.log"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"
"${PYTHON}" humanoidverse/sample_eps.py \
  +simulator=isaacsim +checkpoint="${CKPT}" \
  +eval_command="[0.3,0.0,0.0]" \
  num_envs=64 +num_episodes=64 headless=True \
  > "${EVAL_LOG}" 2>&1
log "Stage 2: eval returned."

VERR=$(grep "Velocity error" "${EVAL_LOG}" 2>/dev/null | tail -1 | sed -E 's/.*Velocity error \(m\/s\): //')
EPLEN=$(grep "Average episode length" "${EVAL_LOG}" 2>/dev/null | tail -1 | sed -E 's/.*: //')
FALLS=$(grep "Total falls" "${EVAL_LOG}" 2>/dev/null | tail -1 | sed -E 's/.*: //')
log "=========================================================="
log "  WALK-REPAIR RESULT (vs v7 baseline vel_err 0.117):"
log "    Velocity error : ${VERR}"
log "    Episode length : ${EPLEN}"
log "    Total falls    : ${FALLS}"
log "  vel_err<=0.13 => walking restored; >0.20 => needs more work."
log "=========================================================="
log "WALK-REPAIR OVERNIGHT DONE."
