#!/usr/bin/env bash
# Paired-control overnight driver (Codex 019e7fcd). Runs both arms sequentially,
# evals each for vel_err @ cmd 0.3, logs the comparison for morning review.
#
# Arm A (rigid): terrain_furrows_stage1_easy, no DFH
# Arm B (soil):  terrain_dfh_stage1_easy + k=8/max=200/floor=-0.05
# Both: v7_std055 warm-start, reward_hunter_locomotion, NO_domain_rand,
#       forward cmd [0.25,0.45], v7-conservative PPO. 800 iters each.
#
# Decision (vs v7 baseline vel_err 0.117):
#   A walks (vel_err<=0.15), B walks       -> mild soil walking WORKS; the v1
#                                             failure was the bad fine-tune stack.
#   A walks, B fails (vel_err>0.25)        -> soil/reward problem; apply Codex's
#                                             progress+asymmetric reward next.
#   A fails                                -> fine-tune stack bad independent of soil.

set -uo pipefail
cd "$(dirname "$0")/../../.."

SEED="${SEED:-1}"
SCRIPTS="extensions/dfh/scripts"
PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"
LOGROOT="logs/DFH_Hunter_ROA"
AP_LOG="${LOGROOT}/DFH_paired_control.log"

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${AP_LOG}"; }

run_arm() {
  local run_name="$1" terrain="$2" dfh_ov="$3"
  log "ARM ${run_name}: training (terrain=${terrain})..."
  SEED="${SEED}" ITERS=800 ENVS=2048 \
    TERRAIN="${terrain}" RUN_NAME="${run_name}" DFH_OVERRIDES="${dfh_ov}" \
    bash "${SCRIPTS}/walk_paired_control_arm.sh" \
    > "${LOGROOT}/${run_name}_train.log" 2>&1
  log "ARM ${run_name}: training returned."

  local rd ckpt
  rd=$(ls -dt ${LOGROOT}/*-${run_name}-locomotion-hunter/ 2>/dev/null | head -1)
  ckpt=$(ls "${rd}"model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}')
  log "ARM ${run_name}: result ckpt ${ckpt}; evaluating vel_err @ cmd 0.3..."
  # shellcheck disable=SC1091
  source "$ISAAC_PATH/setup_python_env.sh"
  "${PYTHON}" humanoidverse/sample_eps.py \
    +simulator=isaacsim +checkpoint="${ckpt}" \
    +eval_command="[0.3,0.0,0.0]" \
    num_envs=64 +num_episodes=64 headless=True \
    > "${LOGROOT}/${run_name}_eval.log" 2>&1
  local verr eplen falls
  verr=$(grep "Velocity error" "${LOGROOT}/${run_name}_eval.log" 2>/dev/null | tail -1 | sed -E 's/.*: //')
  eplen=$(grep "Average episode length" "${LOGROOT}/${run_name}_eval.log" 2>/dev/null | tail -1 | sed -E 's/.*: //')
  falls=$(grep "Total falls" "${LOGROOT}/${run_name}_eval.log" 2>/dev/null | tail -1 | sed -E 's/.*: //')
  log "ARM ${run_name} RESULT: vel_err=${verr} | ep_len=${eplen} | falls=${falls}"
}

log "=========================================================="
log "  PAIRED CONTROL START (seed=${SEED}) — v7 baseline vel_err 0.117"
log "=========================================================="

run_arm "DFH_paired_A_rigid_seed${SEED}"  "terrain_furrows_stage1_easy" ""
run_arm "DFH_paired_B_soil_seed${SEED}"   "terrain_dfh_stage1_easy" \
        "++terrain.dfh.params.sinkage_floor_m=-0.05 ++terrain.dfh.force_coupling.sinkage_drag_k=8.0 ++terrain.dfh.force_coupling.max_drag_force_n=200.0"

log "=========================================================="
log "  PAIRED CONTROL DONE. Compare A (rigid) vs B (soil) vel_err above."
log "  A&B walk => mild soil OK; A walk/B fail => reward fix; A fail => stack bad."
log "=========================================================="
