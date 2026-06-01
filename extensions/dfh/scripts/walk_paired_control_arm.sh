#!/usr/bin/env bash
# Paired-control arm — Codex (thread 019e7fcd). Decisive experiment: is the
# DFH-chain's loss of locomotion caused by SOIL, or by the fine-tune stack
# (stage-2 terrain + survival reward + DR) that walk-repair-v1 wrongly inherited?
#
# Both arms IDENTICAL except DFH force coupling:
#   Arm A (rigid): TERRAIN=terrain_furrows_stage1_easy  (no DFH)
#   Arm B (soil):  TERRAIN=terrain_dfh_stage1_easy + mild k=8/max=200/floor=-0.05
#
# Clean control: v7 walking reward (reward_hunter_locomotion, NOT reward_hunter_dfh),
# NO_domain_rand, v7 forward command [0.25,0.45], v7-conservative PPO, warm-start
# from the patched walker model_4250_std055.pt. NO curriculum preset (that was
# the v1 bug — stage2_dfh dragged in medium terrain + survival reward + DR).
#
# Interpretation:
#   A collapses          -> fine-tune stack is bad independent of soil
#   A holds, B collapses (modest drag) -> reward/objective problem
#   A holds, B collapses (high drag/clip) -> soil dose/model problem
#
# Required env vars: TERRAIN, RUN_NAME. Optional: DFH_OVERRIDES (string), ITERS.

set -euo pipefail
cd "$(dirname "$0")/../../.."

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"

PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: ${PYTHON} not executable"; exit 1; }

: "${SEED:=1}"
: "${ITERS:=1000}"
: "${ENVS:=2048}"
: "${TERRAIN:?must set TERRAIN}"
: "${RUN_NAME:?must set RUN_NAME}"
: "${DFH_OVERRIDES:=}"
: "${WARM_CKPT:=logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250_std055.pt}"
[[ -f "${WARM_CKPT}" ]] || { echo "ERROR: warm ckpt missing: ${WARM_CKPT}"; exit 1; }

echo "=========================================================="
echo "  PAIRED CONTROL arm: ${RUN_NAME}"
echo "  terrain=${TERRAIN}  dfh_overrides='${DFH_OVERRIDES}'"
echo "  warm=${WARM_CKPT}  envs=${ENVS} iters=${ITERS}"
echo "=========================================================="

# shellcheck disable=SC2086
"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion \
  algo=ppo_roa \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  +terrain="${TERRAIN}" \
  +rewards=loco/reward_hunter_locomotion \
  ++rewards.reward_scales.tracking_lin_vel=4.0 \
  +domain_rand=NO_domain_rand \
  checkpoint="${WARM_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++env.config.locomotion_command_ranges.lin_vel_x="[0.25,0.45]" \
  ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
  ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
  ++algo.config.num_learning_iterations="${ITERS}" \
  ++algo.config.num_steps_per_env=24 \
  ++algo.config.desired_kl=0.006 \
  ++algo.config.clip_param=0.12 \
  ++algo.config.entropy_coef=0.002 \
  ++algo.config.load_optimizer=False \
  ++algo.config.save_interval=50 \
  ${DFH_OVERRIDES} \
  project_name=DFH_Hunter_ROA \
  experiment_name="${RUN_NAME}"
