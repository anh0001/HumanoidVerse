#!/usr/bin/env bash
# DFH WALK-REPAIR v1 — Codex (thread 019e7e15).
# DIAGNOSIS CONFIRMED: v7 m4250 walks (vel_err 0.117 @ cmd 0.3) but every DFH
# chain policy only balances (vel_err ~= command, fwd vel ~0). The dose ladder
# optimized drag-SURVIVAL (ep_len + termination gates), never command tracking,
# so it traded away locomotion.
#
# Repair strategy: re-establish forward walking ON soil by warm-starting from the
# KNOWN WALKER (v7) and restoring v7's walking objective, then adding only GENTLE
# soil. Gate on vel_err / forward speed, NOT episode length.
#
# v7 walking objective restored:
#   tracking_lin_vel=4.0 (chain had 2.0), feet_air_time=1.0 (chain had 0.08),
#   forward-only commands lin_vel_x=[0.25,0.45] (chain went bidirectional +-0.3),
#   tighter sigma 0.20.
# Gentlest soil (proven walkable at S1): k=8, max=200, floor=-0.05.
#
# If this restores walking on gentle soil (target vel_err <= 0.13 @ cmd 0.3),
# the dose ladder can be re-run WITH a vel_err gate at every rung.

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
: "${ITERS:=1500}"
: "${ENVS:=2048}"
: "${V7_CKPT:=logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt}"
[[ -f "${V7_CKPT}" ]] || { echo "ERROR: v7 ckpt missing: ${V7_CKPT}"; exit 1; }

RUN_NAME="DFH_walk_repair_v1_from_v7_seed${SEED}"

echo "=========================================================="
echo "  DFH WALK-REPAIR v1 (gentle soil + v7 walking objective)"
echo "  warm=${V7_CKPT}"
echo "  tracking_lin_vel=4.0, feet_air_time=1.0, fwd cmd [0.25,0.45]"
echo "  soil: k=8, max=200, floor=-0.05"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "  TARGET: post-train eval vel_err <= 0.13 @ cmd 0.3 (restored walking)"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${V7_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.05 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=8.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=200.0 \
  ++rewards.reward_scales.tracking_lin_vel=4.0 \
  ++rewards.reward_scales.tracking_ang_vel=1.0 \
  ++rewards.reward_scales.feet_air_time=1.0 \
  ++rewards.reward_scales.penalty_sinkage_excess=0.0 \
  ++rewards.reward_scales.termination=-60.0 \
  ++rewards.reward_tracking_sigma.lin_vel=0.20 \
  ++env.config.locomotion_command_ranges.lin_vel_x="[0.25,0.45]" \
  ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
  ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
  ++algo.config.num_learning_iterations="${ITERS}" \
  ++algo.config.num_steps_per_env=32 \
  ++algo.config.actor_learning_rate=1.0e-4 \
  ++algo.config.critic_learning_rate=1.0e-4 \
  ++algo.config.desired_kl=0.01 \
  ++algo.config.clip_param=0.2 \
  ++algo.config.max_grad_norm=1.0 \
  ++algo.config.entropy_coef=0.005 \
  ++algo.config.load_optimizer=False \
  ++algo.config.save_interval=50 \
  project_name=DFH_Hunter_ROA \
  experiment_name="${RUN_NAME}"
