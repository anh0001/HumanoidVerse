#!/usr/bin/env bash
# Stage 2 DFH training, warm-started from the latest Stage 1 DFH checkpoint.
# Adds feet_air_time reward to pressure the policy toward proper gait timing.
#
# Usage:
#   bash extensions/dfh/scripts/train_dfh_s2_from_s1.sh                 # full 2200 iters
#   S1_CKPT=logs/DFH_Hunter_ROA/DFH_S1_from_v7_seed1/.../model_NNN.pt \
#     bash extensions/dfh/scripts/train_dfh_s2_from_s1.sh               # pin checkpoint

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
: "${SMOKE:=0}"
: "${S1_CKPT:=}"

if [[ -z "${S1_CKPT}" ]]; then
  # HumanoidVerse output dir is `<timestamp>-<exp_name>-<task>-<robot>`; the
  # exp_name embeds the seed. Match the timestamp prefix loosely.
  S1_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S1_from_v7_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S1_CKPT}" || ! -f "${S1_CKPT}" ]]; then
  echo "ERROR: no stage-1 checkpoint found. Set S1_CKPT=... or run S1 first."
  exit 1
fi

if [[ "${SMOKE}" == "1" ]]; then
  ITERS=50; ENVS=256; RUN_NAME="DFH_S2_smoke_seed${SEED}"
else
  ITERS=2200; ENVS=2048; RUN_NAME="DFH_S2_from_S1_seed${SEED}"
fi

echo "=========================================================="
echo "  DFH Stage 2, warm-start from ${S1_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S1_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++rewards.reward_scales.penalty_sinkage_excess=0 \
  ++rewards.reward_scales.feet_air_time=0.5 \
  ++algo.config.num_learning_iterations="${ITERS}" \
  ++algo.config.num_steps_per_env=32 \
  ++algo.config.actor_learning_rate=1.0e-4 \
  ++algo.config.critic_learning_rate=1.0e-4 \
  ++algo.config.desired_kl=0.006 \
  ++algo.config.clip_param=0.12 \
  ++algo.config.max_grad_norm=0.8 \
  ++algo.config.entropy_coef=0.002 \
  ++algo.config.load_optimizer=False \
  ++algo.config.save_interval=50 \
  project_name=DFH_Hunter_ROA \
  experiment_name="${RUN_NAME}"
