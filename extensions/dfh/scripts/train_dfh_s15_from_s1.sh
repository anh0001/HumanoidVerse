#!/usr/bin/env bash
# Stage 1.5 DFH (soil bridge) — S1 geometry, intermediate DFH dynamics.
# Warm-start from best S1 (or S1-extend) checkpoint.
# Authored with Codex (gpt-5.2). See thread 019e645b-9f58-7fb3-a4da-7dde95418fda.

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
: "${S1_CKPT:=}"

# Prefer S1 extension if it exists, fall back to original S1.
if [[ -z "${S1_CKPT}" ]]; then
  S1_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S1_extend_from_5750_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S1_CKPT}" ]]; then
  S1_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S1_from_v7_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S1_CKPT}" || ! -f "${S1_CKPT}" ]]; then
  echo "ERROR: no S1 checkpoint found. Set S1_CKPT=... or run S1 first."
  exit 1
fi

ITERS=1200
ENVS=2048
RUN_NAME="DFH_S15_soil_from_S1_seed${SEED}"

echo "=========================================================="
echo "  DFH Stage 1.5 (soil bridge), warm-start from ${S1_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage15_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S1_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++rewards.reward_scales.penalty_sinkage_excess=0 \
  ++rewards.reward_scales.feet_air_time=0.15 \
  ++algo.config.num_learning_iterations="${ITERS}" \
  ++algo.config.num_steps_per_env=32 \
  ++algo.config.actor_learning_rate=1.0e-4 \
  ++algo.config.critic_learning_rate=1.0e-4 \
  ++algo.config.desired_kl=0.006 \
  ++algo.config.clip_param=0.12 \
  ++algo.config.max_grad_norm=0.8 \
  ++algo.config.entropy_coef=0.003 \
  ++algo.config.load_optimizer=False \
  ++algo.config.save_interval=50 \
  project_name=DFH_Hunter_ROA \
  experiment_name="${RUN_NAME}"
