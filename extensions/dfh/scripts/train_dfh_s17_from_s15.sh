#!/usr/bin/env bash
# Stage 1.7 DFH (geometry bridge) — S2 geometry, intermediate (S1.5) dynamics.
# Warm-start from best S1.5 checkpoint.

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
: "${S15_CKPT:=}"

# Prefer the hardening checkpoint, fall back to retune, then the (failed) v1.
if [[ -z "${S15_CKPT}" ]]; then
  S15_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S15_harden_k10_from_retune_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S15_CKPT}" ]]; then
  S15_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S15_retune_k6_from_S1_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S15_CKPT}" ]]; then
  S15_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S15_soil_from_S1_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S15_CKPT}" || ! -f "${S15_CKPT}" ]]; then
  echo "ERROR: no S1.5 checkpoint found. Set S15_CKPT=... or run S1.5 first."
  exit 1
fi

ITERS=1400
ENVS=2048
RUN_NAME="DFH_S17_geom_from_S15_seed${SEED}"

echo "=========================================================="
echo "  DFH Stage 1.7 (geom bridge), warm-start from ${S15_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage17_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S15_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++rewards.reward_scales.penalty_sinkage_excess=0 \
  ++rewards.reward_scales.feet_air_time=0.10 \
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
