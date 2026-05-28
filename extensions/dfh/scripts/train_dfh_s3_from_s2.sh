#!/usr/bin/env bash
# Stage 3 DFH training, warm-started from the latest Stage 2 DFH checkpoint.
# Enables full-depth/orientation variability terrain, wet-soil DR, and adds
# the sinkage-excess guardrail. Drops LR slightly to harden the policy.

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
: "${S2_CKPT:=}"

if [[ -z "${S2_CKPT}" ]]; then
  S2_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_from_S1_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S2_CKPT}" || ! -f "${S2_CKPT}" ]]; then
  echo "ERROR: no stage-2 checkpoint found. Set S2_CKPT=... or run S2 first."
  exit 1
fi

if [[ "${SMOKE}" == "1" ]]; then
  ITERS=50; ENVS=256; RUN_NAME="DFH_S3_smoke_seed${SEED}"
else
  ITERS=3500; ENVS=2048; RUN_NAME="DFH_S3_from_S2_seed${SEED}"
fi

echo "=========================================================="
echo "  DFH Stage 3, warm-start from ${S2_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS} (LR 7.5e-5)"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +terrain=terrain_dfh_stage3_full \
  +domain_rand=DR_dfh_wet \
  +rewards=loco/reward_hunter_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S2_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++rewards.reward_scales.penalty_sinkage_excess=-20 \
  ++rewards.reward_scales.feet_air_time=1.0 \
  ++rewards.terms.penalty_sinkage_excess.sigma_m=0.065 \
  ++algo.config.num_learning_iterations="${ITERS}" \
  ++algo.config.num_steps_per_env=32 \
  ++algo.config.actor_learning_rate=7.5e-5 \
  ++algo.config.critic_learning_rate=7.5e-5 \
  ++algo.config.desired_kl=0.005 \
  ++algo.config.clip_param=0.12 \
  ++algo.config.max_grad_norm=0.8 \
  ++algo.config.entropy_coef=0.002 \
  ++algo.config.load_optimizer=False \
  ++algo.config.save_interval=50 \
  project_name=DFH_Hunter_ROA \
  experiment_name="${RUN_NAME}"
