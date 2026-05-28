#!/usr/bin/env bash
# Stage 1 DFH extension — resume from S1 m5750 to push mean ep_len ≥16s.
# S1 stopped at 14.6s (5750 iters); promotion gate requires 16s.
#
# Usage:
#   bash extensions/dfh/scripts/train_dfh_s1_extend.sh
#   SEED=1 bash extensions/dfh/scripts/train_dfh_s1_extend.sh

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
: "${S1_CKPT:=logs/DFH_Hunter_ROA/20260526_022909-DFH_S1_from_v7_seed1-locomotion-hunter/model_5750.pt}"

if [[ ! -f "${S1_CKPT}" ]]; then
  echo "ERROR: S1 checkpoint missing at ${S1_CKPT}"
  exit 1
fi

ITERS=1000
ENVS=2048
RUN_NAME="DFH_S1_extend_from_5750_seed${SEED}"

echo "=========================================================="
echo "  DFH Stage 1 extension, warm-start from ${S1_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS} (target ep_len ≥16s)"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage1_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S1_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++rewards.reward_scales.penalty_sinkage_excess=0 \
  ++rewards.reward_scales.feet_air_time=0 \
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
