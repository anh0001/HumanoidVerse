#!/usr/bin/env bash
# Generic DFH stage runner — parameterized by env vars. Used by the S3 autopilot
# driver to run each rung with identical calm-PPO settings, varying only the
# DFH terrain params (floor, k, max_drag) and the warm-start checkpoint.
#
# Required env vars:
#   FLOOR     sinkage_floor_m (e.g. -0.10)
#   KVAL      sinkage_drag_k  (e.g. 25)
#   MAXN      max_drag_force_n (e.g. 520)
#   WARM_CKPT path to warm-start checkpoint
#   RUN_NAME  experiment_name
# Optional:
#   ITERS     num_learning_iterations (default 800)
#   SEED      default 1

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
: "${ITERS:=800}"
: "${FLOOR:?must set FLOOR}"
: "${KVAL:?must set KVAL}"
: "${MAXN:?must set MAXN}"
: "${WARM_CKPT:?must set WARM_CKPT}"
: "${RUN_NAME:?must set RUN_NAME}"
[[ -f "${WARM_CKPT}" ]] || { echo "ERROR: WARM_CKPT not found: ${WARM_CKPT}"; exit 1; }

ENVS=2048
echo "=========================================================="
echo "  DFH generic stage: ${RUN_NAME}"
echo "  floor=${FLOOR}, k=${KVAL}, max=${MAXN}, iters=${ITERS}"
echo "  warm-start: ${WARM_CKPT}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${WARM_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m="${FLOOR}" \
  ++terrain.dfh.force_coupling.sinkage_drag_k="${KVAL}" \
  ++terrain.dfh.force_coupling.max_drag_force_n="${MAXN}" \
  ++rewards.reward_scales.feet_air_time=0.08 \
  ++rewards.reward_scales.penalty_sinkage_excess=0.0 \
  ++rewards.reward_scales.termination=-60.0 \
  ++algo.config.num_learning_iterations="${ITERS}" \
  ++algo.config.num_steps_per_env=32 \
  ++algo.config.actor_learning_rate=7.5e-5 \
  ++algo.config.critic_learning_rate=7.5e-5 \
  ++algo.config.desired_kl=0.005 \
  ++algo.config.clip_param=0.10 \
  ++algo.config.max_grad_norm=0.8 \
  ++algo.config.entropy_coef=0.002 \
  ++algo.config.load_optimizer=False \
  ++algo.config.save_interval=50 \
  project_name=DFH_Hunter_ROA \
  experiment_name="${RUN_NAME}"
