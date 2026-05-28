#!/usr/bin/env bash
# Stage 1 DFH training, warm-started from the v7 m4250 furrow checkpoint with
# action std flattened to 0.55 (less stochasticity on deformable contact).
#
# Recipe authored with Codex (gpt-5.2). See conversation thread
# 019e5e77-612e-7db2-a354-0966ef762c1a for the rationale.
#
# Usage:
#   bash extensions/dfh/scripts/train_dfh_s1_from_v7.sh             # full 1500 iters
#   SMOKE=1 bash extensions/dfh/scripts/train_dfh_s1_from_v7.sh     # 50 iters, 256 envs (~4 min)
#   SEED=2  bash extensions/dfh/scripts/train_dfh_s1_from_v7.sh     # different seed

set -euo pipefail
cd "$(dirname "$0")/../../.."

# IsaacSim env (same pattern as scripts/s0_gate/run_probe.sh).
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"

PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: ${PYTHON} not executable"; exit 1; }

V7_BASE="logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter"
WARM_CKPT="${V7_BASE}/model_4250_std055.pt"

if [[ ! -f "${WARM_CKPT}" ]]; then
  echo "ERROR: warm-start checkpoint missing at ${WARM_CKPT}"
  echo "Patch the v7 actor std with:"
  echo "  python3 extensions/dfh/scripts/patch_actor_std.py"
  exit 1
fi

: "${SEED:=1}"
: "${SMOKE:=0}"

if [[ "${SMOKE}" == "1" ]]; then
  ITERS=50
  ENVS=256
  RUN_NAME="DFH_S1_smoke_seed${SEED}"
else
  ITERS=1500
  ENVS=2048
  RUN_NAME="DFH_S1_from_v7_seed${SEED}"
fi

EXP_DIR="logs/DFH_Hunter_ROA/${RUN_NAME}"

echo "=========================================================="
echo "  DFH Stage 1, warm-start v7 m4250 (std=0.55), seed=${SEED}"
echo "  envs=${ENVS}, iters=${ITERS}, output=${EXP_DIR}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage1_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${WARM_CKPT}" \
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
