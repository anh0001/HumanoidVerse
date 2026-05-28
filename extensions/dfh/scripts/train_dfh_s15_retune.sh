#!/usr/bin/env bash
# Stage 1.5 DFH RETUNE — first S1.5 attempt drag-saturated (137 N, 22.5% body
# weight, sink pinned at -0.065 floor). Codex (thread 019e67b1-...) recipe:
#   - sinkage_drag_k 16 → 6 (S1-equivalent under deeper sink floor)
#   - max_drag_force_n 260 → 220
#   - feet_air_time 0.15 → 0.0 (un-clipped reward was penalizing short steps)
#   - LR 1e-4 → 7.5e-5, entropy 0.003 → 0.002, KL 0.006 → 0.005, clip 0.12 → 0.10
# Resume from S1-extend m6750 (NOT the oscillated S1.5 m7950).

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
: "${S1_CKPT:=logs/DFH_Hunter_ROA/20260526_220243-DFH_S1_extend_from_5750_seed1-locomotion-hunter/model_6750.pt}"

if [[ ! -f "${S1_CKPT}" ]]; then
  echo "ERROR: S1 checkpoint missing at ${S1_CKPT}"
  exit 1
fi

ITERS=1000
ENVS=2048
RUN_NAME="DFH_S15_retune_k6_from_S1_seed${SEED}"

echo "=========================================================="
echo "  DFH Stage 1.5 RETUNE (k=6, no feet_air_time), warm-start from"
echo "  ${S1_CKPT}"
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
  ++terrain.dfh.force_coupling.sinkage_drag_k=6.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=220.0 \
  ++rewards.reward_scales.penalty_sinkage_excess=0 \
  ++rewards.reward_scales.feet_air_time=0.0 \
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
