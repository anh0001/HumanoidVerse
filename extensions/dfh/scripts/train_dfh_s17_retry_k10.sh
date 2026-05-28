#!/usr/bin/env bash
# Stage 1.7 DFH RETRY (k=10) — first S1.7 attempt drag-saturated (23% body
# weight, sink pinned, p95 drag at 260 N cap). The nominal k=16 in
# terrain_dfh_stage17_geom.yaml reproduced the failed S1.5-v1 regime.
# Codex (thread 019e6d04-...) recipe:
#   - Restart from HARDEN m8350 (NOT the contaminated S1.7 m9750)
#   - sinkage_drag_k 16 → 10 (same as harden — isolate geometry change only)
#   - max_drag_force_n 260 → 240
#   - feet_air_time 0.10 → 0.05 (same as harden)
#   - Calmer optim: LR 7.5e-5, KL 0.005, clip 0.10, entropy 0.002
# Early-abort if dfh_applied_drag_pct_weight > 0.12 at iter 300-400.

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
: "${HARDEN_CKPT:=}"

if [[ -z "${HARDEN_CKPT}" ]]; then
  HARDEN_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S15_harden_k10_from_retune_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${HARDEN_CKPT}" || ! -f "${HARDEN_CKPT}" ]]; then
  echo "ERROR: no harden checkpoint found. Set HARDEN_CKPT=... or run harden first."
  exit 1
fi

ITERS=1400
ENVS=2048
RUN_NAME="DFH_S17_geom_k10_from_S15_harden_seed${SEED}"

echo "=========================================================="
echo "  DFH Stage 1.7 RETRY (k=10), warm-start from"
echo "  ${HARDEN_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage17_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${HARDEN_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=10.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=240.0 \
  ++rewards.reward_scales.penalty_sinkage_excess=0 \
  ++rewards.reward_scales.feet_air_time=0.05 \
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
