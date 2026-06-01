#!/usr/bin/env bash
# Stage 2 DFH RETRY (k=12, dose step from S1.7-retry).
# Codex (thread 019e6eb2-ff0a-71f3-9241-d8e2b15d6b1f) recipe:
#   - Warm-start: S1.7-retry m9750 (ep_len 687, clipped_frac 0.022, drag_pct 0.167)
#   - Dose: sinkage_drag_k 10 → 12 (+20% step; +40% or +60% would re-saturate)
#   - max_drag_force_n 240 → 260 (headroom so dose ≠ clipping)
#   - feet_air_time 0.05 → 0.08 (middle step; restore gait pressure)
#   - penalty_sinkage_excess kept at 0.0 (sink is structural, not a learning gate)
#   - Same calm PPO: LR 7.5e-5, KL 0.005, clip 0.10, entropy 0.002
#   - 1400 iters (this is a BRIDGE, not a fresh stage)
# Real failure gate is drag_clipped_frac, NOT drag_pct (which is misleading
# under deepening sink floor). Promotion needs ep_len ≥ 650 and
# clipped_frac ≤ 0.05 over last 200 iters.

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
: "${S17_CKPT:=}"

if [[ -z "${S17_CKPT}" ]]; then
  S17_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S17_geom_k10_from_S15_harden_seed${SEED}-*/model_9750.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S17_CKPT}" ]]; then
  S17_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S17_geom_k10_from_S15_harden_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S17_CKPT}" || ! -f "${S17_CKPT}" ]]; then
  echo "ERROR: no S1.7-retry checkpoint found. Set S17_CKPT=... explicitly."
  exit 1
fi

ITERS=1400
ENVS=2048
RUN_NAME="DFH_S2_retry_k12_from_S17_retry9750_seed${SEED}"

echo "=========================================================="
echo "  DFH Stage 2 RETRY (k=12, +20% dose), warm-start from"
echo "  ${S17_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S17_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=12.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=260.0 \
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
