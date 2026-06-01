#!/usr/bin/env bash
# DFH S3-bridge-1 — Codex (thread 019e7384) recipe.
# After S2-dose-2 (k=15/max=320/floor=-0.08) passed 8/8 gates with clip_frac
# DROPPING to 0.062 (below dose-1's 0.067), the next high-value question is
# sink-axis transfer. Step the sink floor -0.08 → -0.09 while keeping the
# proven k=15/max=320 dose. Same calm PPO.
#
# Codex's bridge green-light criteria (more lenient than dose promotion):
#   ep_len ≥ 600, term ≥ -1.9, clip_frac ≤ 0.08
# If results materially below those numbers, do one more -0.08 rung
# (e.g., k=17/max=360, ratio 0.0472) before stepping sink again.
#
# After this passes: continue dose ladder at floor=-0.09 (S3-dose-1) before
# stepping sink to -0.10. Nominal S3 endgame remains floor=-0.10, k=40, max=400+
# (max may need upward revision per the headroom-neutral principle).

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
: "${DOSE2_CKPT:=}"

if [[ -z "${DOSE2_CKPT}" ]]; then
  DOSE2_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_dose2_k15_max320_from_13950_seed${SEED}-*/model_14750.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${DOSE2_CKPT}" ]]; then
  DOSE2_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_dose2_k15_max320_from_13950_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${DOSE2_CKPT}" || ! -f "${DOSE2_CKPT}" ]]; then
  echo "ERROR: no S2-dose-2 checkpoint found. Set DOSE2_CKPT=... explicitly."
  exit 1
fi

ITERS=800
ENVS=2048
RUN_NAME="DFH_S3_bridge1_floor09_k15_from_14750_seed${SEED}"

echo "=========================================================="
echo "  DFH S3-bridge-1 (floor=-0.09, k=15, max=320), warm-start from"
echo "  ${DOSE2_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  CRITICAL: this is a sink-axis transfer test, not a dose step."
echo "  Drag (k=15/max=320) is unchanged, sink floor deepens -0.08 → -0.09."
echo ""
echo "  BRIDGE GREEN-LIGHT (Codex 019e7384):"
echo "    ep_len ≥ 600, term ≥ -1.9, clip_frac ≤ 0.08"
echo "    materially below → do another -0.08 rung first"
echo ""
echo "  Expected: mean_sink should now sit near -0.089/-0.090."
echo "  Expected clip_frac jump from 0.062 to ~0.08-0.09 (drag = k×sink×N)."
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${DOSE2_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.09 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=15.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=320.0 \
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
