#!/usr/bin/env bash
# DFH S2-proof — Codex (thread 019e71b1-...) recipe.
# After the floor-07 bridge succeeded (m12150: ep_len 761, term -1.14, clip_frac
# 0.040), the next move is to prove the policy can live at the REAL S2 medium
# sink floor of -0.08 with the current drag dose (k=12, max=260). Single-axis
# escalation: sink only.
#
# This is the "S2-proof" stage. If it passes, the chain proceeds with:
#   S2-dose-1 (k=14, max=280), S2-dose-2 (k=16, max=300),
#   S3-bridge-1 (-0.09, k=16, max=300), S3-dose-1 (-0.09, k=20, max=320),
#   S3-bridge-2 (-0.10, k=24, max=340), S3-dose-2 (-0.10, k=30, max=360),
#   S3-dose-3 (-0.10, k=36, max=380), S3-nominal (-0.10, k=40, max=400).
#
# Diagnostic gate at iter 400 (AND, not OR):
#   kill ONLY if ep_len < 580 AND term < -2.0
# If gate fires for the "right" reason, rerun with base_height fallback:
#   ++rewards.reward_scales.base_height=-18.0
#   ++rewards.desired_base_height=0.53

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
: "${BRIDGE_CKPT:=}"

if [[ -z "${BRIDGE_CKPT}" ]]; then
  BRIDGE_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_bridge_floor07_k12_from_11150_seed${SEED}-*/model_12150.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${BRIDGE_CKPT}" ]]; then
  BRIDGE_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_bridge_floor07_k12_from_11150_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${BRIDGE_CKPT}" || ! -f "${BRIDGE_CKPT}" ]]; then
  echo "ERROR: no S2 bridge checkpoint found. Set BRIDGE_CKPT=... explicitly."
  exit 1
fi

ITERS=1000
ENVS=2048
RUN_NAME="DFH_S2_real_floor08_k12_from_12150_seed${SEED}"

echo "=========================================================="
echo "  DFH S2-proof (floor=-0.08, k=12), warm-start from"
echo "  ${BRIDGE_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  PROMOTION (last 3 save-points, every 50 iters):"
echo "    median ep_len ≥ 650, final ckpt ≥ 700"
echo "    ≥2 of last 3 save-points ep_len ≥ 640"
echo "    no save-point clip_frac > 0.055"
echo "    median term ≥ -1.75, median lin_vel ≥ 0.90"
echo "    last 3 mean_sink in [0.076, 0.080]"
echo ""
echo "  ABORT at iter 400 only if (ep_len<580 AND term<-2.0)"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${BRIDGE_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.08 \
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
