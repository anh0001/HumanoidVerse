#!/usr/bin/env bash
# DFH S3-dose-1 — Codex (thread 019e7813) roadmap step 2.
# After S3-bridge-1 extend passed 6/6 gates (floor=-0.09 sink transfer
# confirmed, m15250: ep_len 752, term -0.92, clip 0.073), the next move is a
# headroom-neutral dose step at the new sink floor.
#   15/320 = 0.0469  ✅ proven at floor=-0.09
#   17/360 = 0.0472  ← essentially neutral, real dose +13%
#
# Roadmap to S3-nominal (one axis at a time, never sink+dose together):
#   S3-dose-1: floor=-0.09, k=17, max=360  (THIS)
#   S3-dose-2: floor=-0.09, k=25, max=400
#   S3-sink-2: floor=-0.10, k=25, max=400
#   S3-nominal: floor=-0.10, k=40, max=400-450  (insert k=32 if too sharp)

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

# Warm-start from the extend500 run's highest ckpt (numbering offset — glob for
# the actual max model number in that dir, ~15250).
if [[ -z "${BRIDGE_CKPT}" ]]; then
  BRIDGE_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S3_bridge1_extend500_from_15500_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${BRIDGE_CKPT}" || ! -f "${BRIDGE_CKPT}" ]]; then
  echo "ERROR: no S3-bridge-1 extend checkpoint found. Set BRIDGE_CKPT=... explicitly."
  exit 1
fi

ITERS=800
ENVS=2048
RUN_NAME="DFH_S3_dose1_k17_max360_floor09_seed${SEED}"

echo "=========================================================="
echo "  DFH S3-dose-1 (floor=-0.09, k=17, max=360), warm-start from"
echo "  ${BRIDGE_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  Headroom-neutral dose step at floor=-0.09 (17/360=0.0472)."
echo ""
echo "  PROMOTION (last 3 save-points):"
echo "    median ep_len ≥ 700, worst ≥ 650"
echo "    median term ≥ -1.6, median lin_vel ≥ 1.00"
echo "    median clip_frac ≤ 0.083, worst ≤ 0.090"
echo "    last 3 mean_sink in [0.087, 0.090]"
echo ""
echo "  ITER-400 RULE:"
echo "    continue if ep_len ≥ 650"
echo "    abort if (ep_len<590 AND term<-1.95) OR (clip>0.10 AND ep_len<660)"
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
  ++terrain.dfh.params.sinkage_floor_m=-0.09 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=17.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=360.0 \
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
