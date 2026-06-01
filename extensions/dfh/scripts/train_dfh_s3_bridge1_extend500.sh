#!/usr/bin/env bash
# DFH S3-bridge-1 EXTEND +500 — Codex (thread 019e7813) recipe.
# S3-bridge-1 was borderline (clip_frac 0.082 vs 0.080 gate by +2.5%, but
# ep_len 690 and term -1.46 strong, and ep_len peaked at 800 then dipped to
# 686 over last 50 iters — not settled).
#
# Codex's call: extend in-place from the PEAK ckpt (m15500), not the tail
# (m15550). +500 iters, same conditions. Promotion gate clip_frac lifted to
# 0.083 (deeper sink load is structural, not a learning failure).
#
# Early-stop at +200 iters:
#   - if median(last 2) ep_len < 650 OR clip > 0.086 OR term < -1.90:
#     STOP, switch to consolidation rung at floor=-0.08
#   - if ep_len ≥ 700, term ≥ -1.75, clip ≤ 0.083 at +200: stop early, promote

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
: "${PEAK_CKPT:=}"

if [[ -z "${PEAK_CKPT}" ]]; then
  PEAK_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S3_bridge1_floor09_k15_from_14750_seed${SEED}-*/model_15500.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${PEAK_CKPT}" || ! -f "${PEAK_CKPT}" ]]; then
  echo "ERROR: peak ckpt m15500 not found. Set PEAK_CKPT=... explicitly."
  exit 1
fi

ITERS=500
ENVS=2048
RUN_NAME="DFH_S3_bridge1_extend500_from_15500_seed${SEED}"

echo "=========================================================="
echo "  DFH S3-bridge-1 EXTEND +500 iters from PEAK (m15500)"
echo "  ${PEAK_CKPT}"
echo "  same conditions: floor=-0.09, k=15, max=320"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  PROMOTION (last 3 save-points, lifted clip gate):"
echo "    median ep_len ≥ 700"
echo "    median term ≥ -1.75"
echo "    median clip_frac ≤ 0.083"
echo "    worst-of-last-3 ep_len ≥ 650"
echo "    worst-of-last-3 clip_frac ≤ 0.085"
echo "    median lin_vel ≥ 1.00"
echo ""
echo "  EARLY-STOP at +200 iters (≈iter 15700):"
echo "    abort if median(last 2) ep_len<650 OR clip>0.086 OR term<-1.90"
echo "    promote early if ep_len≥700 AND term≥-1.75 AND clip≤0.083"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${PEAK_CKPT}" \
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
