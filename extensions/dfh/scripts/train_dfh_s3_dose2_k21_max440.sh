#!/usr/bin/env bash
# DFH S3-dose-2 — Codex (thread 019e78f6) recipe.
# S3-dose-1 (k=17/max=360) passed 6/7 with clip_frac at 0.052 (lowest of any
# deep-sink stage). Codex retired the old roadmap's k=25/max=400 (ratio 0.0625,
# +32% saturation pressure) and made headroom-neutral the PERMANENT rule.
#   17/360 = 0.0472  ✅ proven
#   21/440 = 0.0477  ← neutral, +24% real dose
#
# IMPORTANT endgame correction: the nominal S3 yaml (k=40/max=400) is NOT
# learnable. On the working pressure rail, k=40 needs max_drag ≈ 820-880 N.
# Revised ladder from here:
#   21/440 → 25/520 → floor -0.10 @ 25/520 → 32/680 → 40/820-880

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
: "${DOSE1_CKPT:=}"

if [[ -z "${DOSE1_CKPT}" ]]; then
  DOSE1_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S3_dose1_k17_max360_floor09_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${DOSE1_CKPT}" || ! -f "${DOSE1_CKPT}" ]]; then
  echo "ERROR: no S3-dose-1 checkpoint found. Set DOSE1_CKPT=... explicitly."
  exit 1
fi

ITERS=800
ENVS=2048
RUN_NAME="DFH_S3_dose2_k21_max440_floor09_seed${SEED}"

echo "=========================================================="
echo "  DFH S3-dose-2 (floor=-0.09, k=21, max=440), warm-start from"
echo "  ${DOSE1_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  Headroom-neutral (21/440=0.0477), +24% real dose vs 17/360."
echo ""
echo "  PROMOTION (last 3 save-points):"
echo "    median ep_len ≥ 700, worst ≥ 640"
echo "    median term ≥ -1.65, median lin_vel ≥ 0.98"
echo "    median clip_frac ≤ 0.070, worst ≤ 0.080"
echo "    mean_sink ≤ 0.090 (only worry if <0.084 AND weak lin_vel)"
echo ""
echo "  ITER-400 RULE:"
echo "    continue if ep_len ≥ 640"
echo "    continue if (ep_len ≥ 610 AND term ≥ -1.85 AND clip ≤ 0.075)"
echo "    abort if (ep_len<580 AND term<-2.0) OR (clip>0.090 AND ep_len<650)"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${DOSE1_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.09 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=21.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=440.0 \
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
