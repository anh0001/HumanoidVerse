#!/usr/bin/env bash
# DFH S2-dose-2 — Codex (thread 019e7384) recipe.
# After S2-dose-1 (k=13, max=280) passed 8/8 gates with clip_frac stable at
# 0.067, the next headroom-neutral step is k=15, max=320.
#   13/280 = 0.0464  ✅ proven
#   15/320 = 0.0469  ← essentially identical saturation pressure, real dose +8%
# Codex retired the old k=16/max=300 from the pre-fix ladder (that would
# raise both dose AND pressure, exactly the failure shape).
#
# Next stage if this passes: S3-bridge-1 (floor=-0.09, k=15, max=320) — the
# next high-value question becomes sink-axis transfer, not same-floor dose.

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
  DOSE1_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_dose1_k13_max280_from_13150_seed${SEED}-*/model_13950.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${DOSE1_CKPT}" ]]; then
  DOSE1_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_dose1_k13_max280_from_13150_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${DOSE1_CKPT}" || ! -f "${DOSE1_CKPT}" ]]; then
  echo "ERROR: no S2-dose-1 checkpoint found. Set DOSE1_CKPT=... explicitly."
  exit 1
fi

ITERS=800
ENVS=2048
RUN_NAME="DFH_S2_dose2_k15_max320_from_13950_seed${SEED}"

echo "=========================================================="
echo "  DFH S2-dose-2 (k=15, max=320, floor=-0.08), warm-start from"
echo "  ${DOSE1_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  PROMOTION (last 3 save-points):"
echo "    median ep_len ≥ 730, final ckpt ≥ 700"
echo "    ≥2 of last 3 ep_len ≥ 710"
echo "    median term ≥ -1.55, median lin_vel ≥ 1.00"
echo "    last 3 mean_sink in [0.078, 0.081]"
echo "    median clip_frac ≤ 0.075, no save-point > 0.085"
echo ""
echo "  ITER-400 RULE:"
echo "    continue if ep_len ≥ 650"
echo "    continue if (ep_len ≥ 620 AND term ≥ -1.8 AND clip_frac ≤ 0.080)"
echo "    abort if (ep_len<590 AND term<-1.95) OR (clip_frac>0.092 AND ep_len<660)"
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
  ++terrain.dfh.params.sinkage_floor_m=-0.08 \
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
