#!/usr/bin/env bash
# DFH S2-dose-1 — Codex (thread 019e72a4-a2d7-7a60-9895-497b6e8ae715) recipe.
# After S2-proof passed 6/7 gates (m13150: ep_len 783, term -1.14, clip_frac
# 0.063), the next dose step is k=13, max=280 instead of the originally
# planned k=14/max=280.
#
# Reasoning: k/max ratio 13/280=0.0464 ≈ 12/260=0.0462 — almost headroom-neutral.
# This advances the real sinkage dose without raising saturation pressure. The
# k=14 step would simultaneously bump dose AND saturation, which is exactly
# what got us into the bridge mess at S2-retry-v1.
#
# Codex retired the clip_frac ≤ 0.055 gate as too strict for floor=-0.08.
# New regime:
#   - acceptable:  0.060-0.078
#   - caution:     0.078-0.090
#   - alarming:    >0.10 or median last-3 > 0.085
#
# Implication for S3 endgame: max_drag is no longer a side knob; it must scale
# with k. The original k=40/max=400 plan likely needs upward max_drag revision.

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
: "${PROOF_CKPT:=}"

if [[ -z "${PROOF_CKPT}" ]]; then
  PROOF_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_real_floor08_k12_from_12150_seed${SEED}-*/model_13150.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${PROOF_CKPT}" ]]; then
  PROOF_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_real_floor08_k12_from_12150_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${PROOF_CKPT}" || ! -f "${PROOF_CKPT}" ]]; then
  echo "ERROR: no S2-proof checkpoint found. Set PROOF_CKPT=... explicitly."
  exit 1
fi

ITERS=800
ENVS=2048
RUN_NAME="DFH_S2_dose1_k13_max280_from_13150_seed${SEED}"

echo "=========================================================="
echo "  DFH S2-dose-1 (k=13, max=280, floor=-0.08), warm-start from"
echo "  ${PROOF_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  PROMOTION (last 3 save-points):"
echo "    median ep_len ≥ 700, final ckpt ≥ 700"
echo "    ≥2 of last 3 ep_len ≥ 680"
echo "    median term ≥ -1.6, median lin_vel ≥ 1.00"
echo "    last 3 mean_sink in [0.078, 0.081]"
echo "    median clip_frac ≤ 0.078, no save-point > 0.090"
echo ""
echo "  ITER-400 RULE:"
echo "    continue if ep_len ≥ 650"
echo "    continue if (ep_len ≥ 620 AND term ≥ -1.8 AND clip_frac ≤ 0.080)"
echo "    abort if (ep_len<600 AND term<-1.9) OR (clip_frac>0.095 AND ep_len<680)"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${PROOF_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.08 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=13.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=280.0 \
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
