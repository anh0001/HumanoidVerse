#!/usr/bin/env bash
# S2 sink-floor bridge — Codex (thread 019e7044-4473-7541-991c-8bca8054658d) recipe.
# Diagnosis: S2-retry k=12 plateaued at ep_len ~610 because S2 medium stacked
# two structural changes at once (S2 geometry AND deeper sink floor -0.065→-0.08).
# Drag escalation is OK; the sink floor is the cliff.
#
# Fix: keep k=12, max_drag=260, feet_air=0.08, but override sink floor to -0.07
# (a clean missing rung between S1.7's -0.065 and S2 medium's -0.08).
# Warm-start from S2-retry m11150 (most adapted ckpt — do NOT roll back to S1.7).
#
# Diagnostic gate at iter 400: if ep_len < 630 AND rew_termination < -1.9,
# stop and retune base_height (see "fallback overrides" comment below). Do NOT
# just extend longer.

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
: "${S2_CKPT:=}"

if [[ -z "${S2_CKPT}" ]]; then
  S2_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_retry_k12_from_S17_retry9750_seed${SEED}-*/model_11150.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S2_CKPT}" ]]; then
  S2_CKPT=$(ls -t logs/DFH_Hunter_ROA/*-DFH_S2_retry_k12_from_S17_retry9750_seed${SEED}-*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -z "${S2_CKPT}" || ! -f "${S2_CKPT}" ]]; then
  echo "ERROR: no S2-retry checkpoint found. Set S2_CKPT=... explicitly."
  exit 1
fi

ITERS=1000
ENVS=2048
RUN_NAME="DFH_S2_bridge_floor07_k12_from_11150_seed${SEED}"

echo "=========================================================="
echo "  DFH S2 sink-floor BRIDGE (floor=-0.07, k=12), warm-start from"
echo "  ${S2_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "  Promotion (last 200 iters): ep_len≥670, clip_frac≤0.055,"
echo "                              rew_termination≥-1.75, lin_vel≥0.90"
echo "                              final 3 save points all ≥650"
echo "  Abort at iter 400 if ep_len<630 AND rew_termination<-1.9"
echo "=========================================================="

# Fallback overrides if bridge stalls after iter ~400 (Codex suggestion before
# touching termination): ++rewards.reward_scales.base_height=-18.0
#                        ++rewards.desired_base_height=0.53

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${S2_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.07 \
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
