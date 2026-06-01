#!/usr/bin/env bash
# DFH S3-dose-2b RETRY — Codex (thread 019e7ae3) fix D.
# First dose-2b (k=25/max=520) went UNSTABLE: bimodal thrashing, ep_len 390-866,
# never settled. Diagnosis: global clip_frac (~0.012) HID loaded-contact
# saturation. With stance_fraction only ~0.03, stance contacts were pinned at
# the 520N cap (stance_drag_contact_mean_n 408N, p95 520N flat) — an unstable
# impulse band. The policy flip-flopped between a committed stance gait (walks)
# and a protective light-contact gait (falls).
#
# Fix: keep k=25 but CAP max_drag back to 440 (same as dose-2, which was stable
# at stance mean 352N / p95 440N). This bounds per-contact impulse magnitude.
# The k/max ratio rises to 0.0568 — clip_frac will rise, but clip_frac is NOT
# the real gate; loaded-contact mean is.
#
# Warm-start from a good late dose-2 ckpt (m16750), NOT the failed dose-2b and
# NOT a dose-2b peak (same unstable basin).
#
# GATES NOW USE LOADED-CONTACT METRICS, not global clip_frac:
#   promote: median ep_len≥680, worst≥620, median term≥-1.8, median lin_vel≥0.95,
#            median stance_fraction≥0.04, median stance_drag_contact_mean_n≤385,
#            median applied_drag_pct_weight≤0.09
#   iter-200 abort: ep_len<450 & stance_fraction<0.03, OR
#                   stance_drag_contact_mean_n>390 for 2 consecutive saves

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
: "${WARM_CKPT:=logs/DFH_Hunter_ROA/20260530_215958-DFH_S3_dose2_k21_max440_floor09_seed1-locomotion-hunter/model_16750.pt}"
[[ -f "${WARM_CKPT}" ]] || { echo "ERROR: warm ckpt missing: ${WARM_CKPT}"; exit 1; }

ITERS=800
ENVS=2048
RUN_NAME="DFH_S3_dose2b_retry_k25_max440_floor09_seed${SEED}"

echo "=========================================================="
echo "  DFH S3-dose-2b RETRY (k=25, max=440, floor=-0.09)"
echo "  warm=${WARM_CKPT}"
echo "  k/max=0.0568 (clip_frac WILL rise — not the gate; loaded-contact is)"
echo "  envs=${ENVS}, iters=${ITERS}"
echo "=========================================================="

"${PYTHON}" humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  algo=ppo_roa \
  +curriculum=stage2_dfh \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_history_wolinvel \
  checkpoint="${WARM_CKPT}" \
  auto_load_latest=False \
  num_envs="${ENVS}" \
  headless=True \
  seed="${SEED}" \
  ++env.config.env_spacing=2.5 \
  ++terrain.dfh.params.sinkage_floor_m=-0.09 \
  ++terrain.dfh.force_coupling.sinkage_drag_k=25.0 \
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
