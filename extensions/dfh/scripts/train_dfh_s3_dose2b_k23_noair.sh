#!/usr/bin/env bash
# DFH S3-dose-2b v3 — Codex (thread 019e7c43) fix E (A+B hybrid).
# Two prior k=25 attempts (max=520 then max=440) both went unstable: the policy
# retreats to a protective light-contact gait (stance_fraction ~0.03) instead of
# committing to loaded stance, and falls. The max=440 cap fixed per-contact
# IMPULSE (stance_drag_mean 408→372N) but k=25 still won't stabilize.
#
# Two-part fix:
#   A. Smaller dose step: k=21 → 23 (not 25). +10% from the proven dose-2 base.
#   B. feet_air_time 0.08 → 0.0. The term is (air_time-0.5)*first_contact
#      (locomotion.py:161) — it rewards DELAYED touchdown and penalizes quick
#      re-contact, biasing AWAY from committed stance in a drag-heavy regime.
#      The DFH reward yaml disables it by default; the dose scripts wrongly
#      re-enabled it to 0.08. Turn it off during hardening.
#
# Warm-start: dose-2 m16750 (proven k=21 base, stance_drag 345N, stance_fr 0.049).
#
# If k=23 passes → retry k=25/max=440 with the SAME feet_air_time=0.0.
# If k=25 still fails → practical ceiling is k=21-23 @ floor=-0.09; step sink to
# -0.10 at the lower dose. k=32/40 needs REWARD REDESIGN (load-aware stance
# objective: normal-force-weighted stance / reward for F_n>100N stance), not
# more curriculum.

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

ITERS=1000
ENVS=2048
RUN_NAME="DFH_S3_dose2b_k23_noair_floor09_seed${SEED}"

echo "=========================================================="
echo "  DFH S3-dose-2b v3 (k=23, max=440, floor=-0.09, feet_air=0.0)"
echo "  warm=${WARM_CKPT}"
echo "  envs=${ENVS}, iters=${ITERS}"
echo ""
echo "  EARLY ABORT (iter 300-400): ep_len<620 AND stance_fraction<0.035"
echo "  PROMOTE (last 3): med ep_len≥700, worst≥620, med stance_fr≥0.04,"
echo "                    med stance_drag_mean≤380, med drag_pct≤0.09"
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
  ++terrain.dfh.force_coupling.sinkage_drag_k=23.0 \
  ++terrain.dfh.force_coupling.max_drag_force_n=440.0 \
  ++rewards.reward_scales.feet_air_time=0.0 \
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
