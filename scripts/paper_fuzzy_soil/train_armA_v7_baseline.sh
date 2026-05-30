#!/usr/bin/env bash
# ARM A — v7 baseline control, freshly rerun for the fuzzy-soil-on-furrows
# transfer test (docs/experiments/fuzzy_soil_furrows_plan.md).
#
# Replicates the exact recipe that produced FixedStageFv7/model_4250.pt:
#   - DR_mild domain randomization (push_robots vel-based, μ∈[0.7,1.0])
#   - reward_hunter_locomotion.yaml (hard-gated slip penalty)
#   - terrain_furrows_stage1_easy
#   - PPOROA, conservative PPO schedule
# but with a different seed so we get a 3-seed mean ± std for the comparison.
#
# Usage:
#   SEED=1 scripts/paper_fuzzy_soil/train_armA_v7_baseline.sh
#   SEED=2 scripts/paper_fuzzy_soil/train_armA_v7_baseline.sh
#   SEED=3 scripts/paper_fuzzy_soil/train_armA_v7_baseline.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# Only set ISAAC/CARB paths when a binary IsaacSim install exists. On hosts
# with pip-installed isaacsim inside the conda env (e.g. /srv/data/.../isaaclab),
# leaving CARB_APP_PATH/EXP_PATH unset lets the bundled installation discover
# its own plugins. Setting them to nonexistent paths breaks
# `omni::kit::IApp` acquisition with `(pluginName: nullptr)`.
if [ -d "${ISAAC_PATH:-$HOME/isaacsim_4.2}" ]; then
  export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
  export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
  export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
  if [ -f "$ISAAC_PATH/setup_python_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$ISAAC_PATH/setup_python_env.sh"
  fi
fi
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
PY="${PYTHON:-/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python}"
[ -x "$PY" ] || PY="$HOME/miniconda3/envs/isaaclab/bin/python"

SEED="${SEED:-1}"
WARM_CKPT="${WARM_CKPT:-logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt}"
[ -f "$WARM_CKPT" ] || { echo "[armA] warm ckpt missing: $WARM_CKPT"; exit 1; }

ITERS="${ITERS:-750}"
NUM_ENVS="${NUM_ENVS:-2048}"
F_TERRAIN="${F_TERRAIN:-terrain_furrows_stage1_easy}"
LOG_ROOT="logs/FuzzySoilFurrows/ArmA_v7_baseline_seed${SEED}"
mkdir -p "$LOG_ROOT"

echo "[armA seed=$SEED] warm-start=$WARM_CKPT  iters=$ITERS"

"$PY" humanoidverse/train_agent.py \
  +exp=locomotion algo=ppo_roa \
  +domain_rand=DR_mild \
  +rewards=loco/reward_hunter_locomotion \
  +robot=hunter/hunter +simulator=isaacsim \
  +terrain="$F_TERRAIN" \
  +obs=loco/leggedloco_obs_history_wolinvel \
  num_envs="$NUM_ENVS" seed="$SEED" headless=True \
  ++env.config.env_spacing=2.5 \
  ++algo.config.num_learning_iterations="$ITERS" \
  ++algo.config.save_interval=100 \
  ++algo.config.load_optimizer=False \
  ++algo.config.actor_learning_rate=2.5e-4 \
  ++algo.config.critic_learning_rate=2.5e-4 \
  ++algo.config.desired_kl=0.005 \
  ++algo.config.entropy_coef=0.001 \
  ++algo.config.clip_param=0.10 \
  ++checkpoint="$WARM_CKPT" \
  ++env.config.termination.terminate_by_contact=True \
  ++rewards.reward_scales.tracking_lin_vel=4.0 \
  '++env.config.locomotion_command_ranges.lin_vel_x=[0.25,0.45]' \
  '++env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0]' \
  '++env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]' \
  ++env.config.reset_randomization.enable=True \
  project_name=FuzzySoilFurrows experiment_name="armA_seed${SEED}" \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[armA seed=$SEED] DONE"
