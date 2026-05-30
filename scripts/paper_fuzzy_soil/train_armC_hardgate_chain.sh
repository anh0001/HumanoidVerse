#!/usr/bin/env bash
# ARM C — credit-assignment ablation for the fuzzy-soil-on-furrows test.
# EXACTLY Arm B's paper recipe (friction curriculum S2a -> S2b -> S3, warm-started
# from FixedStageFv7/model_4250.pt) with the SINGLE change that the slip penalty
# uses the HARD stance gate (slip_gate.kind=hard) instead of the paper's soft gate
# (kind=soft, beta=4, F_th=1N). Everything else — reward weights (w_slip=0.5,
# w_force=0.05, F_max=150N), DR schedule, terrain, PPO hyperparams — is identical.
#
# Purpose: separate the two factors that Arm B applied together (friction curriculum
# AND soft slip gate). Decision per docs/experiments/fuzzy_soil_furrows_result.md:
#   - If Arm C keeps Arm B's ep_len but DROPS slip  -> the soft gate is harmful.
#   - If Arm C matches Arm B on both                 -> the soft gate is inert; the
#                                                       curriculum is the whole story.
# FULL-chain warm-start (Codex call): hard gate active in ALL THREE stages so the
# only difference from Arm B across the entire curriculum is gate kind (avoids the
# curriculum-history confound a branch-from-S2b would carry).
#
# Usage:
#   SEED=1 scripts/paper_fuzzy_soil/train_armC_hardgate_chain.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
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
[ -f "$WARM_CKPT" ] || { echo "[armC] warm ckpt missing: $WARM_CKPT"; exit 1; }

ITERS_PER_STAGE="${ITERS_PER_STAGE:-250}"
NUM_ENVS="${NUM_ENVS:-2048}"
F_TERRAIN="${F_TERRAIN:-terrain_furrows_stage1_easy}"
LOG_ROOT="logs/FuzzySoilFurrows/ArmC_hardgate_seed${SEED}"
mkdir -p "$LOG_ROOT"

# ---- common training overrides (shared across the 3 stages) ----
# IDENTICAL to train_armB_paper_chain.sh EXCEPT the trailing slip_gate.kind=hard.
common_args=(
  +exp=locomotion algo=ppo_roa
  +rewards=loco/reward_hunter_paper_soil
  +robot=hunter/hunter +simulator=isaacsim
  +terrain="$F_TERRAIN"
  +obs=loco/leggedloco_obs_history_wolinvel
  num_envs="$NUM_ENVS" seed="$SEED" headless=True
  ++env.config.env_spacing=2.5
  ++algo.config.num_learning_iterations="$ITERS_PER_STAGE"
  ++algo.config.save_interval=50
  ++algo.config.load_optimizer=False
  ++algo.config.actor_learning_rate=2.5e-4
  ++algo.config.critic_learning_rate=2.5e-4
  ++algo.config.desired_kl=0.005
  ++algo.config.entropy_coef=0.001
  ++algo.config.clip_param=0.10
  ++env.config.termination.terminate_by_contact=True
  ++rewards.reward_scales.tracking_lin_vel=4.0
  '++env.config.locomotion_command_ranges.lin_vel_x=[0.25,0.45]'
  '++env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0]'
  '++env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]'
  ++env.config.reset_randomization.enable=True
  ++rewards.slip_gate.kind=hard      # <-- THE ONLY DIFFERENCE FROM ARM B
  project_name=FuzzySoilFurrows
)

latest_ckpt_in() {
  find "$1" -type f -name "model_*.pt" 2>/dev/null \
    | awk -F'/' '{n=split($NF,a,"_"); split(a[n],b,"."); print b[1]" "$0}' \
    | sort -n | tail -1 | awk '{print $2}'
}

run_stage() {
  local tag="$1" dr_cfg="$2" warm="$3"
  local stage_log="$LOG_ROOT/$tag"
  mkdir -p "$stage_log"
  echo "[armC seed=$SEED stage=$tag] warm=$warm DR=$dr_cfg iters=$ITERS_PER_STAGE gate=HARD"
  "$PY" humanoidverse/train_agent.py \
    "${common_args[@]}" \
    +domain_rand="$dr_cfg" \
    ++checkpoint="$warm" \
    experiment_name="armC_${tag}_seed${SEED}" \
    2>&1 | tee "$stage_log/train.log"
}

# ---- S2a: μ ∈ [0.6, 0.8] ----
run_stage "S2a" "DR_paper_S2a" "$WARM_CKPT"
s2a_dir="$(ls -dt logs/FuzzySoilFurrows/*armC_S2a_seed${SEED}* 2>/dev/null | head -1 || true)"
s2a_ckpt="$(latest_ckpt_in "$s2a_dir")"
[ -f "$s2a_ckpt" ] || { echo "[armC seed=$SEED] S2a produced no checkpoint"; exit 2; }

# ---- S2b: μ ∈ [0.45, 0.65] ----
run_stage "S2b" "DR_paper_S2b" "$s2a_ckpt"
s2b_dir="$(ls -dt logs/FuzzySoilFurrows/*armC_S2b_seed${SEED}* 2>/dev/null | head -1 || true)"
s2b_ckpt="$(latest_ckpt_in "$s2b_dir")"
[ -f "$s2b_ckpt" ] || { echo "[armC seed=$SEED] S2b produced no checkpoint"; exit 3; }

# ---- S3: μ ∈ [0.35, 0.55] + 35 N lateral pushes ----
run_stage "S3"  "DR_paper_S3"  "$s2b_ckpt"

echo "[armC seed=$SEED] CHAIN DONE: $LOG_ROOT"
