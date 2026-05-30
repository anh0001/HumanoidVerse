#!/usr/bin/env bash
# ARM D — Stage-0 compliant-contact probe.
# EXACTLY the Arm B paper recipe (friction curriculum S2a->S2b->S3 + soft slip gate
# + force cap, warm-started from FixedStageFv7/model_4250.pt) with ONE change: the
# terrain uses PhysX COMPLIANT normal contact (global, all-env-shared) instead of the
# default rigid contact. Tests the last standing hypothesis for the slip non-
# replication: does adding contact compliance change foot slip at all?
#
# Global (not per-env) compliance — kn/cn are a single value across all envs and all
# stages, the softest end of the paper's S3 band (kn=50 kN/m, cn=0.5 kN·s/m). Per-env
# stiffness RANDOMIZATION (the paper's actual scheme) is the expensive Stage 1, gated
# on whether this probe moves slip. See docs/experiments/compliant_contact_feasibility.md.
#
# Usage:  SEED=1 scripts/paper_fuzzy_soil/train_armD_compliant_chain.sh
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
[ -f "$WARM_CKPT" ] || { echo "[armD] warm ckpt missing: $WARM_CKPT"; exit 1; }

ITERS_PER_STAGE="${ITERS_PER_STAGE:-250}"
NUM_ENVS="${NUM_ENVS:-2048}"
F_TERRAIN="${F_TERRAIN:-terrain_furrows_stage1_easy}"
# Global compliant-contact values (softest paper S3 band).
KN="${KN:-50000}"     # N/m  (= 50 kN/m)
CN="${CN:-500}"       # N·s/m (= 0.5 kN·s/m)
LOG_ROOT="logs/FuzzySoilFurrows/ArmD_compliant_seed${SEED}"
mkdir -p "$LOG_ROOT"

# Common args = Arm B's, PLUS the two compliant-contact terrain overrides.
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
  ++terrain.compliant_contact_stiffness="$KN"   # <-- compliant contact (vs Arm B rigid)
  ++terrain.compliant_contact_damping="$CN"     # <--
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
  echo "[armD seed=$SEED stage=$tag] warm=$warm DR=$dr_cfg iters=$ITERS_PER_STAGE kn=$KN cn=$CN"
  "$PY" humanoidverse/train_agent.py \
    "${common_args[@]}" \
    +domain_rand="$dr_cfg" \
    ++checkpoint="$warm" \
    experiment_name="armD_${tag}_seed${SEED}" \
    2>&1 | tee "$stage_log/train.log"
}

run_stage "S2a" "DR_paper_S2a" "$WARM_CKPT"
s2a_dir="$(ls -dt logs/FuzzySoilFurrows/*armD_S2a_seed${SEED}* 2>/dev/null | head -1 || true)"
s2a_ckpt="$(latest_ckpt_in "$s2a_dir")"
[ -f "$s2a_ckpt" ] || { echo "[armD seed=$SEED] S2a produced no checkpoint"; exit 2; }

run_stage "S2b" "DR_paper_S2b" "$s2a_ckpt"
s2b_dir="$(ls -dt logs/FuzzySoilFurrows/*armD_S2b_seed${SEED}* 2>/dev/null | head -1 || true)"
s2b_ckpt="$(latest_ckpt_in "$s2b_dir")"
[ -f "$s2b_ckpt" ] || { echo "[armD seed=$SEED] S2b produced no checkpoint"; exit 3; }

run_stage "S3"  "DR_paper_S3"  "$s2b_ckpt"

echo "[armD seed=$SEED] CHAIN DONE: $LOG_ROOT"
