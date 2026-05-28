#!/usr/bin/env bash
# Pass-4 close-out -> de-saturated seed-1 PILOT (decision gate).
#
# Reviewer (019e3cf3) bar for repair: lower sinkage_drag_k (NOT raise cap) so
# eval stance p95 < ~70-75% of cap and clip < 5%, then check DFH > RIGID.
# This pilot trains ONE de-saturated DFH_FULL (k=12, =0.3x the ultra k=40;
# mu/cap unchanged) seed 1 -> 2000 iters, reuses the existing RIGID_TUNED_s1
# checkpoint, cross-evals both on the de-saturated dfh dose, and reports:
#   (a) de-saturation achieved?  (eval stance_p95 %cap, clip%)
#   (b) paired improvement sign  (DFH@dfh - RIGID@dfh)
# Outcome routes: de-saturated & DFH>RIGID -> full repair (seeds 1-3);
#   de-saturated & DFH<=RIGID -> pivot (clean evidence, no cap confound);
#   still saturated -> lower k again.
#
# All train/eval flags identical to recover_shuffled_phaseBC_ultra.sh except
# sinkage_drag_k (40 -> 12). SHUFFLED omitted (gate question is DFH vs RIGID).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"
PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"

LOG_ROOT="logs/DFHPilot"
TUNED_ROOT="logs/DFHTuned"          # source of the existing RIGID_TUNED_s1 ckpt
ITERS=2000
NUM_ENVS=2048
SEED=1
EVAL_EPISODES=50
EVAL_NUM_ENVS=2048                  # match packed-origins terrain (train env count)
K_DESAT=12.0
mkdir -p "${LOG_ROOT}/crosseval"

# Only sinkage_drag_k changes vs ultra; mu/cap held so anisotropy + cap fixed.
desat_overrides=(
  terrain.dfh.params.anisotropy.mu_across=1.00
  terrain.dfh.params.anisotropy.mu_along=0.60
  terrain.dfh.force_coupling.sinkage_drag_k="${K_DESAT}"
  terrain.dfh.force_coupling.max_drag_force_n=500.0
)
dfh_train_args=(
  +terrain=terrain_dfh_stage1_easy
  +domain_rand=DR_dfh_soft
  terrain.dfh.force_coupling.enabled=True
  terrain.dfh.force_coupling.shuffle_env_ids=False
  terrain.dfh.writeback_enabled=False
)

latest_ckpt() { ls -t "$1"/model_*.pt 2>/dev/null | head -1; }

# ---- 1. Train de-saturated DFH_FULL (skip if already at target iters) ----
DFH_ID="DFH_FULL_pilotk12_s1"
run_dir="${LOG_ROOT}/${DFH_ID}"
if [ -f "${run_dir}/model_${ITERS}.pt" ]; then
  echo "===== SKIP TRAIN ${DFH_ID} (model_${ITERS}.pt exists) ====="
else
  echo "===== TRAIN ${DFH_ID} iters=${ITERS} sinkage_drag_k=${K_DESAT} ====="
  "$PY" humanoidverse/train_agent.py \
    +simulator=isaacsim +exp=locomotion_soil algo=ppo_soil \
    +robot=hunter/hunter \
    +obs=loco/leggedloco_obs_singlestep_withlinvel \
    +rewards=loco/reward_hunter_dfh \
    num_envs="${NUM_ENVS}" headless=True seed="${SEED}" \
    env.config.env_spacing=2.5 \
    algo.config.num_learning_iterations="${ITERS}" \
    rewards.reward_scales.penalty_sinkage_excess=0 \
    project_name=DFHPilot experiment_name="${DFH_ID}" \
    experiment_dir="${run_dir}" save_dir="${run_dir}/.hydra" \
    hydra.run.dir="${run_dir}/.hydra" \
    "${dfh_train_args[@]}" "${desat_overrides[@]}"
fi

# ---- 2. Cross-eval on the de-saturated dfh dose ----
# cross_eval <train_id> <eval_name> <ckpt_path>
cross_eval() {
  local train_id="$1" eval_name="$2" ckpt="$3"
  [ -z "$ckpt" ] && { echo "no ckpt for $train_id" >&2; return 1; }
  local out_dir="${LOG_ROOT}/crosseval/${train_id}_on_${eval_name}"
  mkdir -p "${out_dir}"
  echo "===== EVAL ${train_id} on ${eval_name} (${ckpt}) ====="
  local terrain_args=()
  case "${eval_name}" in
    rigid)
      terrain_args=(+terrain=terrain_furrows_stage1_easy +domain_rand=DR_soil_rigid)
      ;;
    dfh)
      terrain_args=(
        +terrain=terrain_dfh_stage1_easy
        +domain_rand=DR_dfh_soft
        terrain.dfh.force_coupling.enabled=True
        terrain.dfh.force_coupling.shuffle_env_ids=False
        terrain.dfh.writeback_enabled=False
        "${desat_overrides[@]}"
      )
      ;;
  esac
  "$PY" humanoidverse/sample_eps.py \
    +checkpoint="${ckpt}" \
    "${terrain_args[@]}" \
    +exp=locomotion_soil \
    num_envs="${EVAL_NUM_ENVS}" \
    ++env.config.num_envs="${EVAL_NUM_ENVS}" \
    ++env.config.env_spacing=2.5 \
    +num_episodes="${EVAL_EPISODES}" \
    +eval_tb_dir="${out_dir}" \
    eval_name="${train_id}_on_${eval_name}" \
    headless=True \
    hydra.run.dir="${out_dir}/.hydra"
}

DFH_CKPT="$(latest_ckpt "${LOG_ROOT}/${DFH_ID}")"
RIGID_CKPT="$(latest_ckpt "${TUNED_ROOT}/RIGID_TUNED_s1")"
echo "DFH ckpt=${DFH_CKPT}"
echo "RIGID ckpt=${RIGID_CKPT}  (reused, dose-independent rigid training)"

cross_eval "${DFH_ID}"        dfh   "${DFH_CKPT}"
cross_eval "${DFH_ID}"        rigid "${DFH_CKPT}"
cross_eval "RIGID_TUNED_s1"   dfh   "${RIGID_CKPT}"
cross_eval "RIGID_TUNED_s1"   rigid "${RIGID_CKPT}"

# ---- 3. Analyze (paired improvement + de-saturation check) ----
echo "===== analyze pilot (k=${K_DESAT}) ====="
"$PY" -m extensions.dfh.scripts.analyze_regret \
  --strength ultra --root "${LOG_ROOT}" \
  --csv "${LOG_ROOT}/crosseval/pilot_k12.csv" || \
  echo "(analyze_regret verdict classifier may not key on pilot ids; CSV still written -- read it directly)"

echo "===== PILOT k=${K_DESAT} complete ====="
echo "Decision inputs in ${LOG_ROOT}/crosseval/pilot_k12.csv :"
echo "  de-saturation: DFH_FULL_pilotk12_s1,dfh -> stance_drag_p95_n (<360 = <72% of 500 cap), clip_frac (<0.05)"
echo "  gate sign:     ep_len(DFH_FULL_pilotk12_s1,dfh) - ep_len(RIGID_TUNED_s1,dfh)"
