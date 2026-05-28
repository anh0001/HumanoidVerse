#!/usr/bin/env bash
# Pass 3 R2 — dose-response calibration + cross-evaluation.
#
# Phases:
#   A. Calibration sweep (seed 1, 800 iters): mild / medium / strong.
#      Pick weakest stable setting where mean applied_drag_pct_weight ≥ 0.05
#      and ep_len does not collapse.
#   B. Full sweep at selected strength: seeds {1,2}, 2000 iters,
#      both DFH_FULL_TUNED and RIGID_TUNED.
#   C. Cross-eval: each ckpt × {eval_rigid, eval_dfh_<strength>}.
#
# Override env vars:
#   STRENGTH=mild|medium|strong  (skip calibration if set, run only that)
#   ITERS, NUM_ENVS, SEEDS, CALIB_ITERS
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

LOG_ROOT="logs/DFHTuned"
ITERS="${ITERS:-2000}"
CALIB_ITERS="${CALIB_ITERS:-800}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEEDS=(${SEEDS:-1 2})
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-64}"

mkdir -p "${LOG_ROOT}/crosseval"

# Hydra overrides for each dose level. Targets (Hunter ~620 N weight):
#   mild   ~5% (~30 N)   sinkage_drag_k=8,  mu_across=0.70, cap=200
#   medium ~10% (~62 N)  sinkage_drag_k=14, mu_across=0.78, cap=250
#   strong ~20% (~120 N) sinkage_drag_k=20, mu_across=0.85, cap=300
dose_overrides() {
  case "$1" in
    mild)
      echo "terrain.dfh.params.anisotropy.mu_across=0.70 terrain.dfh.force_coupling.sinkage_drag_k=8.0 terrain.dfh.force_coupling.max_drag_force_n=200.0"
      ;;
    medium)
      echo "terrain.dfh.params.anisotropy.mu_across=0.78 terrain.dfh.force_coupling.sinkage_drag_k=14.0 terrain.dfh.force_coupling.max_drag_force_n=250.0"
      ;;
    strong)
      echo "terrain.dfh.params.anisotropy.mu_across=0.85 terrain.dfh.force_coupling.sinkage_drag_k=20.0 terrain.dfh.force_coupling.max_drag_force_n=300.0"
      ;;
    ultra)
      # Pass 4 R2 extension: chase 5-10% body weight target.
      # baseline_mu=0.45 → along-delta=+0.15, across-delta=+0.55. Drag cap 500N.
      echo "terrain.dfh.params.anisotropy.mu_across=1.00 terrain.dfh.params.anisotropy.mu_along=0.60 terrain.dfh.force_coupling.sinkage_drag_k=40.0 terrain.dfh.force_coupling.max_drag_force_n=500.0"
      ;;
    *) echo "" ;;
  esac
}

run_train() {
  local run_id="$1"; shift
  local seed="$1"; shift
  local iters="$1"; shift
  local extra_args=("$@")
  local run_dir="${LOG_ROOT}/${run_id}"
  echo "===== TRAIN ${run_id} iters=${iters} ====="
  "$PY" humanoidverse/train_agent.py \
    +simulator=isaacsim +exp=locomotion_soil algo=ppo_soil \
    +robot=hunter/hunter \
    +obs=loco/leggedloco_obs_singlestep_withlinvel \
    +rewards=loco/reward_hunter_dfh \
    num_envs="${NUM_ENVS}" headless=True seed="${seed}" \
    env.config.env_spacing=2.5 \
    algo.config.num_learning_iterations="${iters}" \
    rewards.reward_scales.penalty_sinkage_excess=0 \
    project_name=DFHTuned experiment_name="${run_id}" \
    experiment_dir="${run_dir}" save_dir="${run_dir}/.hydra" \
    hydra.run.dir="${run_dir}/.hydra" \
    "${extra_args[@]}"
}

dfh_train_args=(
  +terrain=terrain_dfh_stage1_easy
  +domain_rand=DR_dfh_soft
  terrain.dfh.force_coupling.enabled=True
  terrain.dfh.force_coupling.shuffle_env_ids=False
  terrain.dfh.writeback_enabled=False
)
shuffled_train_args=(
  +terrain=terrain_dfh_stage1_easy
  +domain_rand=DR_dfh_soft
  terrain.dfh.force_coupling.enabled=True
  terrain.dfh.force_coupling.shuffle_env_ids=True
  terrain.dfh.writeback_enabled=False
)
rigid_train_args=(
  +terrain=terrain_furrows_stage1_easy
  +domain_rand=DR_soil_rigid
)

latest_ckpt() { ls -t "${LOG_ROOT}/$1"/model_*.pt 2>/dev/null | head -1; }

cross_eval() {
  local train_id="$1" eval_name="$2" strength="$3" seed="$4"
  local ckpt; ckpt="$(latest_ckpt "${train_id}")"
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
      )
      # shellcheck disable=SC2206
      local extra=($(dose_overrides "$strength"))
      terrain_args+=("${extra[@]}")
      ;;
  esac

  "$PY" humanoidverse/sample_eps.py \
    +checkpoint="${ckpt}" \
    "${terrain_args[@]}" \
    +exp=locomotion_soil \
    num_envs="${EVAL_NUM_ENVS}" \
    +num_episodes="${EVAL_EPISODES}" \
    headless=True \
    project_name=DFHCrossEval \
    experiment_name="${train_id}_on_${eval_name}" \
    experiment_dir="${out_dir}" \
    save_dir="${out_dir}/.hydra" \
    hydra.run.dir="${out_dir}/.hydra"
}

# ---------------- Phase A: Calibration ----------------
# Skip if STRENGTH explicitly set.
SELECTED_STRENGTH="${STRENGTH:-}"
CALIB_LEVELS=(${CALIB_LEVELS:-mild medium strong ultra})
if [ -z "$SELECTED_STRENGTH" ]; then
  for strength in "${CALIB_LEVELS[@]}"; do
    run_id="CALIB_${strength}_s1"
    if [ -d "${LOG_ROOT}/${run_id}" ] && [ -n "$(latest_ckpt "${run_id}" || true)" ]; then
      echo "===== SKIP ${run_id} (already trained) ====="
      continue
    fi
    # shellcheck disable=SC2206
    overrides=($(dose_overrides "$strength"))
    run_train "$run_id" 1 "$CALIB_ITERS" "${dfh_train_args[@]}" "${overrides[@]}"
  done
  echo "===== Calibration complete. Inspect logs/DFHTuned/CALIB_* and pick STRENGTH. ====="
  echo "Re-run with: STRENGTH=mild|medium|strong|ultra $0"
  exit 0
fi

# ---------------- Phase B: Full Sweep ----------------
# shellcheck disable=SC2206
sel_overrides=($(dose_overrides "$SELECTED_STRENGTH"))
echo "===== Phase B: SELECTED_STRENGTH=${SELECTED_STRENGTH} ====="
# Train FULL, RIGID, and SHUFFLED (negative control) per seed.
for s in "${SEEDS[@]}"; do
  run_train "DFH_FULL_${SELECTED_STRENGTH}_s${s}" "$s" "$ITERS" \
    "${dfh_train_args[@]}" "${sel_overrides[@]}"
  run_train "RIGID_TUNED_s${s}" "$s" "$ITERS" \
    "${rigid_train_args[@]}"
  run_train "SHUFFLED_${SELECTED_STRENGTH}_s${s}" "$s" "$ITERS" \
    "${shuffled_train_args[@]}" "${sel_overrides[@]}"
done

# ---------------- Phase C: Cross-eval ----------------
for s in "${SEEDS[@]}"; do
  for tr in "DFH_FULL_${SELECTED_STRENGTH}_s${s}" \
            "RIGID_TUNED_s${s}" \
            "SHUFFLED_${SELECTED_STRENGTH}_s${s}"; do
    for ev in rigid dfh; do
      cross_eval "$tr" "$ev" "$SELECTED_STRENGTH" "$s"
    done
  done
done

echo "===== Pass 3 R2 sweep + cross-eval complete (strength=${SELECTED_STRENGTH}) ====="
