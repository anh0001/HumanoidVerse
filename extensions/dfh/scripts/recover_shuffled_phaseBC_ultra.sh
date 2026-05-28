#!/usr/bin/env bash
# Recovery for the dead Pass-4 ultra sweep (bg pid 111152 killed at SHUFFLED
# iter 973/2000). DFH_FULL_ultra_s1 and RIGID_TUNED_s1 are already complete
# (model_2000.pt) and are NOT retrained here. This:
#   1. Archives the partial SHUFFLED_ultra_s1 dir.
#   2. Retrains SHUFFLED_ultra_s1 fresh -> 2000 iters (negative control).
#   3. Phase C cross-eval: {DFH_FULL,RIGID_TUNED,SHUFFLED}_*_s1 x {rigid,dfh}.
#   4. analyze_regret --strength ultra.
# All args byte-identical to run_tuned_sweep_crosseval.sh / the completed
# DFH_FULL_ultra_s1 overrides.yaml.
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
ITERS=2000
NUM_ENVS=2048
SEED=1
EVAL_EPISODES=50
# Eval at the training env count: the packed per-env spawn-anchor terrain path
# (commits fb68f9c/28549dd) is sized to num_envs=2048; eval terrain must match
# the layout the policies trained on. 2048 is validated on both eval terrains
# (RIGID_TUNED + DFH_FULL trained at 2048 on these exact terrains).
EVAL_NUM_ENVS=2048
STRENGTH=ultra
mkdir -p "${LOG_ROOT}/crosseval"

ultra_overrides=(
  terrain.dfh.params.anisotropy.mu_across=1.00
  terrain.dfh.params.anisotropy.mu_along=0.60
  terrain.dfh.force_coupling.sinkage_drag_k=40.0
  terrain.dfh.force_coupling.max_drag_force_n=500.0
)
shuffled_train_args=(
  +terrain=terrain_dfh_stage1_easy
  +domain_rand=DR_dfh_soft
  terrain.dfh.force_coupling.enabled=True
  terrain.dfh.force_coupling.shuffle_env_ids=True
  terrain.dfh.writeback_enabled=False
)

latest_ckpt() { ls -t "${LOG_ROOT}/$1"/model_*.pt 2>/dev/null | head -1; }

# ---- 1. Archive partial SHUFFLED (only if not already fully trained) ----
if [ ! -f "${LOG_ROOT}/SHUFFLED_ultra_s1/model_${ITERS}.pt" ] && [ -d "${LOG_ROOT}/SHUFFLED_ultra_s1" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "===== ARCHIVE partial SHUFFLED_ultra_s1 -> SHUFFLED_ultra_s1.partial_${ts} ====="
  mv "${LOG_ROOT}/SHUFFLED_ultra_s1" "${LOG_ROOT}/SHUFFLED_ultra_s1.partial_${ts}"
fi

# ---- 2. Retrain SHUFFLED fresh (skip if already at target iters) ----
run_dir="${LOG_ROOT}/SHUFFLED_ultra_s1"
if [ -f "${run_dir}/model_${ITERS}.pt" ]; then
  echo "===== SKIP TRAIN SHUFFLED_ultra_s1 (model_${ITERS}.pt exists) ====="
else
echo "===== TRAIN SHUFFLED_ultra_s1 iters=${ITERS} (fresh) ====="
"$PY" humanoidverse/train_agent.py \
  +simulator=isaacsim +exp=locomotion_soil algo=ppo_soil \
  +robot=hunter/hunter \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  +rewards=loco/reward_hunter_dfh \
  num_envs="${NUM_ENVS}" headless=True seed="${SEED}" \
  env.config.env_spacing=2.5 \
  algo.config.num_learning_iterations="${ITERS}" \
  rewards.reward_scales.penalty_sinkage_excess=0 \
  project_name=DFHTuned experiment_name=SHUFFLED_ultra_s1 \
  experiment_dir="${run_dir}" save_dir="${run_dir}/.hydra" \
  hydra.run.dir="${run_dir}/.hydra" \
  "${shuffled_train_args[@]}" "${ultra_overrides[@]}"
fi

# ---- 3. Phase C cross-eval ----
cross_eval() {
  local train_id="$1" eval_name="$2"
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
        "${ultra_overrides[@]}"
      )
      ;;
  esac
  # sample_eps.py uses config_name=base_eval: only num_envs/+num_episodes/
  # +eval_tb_dir/eval_name/hydra.run.dir are valid. project_name/
  # experiment_name/experiment_dir/save_dir are NOT in its struct and were
  # silently never honored -- tfevents are routed via +eval_tb_dir, which
  # analyze_regret.py globs at crosseval/<train>_on_<eval>/.
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

for tr in "DFH_FULL_${STRENGTH}_s${SEED}" "RIGID_TUNED_s${SEED}" "SHUFFLED_${STRENGTH}_s${SEED}"; do
  for ev in rigid dfh; do
    cross_eval "$tr" "$ev"
  done
done

# ---- 4. Paired-improvement / regret analysis ----
echo "===== analyze_regret strength=${STRENGTH} ====="
"$PY" -m extensions.dfh.scripts.analyze_regret \
  --strength "${STRENGTH}" --root "${LOG_ROOT}" \
  --csv "${LOG_ROOT}/crosseval/regret_${STRENGTH}.csv"

echo "===== Recovery complete (strength=${STRENGTH}) ====="
