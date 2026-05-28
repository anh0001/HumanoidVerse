#!/usr/bin/env bash
# Paired evidence matrix for DFH v0.2 surrogate.
#
# Conditions × seeds = 4 × 3 = 12 runs. ~2h per run at num_envs=2048,
# ~25-30h total wall-clock on a single GPU. Sequential to avoid VRAM
# contention on shared IsaacSim instance.
#
# Each run writes to logs/DFHEvidence/<condition>_s<seed>/ with full
# tfevents and Hydra config snapshot. Post-hoc analysis scripts can
# walk that tree and produce paired deltas per seed.
set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# IsaacSim env (mirrors the one used in /tmp/run_dfh_smoke.sh).
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"
PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"

LOG_ROOT="logs/DFHEvidence"
ITERS="${ITERS:-5000}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEEDS=(${SEEDS:-0 1 2})
CONDITIONS=(${CONDITIONS:-DFH_FULL SHADOW RIGID SHUFFLED})

mkdir -p "${LOG_ROOT}"

run_condition() {
  local condition="$1"
  local seed="$2"
  local run_id="${condition}_s${seed}"
  local run_dir="${LOG_ROOT}/${run_id}"

  local common=(
    "$PY" humanoidverse/train_agent.py
    +simulator=isaacsim
    +exp=locomotion_soil
    algo=ppo_soil
    +robot=hunter/hunter
    +obs=loco/leggedloco_obs_singlestep_withlinvel
    +rewards=loco/reward_hunter_dfh
    num_envs="${NUM_ENVS}"
    headless=True
    seed="${seed}"
    env.config.env_spacing=2.5
    algo.config.num_learning_iterations="${ITERS}"
    rewards.reward_scales.penalty_sinkage_excess=0
    project_name=DFHEvidence
    experiment_name="${run_id}"
    experiment_dir="${run_dir}"
    save_dir="${run_dir}/.hydra"
    hydra.run.dir="${run_dir}/.hydra"
  )

  case "${condition}" in
    DFH_FULL)
      "${common[@]}" \
        +terrain=terrain_dfh_stage1_easy \
        +domain_rand=DR_dfh_soft \
        terrain.dfh.force_coupling.enabled=True \
        terrain.dfh.force_coupling.shuffle_env_ids=False \
        terrain.dfh.writeback_enabled=False
      ;;
    SHADOW)
      "${common[@]}" \
        +terrain=terrain_dfh_stage1_easy \
        +domain_rand=DR_dfh_soft \
        terrain.dfh.force_coupling.enabled=False \
        terrain.dfh.force_coupling.shuffle_env_ids=False \
        terrain.dfh.writeback_enabled=False
      ;;
    RIGID)
      "${common[@]}" \
        +terrain=terrain_furrows_stage1_easy \
        +domain_rand=DR_soil_rigid
      ;;
    SHUFFLED)
      "${common[@]}" \
        +terrain=terrain_dfh_stage1_easy \
        +domain_rand=DR_dfh_soft \
        terrain.dfh.force_coupling.enabled=True \
        terrain.dfh.force_coupling.shuffle_env_ids=True \
        terrain.dfh.writeback_enabled=False
      ;;
    *)
      echo "Unknown condition: ${condition}" >&2
      exit 2
      ;;
  esac
}

for condition in "${CONDITIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "===== Starting ${condition} seed=${seed} ====="
    run_condition "${condition}" "${seed}"
    echo "===== Finished ${condition} seed=${seed} ====="
  done
done
