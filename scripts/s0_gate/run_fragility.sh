#!/usr/bin/env bash
# §0-A — Baseline fragility measurement (EXPERIMENT_PLAN.md §0).
#
# Question (reviewer Q1): is there "drastic" headroom? Only true if the
# current clean baseline is FRAGILE under harder terrain + domain rand.
#
# Baseline: logs/CurriculumStage2/model_910050.pt (clean curriculum-trained,
# NOT a DFH variant). Evaluated on 4 stress conditions, sequential, blind.
# Budget: modest num_envs/episodes so total < ~1.5 GPU-h on the
# externally-contended GPU. Logs → logs/S0Gate/<cond>/.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# IsaacSim env (mirrors extensions/dfh/scripts/run_paired_evidence.sh).
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"
PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"

CKPT="${CKPT:-logs/CurriculumStage2/model_910050.pt}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_EPISODES="${NUM_EPISODES:-64}"
CMD="${CMD:-[0.3,0.0,0.0]}"
LOG_ROOT="logs/S0Gate"
mkdir -p "$LOG_ROOT"

# condition : terrain : domain_rand : extra_overrides
# soil terrains are 1x1-tile (capacity 9/tile); bump to 6x6=36 tiles -> cap 324 >= num_envs.
CONDS=(
  "soil_moderate:terrain_soil_moderate_tilled:DR_soil_moderate:++terrain.num_rows=6 ++terrain.num_cols=6"
  "soil_challenging:terrain_soil_challenging_wet:DR_soil_challenging:++terrain.num_rows=6 ++terrain.num_cols=6"
  "furrows_s2:terrain_furrows_stage2_medium:YES_domain_rand:"
  "furrows_s3:terrain_furrows_stage3_full:YES_domain_rand:"
)

echo "[s0-A] baseline=$CKPT num_envs=$NUM_ENVS episodes=$NUM_EPISODES cmd=$CMD"
for entry in "${CONDS[@]}"; do
  IFS=":" read -r name terrain dr extra <<< "$entry"
  out="$LOG_ROOT/$name"
  mkdir -p "$out"
  echo "[s0-A] === $name  (terrain=$terrain dr=$dr extra='$extra') ==="
  "$PY" humanoidverse/sample_eps.py \
    +checkpoint="$CKPT" \
    +terrain="$terrain" \
    +domain_rand="$dr" \
    +eval_command="$CMD" \
    num_envs="$NUM_ENVS" \
    +num_episodes="$NUM_EPISODES" \
    headless=True \
    $extra \
    2>&1 | tee "$out/eval.log"
  echo "[s0-A] $name done -> $out/eval.log"
done
echo "[s0-A] ALL CONDITIONS COMPLETE"
