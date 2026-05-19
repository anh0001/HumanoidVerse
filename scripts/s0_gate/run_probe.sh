#!/usr/bin/env bash
# §0-B — latent-identifiability probe (EXPERIMENT_PLAN.md §0, the falsification gate).
# Collect the exact policy-input stream under 4 regimes, then test whether a
# short proprio-history window identifies the regime early (pre-fall).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"
PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"

CKPT="${CKPT:-logs/CurriculumStage2/model_910050.pt}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-250}"
OUT="logs/S0Gate/probe"
SOIL_TILES="++terrain.num_rows=6 ++terrain.num_cols=6"
mkdir -p "$OUT"

# regime:terrain:domain_rand:extra
REGIMES=(
  "0:plane:terrain_locomotion_plane:NO_domain_rand:"
  "1:soil_moderate:terrain_soil_moderate_tilled:DR_soil_moderate:$SOIL_TILES"
  "2:soil_challenging:terrain_soil_challenging_wet:DR_soil_challenging:$SOIL_TILES"
  "3:furrows_s2:terrain_furrows_stage2_medium:YES_domain_rand:"
)

for entry in "${REGIMES[@]}"; do
  IFS=":" read -r rid name terrain dr extra <<< "$entry"
  echo "[s0-B] === collect regime $rid=$name (terrain=$terrain dr=$dr) ==="
  "$PY" humanoidverse/collect_probe_data.py \
    +checkpoint="$CKPT" \
    +terrain="$terrain" \
    +domain_rand="$dr" \
    +eval_command=[0.3,0.0,0.0] \
    num_envs="$NUM_ENVS" \
    +num_episodes=1 \
    +probe_out="$OUT/$name.npz" \
    +probe_steps="$STEPS" \
    +probe_regime="$rid" \
    +probe_regime_name="$name" \
    headless=True \
    $extra \
    2>&1 | tee "$OUT/${name}_collect.log"
  echo "[s0-B] regime $name done"
done

echo "[s0-B] === training probe ==="
"$PY" scripts/s0_gate/train_probe.py --data "$OUT" --report refine-logs/S0B_PROBE_REPORT.md 2>&1 | tee "$OUT/train_probe.log"
echo "[s0-B] COMPLETE"
