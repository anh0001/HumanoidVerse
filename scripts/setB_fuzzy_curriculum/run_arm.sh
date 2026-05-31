#!/usr/bin/env bash
# Set B — run ONE arm x ONE seed of the block-chained friction curriculum.
# Sets up the IsaacSim/IsaacLab env (same block as the Arm B / sweep scripts),
# then hands off to the Python orchestrator run_block_chain.py.
#
# Usage:
#   ARM=fuzzy SEED=1 scripts/setB_fuzzy_curriculum/run_arm.sh
#   ARM=crisp SEED=2 BLOCKS=8 ITERS=94 NUM_ENVS=2048 scripts/setB_fuzzy_curriculum/run_arm.sh
#   # pre-flight smoke (tiny):
#   ARM=crisp SEED=1 BLOCKS=2 ITERS=4 NUM_ENVS=64 SMOKE=1 scripts/setB_fuzzy_curriculum/run_arm.sh
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

ARM="${ARM:?set ARM=fixed|crisp|fuzzy}"
SEED="${SEED:-1}"
BLOCKS="${BLOCKS:-8}"
ITERS="${ITERS:-94}"
NUM_ENVS="${NUM_ENVS:-2048}"
LADDER="${LADDER:-0.80,0.65,0.50,0.35}"
EP_LEN_CAP="${EP_LEN_CAP:-1000}"
CRISP_T="${CRISP_T:-0.50}"
FUZZY_BREAKS="${FUZZY_BREAKS:-0.35,0.50,0.65}"
WARM_CKPT="${WARM_CKPT:-logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt}"

extra=()
[ "${SMOKE:-0}" = "1" ] && extra+=(--smoke)

exec "$PY" "$SCRIPT_DIR/run_block_chain.py" \
  --arm "$ARM" --seed "$SEED" --blocks "$BLOCKS" --iters-per-block "$ITERS" \
  --num-envs "$NUM_ENVS" --ladder "$LADDER" --ep-len-cap "$EP_LEN_CAP" \
  --crisp-threshold "$CRISP_T" --fuzzy-breaks "$FUZZY_BREAKS" \
  --warm-ckpt "$WARM_CKPT" --py "$PY" "${extra[@]}"
