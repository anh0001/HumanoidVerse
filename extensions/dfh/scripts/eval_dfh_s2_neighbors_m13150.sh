#!/usr/bin/env bash
# Cross-condition eval of S2-proof m13150 to validate it isn't overfit to k=12.
# Codex (thread 019e72a4) recommendation. Runs 3 conditions SEQUENTIALLY
# (parallel would OOM alongside the S2-dose-1 training that's also on the GPU).
#
# Conditions:
#   1. floor=-0.08, k=12, max=280   (headroom test — does new cap help baseline?)
#   2. floor=-0.08, k=13, max=280   (the dose we're about to train into — baseline)
#   3. floor=-0.09, k=13, max=280   (forward S3-bridge probe — ahead of training)
#
# Each run: num_envs=64, num_episodes=64, forward velocity 0.2 m/s.
# ~2-3 min per condition incl IsaacSim boot.

set -euo pipefail
cd "$(dirname "$0")/../../.."

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"

PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"

CKPT="logs/DFH_Hunter_ROA/20260529_121443-DFH_S2_real_floor08_k12_from_12150_seed1-locomotion-hunter/model_13150.pt"
[[ -f "$CKPT" ]] || { echo "ERROR: $CKPT not found"; exit 1; }

OUTDIR="logs/DFH_Hunter_ROA/eval_m13150_neighbors"
mkdir -p "$OUTDIR"

run_eval () {
  local tag="$1" floor="$2" k="$3" maxn="$4"
  local log="$OUTDIR/eval_${tag}.log"
  echo ""
  echo "==========================================================" | tee -a "$log"
  echo "  [${tag}] floor=${floor}, k=${k}, max=${maxn}  $(date +%H:%M:%S)" | tee -a "$log"
  echo "==========================================================" | tee -a "$log"
  "${PYTHON}" humanoidverse/sample_eps.py \
    +simulator=isaacsim \
    +checkpoint="$CKPT" \
    +terrain=terrain_dfh_stage2_medium \
    +eval_command="[0.2,0.0,0.0]" \
    num_envs=64 \
    +num_episodes=64 \
    headless=True \
    ++terrain.dfh.params.sinkage_floor_m="${floor}" \
    ++terrain.dfh.force_coupling.sinkage_drag_k="${k}" \
    ++terrain.dfh.force_coupling.max_drag_force_n="${maxn}" \
    2>&1 | tee -a "$log"
}

run_eval "01_floor08_k12_max280" "-0.08" "12.0" "280.0"
run_eval "02_floor08_k13_max280" "-0.08" "13.0" "280.0"
run_eval "03_floor09_k13_max280" "-0.09" "13.0" "280.0"

echo ""
echo "===== NEIGHBOR EVAL SUMMARY ($(date)) ====="
for tag in 01_floor08_k12_max280 02_floor08_k13_max280 03_floor09_k13_max280; do
  echo ""
  echo "[$tag]"
  grep -E "Episodes completed|Total falls|Average episode length|Average distance" "$OUTDIR/eval_${tag}.log" 2>/dev/null | tail -10
done
