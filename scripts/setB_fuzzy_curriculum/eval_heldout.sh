#!/usr/bin/env bash
# Set B — held-out robustness eval of ONE final checkpoint over a fixed-mu sweep.
# Mirrors scripts/paper_fuzzy_soil/run_fuzzy_sweep_repeats.sh (same parse/ANSI-strip),
# but takes an arbitrary CKPT and writes a per-(mu,rep) robustness CSV.
#
# Usage:
#   CKPT=logs/.../model_xxxx.pt OUT_CSV=logs/FuzzySoilFurrowsSetB/eval/<tag>.csv \
#     scripts/setB_fuzzy_curriculum/eval_heldout.sh
#   # quick smoke (one mu, few eps):
#   CKPT=... OUT_CSV=... MUS="0.50" REPEATS=1 EVAL_EPS=5 EVAL_ENVS=64 \
#     scripts/setB_fuzzy_curriculum/eval_heldout.sh
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

CKPT="${CKPT:?set CKPT=path/to/model.pt}"
[ -f "$CKPT" ] || { echo "ckpt missing: $CKPT"; exit 1; }
OUT_CSV="${OUT_CSV:?set OUT_CSV=path/to/out.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
LOG_DIR="$(dirname "$OUT_CSV")/logs_$(basename "${OUT_CSV%.csv}")"
mkdir -p "$LOG_DIR"

EVAL_TERRAIN="${EVAL_TERRAIN:-terrain_furrows_with_maize}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-100}"
EVAL_CMD="${EVAL_CMD:-[0.3,0.0,0.0]}"
MUS="${MUS:-0.35 0.45 0.55 0.65 0.75 0.85}"
REPEATS="${REPEATS:-3}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [setB-eval] $*"; }
strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }

echo "mu,rep,seed,episodes,ep_len_steps,distance_m,slip_per_100m,falls_per_100m,vel_err" > "$OUT_CSV"
log "CKPT=$CKPT  MUS=[$MUS]  REPEATS=$REPEATS  -> $OUT_CSV"

for mu in $MUS; do
  for rep in $(seq 1 "$REPEATS"); do
    out="$LOG_DIR/mu${mu}_r${rep}.log"
    log "mu=$mu rep=$rep seed=$rep"
    "$PY" humanoidverse/sample_eps.py +checkpoint="$CKPT" \
      +terrain="$EVAL_TERRAIN" +domain_rand=DR_paper_S3 \
      +eval_command="$EVAL_CMD" +seed="$rep" \
      ++terrain.static_friction="$mu" ++terrain.dynamic_friction="$mu" \
      num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
      ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
      ++env.config.termination.terminate_by_contact=True \
      2>&1 | tee "$out" >/dev/null
    field() { grep -E "$1" "$out" 2>/dev/null | strip_ansi | tail -1 | sed -E 's/.*:[[:space:]]*//' | awk '{print $1}'; }
    e=$(field "Episodes completed:")
    el=$(field "Average episode length:")
    ds=$(field "Average distance per episode:")
    sp=$(field "Slip distance per 100m:")
    fp=$(field "Falls per 100m:")
    ve=$(field "Velocity error")
    echo "$mu,$rep,$rep,$e,$el,$ds,$sp,$fp,$ve" >> "$OUT_CSV"
    log "  -> ep_len=$el slip100=$sp falls100=$fp vel=$ve"
  done
done
log "DONE -> $OUT_CSV"
