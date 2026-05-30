#!/usr/bin/env bash
# Set-A experiment: sweep friction mu at controlled conditions to validate the
# fuzzy difficulty index as a descriptor of measured task difficulty.
#
# Holds contact RIGID (no compliance) and DR OFF (no pushes / no friction
# randomization) so each condition has a single known mu -> a clean mapping
# condition -> fuzzy D. Evaluates a fixed trained policy (Arm B seed1) at each mu.
# Records mu + slip/falls/ep_len/vel_err. analyze_fuzzy.py then computes D and
# tests monotonicity + fuzzy-vs-linear predictive power.
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/run_fuzzy_sweep.sh > logs/FuzzySoilFurrows/fuzzy_sweep.log 2>&1 &
#   disown
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

ROOT="logs/FuzzySoilFurrows"
SWEEP_CSV="$ROOT/fuzzy_sweep.csv"
CKPT="${CKPT:-$ROOT/20260529_015641-armB_S3_seed1-locomotion-hunter/model_5000.pt}"
EVAL_TERRAIN="${EVAL_TERRAIN:-terrain_furrows_with_maize}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-100}"
EVAL_CMD="${EVAL_CMD:-[0.3,0.0,0.0]}"
MUS="${MUS:-0.35 0.45 0.55 0.65 0.75 0.85}"   # spans traction Low(<0.45)/Med(~0.6)/High(>0.75)
mkdir -p "$ROOT"
[ -f "$CKPT" ] || { echo "ckpt missing: $CKPT"; exit 1; }

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fuzzy-sweep] $*"; }
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "mu,kn_knm,episodes,falls,ep_len_steps,distance_m,slip_per_100m,falls_per_100m,vel_err" > "$SWEEP_CSV"

for mu in $MUS; do
  out="$ROOT/fuzzy_mu${mu}.log"
  log "mu=$mu (DR_paper_S3 held constant; only terrain ground friction varies)"
  # Keep the FULL DR the policy was trained with (pushes, gains, ctrl-delay) so it
  # stays in-distribution (~450 ep_len). DR's randomize_friction touches JOINT
  # friction (mdp.randomize_joint_parameters), which is independent of the terrain
  # ground material friction we override here -> mu is the lone difficulty variable.
  # Override terrain friction AFTER the DR include so it wins for the ground material.
  "$PY" humanoidverse/sample_eps.py +checkpoint="$CKPT" \
    +terrain="$EVAL_TERRAIN" +domain_rand=DR_paper_S3 \
    +eval_command="$EVAL_CMD" \
    ++terrain.static_friction="$mu" ++terrain.dynamic_friction="$mu" \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out" >/dev/null
  e=$(extract "Episodes completed:" "$out")
  fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out")
  ds=$(extract "Average distance" "$out")
  sp=$(extract "Slip distance per 100m:" "$out")
  # vel_err line is "Velocity error (m/s): 0.497 ± 0.024" — take only the first number.
  ve=$(grep -E "Velocity error" "$out" 2>/dev/null | tail -1 | sed -E 's/.*:\s*//' | awk '{print $1}')
  fp=""
  if [ -n "$fl" ] && [ -n "$ds" ] && [ -n "$e" ]; then
    fp=$(python3 -c "
fl,ds,e=float('$fl'),float('$ds'),float('$e'); td=ds*e
print(f'{(fl/(td/100.0)):.3f}') if td>0 else print('')")
  fi
  echo "$mu,1000000,$e,$fl,$el,$ds,$sp,$fp,$ve" >> "$SWEEP_CSV"
  log "  -> ep_len=$el slip100=$sp falls100=$fp vel_err=$ve"
done

log "sweep done — running fuzzy analysis"
python3 "$SCRIPT_DIR/analyze_fuzzy.py" --sweep "$SWEEP_CSV" 2>&1 | tee "$ROOT/fuzzy_analysis.txt"
log "DONE — see $ROOT/fuzzy_analysis.txt and $ROOT/fuzzy_validation.png"
