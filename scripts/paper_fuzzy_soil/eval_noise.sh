#!/usr/bin/env bash
# Quantify EVAL noise: re-run the same checkpoint K times and measure the spread
# of ep_len and slip_per_100m. Motivated by a 23% ep_len swing observed on the SAME
# Arm B seed-1 checkpoint between two evals (GPU-PhysX / cuDNN nondeterminism; note
# sample_eps.py does not call seeding(), so runs are not bitwise reproducible).
#
# Sweeps the two checkpoints we just compared (Arm B seed1, Arm C seed1), REPEATS
# each, varying +seed to also stir env-init RNG on top of the inherent nondeterminism.
# Writes one row per run to logs/FuzzySoilFurrows/eval_noise.csv, then summarizes.
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/eval_noise.sh > logs/FuzzySoilFurrows/eval_noise.log 2>&1 &
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
NOISE_CSV="$ROOT/eval_noise.csv"
EVAL_TERRAIN="${EVAL_TERRAIN:-terrain_furrows_with_maize}"
EVAL_DR="${EVAL_DR:-DR_paper_S3}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-100}"
EVAL_CMD="${EVAL_CMD:-[0.3,0.0,0.0]}"
REPEATS="${REPEATS:-5}"
mkdir -p "$ROOT"

B_CKPT="$ROOT/20260529_015641-armB_S3_seed1-locomotion-hunter/model_5000.pt"
C_CKPT="$ROOT/20260529_131748-armC_S3_seed1-locomotion-hunter/model_5000.pt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [eval-noise] $*"; }
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "label,ckpt,rep,seed,episodes,ep_len_steps,distance_m,slip_per_100m" > "$NOISE_CSV"

run_rep() {
  local label="$1" ckpt="$2" rep="$3" seed="$4"
  [ -f "$ckpt" ] || { log "MISSING ckpt $ckpt — skip"; return; }
  local out="$ROOT/noise_${label}_rep${rep}.log"
  log "label=$label rep=$rep seed=$seed"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ckpt" \
    +terrain="$EVAL_TERRAIN" +domain_rand="$EVAL_DR" \
    +eval_command="$EVAL_CMD" +seed="$seed" \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out" >/dev/null
  local e el ds sp
  e=$(extract "Episodes completed:" "$out")
  el=$(extract "Average episode length:" "$out")
  ds=$(extract "Average distance" "$out")
  sp=$(extract "Slip distance per 100m:" "$out")
  echo "$label,$ckpt,$rep,$seed,$e,$el,$ds,$sp" >> "$NOISE_CSV"
  log "  -> ep_len=$el slip100=$sp dist=$ds"
}

for rep in $(seq 1 "$REPEATS"); do
  run_rep "B" "$B_CKPT" "$rep" "$rep"
  run_rep "C" "$C_CKPT" "$rep" "$rep"
done

log "sweep done — summarizing"
python3 - "$NOISE_CSV" <<'PYEOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
def f(x):
    try: return float(x)
    except: return None
print("\n== Eval-noise summary (same checkpoint, repeated evals) ==")
print(f"{'label':>6} | {'n':>2} | {'ep_len mean±sd (cv%)':>26} | {'slip100 mean±sd (cv%)':>26}")
for label in ("B","C"):
    rs = [r for r in rows if r["label"]==label]
    els = [f(r["ep_len_steps"]) for r in rs if f(r["ep_len_steps"]) is not None]
    sps = [f(r["slip_per_100m"]) for r in rs if f(r["slip_per_100m"]) is not None]
    def stat(v):
        if len(v)<2: return "n/a"
        m=st.mean(v); sd=st.pstdev(v); cv=100*sd/m if m else float('nan')
        return f"{m:8.1f} ± {sd:6.1f} ({cv:4.1f}%)"
    print(f"{label:>6} | {len(rs):>2} | {stat(els):>26} | {stat(sps):>26}")
# Is C distinguishable from B given this noise?
def mean_of(label, col):
    v=[f(r[col]) for r in rows if r["label"]==label and f(r[col]) is not None]
    return (st.mean(v), st.pstdev(v), len(v)) if v else (None,None,0)
mb,sb,nb = mean_of("B","ep_len_steps"); mc,sc,nc = mean_of("C","ep_len_steps")
if mb and mc and nb>1 and nc>1:
    pooled = ((sb**2/nb)+(sc**2/nc))**0.5
    diff = mc-mb
    print(f"\nep_len: C-B = {diff:+.1f} steps; pooled SE = {pooled:.1f}; "
          f"|diff|/SE = {abs(diff)/pooled:.2f}")
    print("  => " + ("C vs B SEPARABLE (>2 SE)" if abs(diff)>2*pooled
                      else "C vs B WITHIN eval noise (not separable at 1 seed)"))
PYEOF
log "DONE — see $NOISE_CSV"
