#!/usr/bin/env bash
# Eval-noise sweep for Arm D (compliant), to firm up the borderline ~1.3σ slip drop.
# Runs REPEATS compliant evals of the Arm D seed-1 checkpoint (compliance ON, matching
# training/eval), appends rows labeled "D" to eval_noise.csv, then does a proper
# two-sample (Welch) test of D vs B (B's 5 rigid reps already in the CSV).
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/eval_noise_D.sh > logs/FuzzySoilFurrows/eval_noise_D.log 2>&1 &
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
KN="${KN:-50000}"
CN="${CN:-500}"
D_CKPT="$ROOT/20260529_165729-armD_S3_seed1-locomotion-hunter/model_5000.pt"
[ -f "$D_CKPT" ] || { echo "Arm D ckpt missing: $D_CKPT"; exit 1; }
[ -f "$NOISE_CSV" ] || echo "label,ckpt,rep,seed,episodes,ep_len_steps,distance_m,slip_per_100m" > "$NOISE_CSV"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [eval-noise-D] $*"; }
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

for rep in $(seq 1 "$REPEATS"); do
  out="$ROOT/noise_D_rep${rep}.log"
  log "label=D rep=$rep seed=$rep (compliant kn=$KN cn=$CN)"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$D_CKPT" \
    +terrain="$EVAL_TERRAIN" +domain_rand="$EVAL_DR" \
    +eval_command="$EVAL_CMD" +seed="$rep" \
    ++terrain.compliant_contact_stiffness="$KN" ++terrain.compliant_contact_damping="$CN" \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out" >/dev/null
  e=$(extract "Episodes completed:" "$out")
  el=$(extract "Average episode length:" "$out")
  ds=$(extract "Average distance" "$out")
  sp=$(extract "Slip distance per 100m:" "$out")
  echo "D,$D_CKPT,$rep,$rep,$e,$el,$ds,$sp" >> "$NOISE_CSV"
  log "  -> ep_len=$el slip100=$sp dist=$ds"
done

log "sweep done — two-sample test D vs B"
python3 - "$NOISE_CSV" <<'PYEOF'
import csv, sys, statistics as st, math
rows = list(csv.DictReader(open(sys.argv[1])))
def f(x):
    try: return float(x)
    except: return None
def col(label, c):
    return [f(r[c]) for r in rows if r["label"]==label and f(r[c]) is not None]
print("\n== Arm D (compliant) vs Arm B (rigid) — slip/100m, repeated evals ==")
for lab in ("B","D"):
    v=col(lab,"slip_per_100m")
    if len(v)>=2:
        m,sd=st.mean(v),st.stdev(v)
        print(f"  {lab}: n={len(v)}  slip/100m = {m:7.1f} ± {sd:5.1f} (sd)   ep_len={st.mean(col(lab,'ep_len_steps')):.1f}")
b=col("B","slip_per_100m"); d=col("D","slip_per_100m")
if len(b)>=2 and len(d)>=2:
    mb,sb,nb=st.mean(b),st.stdev(b),len(b)
    md,sd_,nd=st.mean(d),st.stdev(d),len(d)
    se=math.sqrt(sb*sb/nb + sd_*sd_/nd)          # Welch SE
    t=(md-mb)/se if se else float('nan')
    # Welch–Satterthwaite df
    num=(sb*sb/nb + sd_*sd_/nd)**2
    den=(sb*sb/nb)**2/(nb-1) + (sd_*sd_/nd)**2/(nd-1)
    df=num/den if den else float('nan')
    print(f"\n  D-B = {md-mb:+.1f} slip/100m   Welch SE={se:.1f}   t={t:.2f}   df≈{df:.1f}")
    pct=100*(md-mb)/mb if mb else float('nan')
    print(f"  relative change: {pct:+.1f}%")
    sig = abs(t)>2.0
    print("  => " + ("D vs B SEPARABLE at ~2 SE — compliance lowers slip; Stage 1 JUSTIFIED."
                     if (sig and md<mb) else
                     "D vs B WITHIN noise (<2 SE) — global compliance is a wash; Stage 1 NOT justified on this evidence."))
PYEOF
log "DONE — see $NOISE_CSV"
