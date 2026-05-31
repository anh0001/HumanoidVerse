#!/usr/bin/env bash
# Set B — autonomous continuation ("phase 2"). Runs unattended to completion.
#
# Assumes the FIXED arm is already running (campaign_fixed.log). This script:
#   1. waits for the fixed arm to finish (or aborts if it failed),
#   2. CALIBRATES crisp_threshold + fuzzy breakpoints from the fixed arm's
#      block-0 (mu=0.80) mean ep_len averaged over its 3 seeds (read from file,
#      never from memory),
#   3. runs the CRISP arm (3 seeds) then the FUZZY arm (3 seeds) with the
#      identical calibrated constants,
#   4. held-out eval of all completed final checkpoints over the mu sweep,
#   5. analyze_setB.py -> setB_analysis.txt.
# Best-effort: a failing arm is logged but eval/analyze still run on whatever
# completed. A STATUS file records the final state.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python}"
[ -x "$PY" ] || PY="$HOME/miniconda3/envs/isaaclab/bin/python"

ROOT="logs/FuzzySoilFurrowsSetB"
FIXED_LOG="$ROOT/campaign_fixed.log"
EVAL_DIR="$ROOT/eval"
STATUS="$ROOT/STATUS.txt"
mkdir -p "$EVAL_DIR"
ts() { date '+%F %T'; }
say() { echo "[$(ts)] [phase2] $*"; }
set_status() { echo "[$(ts)] $*" > "$STATUS"; }

EVAL_MUS="${EVAL_MUS:-0.35 0.45 0.55 0.65 0.75 0.85}"
EVAL_REPEATS="${EVAL_REPEATS:-2}"
EVAL_EPS="${EVAL_EPS:-100}"
EVAL_ENVS="${EVAL_ENVS:-256}"

set_status "RUNNING: waiting for fixed arm"
say "waiting for fixed arm to finish ..."
# ---- 1. wait for fixed arm (poll log; bail on ABORT) ----
while true; do
  if grep -q "ARM=fixed all seeds done" "$FIXED_LOG" 2>/dev/null; then
    say "fixed arm done."; break
  fi
  if grep -q "ABORT" "$FIXED_LOG" 2>/dev/null; then
    say "fixed arm ABORTED — continuing to calibration with whatever exists."; break
  fi
  sleep 60
done

# ---- 2. calibrate from fixed block-0 mean ep_len (averaged over seeds) ----
say "calibrating crisp_threshold + fuzzy breaks from fixed block-0 ep_len ..."
CALIB_OUT="$("$PY" - <<'PYEOF'
import csv, glob, os, statistics as st
vals = []
for dec in glob.glob("logs/FuzzySoilFurrowsSetB/setB_fixed_seed*/decisions.csv"):
    try:
        with open(dec) as f:
            for r in csv.DictReader(f):
                if r.get("block") == "0":
                    v = float(r["mean_ep_len"]); vals.append(v); break
    except Exception:
        pass
cap = 1000.0
if vals:
    mastery0 = st.mean(vals) / cap
else:
    mastery0 = 0.45  # fallback if no fixed data; documented
T = min(0.55, max(0.20, round(0.75 * mastery0, 2)))
lo = max(0.05, round(T - 0.12, 2)); hi = min(0.95, round(T + 0.12, 2))
print(f"{T} {lo},{T},{hi} {round(mastery0,4)} {len(vals)}")
PYEOF
)"
CRISP_T="$(echo "$CALIB_OUT" | awk '{print $1}')"
FUZZY_BREAKS="$(echo "$CALIB_OUT" | awk '{print $2}')"
MASTERY0="$(echo "$CALIB_OUT" | awk '{print $3}')"
NSEEDS="$(echo "$CALIB_OUT" | awk '{print $4}')"
{
  echo "# Set B calibration (from fixed arm block-0 mu=0.80 mean ep_len)"
  echo "# averaged over $NSEEDS fixed seed(s); ep_len_cap=1000; mastery0=$MASTERY0"
  echo "# crisp advances when mastery>=CRISP_T; fuzzy breaks centred on CRISP_T"
  echo "CRISP_T=$CRISP_T"
  echo "FUZZY_BREAKS=$FUZZY_BREAKS"
} > "$ROOT/calib.env"
say "calib: mastery0=$MASTERY0 (n=$NSEEDS)  CRISP_T=$CRISP_T  FUZZY_BREAKS=$FUZZY_BREAKS"
export CRISP_T FUZZY_BREAKS

# ---- 3. crisp then fuzzy (3 seeds each), identical calibrated constants ----
set_status "RUNNING: crisp arm (CRISP_T=$CRISP_T)"
say "=== CRISP arm ==="
ARM=crisp CRISP_T="$CRISP_T" FUZZY_BREAKS="$FUZZY_BREAKS" \
  bash "$SCRIPT_DIR/run_all_seeds.sh" >> "$ROOT/campaign_crisp.log" 2>&1
say "crisp arm rc=$?"

set_status "RUNNING: fuzzy arm (FUZZY_BREAKS=$FUZZY_BREAKS)"
say "=== FUZZY arm ==="
ARM=fuzzy CRISP_T="$CRISP_T" FUZZY_BREAKS="$FUZZY_BREAKS" \
  bash "$SCRIPT_DIR/run_all_seeds.sh" >> "$ROOT/campaign_fuzzy.log" 2>&1
say "fuzzy arm rc=$?"

# ---- 4. held-out eval of all completed final checkpoints ----
set_status "RUNNING: held-out eval"
say "=== held-out eval (mus=[$EVAL_MUS] reps=$EVAL_REPEATS eps=$EVAL_EPS) ==="
for arm in fixed crisp fuzzy; do
  for seed in 1 2 3; do
    man="$ROOT/setB_${arm}_seed${seed}/manifest.json"
    [ -f "$man" ] || { say "skip eval: no manifest $man"; continue; }
    ckpt="$("$PY" -c "import json;print(json.load(open('$man'))['final_ckpt'])" 2>/dev/null)"
    [ -f "$ckpt" ] || { say "skip eval: final_ckpt missing for $arm seed$seed"; continue; }
    out="$EVAL_DIR/setB_${arm}_seed${seed}.csv"
    say "eval $arm seed$seed -> $out"
    CKPT="$ckpt" OUT_CSV="$out" MUS="$EVAL_MUS" REPEATS="$EVAL_REPEATS" \
      EVAL_EPS="$EVAL_EPS" EVAL_ENVS="$EVAL_ENVS" \
      bash "$SCRIPT_DIR/eval_heldout.sh" >> "$ROOT/eval_all.log" 2>&1
    say "  eval $arm seed$seed rc=$?"
  done
done

# ---- 5. analyze ----
set_status "RUNNING: analyze"
say "=== analyze ==="
"$PY" "$SCRIPT_DIR/analyze_setB.py" --eval-dir "$EVAL_DIR" --runs-root "$ROOT" \
  --out "$ROOT/setB_analysis.txt" >> "$ROOT/analyze.log" 2>&1
say "analyze rc=$?"
set_status "DONE: see $ROOT/setB_analysis.txt"
say "ALL DONE. Verdict -> $ROOT/setB_analysis.txt"
