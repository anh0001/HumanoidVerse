#!/usr/bin/env bash
# Set-A fuzzy validation, REPEATED evals per condition to average out the ~8% eval
# noise and catch logging artifacts (e.g. the suspicious duplicate slip values in
# the single-shot run). Sweeps terrain friction mu with the FULL DR_paper_S3 held
# constant (keeps the policy in-distribution). REPEATS evals per mu, each with a
# distinct +seed; writes EVERY raw per-eval row to fuzzy_sweep_repeats_raw.csv,
# then aggregates to fuzzy_sweep.csv (mean per mu). No hand-entered numbers.
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/run_fuzzy_sweep_repeats.sh > logs/FuzzySoilFurrows/fuzzy_repeats_queue.log 2>&1 &
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
RAW_CSV="$ROOT/fuzzy_sweep_repeats_raw.csv"
AGG_CSV="$ROOT/fuzzy_sweep.csv"
CKPT="${CKPT:-$ROOT/20260529_015641-armB_S3_seed1-locomotion-hunter/model_5000.pt}"
EVAL_TERRAIN="${EVAL_TERRAIN:-terrain_furrows_with_maize}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-100}"
EVAL_CMD="${EVAL_CMD:-[0.3,0.0,0.0]}"
MUS="${MUS:-0.35 0.45 0.55 0.65 0.75 0.85}"
REPEATS="${REPEATS:-3}"
mkdir -p "$ROOT"
[ -f "$CKPT" ] || { echo "ckpt missing: $CKPT"; exit 1; }

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fuzzy-rep] $*"; }

echo "mu,rep,seed,episodes,ep_len_steps,distance_m,slip_per_100m,falls_per_100m,vel_err" > "$RAW_CSV"

for mu in $MUS; do
  for rep in $(seq 1 "$REPEATS"); do
    out="$ROOT/fuzzy_rep_mu${mu}_r${rep}.log"
    log "mu=$mu rep=$rep seed=$rep"
    "$PY" humanoidverse/sample_eps.py +checkpoint="$CKPT" \
      +terrain="$EVAL_TERRAIN" +domain_rand=DR_paper_S3 \
      +eval_command="$EVAL_CMD" +seed="$rep" \
      ++terrain.static_friction="$mu" ++terrain.dynamic_friction="$mu" \
      num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
      ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
      ++env.config.termination.terminate_by_contact=True \
      2>&1 | tee "$out" >/dev/null
    # Parse THIS eval's own RESULTS block. Strip ANSI color codes (loguru lines
    # carry \e[..m) BEFORE extracting, else they leak into fields like "100\e[0m".
    strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }
    field() { grep -E "$1" "$out" 2>/dev/null | strip_ansi | tail -1 | sed -E 's/.*:[[:space:]]*//' | awk '{print $1}'; }
    e=$(field "Episodes completed:")
    el=$(field "Average episode length:")
    ds=$(field "Average distance per episode:")
    sp=$(field "Slip distance per 100m:")
    fp=$(field "Falls per 100m:")
    ve=$(field "Velocity error")
    echo "$mu,$rep,$rep,$e,$el,$ds,$sp,$fp,$ve" >> "$RAW_CSV"
    log "  -> ep_len=$el slip100=$sp falls100=$fp vel=$ve"
  done
done

log "all evals done — aggregating raw -> mean per mu"
"$PY" - "$RAW_CSV" "$AGG_CSV" <<'PYEOF'
import csv, sys
from collections import defaultdict
raw_path, agg_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(raw_path)))
by_mu = defaultdict(list)
for r in rows:
    by_mu[r["mu"]].append(r)
def fmean(rs, col):
    vals = []
    for r in rs:
        try: vals.append(float(r[col]))
        except (ValueError, KeyError): pass
    return sum(vals)/len(vals) if vals else float("nan")
with open(agg_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mu","kn_knm","episodes","falls","ep_len_steps","distance_m",
                "slip_per_100m","falls_per_100m","vel_err"])
    for mu in sorted(by_mu, key=float):
        rs = by_mu[mu]
        w.writerow([mu, 1000000, 1000, 1000,
                    f"{fmean(rs,'ep_len_steps'):.2f}",
                    f"{fmean(rs,'distance_m'):.3f}",
                    f"{fmean(rs,'slip_per_100m'):.3f}",
                    f"{fmean(rs,'falls_per_100m'):.3f}",
                    f"{fmean(rs,'vel_err'):.3f}"])
print(f"[agg] wrote {agg_path} from {len(rows)} raw evals over {len(by_mu)} mu conditions")
PYEOF

log "running fuzzy analysis on aggregated means"
"$PY" "$SCRIPT_DIR/analyze_fuzzy.py" --sweep "$AGG_CSV" 2>/dev/null | tee "$ROOT/fuzzy_analysis.txt"
log "DONE — raw=$RAW_CSV agg=$AGG_CSV analysis=$ROOT/fuzzy_analysis.txt"
