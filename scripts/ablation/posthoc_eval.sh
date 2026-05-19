#!/usr/bin/env bash
# Post-hoc held-out eval for §2 ablation matrix.
#
# Workaround for the in-wave eval crash: the saved train config bakes
# `env.config.simulator.scene.num_envs: 2048` as a LITERAL (not an
# interpolation of top-level `num_envs`), so the CLI `num_envs=256`
# override is shadowed -> env builds at 2048 and crashes. We override
# both the top-level `num_envs` AND that nested literal here.
#
# Idempotent: skips runs whose eval_<cond>_v2.log already shows a summary.
# Run while the train wave is still ongoing (only touches DONE arms by
# requiring `model_3000.pt`).
#
# Usage:
#   bash scripts/ablation/posthoc_eval.sh                # eval all DONE arms
#   ARMS="A0_baseline_s1 A9_full_s1" bash ...            # specific runs
#   EVAL_ENVS=256 EVAL_EPS=64 bash ...                   # tune
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

EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-64}"
LOG_ROOT="logs/AblationMatrix"

# Held-out conditions (same as in-wave eval). Soil tile-bump matches §0-A.
CONDS=(
  "furrows_s3:terrain_furrows_stage3_full:YES_domain_rand:"
  "soil_chal:terrain_soil_challenging_wet:DR_soil_challenging:++terrain.num_rows=6 ++terrain.num_cols=6"
)

# Discover (arm,seed) by completed-train criterion (model_3000.pt exists).
if [ -n "${ARMS:-}" ]; then
  RUNS=(${ARMS})
else
  RUNS=()
  for r in A0_baseline A3_hist50 A5_roa A8_gait A9_full; do
    for s in 1 2 3; do
      run="${r}_s${s}"
      ckpt="$(ls -t "$LOG_ROOT"/*"$run"-locomotion-hunter/model_3000.pt 2>/dev/null | head -1)"
      [ -n "$ckpt" ] && RUNS+=("$run")
    done
  done
fi

if [ ${#RUNS[@]} -eq 0 ]; then
  echo "[posthoc] no completed runs (no model_3000.pt found)"; exit 0
fi
echo "[posthoc] runs to eval: ${RUNS[*]}"

summary_csv="$LOG_ROOT/posthoc_summary.csv"
if [ ! -f "$summary_csv" ]; then
  echo "arm,seed,condition,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary_csv"
fi

extract() {  # extract <field> <file>
  grep -E "$1" "$2" | tail -1 | sed -E 's/.*: //;s/[^0-9.]//g'
}

for run in "${RUNS[@]}"; do
  arm="${run%_s*}"; seed="${run##*_s}"
  rd="$LOG_ROOT/$run"
  mkdir -p "$rd"
  ckpt="$(ls -t "$LOG_ROOT"/*"$run"-locomotion-hunter/model_3000.pt 2>/dev/null | head -1)"
  if [ -z "$ckpt" ]; then echo "[posthoc] skip $run (no model_3000.pt)"; continue; fi
  echo "[posthoc] === $run  ckpt=$ckpt ==="

  for ev in "${CONDS[@]}"; do
    IFS=":" read -r en et ed ex <<< "$ev"
    out="$rd/eval_${en}_v2.log"
    # Idempotent skip: already has a valid summary
    if grep -q "Episodes completed:" "$out" 2>/dev/null && ! grep -q "Error" "$out" 2>/dev/null; then
      echo "[posthoc] $run @ $en already done -> $out"
    else
      "$PY" humanoidverse/sample_eps.py \
        +checkpoint="$ckpt" +terrain="$et" +domain_rand="$ed" \
        +eval_command=[0.3,0.0,0.0] \
        num_envs="$EVAL_ENVS" \
        ++simulator.config.scene.num_envs="$EVAL_ENVS" \
        +num_episodes="$EVAL_EPS" \
        headless=True $ex 2>&1 | tee "$out"
    fi
    eps=$(extract "Episodes completed:" "$out" 2>/dev/null)
    falls=$(extract "Total falls:" "$out" 2>/dev/null)
    eplen=$(extract "Average episode length:" "$out" 2>/dev/null)
    dist=$(extract "Average distance per episode:" "$out" 2>/dev/null)
    if [ -n "$eps" ] && [ -n "$falls" ]; then
      fr=$(python3 -c "print(f'{float($falls)/float($eps):.3f}')" 2>/dev/null || echo NA)
      echo "$arm,$seed,$en,3000,$eps,$falls,$fr,$eplen,$dist" >> "$summary_csv"
      echo "[posthoc]  -> falls=$falls/$eps (rate $fr)  ep_len=$eplen  dist=$dist"
    else
      echo "[posthoc]  -> $run @ $en eval FAILED (no summary in log)"
    fi
  done
done

echo "[posthoc] DONE.  Summary -> $summary_csv"
column -t -s, "$summary_csv" 2>/dev/null || cat "$summary_csv"
