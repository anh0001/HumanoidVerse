#!/usr/bin/env bash
# Arm C credit-assignment runner (1-seed directional check).
#   1. Train the Arm C hard-gate FULL chain (S2a->S2b->S3) for SEED (default 1).
#   2. Eval Arm C seed on terrain_furrows_with_maize + DR_paper_S3 (100 eps).
#   3. Re-eval Arm B same seed (its original eval predates the per_episode.csv dump,
#      and we need per-episode {distance, slip_distance} for the matched comparison).
#   4. Append aggregate rows to results.csv and run the slip-vs-distance analysis.
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/run_armC.sh > logs/FuzzySoilFurrows/queue_armC.log 2>&1 &
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

SEED="${SEED:-1}"
ROOT="logs/FuzzySoilFurrows"
RESULTS="$ROOT/results.csv"
EVAL_TERRAIN="${EVAL_TERRAIN:-terrain_furrows_with_maize}"
EVAL_DR="${EVAL_DR:-DR_paper_S3}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-100}"
EVAL_CMD="${EVAL_CMD:-[0.3,0.0,0.0]}"
mkdir -p "$ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [armC-queue] $*"; }

latest_ckpt_in_glob() {
  local pattern="$1"
  # shellcheck disable=SC2086
  ls -1 $pattern 2>/dev/null \
    | awk -F'/' '{n=split($NF,a,"_"); split(a[n],b,"."); print b[1]" "$0}' \
    | sort -n | tail -1 | awk '{print $2}'
}
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

eval_one() {
  # $1=arm label, $2=ckpt path, $3=per-episode copy destination
  local arm="$1" ckpt="$2" pe_dest="$3"
  local out="$ROOT/eval_${arm}_seed${SEED}.log"
  log "eval arm=$arm seed=$SEED ckpt=$ckpt"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ckpt" \
    +terrain="$EVAL_TERRAIN" +domain_rand="$EVAL_DR" \
    +eval_command="$EVAL_CMD" \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out"

  local e fl el ds sp fp
  e=$(extract "Episodes completed:" "$out")
  fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out")
  ds=$(extract "Average distance" "$out")
  sp=$(extract "Slip distance per 100m:" "$out")
  fp=""
  if [ -n "$fl" ] && [ -n "$ds" ] && [ -n "$e" ]; then
    fp=$(python3 -c "
fl, ds, e = float('$fl'), float('$ds'), float('$e')
td = ds * e
print(f'{(fl/(td/100.0)):.3f}') if td > 0 else print('')
")
  fi
  [ -n "$e" ] && echo "$arm,$SEED,$ckpt,$e,$fl,$el,$ds,$sp,$fp" >> "$RESULTS"

  # Grab the per_episode.csv that sample_eps.py just wrote (newest eval output dir).
  local pe_src
  pe_src=$(ls -t logs_eval/TEST/*/per_episode.csv 2>/dev/null | head -1)
  if [ -n "$pe_src" ] && [ -f "$pe_src" ]; then
    cp "$pe_src" "$pe_dest"
    log "per-episode -> $pe_dest"
  else
    log "WARNING: no per_episode.csv found for arm=$arm"
  fi
}

# ---- 1. Train Arm C chain (skip if final S3 ckpt already exists) ----
if ls "$ROOT"/*-armC_S3_seed${SEED}-locomotion-*/model_*.pt >/dev/null 2>&1; then
  log "Arm C training SKIP (S3 checkpoint already exists)"
else
  log "Arm C training START (FULL chain, hard gate)"
  SEED="$SEED" "$SCRIPT_DIR/train_armC_hardgate_chain.sh"
  rc=$?
  [ $rc -eq 0 ] || { log "Arm C training FAIL exit=$rc — aborting"; exit $rc; }
  log "Arm C training DONE"
fi

# ---- 2. Eval Arm C ----
c_ckpt="$(latest_ckpt_in_glob "$ROOT/*-armC_S3_seed${SEED}-locomotion-*/model_*.pt")"
[ -n "$c_ckpt" ] && [ -f "$c_ckpt" ] || { log "no Arm C ckpt — aborting"; exit 2; }
eval_one "C" "$c_ckpt" "$ROOT/per_episode_C_seed${SEED}.csv"

# ---- 3. Re-eval Arm B same seed (for fresh per-episode data) ----
b_ckpt="$(latest_ckpt_in_glob "$ROOT/*-armB_S3_seed${SEED}-locomotion-*/model_*.pt")"
if [ -n "$b_ckpt" ] && [ -f "$b_ckpt" ]; then
  eval_one "B_recheck" "$b_ckpt" "$ROOT/per_episode_B_seed${SEED}.csv"
else
  log "WARNING: no Arm B ckpt for seed $SEED — matched comparison will be one-sided"
fi

# ---- 4. Analysis ----
log "running slip-vs-distance analysis"
python3 "$SCRIPT_DIR/analyze_armC.py" \
  --b "$ROOT/per_episode_B_seed${SEED}.csv" \
  --c "$ROOT/per_episode_C_seed${SEED}.csv" \
  --seed "$SEED" 2>&1 | tee "$ROOT/armC_analysis_seed${SEED}.txt"

log "ARM C QUEUE COMPLETE — see $ROOT/armC_analysis_seed${SEED}.txt and $RESULTS"
