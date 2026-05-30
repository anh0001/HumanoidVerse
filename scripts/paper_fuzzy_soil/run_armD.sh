#!/usr/bin/env bash
# Stage-0 compliant-contact probe runner (1 seed).
#   1. Train Arm D (Arm B recipe + global compliant contact) chain.
#   2. Eval Arm D on terrain_furrows_with_maize + DR_paper_S3 WITH compliance on
#      (the slip mechanic only manifests if the eval world is also compliant).
#   3. Append aggregate row + dump per-episode, then compare slip vs Arm B (rigid).
#
# Usage:
#   nohup scripts/paper_fuzzy_soil/run_armD.sh > logs/FuzzySoilFurrows/queue_armD.log 2>&1 &
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
KN="${KN:-50000}"
CN="${CN:-500}"
mkdir -p "$ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [armD-queue] $*"; }
latest_ckpt_in_glob() {
  local pattern="$1"
  # shellcheck disable=SC2086
  ls -1 $pattern 2>/dev/null \
    | awk -F'/' '{n=split($NF,a,"_"); split(a[n],b,"."); print b[1]" "$0}' \
    | sort -n | tail -1 | awk '{print $2}'
}
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

# ---- 1. Train ----
if ls "$ROOT"/*-armD_S3_seed${SEED}-locomotion-*/model_*.pt >/dev/null 2>&1; then
  log "Arm D training SKIP (S3 checkpoint exists)"
else
  log "Arm D training START (Arm B recipe + compliant contact kn=$KN cn=$CN)"
  SEED="$SEED" KN="$KN" CN="$CN" "$SCRIPT_DIR/train_armD_compliant_chain.sh"
  rc=$?
  [ $rc -eq 0 ] || { log "Arm D training FAIL exit=$rc — aborting"; exit $rc; }
  log "Arm D training DONE"
fi

# ---- 2. Eval Arm D (compliance ON in eval too) ----
d_ckpt="$(latest_ckpt_in_glob "$ROOT/*-armD_S3_seed${SEED}-locomotion-*/model_*.pt")"
[ -n "$d_ckpt" ] && [ -f "$d_ckpt" ] || { log "no Arm D ckpt — aborting"; exit 2; }
out="$ROOT/eval_D_seed${SEED}.log"
log "eval arm=D seed=$SEED ckpt=$d_ckpt (compliant eval kn=$KN cn=$CN)"
"$PY" humanoidverse/sample_eps.py +checkpoint="$d_ckpt" \
  +terrain="$EVAL_TERRAIN" +domain_rand="$EVAL_DR" \
  +eval_command="$EVAL_CMD" \
  ++terrain.compliant_contact_stiffness="$KN" ++terrain.compliant_contact_damping="$CN" \
  num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
  ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
  ++env.config.termination.terminate_by_contact=True \
  2>&1 | tee "$out"

e=$(extract "Episodes completed:" "$out")
fl=$(extract "Total falls:" "$out")
el=$(extract "Average episode length:" "$out")
ds=$(extract "Average distance" "$out")
sp=$(extract "Slip distance per 100m:" "$out")
fp=""
if [ -n "$fl" ] && [ -n "$ds" ] && [ -n "$e" ]; then
  fp=$(python3 -c "
fl, ds, e = float('$fl'), float('$ds'), float('$e')
td = ds*e; print(f'{(fl/(td/100.0)):.3f}') if td>0 else print('')")
fi
[ -n "$e" ] && echo "D,$SEED,$d_ckpt,$e,$fl,$el,$ds,$sp,$fp" >> "$RESULTS"
pe_src=$(ls -t logs_eval/TEST/*/per_episode.csv 2>/dev/null | head -1)
[ -n "$pe_src" ] && cp "$pe_src" "$ROOT/per_episode_D_seed${SEED}.csv" && log "per-episode -> $ROOT/per_episode_D_seed${SEED}.csv"

# ---- 3. Compare vs Arm B (rigid) ----
log "comparison vs Arm B (rigid baseline)"
{
  echo "== Stage-0 compliant-contact probe — Arm D vs Arm B (rigid) — seed $SEED =="
  echo "  Arm D: terrain compliant (kn=$KN N/m, cn=$CN N·s/m), Arm B recipe (soft gate + curriculum)"
  echo "  Arm B reference: rigid contact, same recipe."
  echo
  echo "  Arm D this run:  ep_len=$el  slip/100m=$sp  dist=$ds  falls/100m=$fp"
  echo
  echo "  Arm B rigid (eval-noise sweep, n=5):  ep_len 409.1 ± 33.8   slip/100m 284.1 ± 11.7"
  echo "  Arm B rigid (original single eval):   ep_len 368.3          slip/100m 288.2"
  echo
  python3 -c "
sp_d = float('$sp') if '$sp' else None
B_MU, B_SD = 284.1, 11.7   # Arm B rigid slip/100m, 5-rep mean±sd
if sp_d is not None:
    z = (sp_d - B_MU) / B_SD
    print(f'  slip/100m:  D={sp_d:.1f}  vs  B={B_MU:.1f}±{B_SD:.1f}   (D is {z:+.1f} sd from B)')
    if abs(z) < 2:
        print('  => Compliance did NOT move slip beyond eval noise. Hypothesis WEAKENED;')
        print('     per-env stiffness randomization (Stage 1) unlikely to rescue the slip claim.')
    elif z < -2:
        print('  => Compliance LOWERED slip materially. Hypothesis SUPPORTED;')
        print('     Stage 1 (per-env kn/cn randomization) is now worth costing.')
    else:
        print('  => Compliance raised slip. Unexpected; inspect gait before concluding.')
"
} | tee "$ROOT/armD_compliance_seed${SEED}.txt"

log "ARM D QUEUE COMPLETE — see $ROOT/armD_compliance_seed${SEED}.txt"
