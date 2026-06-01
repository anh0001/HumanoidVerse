#!/usr/bin/env bash
# Walk-ladder driver (Codex 019e80a8): find the DRAG CEILING for a walker.
# Anchored on the mild-soil walker m5050 (vel_err 0.098, ~0.119 BW drag).
# Each rung warm-starts from the previous rung's ckpt, trains, evals vel_err +
# forward-progress, and gates. One-axis-at-a-time; max_drag follows k.
# Fixed all night: mu_along=0.55, mu_across=0.70, forward cmd [0.25,0.45],
# tracking=4.0, feet_air=1.0, +exp=locomotion, NO_domain_rand, no reward_hunter_dfh.
#
# Gate (walk_ladder_gate.py): PROMOTE -> next rung; BORDERLINE -> extend +400
# once, re-eval, stop if still borderline; KILL -> stop. Previous clean rung = ceiling.
#
# Rungs (k, max_drag, sinkage_floor, iters):
#   R1 10/220/-0.05/600   R2 10/220/-0.06/800   R3 12/260/-0.06/600
#   R4 12/260/-0.07/800   R5 14/300/-0.07/600   R6 14/300/-0.08/800
#   R7 16/340/-0.08/600   R8 16/340/-0.09/800   R9 18/380/-0.09/600 (stress)

set -uo pipefail
cd "$(dirname "$0")/../../.."

SEED="${SEED:-1}"
SCRIPTS="extensions/dfh/scripts"
PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"
LOGROOT="logs/DFH_Hunter_ROA"
AP_LOG="${LOGROOT}/DFH_walk_ladder.log"
STOP_MARKER="${LOGROOT}/DFH_walk_ladder.STOP"
rm -f "${STOP_MARKER}"

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${AP_LOG}"; }

MU_OV="++terrain.dfh.params.anisotropy.mu_along=0.55 ++terrain.dfh.params.anisotropy.mu_across=0.70"

# Anchor walker.
PREV_CKPT="logs/DFH_Hunter_ROA/20260601_080848-DFH_paired_B_soil_seed1-locomotion-hunter/model_5050.pt"

# Rung definitions (parallel arrays).
RK=(  10   10   12   12   14   14   16   16   18 )
RMAX=(220  220  260  260  300  300  340  340  380 )
RFLR=(-0.05 -0.06 -0.06 -0.07 -0.07 -0.08 -0.08 -0.09 -0.09)
RIT=( 600  800  600  800  600  800  600  800  600 )

find_latest_ckpt() {
  local run_name="$1" d
  d=$(ls -dt ${LOGROOT}/*-${run_name}-locomotion-hunter/ 2>/dev/null | head -1)
  [[ -z "$d" ]] && return 1
  ls "${d}"model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}'
}

# Train one rung from PREV_CKPT, then eval. Sets EVAL_LOG + RUN_NAME globals.
train_and_eval() {
  local run_name="$1" k="$2" maxn="$3" flr="$4" iters="$5" warm="$6"
  log "  training ${run_name}: k=${k} max=${maxn} floor=${flr} iters=${iters} warm=$(basename "${warm}")"
  local dfh_ov="++terrain.dfh.force_coupling.sinkage_drag_k=${k} ++terrain.dfh.force_coupling.max_drag_force_n=${maxn} ++terrain.dfh.params.sinkage_floor_m=${flr} ${MU_OV}"
  SEED="${SEED}" ITERS="${iters}" ENVS=2048 \
    TERRAIN="terrain_dfh_stage1_easy" RUN_NAME="${run_name}" \
    DFH_OVERRIDES="${dfh_ov}" WARM_CKPT="${warm}" \
    bash "${SCRIPTS}/walk_paired_control_arm.sh" \
    > "${LOGROOT}/${run_name}_train.log" 2>&1

  local ckpt; ckpt=$(find_latest_ckpt "${run_name}")
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    log "  ERROR: no ckpt from ${run_name}"; return 9
  fi
  RUNG_CKPT="${ckpt}"
  EVAL_LOG="${LOGROOT}/${run_name}_eval.log"
  log "  evaluating ${run_name} (cmd 0.3)..."
  # shellcheck disable=SC1091
  source "$ISAAC_PATH/setup_python_env.sh"
  "${PYTHON}" humanoidverse/sample_eps.py \
    +simulator=isaacsim +checkpoint="${ckpt}" \
    +eval_command="[0.3,0.0,0.0]" num_envs=64 +num_episodes=64 headless=True \
    > "${EVAL_LOG}" 2>&1
  "${PYTHON}" "${SCRIPTS}/walk_ladder_gate.py" "${EVAL_LOG}" | tee -a "${AP_LOG}"
  return "${PIPESTATUS[0]}"
}

log "=========================================================="
log "  WALK-LADDER START (seed=${SEED}) — anchor m5050, v7 baseline vel_err 0.117"
log "  ceiling target ~0.16-0.18 BW drag; kill if vel_err>0.16 or dist<3.5m"
log "=========================================================="

CEILING="R0 (anchor m5050, k=8/floor=-0.05)"
N=${#RK[@]}
for ((i=0; i<N; i++)); do
  rung="R$((i+1))"
  name="DFH_walk_${rung}_k${RK[$i]}_f${RFLR[$i]}_seed${SEED}"
  log ">>> ${rung}: k=${RK[$i]} max=${RMAX[$i]} floor=${RFLR[$i]}"
  train_and_eval "${name}" "${RK[$i]}" "${RMAX[$i]}" "${RFLR[$i]}" "${RIT[$i]}" "${PREV_CKPT}"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    log "  ${rung} PROMOTE. New anchor."
    PREV_CKPT="${RUNG_CKPT}"; CEILING="${rung} (k=${RK[$i]}/floor=${RFLR[$i]})"
  elif [[ $rc -eq 2 ]]; then
    log "  ${rung} BORDERLINE -> extending +400 iters once..."
    train_and_eval "${name}_ext" "${RK[$i]}" "${RMAX[$i]}" "${RFLR[$i]}" 400 "${RUNG_CKPT}"
    rc2=$?
    if [[ $rc2 -eq 0 ]]; then
      log "  ${rung} (extended) PROMOTE. New anchor."
      PREV_CKPT="${RUNG_CKPT}"; CEILING="${rung}+ext (k=${RK[$i]}/floor=${RFLR[$i]})"
    else
      log "  ${rung} still not clean after extend (rc=${rc2}). STOP."
      echo "ceiling at ${CEILING}; ${rung} failed" > "${STOP_MARKER}"; break
    fi
  else
    log "  ${rung} KILL (rc=${rc}). STOP."
    echo "ceiling at ${CEILING}; ${rung} killed" > "${STOP_MARKER}"; break
  fi
done

log "=========================================================="
log "  WALK-LADDER DONE. Walker drag CEILING = ${CEILING}"
log "=========================================================="
