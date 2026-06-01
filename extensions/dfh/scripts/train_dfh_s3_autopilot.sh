#!/usr/bin/env bash
# DFH S3 AUTOPILOT — overnight unattended run of the remaining dose ladder.
# Codex ladder (thread 019e78f6), all headroom-neutral (k/max ratio ~0.047):
#   [running now] S3-dose-2  : floor=-0.09, k=21, max=440
#   S3-dose-2b              : floor=-0.09, k=25, max=520
#   S3-sink-2               : floor=-0.10, k=25, max=520   (sink-axis step)
#   S3-dose-3               : floor=-0.10, k=32, max=680
#   S3-nominal              : floor=-0.10, k=40, max=850
#
# Behaviour:
#   1. Wait for the currently-running S3-dose-2 (PID-agnostic: waits on the
#      train_agent process for that run name) to finish.
#   2. Gate-check each completed stage via dfh_gate_check.py.
#   3. PASS  -> warm-start the next stage from this stage's highest ckpt.
#      STOP  -> halt the chain, write a STOP marker explaining why, exit.
#   4. Each stage's stdout -> its own log; driver progress -> autopilot log.
#
# Stops at the FIRST gate failure so no compute is wasted on a broken chain.
# Everything left for morning review in the autopilot log + STOP marker.

set -uo pipefail
cd "$(dirname "$0")/../../.."

SEED="${SEED:-1}"
SCRIPTS="extensions/dfh/scripts"
PYTHON="${PYTHON:-/home/anhar/miniconda3/envs/isaaclab/bin/python}"
LOGROOT="logs/DFH_Hunter_ROA"
AP_LOG="${LOGROOT}/DFH_S3_autopilot.log"
STOP_MARKER="${LOGROOT}/DFH_S3_autopilot.STOP"
rm -f "${STOP_MARKER}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${AP_LOG}"; }

# Highest model_*.pt in the newest run dir matching a RUN_NAME.
find_latest_ckpt() {
  local run_name="$1"
  local d
  d=$(ls -dt ${LOGROOT}/*-${run_name}-locomotion-hunter/ 2>/dev/null | head -1 || true)
  [[ -z "$d" ]] && return 1
  ls "${d}"model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}'
}
find_run_dir() {
  ls -dt ${LOGROOT}/*-$1-locomotion-hunter/ 2>/dev/null | head -1
}

# Block until no train_agent.py process for the given run name is alive.
wait_for_run() {
  local run_name="$1"
  log "waiting for run '${run_name}' to finish..."
  while pgrep -fa "train_agent.py" 2>/dev/null | grep -q "experiment_name=${run_name}"; do
    sleep 60
  done
  sleep 10  # let final ckpt flush
  log "run '${run_name}' is no longer active."
}

# Run one stage: gate-check the PRIOR stage, then launch this stage.
# args: PRIOR_RUN  THIS_RUN  FLOOR  K  MAXN
run_stage() {
  local prior_run="$1" this_run="$2" floor="$3" kval="$4" maxn="$5"

  # Gate-check the prior stage.
  local prior_dir; prior_dir=$(find_run_dir "${prior_run}")
  if [[ -z "${prior_dir}" ]]; then
    log "STOP: cannot find prior run dir for '${prior_run}'"
    echo "no prior run dir for ${prior_run}" > "${STOP_MARKER}"
    return 1
  fi
  log "gate-checking prior stage '${prior_run}' (${prior_dir})"
  if ! "${PYTHON}" "${SCRIPTS}/dfh_gate_check.py" "${prior_dir}" >> "${AP_LOG}" 2>&1; then
    log "STOP: gate failed/borderline for '${prior_run}'. Halting chain for review."
    echo "gate failed at ${prior_run}; see ${AP_LOG}" > "${STOP_MARKER}"
    return 1
  fi
  log "gate PASS for '${prior_run}'."

  # Warm-start checkpoint = prior stage highest ckpt.
  local warm; warm=$(find_latest_ckpt "${prior_run}")
  if [[ -z "${warm}" || ! -f "${warm}" ]]; then
    log "STOP: no warm-start ckpt from '${prior_run}'"
    echo "no ckpt from ${prior_run}" > "${STOP_MARKER}"
    return 1
  fi

  log "launching '${this_run}': floor=${floor}, k=${kval}, max=${maxn}, warm=${warm}"
  local stage_log="${LOGROOT}/${this_run}.log"
  FLOOR="${floor}" KVAL="${kval}" MAXN="${maxn}" WARM_CKPT="${warm}" \
    RUN_NAME="${this_run}" ITERS=800 SEED="${SEED}" \
    bash "${SCRIPTS}/train_dfh_stage_generic.sh" > "${stage_log}" 2>&1

  wait_for_run "${this_run}"
  return 0
}

log "=========================================================="
log "  DFH S3 AUTOPILOT START (seed=${SEED})"
log "  ladder: dose-2(k21) -> dose-2b(k25) -> sink-2(-0.10) -> dose-3(k32) -> nominal(k40)"
log "=========================================================="

# 0. Wait for the in-flight S3-dose-2 (k=21) to finish.
wait_for_run "DFH_S3_dose2_k21_max440_floor09_seed${SEED}"

# Chain. Each call gate-checks the PRIOR run, then launches the named stage.
run_stage "DFH_S3_dose2_k21_max440_floor09_seed${SEED}" \
          "DFH_S3_dose2b_k25_max520_floor09_seed${SEED}" "-0.09" "25" "520" || exit 1

run_stage "DFH_S3_dose2b_k25_max520_floor09_seed${SEED}" \
          "DFH_S3_sink2_k25_max520_floor10_seed${SEED}"  "-0.10" "25" "520" || exit 1

run_stage "DFH_S3_sink2_k25_max520_floor10_seed${SEED}" \
          "DFH_S3_dose3_k32_max680_floor10_seed${SEED}"  "-0.10" "32" "680" || exit 1

run_stage "DFH_S3_dose3_k32_max680_floor10_seed${SEED}" \
          "DFH_S3_nominal_k40_max850_floor10_seed${SEED}" "-0.10" "40" "850" || exit 1

# Final gate-check on S3-nominal.
NOM_DIR=$(find_run_dir "DFH_S3_nominal_k40_max850_floor10_seed${SEED}")
log "gate-checking final S3-nominal (${NOM_DIR})"
if "${PYTHON}" "${SCRIPTS}/dfh_gate_check.py" "${NOM_DIR}" >> "${AP_LOG}" 2>&1; then
  log "S3-NOMINAL PASSED. Full DFH ladder complete. floor=-0.10, k=40, max=850."
else
  log "S3-nominal finished but gate did not pass cleanly — review ${AP_LOG}."
fi

log "=========================================================="
log "  DFH S3 AUTOPILOT DONE"
log "=========================================================="
