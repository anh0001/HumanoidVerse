#!/usr/bin/env bash
# Resume the DFH chain at Stage 2 (S1 already finished).
set -uo pipefail
cd "$(dirname "$0")/../../.."
: "${SEED:=1}"
SUP_LOG="logs/DFH_Hunter_ROA/DFH_chain_seed${SEED}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUP_LOG}"; }
run_stage() {
  local name="$1" script="$2" log_file="$3"
  log "==== START ${name} ===="
  SEED="${SEED}" bash "${script}" > "${log_file}" 2>&1
  local ec=$?
  if [[ ${ec} -ne 0 ]]; then
    log "FATAL: ${name} exited ${ec}. Tail:"
    tail -30 "${log_file}" | tee -a "${SUP_LOG}"
    exit ${ec}
  fi
  log "==== DONE  ${name} ===="
}
log "Resuming chain: S2 → S3 (seed=${SEED})"
run_stage "Stage 2" extensions/dfh/scripts/train_dfh_s2_from_s1.sh "logs/DFH_Hunter_ROA/DFH_S2_seed${SEED}.log"
run_stage "Stage 3" extensions/dfh/scripts/train_dfh_s3_from_s2.sh "logs/DFH_Hunter_ROA/DFH_S3_seed${SEED}.log"
log "DFH chain (S2+S3) complete. Final ckpt: $(ls -t logs/DFH_Hunter_ROA/*-DFH_S3_from_S2_seed${SEED}-*/model_*.pt | head -1)"
