#!/usr/bin/env bash
# Run DFH Stage 1 → Stage 2 → Stage 3 sequentially, auto-promoting on the
# latest checkpoint produced by each stage.
#
# Single GPU, sequential. Total wall time: ~7-10 h on RTX 6000 Ada Generation.
#
# Logs:
#   logs/DFH_Hunter_ROA/DFH_chain_seedN.log   — supervisor stdout
#   logs/DFH_Hunter_ROA/DFH_S{1,2,3}_seedN.log — per-stage stdout

set -uo pipefail
cd "$(dirname "$0")/../../.."

: "${SEED:=1}"
SUP_LOG="logs/DFH_Hunter_ROA/DFH_chain_seed${SEED}.log"
mkdir -p logs/DFH_Hunter_ROA

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUP_LOG}"; }

run_stage() {
  local name="$1" script="$2" log_file="$3"
  log "==== START ${name} ===="
  log "  script: ${script}"
  log "  log   : ${log_file}"
  SEED="${SEED}" bash "${script}" > "${log_file}" 2>&1
  local ec=$?
  if [[ ${ec} -ne 0 ]]; then
    log "FATAL: ${name} exited ${ec}. Aborting chain. Tail:"
    tail -30 "${log_file}" | tee -a "${SUP_LOG}"
    exit ${ec}
  fi
  log "==== DONE  ${name} ===="
}

log "DFH training chain start (seed=${SEED})"
log "Stages: S1 (1500 iters) → S2 (2200 iters) → S3 (3500 iters)"

run_stage "Stage 1" \
  "extensions/dfh/scripts/train_dfh_s1_from_v7.sh" \
  "logs/DFH_Hunter_ROA/DFH_S1_seed${SEED}.log"

run_stage "Stage 2" \
  "extensions/dfh/scripts/train_dfh_s2_from_s1.sh" \
  "logs/DFH_Hunter_ROA/DFH_S2_seed${SEED}.log"

run_stage "Stage 3" \
  "extensions/dfh/scripts/train_dfh_s3_from_s2.sh" \
  "logs/DFH_Hunter_ROA/DFH_S3_seed${SEED}.log"

log "DFH training chain complete."
log "Final checkpoint: $(ls -t logs/DFH_Hunter_ROA/DFH_S3_from_S2_seed${SEED}/*/model_*.pt | head -1)"
