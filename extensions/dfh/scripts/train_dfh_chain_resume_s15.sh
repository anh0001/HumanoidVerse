#!/usr/bin/env bash
# Resume the bridged DFH chain starting at S1.5 (S1-extend already done).
# Launches S1.5 → S1.7 → S2-retry. Each stage auto-discovers the prior
# checkpoint via experiment-name glob.

set -euo pipefail
cd "$(dirname "$0")/../../.."

: "${SEED:=1}"

LOG="logs/DFH_Hunter_ROA/DFH_bridged_chain_resume_s15_seed${SEED}.log"
mkdir -p logs/DFH_Hunter_ROA

echo "==========================================================" | tee -a "$LOG"
echo "  Bridged DFH chain RESUME at S1.5 (seed=${SEED}) $(date)"   | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"

run_stage() {
  local script="$1"
  echo ""                                                   | tee -a "$LOG"
  echo "----- launching ${script} at $(date) -----"          | tee -a "$LOG"
  SEED="${SEED}" bash "extensions/dfh/scripts/${script}"   2>&1 | tee -a "$LOG"
}

run_stage train_dfh_s15_from_s1.sh
run_stage train_dfh_s17_from_s15.sh
run_stage train_dfh_s2_from_s17.sh

echo "" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
echo "  Bridged DFH chain (resume) done $(date)" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
