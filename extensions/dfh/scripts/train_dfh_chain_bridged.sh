#!/usr/bin/env bash
# Chains the bridged curriculum: S1 extend → S1.5 soil → S1.7 geom → S2 retry.
# After each stage, the next script picks up the latest matching checkpoint.
# Logs go to logs/DFH_Hunter_ROA/DFH_bridged_chain_seed${SEED}.log.

set -euo pipefail
cd "$(dirname "$0")/../../.."

: "${SEED:=1}"

LOG="logs/DFH_Hunter_ROA/DFH_bridged_chain_seed${SEED}.log"
mkdir -p logs/DFH_Hunter_ROA

echo "==========================================================" | tee -a "$LOG"
echo "  Bridged DFH chain (seed=${SEED}) starting $(date)" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"

run_stage() {
  local script="$1"
  echo ""                                                  | tee -a "$LOG"
  echo "----- launching ${script} at $(date) -----"         | tee -a "$LOG"
  SEED="${SEED}" bash "extensions/dfh/scripts/${script}"   2>&1 | tee -a "$LOG"
}

run_stage train_dfh_s1_extend.sh
run_stage train_dfh_s15_from_s1.sh
run_stage train_dfh_s17_from_s15.sh
run_stage train_dfh_s2_from_s17.sh

echo "" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
echo "  Bridged DFH chain done $(date)" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
