#!/usr/bin/env bash
# Set B — run all 3 seeds of ONE arm sequentially (one GPU).
# Calibration constants (CRISP_T, FUZZY_BREAKS, EP_LEN_CAP) are passed through to
# run_arm.sh; set them identically for the crisp & fuzzy arms (see calib note in
# docs/experiments/fuzzy_setB_plan.md). The fixed arm ignores them.
#
# Usage:
#   ARM=fixed scripts/setB_fuzzy_curriculum/run_all_seeds.sh
#   ARM=crisp CRISP_T=0.30 scripts/setB_fuzzy_curriculum/run_all_seeds.sh
#   ARM=fuzzy FUZZY_BREAKS=0.15,0.30,0.45 scripts/setB_fuzzy_curriculum/run_all_seeds.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARM="${ARM:?set ARM=fixed|crisp|fuzzy}"
SEEDS="${SEEDS:-1 2 3}"
for s in $SEEDS; do
  echo "######## [campaign] ARM=$ARM SEED=$s start $(date '+%F %T') ########"
  ARM="$ARM" SEED="$s" "$SCRIPT_DIR/run_arm.sh"
  rc=$?
  echo "######## [campaign] ARM=$ARM SEED=$s end rc=$rc $(date '+%F %T') ########"
  [ "$rc" -ne 0 ] && { echo "[campaign] ABORT: $ARM seed $s failed rc=$rc"; exit "$rc"; }
done
echo "[campaign] ARM=$ARM all seeds done."
