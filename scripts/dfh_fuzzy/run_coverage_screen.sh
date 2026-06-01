#!/usr/bin/env bash
# Codex coverage screen: which (if any) frozen policy WALKS across the support-varying
# K range? 4 checkpoints x mu{0.50,0.56,0.62} x K{1,2,4,8,16} x seed{1} = 60 evals.
# Walking coverage qualifies a policy for the descriptor test; survival coverage is
# context only (balancers cannot rescue the locomotion descriptor test).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO"
PY="${PYTHON:-/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python}"
M=logs/DFH_hf_models
declare -A POL=(
  [mildsoil]="$M/walkers/mildsoil_walker/model_5050.pt"
  [rigid]="$M/walkers/rigid_walker/model_5050.pt"
  [chain11]="$M/dfh_chain/11_s3_bridge1_floor09/model_15500.pt"
  [chain12]="$M/dfh_chain/12_s3_bridge1_ext500/model_15250.pt"
)
for name in mildsoil rigid chain11 chain12; do
  ckpt="${POL[$name]}"
  echo "######## SCREEN policy=$name ($ckpt) $(date '+%F %T') ########"
  "$PY" "$SCRIPT_DIR/run_dfh_fuzzy_sweep.py" \
    --policy "$ckpt" --mus 0.50,0.56,0.62 --Ks 1.0,2.0,4.0,8.0,16.0 --seeds 1 \
    --force-coupling True --num-eps 50 --num-envs 64 \
    --out-csv "logs/DFH_fuzzy/screen_${name}.csv" --tag "screen_${name}"
done
echo "######## SCREEN DONE $(date '+%F %T') ########"
