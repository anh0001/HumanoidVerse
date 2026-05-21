#!/usr/bin/env bash
# Cliff Decomposition Diagnostic (Codex thread 019e4a98, option B)
#
# A9 (ROA+gait) failed Stage F on furrows_s2 + YES_DR. Was the killer
# geometry, dynamics randomization, or their interaction?  Take the SAME
# verified-walking A9 plane checkpoint, fine-tune separately on each
# factor in isolation, and on both together, with pre-PPO eval first so we
# can tell "policy dies on warm-load" from "PPO destroys it during fine-tune".
#
# Conditions (all warm-load from logs/WarmstartPilot/2026...A9_full_P/model_2000.pt):
#   B1_geom_only : furrows_s2_medium  + NO_domain_rand   (geometry only)
#   B2_dr_only   : plane              + YES_domain_rand  (dynamics only)
#   B3_both      : furrows_s2_medium  + YES_domain_rand  (the cliff)
#
# For each condition: pre-PPO eval (no training), 1000 PPO iters, then
# post-train evals at saved ckpts. Output table to refine-logs/CLIFF_DIAG.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"
PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"

PLANE_CKPT="$(ls -t logs/WarmstartPilot/2026*A9_full_P-locomotion-hunter/model_2000.pt 2>/dev/null | head -1)"
if [ -z "$PLANE_CKPT" ]; then echo "[cliff] no A9 plane ckpt"; exit 1; fi
echo "[cliff] warm-start ckpt = $PLANE_CKPT"

ITERS="${ITERS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-32}"
LOG_ROOT="logs/CliffDiag"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/cliff_summary.csv"
[ ! -f "$summary" ] && \
  echo "phase,condition,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,tracking_lin_vel,distance_m" > "$summary"

# condition : terrain : domain_rand : env_spacing
CONDS=(
  "B1_geom_only:terrain_furrows_stage2_medium:NO_domain_rand:2.5"
  "B2_dr_only:terrain_locomotion_plane:YES_domain_rand:2.0"
  "B3_both:terrain_furrows_stage2_medium:YES_domain_rand:2.5"
)

extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

eval_ckpt() {  # eval_ckpt <ckpt> <phase_label> <condname> <terrain> <dr> <spacing>
  local ckpt="$1" phase="$2" cn="$3" et="$4" ed="$5" sp="$6"
  local cit
  cit="$(basename "$ckpt" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  local out="$LOG_ROOT/$cn/eval_${phase}_iter${cit}.log"
  mkdir -p "$LOG_ROOT/$cn"
  if grep -q "Episodes completed:" "$out" 2>/dev/null && ! grep -q "Error in call" "$out" 2>/dev/null; then
    echo "[cliff]   $phase $cn iter $cit already done"
  else
    "$PY" humanoidverse/sample_eps.py \
      +checkpoint="$ckpt" +terrain="$et" +domain_rand="$ed" \
      +eval_command=[0.3,0.0,0.0] \
      num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
      ++env.config.env_spacing="$sp" \
      +num_episodes="$EVAL_EPS" headless=True 2>&1 | tee "$out"
  fi
  local eps falls eplen dist tlv fr
  eps=$(extract "Episodes completed:" "$out")
  falls=$(extract "Total falls:" "$out")
  eplen=$(extract "Average episode length:" "$out")
  dist=$(extract "Average distance per episode:" "$out")
  # tracking_lin_vel is logged in train.log but not sample_eps; use a sentinel
  tlv="NA"
  if [ -n "$eps" ] && [ -n "$falls" ]; then
    fr=$(python3 -c "print(f'{float($falls)/float($eps):.3f}')" 2>/dev/null || echo NA)
    echo "$phase,$cn,$cit,$eps,$falls,$fr,$eplen,$tlv,$dist" >> "$summary"
    echo "[cliff]   $phase $cn iter $cit -> falls=$falls/$eps ($fr) ep_len=$eplen"
  fi
}

for entry in "${CONDS[@]}"; do
  IFS=":" read -r cn et ed sp <<< "$entry"
  rd="$LOG_ROOT/$cn"
  mkdir -p "$rd"
  echo "[cliff] ========== $cn  terrain=$et  dr=$ed  spacing=$sp =========="

  echo "[cliff] --- PRE-PPO eval (warm-load only, no training) ---"
  eval_ckpt "$PLANE_CKPT" "pre" "$cn" "$et" "$ed" "$sp"

  echo "[cliff] --- PPO fine-tune $ITERS iters ---"
  "$PY" humanoidverse/train_agent.py \
    +exp=locomotion algo=ppo_roa \
    +domain_rand="$ed" \
    +rewards=loco/reward_hunter_locomotion \
    +robot=hunter/hunter +simulator=isaacsim \
    +terrain="$et" \
    +obs=loco/leggedloco_obs_history_wolinvel \
    num_envs="$NUM_ENVS" seed=1 headless=True \
    ++env.config.env_spacing="$sp" \
    ++algo.config.num_learning_iterations="$ITERS" \
    ++algo.config.save_interval="$SAVE_EVERY" \
    ++algo.config.load_optimizer=False \
    ++checkpoint="$PLANE_CKPT" \
    project_name=CliffDiag experiment_name="$cn" \
    2>&1 | tee "$rd/train.log"

  echo "[cliff] --- POST-PPO eval at each saved ckpt ---"
  for ck in "$LOG_ROOT"/*"$cn"-locomotion-hunter/model_*.pt; do
    [ -f "$ck" ] || continue
    cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
    [ "$cit" = "0" ] && continue
    eval_ckpt "$ck" "post" "$cn" "$et" "$ed" "$sp"
  done
  echo "[cliff] === $cn DONE ==="
done

echo "[cliff] ALL CONDITIONS DONE"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
