#!/usr/bin/env bash
# Softest-rung pilot (Codex 019e4a98 -> option E, "much softer" reset).
#
# The cliff diagnostic showed BOTH geometry and DR independently collapse a
# NO_DR plane policy. Two design fixes tested here:
#   (1) warm up WITH mild DR (DR_mild) so the policy is not DR-naive.
#   (2) first furrows rung = furrows_stage1_easy (shallowest), not s2.
#
# A9 architecture only (ppo_roa + history obs). Stages:
#   Stage P : plane + DR_mild, 2000 iters  (DR-aware walking policy)
#   [KEY]   : PRE-PPO eval of the Stage-P policy on the Stage-F condition.
#             If pre-PPO ep_len stays high (>~150) the rung is small enough
#             and the curriculum approach is viable. If it is still ~30, the
#             rung is too big -> curriculum cannot work, negative result.
#   Stage F : furrows_stage1_easy + DR_mild, 1500 iters
#   post-eval at saved Stage F ckpts.
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

PLANE_ITERS="${PLANE_ITERS:-2000}"
FURROW_ITERS="${FURROW_ITERS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-32}"
OBS="loco/leggedloco_obs_history_wolinvel"
DR="DR_mild"
F_TERRAIN="terrain_furrows_stage1_easy"
LOG_ROOT="logs/SoftRung"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/softrung_summary.csv"
[ ! -f "$summary" ] && \
  echo "phase,ckpt_iter,terrain,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary"

extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

eval_ckpt() {  # eval_ckpt <ckpt> <phase> <terrain> <spacing>
  local ckpt="$1" phase="$2" et="$3" sp="$4"
  local cit out
  cit="$(basename "$ckpt" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  out="$LOG_ROOT/eval_${phase}_iter${cit}.log"
  if grep -q "Episodes completed:" "$out" 2>/dev/null && ! grep -q "Error in call" "$out" 2>/dev/null; then
    echo "[soft] $phase iter $cit already evaluated"
  else
    "$PY" humanoidverse/sample_eps.py \
      +checkpoint="$ckpt" +terrain="$et" +domain_rand="$DR" \
      +eval_command=[0.3,0.0,0.0] \
      num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
      ++env.config.env_spacing="$sp" \
      +num_episodes="$EVAL_EPS" headless=True 2>&1 | tee "$out"
  fi
  local eps falls eplen dist fr
  eps=$(extract "Episodes completed:" "$out")
  falls=$(extract "Total falls:" "$out")
  eplen=$(extract "Average episode length:" "$out")
  dist=$(extract "Average distance per episode:" "$out")
  if [ -n "$eps" ] && [ -n "$falls" ]; then
    fr=$(python3 -c "print(f'{float($falls)/float($eps):.3f}')" 2>/dev/null || echo NA)
    echo "$phase,$cit,$et,$eps,$falls,$fr,$eplen,$dist" >> "$summary"
    echo "[soft] $phase iter $cit ($et) -> falls=$falls/$eps ($fr) ep_len=$eplen"
  fi
}

##### Stage P : plane + DR_mild #####
echo "[soft] ===== STAGE P (plane + $DR, $PLANE_ITERS iters) ====="
"$PY" humanoidverse/train_agent.py \
  +exp=locomotion algo=ppo_roa \
  +domain_rand="$DR" \
  +rewards=loco/reward_hunter_locomotion \
  +robot=hunter/hunter +simulator=isaacsim \
  +terrain=terrain_locomotion_plane \
  +obs="$OBS" \
  num_envs="$NUM_ENVS" seed=1 headless=True \
  ++env.config.env_spacing=2.0 \
  ++algo.config.num_learning_iterations="$PLANE_ITERS" \
  ++algo.config.save_interval="$SAVE_EVERY" \
  project_name=SoftRung experiment_name=softrung_P \
  2>&1 | tee "$LOG_ROOT/stageP.log"

plane_ckpt="$(ls -t "$LOG_ROOT"/*softrung_P-locomotion-hunter/model_*.pt 2>/dev/null | head -1)"
if [ -z "$plane_ckpt" ]; then echo "[soft] !! no plane ckpt"; exit 1; fi
echo "[soft] plane_ckpt=$plane_ckpt"

##### KEY: pre-PPO transfer eval onto the Stage-F condition #####
echo "[soft] ===== PRE-PPO transfer eval: Stage-P policy on $F_TERRAIN + $DR ====="
eval_ckpt "$plane_ckpt" "preF" "$F_TERRAIN" "2.5"

##### Stage F : furrows_stage1_easy + DR_mild #####
echo "[soft] ===== STAGE F ($F_TERRAIN + $DR, $FURROW_ITERS iters) ====="
"$PY" humanoidverse/train_agent.py \
  +exp=locomotion algo=ppo_roa \
  +domain_rand="$DR" \
  +rewards=loco/reward_hunter_locomotion \
  +robot=hunter/hunter +simulator=isaacsim \
  +terrain="$F_TERRAIN" \
  +obs="$OBS" \
  num_envs="$NUM_ENVS" seed=1 headless=True \
  ++env.config.env_spacing=2.5 \
  ++algo.config.num_learning_iterations="$FURROW_ITERS" \
  ++algo.config.save_interval="$SAVE_EVERY" \
  ++algo.config.load_optimizer=False \
  ++checkpoint="$plane_ckpt" \
  project_name=SoftRung experiment_name=softrung_F \
  2>&1 | tee "$LOG_ROOT/stageF.log"

echo "[soft] ===== POST-PPO eval at each Stage F ckpt ====="
for ck in "$LOG_ROOT"/*softrung_F-locomotion-hunter/model_*.pt; do
  [ -f "$ck" ] || continue
  cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  [ "$cit" = "0" ] && continue
  eval_ckpt "$ck" "postF" "$F_TERRAIN" "2.5"
done

echo "[soft] ALL DONE"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
