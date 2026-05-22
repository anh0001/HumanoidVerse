#!/usr/bin/env bash
# Decisive post-spawn-fix test: can a policy LEARN furrows once the spawn bug
# is fixed?  Warm-load the existing (valid) softrung Stage-P plane checkpoint
# and fine-tune on fixed furrows_stage1_easy + DR_mild. Eval each ckpt on the
# same condition. Key signal: does ep_len climb well above the ~46 zero-action
# floor during training?
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

PLANE_CKPT="${PLANE_CKPT:-$(ls -t logs/SoftRung/2026*softrung_P-locomotion-hunter/model_2000.pt 2>/dev/null | head -1)}"
[ -z "$PLANE_CKPT" ] && { echo "[fixedF] no plane ckpt"; exit 1; }
ITERS="${ITERS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-250}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-32}"
F_TERRAIN="terrain_furrows_stage1_easy"
LOG_ROOT="logs/FixedStageFv2"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/summary.csv"
[ ! -f "$summary" ] && echo "phase,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary"
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "[fixedF] warm-start ckpt = $PLANE_CKPT"

# Pre-PPO transfer eval (sanity: ~46 expected pre-training, post-spawn-fix).
echo "[fixedF] --- PRE-PPO eval on $F_TERRAIN + DR_mild ---"
pre_out="$LOG_ROOT/eval_pre.log"
"$PY" humanoidverse/sample_eps.py +checkpoint="$PLANE_CKPT" \
  +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
  num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
  ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True ++env.config.termination.terminate_by_contact=True \
  2>&1 | tee "$pre_out"
e=$(extract "Episodes completed:" "$pre_out"); fl=$(extract "Total falls:" "$pre_out")
el=$(extract "Average episode length:" "$pre_out"); ds=$(extract "Average distance" "$pre_out")
[ -n "$e" ] && echo "pre,0,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"

echo "[fixedF] --- TRAIN furrows_s1 + DR_mild, $ITERS iters (FIXED spawn code) ---"
"$PY" humanoidverse/train_agent.py \
  +exp=locomotion algo=ppo_roa \
  +domain_rand=DR_mild \
  +rewards=loco/reward_hunter_locomotion \
  +robot=hunter/hunter +simulator=isaacsim \
  +terrain="$F_TERRAIN" \
  +obs=loco/leggedloco_obs_history_wolinvel \
  num_envs="$NUM_ENVS" seed=1 headless=True \
  ++env.config.env_spacing=2.5 \
  ++algo.config.num_learning_iterations="$ITERS" \
  ++algo.config.save_interval="$SAVE_EVERY" \
  ++algo.config.load_optimizer=False \
  ++checkpoint="$PLANE_CKPT" \
  ++env.config.termination.terminate_by_contact=True \
    project_name=FixedStageFv2 experiment_name=fixedFv2 \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[fixedF] --- POST-PPO eval at each ckpt ---"
for ck in "$LOG_ROOT"/*fixedFv2-locomotion-hunter/model_*.pt; do
  [ -f "$ck" ] || continue
  cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  [ "$cit" = "0" ] && continue
  out="$LOG_ROOT/eval_post_iter${cit}.log"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ck" \
    +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out"
  e=$(extract "Episodes completed:" "$out"); fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out"); ds=$(extract "Average distance" "$out")
  [ -n "$e" ] && echo "post,$cit,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"
done

echo "[fixedF] ALL DONE"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
