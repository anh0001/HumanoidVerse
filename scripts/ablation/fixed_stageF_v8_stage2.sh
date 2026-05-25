#!/usr/bin/env bash
# FixedStageF v8: STAGE 2 PROMOTION (Codex prescription, thread 019e54fe).
#
# v7 m4250 hit ep_len 1029.3 in 96 eps (~20.6s walking on furrows_s1), +147%
# over v3 baseline. Codex: do NOT push v7 further on stage1; that risks
# overfitting easy geometry. Instead promote to furrows_stage2_medium with
# the same conservative PPO schedule that made v7 work.
#
# Stage2 terrain: depth_range deeper than stage1 (configurable but typically
# 0.12-0.16 vs 0.08-0.12), same spacing/orientation generation.
#
# Stop criterion (eval every 100 saved iters, 96 eps each):
#   - ep_len >= 700 by iter 5000  -> stage2 solved, promote to stage3
#   - 400-700 plateau              -> good progress, run more iters
#   - <400                          -> stage2 too hard, drop back to s1 first
#   - eval regression >15% from running best -> early stop (per v7 rule)
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

WARM_CKPT="${WARM_CKPT:-logs/FixedStageFv7/20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt}"
[ -f "$WARM_CKPT" ] || { echo "[v8s2] warm ckpt missing: $WARM_CKPT"; exit 1; }

ITERS="${ITERS:-750}"
SAVE_EVERY="${SAVE_EVERY:-100}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-96}"  # unbiased eval (32-eps had ~40% selection bias)
S2_TERRAIN="terrain_furrows_stage2_medium"
LOG_ROOT="logs/FixedStageFv8_s2"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/summary.csv"
[ ! -f "$summary" ] && echo "phase,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary"
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "[v8s2] warm-start ckpt = $WARM_CKPT"
echo "[v8s2] promoting to $S2_TERRAIN with conservative PPO (v7 recipe)"

# Pre-eval on stage2: zero-shot transfer of v7 m4250 onto stage2 terrain.
echo "[v8s2] --- PRE-PPO eval (zero-shot transfer on stage2) ---"
pre_out="$LOG_ROOT/eval_pre.log"
"$PY" humanoidverse/sample_eps.py +checkpoint="$WARM_CKPT" \
  +terrain="$S2_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
  num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
  ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
  ++env.config.termination.terminate_by_contact=True \
  2>&1 | tee "$pre_out"
e=$(extract "Episodes completed:" "$pre_out"); fl=$(extract "Total falls:" "$pre_out")
el=$(extract "Average episode length:" "$pre_out"); ds=$(extract "Average distance" "$pre_out")
[ -n "$e" ] && echo "pre,4250,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"

echo "[v8s2] --- TRAIN on $S2_TERRAIN, conservative PPO, $ITERS iters ---"
"$PY" humanoidverse/train_agent.py \
  +exp=locomotion algo=ppo_roa \
  +domain_rand=DR_mild \
  +rewards=loco/reward_hunter_locomotion \
  +robot=hunter/hunter +simulator=isaacsim \
  +terrain="$S2_TERRAIN" \
  +obs=loco/leggedloco_obs_history_wolinvel \
  num_envs="$NUM_ENVS" seed=1 headless=True \
  ++env.config.env_spacing=2.5 \
  ++algo.config.num_learning_iterations="$ITERS" \
  ++algo.config.save_interval="$SAVE_EVERY" \
  ++algo.config.load_optimizer=False \
  ++algo.config.actor_learning_rate=2.5e-4 \
  ++algo.config.critic_learning_rate=2.5e-4 \
  ++algo.config.desired_kl=0.005 \
  ++algo.config.entropy_coef=0.001 \
  ++algo.config.clip_param=0.10 \
  ++checkpoint="$WARM_CKPT" \
  ++env.config.termination.terminate_by_contact=True \
  ++rewards.reward_scales.tracking_lin_vel=4.0 \
  '++env.config.locomotion_command_ranges.lin_vel_x=[0.25,0.45]' \
  '++env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0]' \
  '++env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]' \
  ++env.config.reset_randomization.enable=True \
  '++env.config.reset_randomization.dof_pos_scale_range=[1.0,1.0]' \
  '++env.config.reset_randomization.root_lin_vel_range=[0.0,0.0]' \
  '++env.config.reset_randomization.root_ang_vel_range=[0.0,0.0]' \
  project_name=FixedStageFv8_s2 experiment_name=v8s2 \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[v8s2] --- POST-PPO eval at each ckpt (96 eps each) ---"
best_eplen=0
best_ck=""
for ck in "$LOG_ROOT"/*v8s2-locomotion-hunter/model_*.pt; do
  [ -f "$ck" ] || continue
  cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  out="$LOG_ROOT/eval_post_iter${cit}.log"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ck" \
    +terrain="$S2_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out"
  e=$(extract "Episodes completed:" "$out"); fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out"); ds=$(extract "Average distance" "$out")
  if [ -n "$e" ]; then
    echo "post,$cit,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"
    el_int=$(python3 -c "print(int(float('$el')))")
    if [ "$el_int" -gt "$best_eplen" ]; then
      best_eplen=$el_int
      best_ck="$ck"
      echo "[v8s2] new best: $(basename $ck) ep_len=$el"
    fi
  fi
done

echo "[v8s2] ALL DONE"
echo "[v8s2] best stage2 ckpt: $best_ck  ep_len=$best_eplen"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
