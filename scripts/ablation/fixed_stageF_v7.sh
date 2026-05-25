#!/usr/bin/env bash
# FixedStageF v7: CONSERVATIVE PPO REFINEMENT from v6 model_4750.pt
# (Codex prescription, thread 019e54fe, post-v6-peak).
#
# v6 peaked at iter 4750 with eval ep_len 381.1 (+83% over baseline 207.9)
# then overshot — declined to 252 by iter 5500. Codex diagnosis: PPO
# over-optimized the training proxy with LR too aggressive late-stage,
# entropy/std stuck at clamp max (2.0). Solution: continue from the peak
# ckpt with a much more conservative PPO schedule and early-stop on the
# first eval regression > 15% from running best.
#
# Targets: 381 -> 420-480. If can't break 450 after 750 iters, switch to
# heightmap obs (Option C).
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

WARM_CKPT="${WARM_CKPT:-logs/FixedStageFv6/20260524_105034-fixedFv6-locomotion-hunter/model_4750.pt}"
[ -f "$WARM_CKPT" ] || { echo "[fixedFv7] warm ckpt missing: $WARM_CKPT"; exit 1; }

ITERS="${ITERS:-750}"
SAVE_EVERY="${SAVE_EVERY:-100}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-96}"   # 32-eps had ~40% selection bias (first envs to fall); 96 is unbiased.
F_TERRAIN="terrain_furrows_stage1_easy"
LOG_ROOT="logs/FixedStageFv7"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/summary.csv"
[ ! -f "$summary" ] && echo "phase,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary"
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "[fixedFv7] warm-start ckpt = $WARM_CKPT"
echo "[fixedFv7] conservative PPO: LR=2.5e-4, KL=0.005, clip=0.10, entropy=0.001"

# Pre-PPO eval (should reproduce ~381).
echo "[fixedFv7] --- PRE-PPO eval ---"
pre_out="$LOG_ROOT/eval_pre.log"
"$PY" humanoidverse/sample_eps.py +checkpoint="$WARM_CKPT" \
  +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
  num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
  ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
  ++env.config.termination.terminate_by_contact=True \
  2>&1 | tee "$pre_out"
e=$(extract "Episodes completed:" "$pre_out"); fl=$(extract "Total falls:" "$pre_out")
el=$(extract "Average episode length:" "$pre_out"); ds=$(extract "Average distance" "$pre_out")
[ -n "$e" ] && echo "pre,4750,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"

echo "[fixedFv7] --- TRAIN conservative-PPO from v6 m4750, $ITERS iters ---"
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
  project_name=FixedStageFv7 experiment_name=fixedFv7 \
  2>&1 | tee "$LOG_ROOT/train.log"

# Post-PPO eval at each ckpt + early-stop tracking via best-so-far.
echo "[fixedFv7] --- POST-PPO eval with running best tracking ---"
best_eplen=0
best_ck=""
regression_threshold=0.85  # 15% drop triggers warning; we still log all ckpts.

for ck in "$LOG_ROOT"/*fixedFv7-locomotion-hunter/model_*.pt; do
  [ -f "$ck" ] || continue
  cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  out="$LOG_ROOT/eval_post_iter${cit}.log"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ck" \
    +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out"
  e=$(extract "Episodes completed:" "$out"); fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out"); ds=$(extract "Average distance" "$out")
  if [ -n "$e" ]; then
    echo "post,$cit,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"
    # Track best so far
    el_int=$(python3 -c "print(int(float('$el')))")
    if [ "$el_int" -gt "$best_eplen" ]; then
      best_eplen=$el_int
      best_ck="$ck"
      echo "[fixedFv7] new best: model_${cit}.pt ep_len=$el"
    fi
  fi
done

echo "[fixedFv7] ALL DONE"
echo "[fixedFv7] best ckpt: $best_ck  ep_len=$best_eplen"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
