#!/usr/bin/env bash
# FixedStageF v6: RESUME-V3 + CLEAN-RESETS (Codex Option E, thread 019e54fe).
#
# Diagnosis (2026-05-24): the train/eval gap that consumed v4 and v5 was NOT
# an eval-pipeline bug. v3 model_3500 re-evals to ep_len 230.7 today (orig.
# 209.2). v4's termination=-10 reward and v5's added survival shaping each
# broke the v3 policy into a degenerate basin that scored well on the
# training-time lenbuffer (~520) by minimizing reset rate but collapsed to
# ep_len ~35 in eval. Discard v4/v5.
#
# v6: warm-load v3 model_3500 with v3's reward setup (NO termination penalty,
# NO orientation/base_height/contact-force shaping changes) and tighten reset
# randomization to IsaacLab H1's convention (joint scale (1.0, 1.0), zero root
# velocity). This is the single high-signal change from upstream IsaacLab.
#
# Stop criterion (eval every 250 saved ckpts):
#   - ep_len >= 450 and still rising  -> promote, train longer
#   - plateau below 350               -> switch to heightmap obs (Option C)
#   - collapse like v4 (<100)         -> revert to pure A (no reset change)
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

# Warm-load: v3's final checkpoint (the last "honestly improving" ckpt).
WARM_CKPT="${WARM_CKPT:-logs/FixedStageFv3/20260523_152906-fixedFv3-locomotion-hunter/model_3500.pt}"
[ -f "$WARM_CKPT" ] || { echo "[fixedFv6] warm ckpt missing: $WARM_CKPT"; exit 1; }

ITERS="${ITERS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-32}"
F_TERRAIN="terrain_furrows_stage1_easy"
LOG_ROOT="logs/FixedStageFv6"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/summary.csv"
[ ! -f "$summary" ] && echo "phase,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary"
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "[fixedFv6] warm-start ckpt = $WARM_CKPT"
echo "[fixedFv6] clean-reset overrides: dof_pos_scale=(1.0,1.0), root_lin_vel=0, root_ang_vel=0"

# Pre-PPO transfer eval (sanity: ~209-230 expected, matching v3 re-eval).
echo "[fixedFv6] --- PRE-PPO eval on $F_TERRAIN + DR_mild ---"
pre_out="$LOG_ROOT/eval_pre.log"
"$PY" humanoidverse/sample_eps.py +checkpoint="$WARM_CKPT" \
  +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
  num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
  ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
  ++env.config.termination.terminate_by_contact=True \
  2>&1 | tee "$pre_out"
e=$(extract "Episodes completed:" "$pre_out"); fl=$(extract "Total falls:" "$pre_out")
el=$(extract "Average episode length:" "$pre_out"); ds=$(extract "Average distance" "$pre_out")
[ -n "$e" ] && echo "pre,0,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"

echo "[fixedFv6] --- TRAIN v3-config + clean reset, $ITERS iters ---"
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
  project_name=FixedStageFv6 experiment_name=fixedFv6 \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[fixedFv6] --- POST-PPO eval at each ckpt ---"
for ck in "$LOG_ROOT"/*fixedFv6-locomotion-hunter/model_*.pt; do
  [ -f "$ck" ] || continue
  cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  [ "$cit" = "0" ] && continue
  out="$LOG_ROOT/eval_post_iter${cit}.log"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ck" \
    +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$out"
  e=$(extract "Episodes completed:" "$out"); fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out"); ds=$(extract "Average distance" "$out")
  [ -n "$e" ] && echo "post,$cit,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"
done

echo "[fixedFv6] ALL DONE"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
