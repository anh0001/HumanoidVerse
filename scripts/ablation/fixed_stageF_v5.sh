#!/usr/bin/env bash
# FixedStageF v5: GEOMETRY RUNG (Codex prescription, thread 019e54fe).
# v4 plateaued at ~520 ep_len on depth 0.08-0.12. Codex: this is NOT a hard
# ceiling but a "mid-horizon falling mode." Reduce furrow depth to 0.05-0.08
# (fix geometry rung first), then ramp back. Add modest survival shaping:
# termination -25, stronger orientation/base_height/contact-force penalties.
# Warm-load from v4 model_5000.pt. Eval each ckpt on same condition.
# Promote criterion (back to 0.08-0.12 depth): ep_len >= 850 AND fall_rate <= 0.20.
# Bail criterion (move to perceptive obs/heightmap): ep_len < 700 or fall_rate > 0.5
# after the full 1500 iters here.
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

# Default warm-load: v4's final ckpt. Override with WARM_CKPT=... if v4 hasn't
# saved model_5000.pt yet (fall back to model_4750.pt or model_4500.pt).
WARM_CKPT="${WARM_CKPT:-$(ls -t logs/FixedStageFv4/*fixedFv4-locomotion-hunter/model_5000.pt 2>/dev/null | head -1)}"
if [ -z "$WARM_CKPT" ] || [ ! -f "$WARM_CKPT" ]; then
  WARM_CKPT="$(ls -t logs/FixedStageFv4/*fixedFv4-locomotion-hunter/model_*.pt 2>/dev/null | head -1)"
fi
[ -z "$WARM_CKPT" ] && { echo "[fixedFv5] no v4 ckpt found under logs/FixedStageFv4/"; exit 1; }

ITERS="${ITERS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-250}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-32}"
F_TERRAIN="terrain_furrows_stage1_easy"
LOG_ROOT="logs/FixedStageFv5"
mkdir -p "$LOG_ROOT"
summary="$LOG_ROOT/summary.csv"
[ ! -f "$summary" ] && echo "phase,ckpt_iter,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary"
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

echo "[fixedFv5] warm-start ckpt = $WARM_CKPT"
echo "[fixedFv5] geometry rung: depth_range [0.05, 0.08] (was [0.08, 0.12])"

# Pre-PPO transfer eval on the REDUCED-depth terrain (sanity check before training).
echo "[fixedFv5] --- PRE-PPO eval on $F_TERRAIN (depth 0.05-0.08) + DR_mild ---"
pre_out="$LOG_ROOT/eval_pre.log"
"$PY" humanoidverse/sample_eps.py +checkpoint="$WARM_CKPT" \
  +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
  num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
  ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
  ++env.config.termination.terminate_by_contact=True \
  '++terrain.terrain_kwargs.depth_range=[0.05,0.08]' \
  2>&1 | tee "$pre_out"
e=$(extract "Episodes completed:" "$pre_out"); fl=$(extract "Total falls:" "$pre_out")
el=$(extract "Average episode length:" "$pre_out"); ds=$(extract "Average distance" "$pre_out")
[ -n "$e" ] && echo "pre,0,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"

echo "[fixedFv5] --- TRAIN furrows_s1 depth 0.05-0.08 + DR_mild, $ITERS iters ---"
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
  '++terrain.terrain_kwargs.depth_range=[0.05,0.08]' \
  ++rewards.reward_scales.tracking_lin_vel=4.0 \
  ++rewards.reward_scales.termination=-25.0 \
  ++rewards.reward_scales.penalty_orientation=-2.0 \
  ++rewards.reward_scales.base_height=-8.0 \
  ++rewards.reward_scales.penalty_feet_contact_forces=-0.02 \
  '++env.config.locomotion_command_ranges.lin_vel_x=[0.25,0.45]' \
  '++env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0]' \
  '++env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0]' \
  project_name=FixedStageFv5 experiment_name=fixedFv5 \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[fixedFv5] --- POST-PPO eval at each ckpt (same depth 0.05-0.08) ---"
for ck in "$LOG_ROOT"/*fixedFv5-locomotion-hunter/model_*.pt; do
  [ -f "$ck" ] || continue
  cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
  [ "$cit" = "0" ] && continue
  out="$LOG_ROOT/eval_post_iter${cit}.log"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$ck" \
    +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    '++terrain.terrain_kwargs.depth_range=[0.05,0.08]' \
    2>&1 | tee "$out"
  e=$(extract "Episodes completed:" "$out"); fl=$(extract "Total falls:" "$out")
  el=$(extract "Average episode length:" "$out"); ds=$(extract "Average distance" "$out")
  [ -n "$e" ] && echo "post,$cit,$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"
done

# Promotion eval: best v5 ckpt evaluated on ORIGINAL depth (0.08-0.12) to test
# whether geometry-rung learning transfers back up. Only meaningful if the
# in-distribution post evals look good.
BEST_CK="$(ls -t "$LOG_ROOT"/*fixedFv5-locomotion-hunter/model_*.pt 2>/dev/null | head -1)"
if [ -n "$BEST_CK" ]; then
  echo "[fixedFv5] --- PROMOTION eval (latest v5 ckpt on FULL depth 0.08-0.12) ---"
  prom_out="$LOG_ROOT/eval_promotion.log"
  "$PY" humanoidverse/sample_eps.py +checkpoint="$BEST_CK" \
    +terrain="$F_TERRAIN" +domain_rand=DR_mild +eval_command=[0.3,0.0,0.0] \
    num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
    ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
    ++env.config.termination.terminate_by_contact=True \
    2>&1 | tee "$prom_out"
  e=$(extract "Episodes completed:" "$prom_out"); fl=$(extract "Total falls:" "$prom_out")
  el=$(extract "Average episode length:" "$prom_out"); ds=$(extract "Average distance" "$prom_out")
  [ -n "$e" ] && echo "promo_full_depth,$(basename $BEST_CK .pt | sed -E 's/model_//'),$e,$fl,$(python3 -c "print(f'{float($fl)/float($e):.3f}')"),$el,$ds" >> "$summary"
fi

echo "[fixedFv5] ALL DONE"
column -t -s, "$summary" 2>/dev/null || cat "$summary"
