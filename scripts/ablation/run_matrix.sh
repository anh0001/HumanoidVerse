#!/usr/bin/env bash
# §2 ablation matrix (EXPERIMENT_PLAN.md). Sequenced (one run at a time —
# GPU is externally contended). Each (arm,seed): train on furrows_s2+DR for
# ITERS, then held-out eval on furrows_s3 and soil_challenging.
#
# Arms (isolation):
#   A0_baseline  ppo  + singlestep obs        + gait OFF   (baseline)
#   A3_hist50    ppo  + 50-step history obs   + gait OFF   (memory alone)
#   A5_roa       ppo_roa + 5-step history obs + gait OFF   (adaptation)
#   A8_gait      ppo  + singlestep obs        + gait ON    (gait reg alone)
#   A9_full      ppo_roa + 5-step history obs + gait ON    (proposed stack)
#
# Calibration: ITERS=30 SEEDS=1 NUM_ENVS=64 ARMS="A0_baseline A9_full" bash ...
# Core wave (default): ITERS=3000 SEEDS="1 2 3" NUM_ENVS=2048, all 5 arms.
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

ITERS="${ITERS:-3000}"
SEEDS=(${SEEDS:-1 2 3})
NUM_ENVS="${NUM_ENVS:-2048}"
ARMS=(${ARMS:-A0_baseline A3_hist50 A5_roa A8_gait A9_full})
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-64}"
LOG_ROOT="logs/AblationMatrix"
GAIT_OFF="++rewards.reward_scales.gait_phase=0 ++rewards.reward_scales.penalty_gait_asymmetry=0"
mkdir -p "$LOG_ROOT"

# arm -> "algo|obs|gait"
arm_spec() {
  case "$1" in
    A0_baseline) echo "ppo|loco/leggedloco_obs_singlestep_withlinvel|off" ;;
    A3_hist50)   echo "ppo|loco/leggedloco_obs_hist50_wolinvel|off" ;;
    A5_roa)      echo "ppo_roa|loco/leggedloco_obs_history_wolinvel|off" ;;
    A8_gait)     echo "ppo|loco/leggedloco_obs_singlestep_withlinvel|on" ;;
    A9_full)     echo "ppo_roa|loco/leggedloco_obs_history_wolinvel|on" ;;
    *) echo "ERR"; return 1 ;;
  esac
}

for arm in "${ARMS[@]}"; do
  spec="$(arm_spec "$arm")" || { echo "[matrix] unknown arm $arm"; continue; }
  IFS="|" read -r algo obs gait <<< "$spec"
  gait_ov=""; [ "$gait" = "off" ] && gait_ov="$GAIT_OFF"
  for s in "${SEEDS[@]}"; do
    run="${arm}_s${s}"
    rd="$LOG_ROOT/$run"
    mkdir -p "$rd"
    echo "[matrix] === TRAIN $run (algo=$algo obs=$obs gait=$gait iters=$ITERS) ==="
    "$PY" humanoidverse/train_agent.py \
      +exp=locomotion algo="$algo" \
      +domain_rand=YES_domain_rand \
      +rewards=loco/reward_hunter_locomotion \
      +robot=hunter/hunter +simulator=isaacsim \
      +terrain=terrain_furrows_stage2_medium \
      +obs="$obs" \
      num_envs="$NUM_ENVS" seed="$s" headless=True \
      ++env.config.env_spacing=2.5 \
      ++algo.config.num_learning_iterations="$ITERS" \
      project_name=AblationMatrix experiment_name="$run" \
      $gait_ov \
      2>&1 | tee "$rd/train.log"

    ckpt="$(ls -t logs/AblationMatrix/*"$run"*/model_*.pt 2>/dev/null | head -1)"
    if [ -z "$ckpt" ]; then echo "[matrix] !! no ckpt for $run, skip eval"; continue; fi
    echo "[matrix] ckpt=$ckpt"
    for ev in "furrows_s3:terrain_furrows_stage3_full:YES_domain_rand:" \
              "soil_chal:terrain_soil_challenging_wet:DR_soil_challenging:++terrain.num_rows=6 ++terrain.num_cols=6"; do
      IFS=":" read -r en et ed ex <<< "$ev"
      echo "[matrix] --- EVAL $run @ $en ---"
      "$PY" humanoidverse/sample_eps.py \
        +checkpoint="$ckpt" +terrain="$et" +domain_rand="$ed" \
        +eval_command=[0.3,0.0,0.0] num_envs="$EVAL_ENVS" +num_episodes="$EVAL_EPS" \
        headless=True $ex 2>&1 | tee "$rd/eval_${en}.log"
    done
    echo "[matrix] $run DONE"
  done
done
echo "[matrix] ALL ARMS COMPLETE"
