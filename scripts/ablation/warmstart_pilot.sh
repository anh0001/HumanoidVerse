#!/usr/bin/env bash
# Warm-start pilot (per Codex review, option E) — 4 arms × seed 1, 2-stage
# curriculum per arm. Honors per-arm architecture: each arm gets its OWN
# plane-warm-up ckpt so even A3/A5/A9 (different actor archs from baseline)
# can be fine-tuned on furrows_s2+DR without arch-mismatched warm-loading.
#
# Stage P (Plane, NO_DR, ~500 iters) -> arm-specific plane-pretrained policy
# Stage F (furrows_s2+YES_DR, 1000 iters, save every 250) -> per-arm fine-tune
# Held-out eval at each F ckpt via posthoc_eval logic (correct nested
# num_envs override).
#
# Decision rule (Codex):
#  - A9 nonzero locomotion AND A0/A3/A8 lower -> restart full matrix from
#    warm-start.
#  - all arms fail -> terrain too hard, build staged curriculum first.
#  - A0 also solves -> ROA+gait may not be necessary, re-scope.
#  - A3 or A8 matches A9 -> stack thesis weak.
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

PLANE_ITERS="${PLANE_ITERS:-500}"
FURROW_ITERS="${FURROW_ITERS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-64}"
ARMS=(${ARMS:-A0_baseline A3_hist50 A8_gait A9_full})
SEED=1
LOG_ROOT="logs/WarmstartPilot"
GAIT_OFF="++rewards.reward_scales.gait_phase=0 ++rewards.reward_scales.penalty_gait_asymmetry=0"
mkdir -p "$LOG_ROOT"

# arm -> "algo|obs|gait"  (mirrors run_matrix.sh)
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

CONDS=(
  "furrows_s3:terrain_furrows_stage3_full:YES_domain_rand:"
  "soil_chal:terrain_soil_challenging_wet:DR_soil_challenging:++terrain.num_rows=6 ++terrain.num_cols=6"
)

summary_csv="$LOG_ROOT/pilot_summary.csv"
[ ! -f "$summary_csv" ] && \
  echo "arm,seed,stage,ckpt_iter,condition,episodes,falls,fall_rate,ep_len_steps,distance_m" > "$summary_csv"

extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.]//g'; }

eval_one() {  # eval_one <ckpt> <arm> <stage_label> <ckpt_iter> <out_dir>
  local ckpt="$1" arm="$2" stage="$3" cit="$4" rd="$5"
  for ev in "${CONDS[@]}"; do
    IFS=":" read -r en et ed ex <<< "$ev"
    out="$rd/eval_${stage}_iter${cit}_${en}.log"
    if grep -q "Episodes completed:" "$out" 2>/dev/null && ! grep -q "Error in call" "$out" 2>/dev/null; then
      echo "[pilot]   $arm @ $stage iter $cit @ $en already done"
    else
      "$PY" humanoidverse/sample_eps.py \
        +checkpoint="$ckpt" +terrain="$et" +domain_rand="$ed" \
        +eval_command=[0.3,0.0,0.0] \
        num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
        +num_episodes="$EVAL_EPS" headless=True $ex 2>&1 | tee "$out"
    fi
    local eps falls eplen dist fr
    eps=$(extract "Episodes completed:" "$out")
    falls=$(extract "Total falls:" "$out")
    eplen=$(extract "Average episode length:" "$out")
    dist=$(extract "Average distance per episode:" "$out")
    if [ -n "$eps" ] && [ -n "$falls" ]; then
      fr=$(python3 -c "print(f'{float($falls)/float($eps):.3f}')" 2>/dev/null || echo NA)
      echo "$arm,$SEED,$stage,$cit,$en,$eps,$falls,$fr,$eplen,$dist" >> "$summary_csv"
      echo "[pilot]   -> falls=$falls/$eps ($fr) ep_len=$eplen dist=$dist"
    fi
  done
}

for arm in "${ARMS[@]}"; do
  spec="$(arm_spec "$arm")" || { echo "[pilot] unknown $arm"; continue; }
  IFS="|" read -r algo obs gait <<< "$spec"
  gait_ov=""; [ "$gait" = "off" ] && gait_ov="$GAIT_OFF"
  rd="$LOG_ROOT/$arm"
  mkdir -p "$rd"

  ##### STAGE P — plane warm-up #####
  pname="${arm}_P"
  echo "[pilot] ===== STAGE P (plane,NO_DR,${PLANE_ITERS} iters): $pname ====="
  "$PY" humanoidverse/train_agent.py \
    +exp=locomotion algo="$algo" \
    +domain_rand=NO_domain_rand \
    +rewards=loco/reward_hunter_locomotion \
    +robot=hunter/hunter +simulator=isaacsim \
    +terrain=terrain_locomotion_plane \
    +obs="$obs" \
    num_envs="$NUM_ENVS" seed="$SEED" headless=True \
    ++env.config.env_spacing=2.0 \
    ++algo.config.num_learning_iterations="$PLANE_ITERS" \
    ++algo.config.save_interval="$SAVE_EVERY" \
    project_name=WarmstartPilot experiment_name="$pname" \
    $gait_ov 2>&1 | tee "$rd/stageP.log"

  plane_ckpt="$(ls -t "$LOG_ROOT"/*"$pname"-locomotion-hunter/model_*.pt 2>/dev/null | head -1)"
  if [ -z "$plane_ckpt" ]; then echo "[pilot] !! no plane ckpt for $arm, skip"; continue; fi
  echo "[pilot] plane_ckpt=$plane_ckpt"

  ##### STAGE F — furrows_s2+DR fine-tune #####
  fname="${arm}_F"
  echo "[pilot] ===== STAGE F (furrows_s2+DR,${FURROW_ITERS} iters): $fname ====="
  "$PY" humanoidverse/train_agent.py \
    +exp=locomotion algo="$algo" \
    +domain_rand=YES_domain_rand \
    +rewards=loco/reward_hunter_locomotion \
    +robot=hunter/hunter +simulator=isaacsim \
    +terrain=terrain_furrows_stage2_medium \
    +obs="$obs" \
    num_envs="$NUM_ENVS" seed="$SEED" headless=True \
    ++env.config.env_spacing=2.5 \
    ++algo.config.num_learning_iterations="$FURROW_ITERS" \
    ++algo.config.save_interval="$SAVE_EVERY" \
    ++algo.config.load_optimizer=False \
    ++checkpoint="$plane_ckpt" \
    project_name=WarmstartPilot experiment_name="$fname" \
    $gait_ov 2>&1 | tee "$rd/stageF.log"

  ##### Held-out eval at each F ckpt #####
  for ck in "$LOG_ROOT"/*"$fname"-locomotion-hunter/model_*.pt; do
    [ -f "$ck" ] || continue
    cit="$(basename "$ck" | sed -E 's/model_([0-9]+)\.pt/\1/')"
    [ "$cit" = "0" ] && continue   # skip the snapshot taken before any update
    echo "[pilot] -- EVAL $arm @ F iter $cit --"
    eval_one "$ck" "$arm" "F" "$cit" "$rd"
  done
  echo "[pilot] === $arm DONE ==="
done

echo "[pilot] ALL ARMS COMPLETE"
echo "[pilot] summary -> $summary_csv"
column -t -s, "$summary_csv" 2>/dev/null || cat "$summary_csv"
