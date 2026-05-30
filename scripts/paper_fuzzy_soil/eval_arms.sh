#!/usr/bin/env bash
# Evaluate Arm A and Arm B checkpoints on the proven `terrain_furrows_with_maize`
# eval surface. Mirrors the metrics columns of paper Table II.
#
# Usage:
#   scripts/paper_fuzzy_soil/eval_arms.sh                   # all seeds, both arms
#   ARM=A SEED=1 scripts/paper_fuzzy_soil/eval_arms.sh      # one cell
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
if [ -d "${ISAAC_PATH:-$HOME/isaacsim_4.2}" ]; then
  export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim_4.2}"
  export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
  export CARB_APP_PATH="${CARB_APP_PATH:-$ISAAC_PATH/kit}"
  if [ -f "$ISAAC_PATH/setup_python_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$ISAAC_PATH/setup_python_env.sh"
  fi
fi
export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/IsaacLab}"
PY="${PYTHON:-/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python}"
[ -x "$PY" ] || PY="$HOME/miniconda3/envs/isaaclab/bin/python"

EVAL_TERRAIN="${EVAL_TERRAIN:-terrain_furrows_with_maize}"
EVAL_DR="${EVAL_DR:-DR_paper_S3}"           # paper's hardest condition
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_EPS="${EVAL_EPS:-100}"                 # paper uses 100 eps per seed
EVAL_CMD="${EVAL_CMD:-[0.3,0.0,0.0]}"

ARMS="${ARM:-A B}"
SEEDS="${SEED:-1 2 3}"

RESULTS="logs/FuzzySoilFurrows/results.csv"
mkdir -p "$(dirname "$RESULTS")"
# Columns include slip_per_100m and falls_per_100m — required by the preregistered
# decision rule in docs/experiments/fuzzy_soil_furrows_plan.md (slip_B ≤ 0.5 × slip_A).
[ ! -f "$RESULTS" ] && echo "arm,seed,ckpt,episodes,falls,ep_len_steps,distance_m,slip_per_100m,falls_per_100m" > "$RESULTS"

latest_ckpt_in_glob() {
  # Find the highest-iter model_*.pt across all dirs matching the given glob.
  # Hydra writes runs into logs/FuzzySoilFurrows/<timestamp>-<experiment_name>-<task>-<robot>/,
  # which is a SIBLING of the tracking dir — so we search by the experiment_name
  # fragment in the timestamped dir name, not by tracking-dir parent.
  local pattern="$1"
  # shellcheck disable=SC2086
  ls -1 $pattern 2>/dev/null \
    | awk -F'/' '{n=split($NF,a,"_"); split(a[n],b,"."); print b[1]" "$0}' \
    | sort -n | tail -1 | awk '{print $2}'
}
extract() { grep -E "$1" "$2" 2>/dev/null | tail -1 | sed -E 's/.*: //;s/[^0-9.-]//g'; }

for arm in $ARMS; do
  for s in $SEEDS; do
    case "$arm" in
      # Arm A: one timestamped dir per seed → ...-armA_seed${s}-locomotion-...
      A) pattern="logs/FuzzySoilFurrows/*-armA_seed${s}-locomotion-*/model_*.pt" ;;
      # Arm B: three timestamped dirs per seed (S2a/S2b/S3); take the final S3 ckpt.
      B) pattern="logs/FuzzySoilFurrows/*-armB_S3_seed${s}-locomotion-*/model_*.pt" ;;
      *) echo "[eval] unknown arm $arm"; continue ;;
    esac
    ckpt="$(latest_ckpt_in_glob "$pattern")"
    [ -n "$ckpt" ] && [ -f "$ckpt" ] || { echo "[eval] no ckpt matching $pattern — skipping"; continue; }

    out="logs/FuzzySoilFurrows/eval_${arm}_seed${s}.log"
    echo "[eval] arm=$arm seed=$s ckpt=$ckpt"
    "$PY" humanoidverse/sample_eps.py +checkpoint="$ckpt" \
      +terrain="$EVAL_TERRAIN" +domain_rand="$EVAL_DR" \
      +eval_command="$EVAL_CMD" \
      num_envs="$EVAL_ENVS" ++simulator.config.scene.num_envs="$EVAL_ENVS" \
      ++env.config.env_spacing=2.5 +num_episodes="$EVAL_EPS" headless=True \
      ++env.config.termination.terminate_by_contact=True \
      2>&1 | tee "$out"

    e=$(extract "Episodes completed:" "$out")
    fl=$(extract "Total falls:" "$out")
    el=$(extract "Average episode length:" "$out")
    ds=$(extract "Average distance" "$out")
    sp=$(extract "Slip distance per 100m:" "$out")
    # falls/100m = falls / (total distance / 100)
    fp=""
    if [ -n "$fl" ] && [ -n "$ds" ] && [ -n "$e" ]; then
      fp=$(python3 -c "
fl, ds, e = float('$fl'), float('$ds'), float('$e')
total_dist = ds * e
print(f'{(fl / (total_dist/100.0)):.3f}') if total_dist > 0 else print('')
")
    fi
    [ -n "$e" ] && echo "$arm,$s,$ckpt,$e,$fl,$el,$ds,$sp,$fp" >> "$RESULTS"
  done
done

echo "[eval] DONE — see $RESULTS"
column -t -s, "$RESULTS" 2>/dev/null || cat "$RESULTS"
