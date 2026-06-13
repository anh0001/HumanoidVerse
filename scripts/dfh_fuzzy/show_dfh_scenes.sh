#!/usr/bin/env bash
# Launch DFH visualization windows on a display (default :30540), staggered so the
# three Kit instances don't thrash at startup. Each = 1 robot, writeback ON (visible
# soil deformation). Robust/no-set-u; mirrors the proven inline launch.
#   SOFT  (K=1,  deep sink ~5 cm)        terrain_dfh_stage1_easy
#   FIRM  (K=8,  shallow sink ~1.5 cm)   terrain_dfh_stage1_easy (Bekker x8)
#   MAIZE (K=1 + maize row-crop scene)   terrain_dfh_with_maize
#
# Usage:  scripts/dfh_fuzzy/show_dfh_scenes.sh        # uses :30540
#         VIZ_DISPLAY=:0 scripts/dfh_fuzzy/show_dfh_scenes.sh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${PYTHON:-/srv/data/users/anhar/miniconda3/envs/isaaclab/bin/python}"
CKPT="${CKPT:-logs/DFH_hf_models/walkers/rigid_walker/model_5050.pt}"
DISP="${VIZ_DISPLAY:-:30540}"
PRELUDE=/tmp/dfh_env_prelude.sh
if [ ! -f "$PRELUDE" ]; then
  cat > "$PRELUDE" <<'EOF'
export OMNI_KIT_ACCEPT_EULA=YES
unset CARB_APP_PATH EXP_PATH ISAAC_PATH ISAACSIM_PATH
export ISAACLAB_PATH="${ISAACLAB_PATH:-$PWD/IsaacLab}"
EOF
fi

launch() {  # name  delay_s  terrain  extra_overrides...
  local name="$1" delay="$2" terr="$3"; shift 3
  nohup bash -c "sleep $delay; source $PRELUDE; export DISPLAY=$DISP; cd $REPO
    $PY humanoidverse/sample_eps.py +checkpoint=$CKPT +terrain=$terr \
      +eval_command='[0.3,0.0,0.0]' num_envs=1 ++simulator.config.scene.num_envs=1 \
      +num_episodes=1000 headless=False ++terrain.dfh.params.sinkage_floor_m=-0.05 \
      ++terrain.dfh.writeback_enabled=True $* +experiment_name=show_$name" \
    > "logs/DFH_fuzzy/viz_$name.log" 2>&1 &
  echo "[show] $name scheduled (+${delay}s) on $DISP pid=$!"
}

launch SOFT  0  terrain_dfh_stage1_easy
launch FIRM  40 terrain_dfh_stage1_easy ++terrain.dfh.params.bekker.kc=11200 ++terrain.dfh.params.bekker.k_phi=6560000
launch MAIZE 80 terrain_dfh_with_maize
echo "[show] all three scheduled; windows appear on $DISP as each Kit finishes startup."
