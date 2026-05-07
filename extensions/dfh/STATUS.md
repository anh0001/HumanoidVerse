# DFH Status — v0.2.1

**Date:** 2026-05-03
**Validated against:** IsaacSim 4.2.0, IsaacLab 1.4.1, Hunter robot

## Working ✅

| Component | Evidence |
|-----------|----------|
| `ext_dfh` package install (`pip install -e extensions/dfh`) | clean install in `isaaclab` conda env |
| Hydra config composition (`+curriculum=stage1_dfh`) | `terrain.dfh_enabled=true`, full DFH params resolved |
| DFH adapter attach at IsaacSim init | `DFH terrain layer attached: feet=['left_ankle_link', 'right_ankle_link'], grid=120x120, writeback=on` |
| Lazy foot body resolution | survives `_root_physx_view` not-yet-ready at init |
| Per-step `H_plastic` update from contact forces | runs every physics step, no errors over 214 PPO iter |
| **Path A: PhysX heightfield USD mutation** | `mesh.CreatePointsAttr().Set(Vt.Vec3fArray(...))` succeeds at K=4 cadence |
| Per-vertex cell map for triangulated/decimated meshes | handles `6593 vertices on 881x881 grid` correctly |
| Grid remap (terrain mesh grid → DFH config grid) | `881x881 → 120x120` index scaling |
| Graceful degradation outside Kit / on USD failure | no crashes; one-shot warning then no-op |
| Per-env reset (`dfh_reset` in `_reset_buffers_callback`) | wired |
| Unit + mock-integration tests | 27/27 passing |

## Performance ✅

- ~0.78 s / PPO iteration at `num_envs=4` (no measurable overhead from DFH writeback at K=4)
- `physx_writeback_every_k_steps=4` default tunable per terrain config

## Known limitations / v0.3 work

| Gap | Impact | Path |
|-----|--------|------|
| Single shared terrain prim across envs (cloner instances all share one mesh) | Cross-env aggregation = `min` reduction; one env's deformation affects all | Per-env mini-heightfield prim under each `env_<id>` root |
| Furrow direction defaults to 0° everywhere | Anisotropy magnitude correct, direction uniform | Intercept furrow generator at `isaacsim.py:442` to store θ per cell |
| `refresh_physics_objects` API not on every IsaacSim build | Falls back to PhysX's USD dirty detection | Try/except wrapped, no crash |
| Bekker params not yet calibrated against DEM | Defaults from Wong's loose dry sand; physically reasonable but uncalibrated | Run `calibration/run_chrono_bevameter.py` (TODO) + `fit_bekker.py` (TODO) |
| Visual verification of terrain deformation | Not yet inspected in `headless=False` | Launch with DISPLAY + sample sinkage at known load |
| `num_envs=2048` long-run stability | Untested | Phase 5 of `EXPERIMENT_PLAN.md` |

## Reproducer

```bash
export ISAAC_PATH=~/isaacsim_4.2
export EXP_PATH=$ISAAC_PATH/apps
export CARB_APP_PATH=$ISAAC_PATH/kit
export ISAACLAB_PATH=$(pwd)/IsaacLab
source ~/isaacsim_4.2/setup_python_env.sh

~/miniconda3/envs/isaaclab/bin/python humanoidverse/train_agent.py \
  +curriculum=stage1_dfh \
  +robot=hunter/hunter \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=4 headless=True \
  project_name=DFH_smoke experiment_name=Hunter_DFH_S1_smoke
```

Expected log lines:
- `Baseline mesh: <N> vertices on a <H> x <W> cell grid (triangulation; per-vertex cell map built).`
- `DFH terrain layer attached: feet=['left_ankle_link', 'right_ankle_link'], grid=120x120, writeback=on`
- No `DFH writeback failed` warnings.
- Steady `Iteration time: ~0.8 s`.

## Test

```bash
cd extensions/dfh
PYTHONPATH=source ~/miniconda3/envs/isaaclab/bin/python -m pytest tests --no-header -q
# 27 passed
```
