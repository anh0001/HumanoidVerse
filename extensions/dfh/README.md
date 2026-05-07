# `ext_dfh` — Deformable Furrowed Heightfield for IsaacSim

Drop-in plastic heightfield contact layer for NVIDIA IsaacSim / IsaacLab. Replaces the rigid `furrows` / `perlin` heightfield used in the HumanoidVerse Hunter pipeline with a **GPU-resident deformable surface** that captures:

- **Bekker–Wong pressure-sinkage** — feet sink into soil under load.
- **Janosi–Hanamoto shear** — tangential slip generates resistive shear.
- **Slip-sinkage coupling (Wong–Reece)** — dragging feet sink deeper.
- **Bulldozing** — displaced soil mass redistributes laterally.
- **Furrow-anisotropic friction** — μ depends on stride direction relative to furrow.

Designed for the **Hunter humanoid biped on tilled agricultural soil**. Built to release as a standalone IsaacLab extension.

## Scope

- **Sim:** IsaacSim / IsaacLab only (no MuJoCo, no IsaacGym, no Genesis).
- **Robot:** validated against Hunter; foot collider assumed box, ~3.45 × 10⁻³ m² contact area.
- **Calibration:** Bekker parameters fitted offline from DEM ground truth (see `calibration/`).

## Layout

```
extensions/dfh/
├── pyproject.toml
├── source/ext_dfh/
│   ├── config.py          # DFHParams, DFHConfig dataclasses
│   ├── dfh_layer.py       # DFHTerrainLayer — public API
│   ├── kernels.py         # GPU kernels: Bekker, slip-sinkage, bulldozing, anisotropy
│   └── integration.py     # IsaacSim per-step hook + heightfield write-back
├── configs/               # Hydra YAML for terrain / DR / curriculum / rewards
├── calibration/           # Hu-2023-style virtual bevameter workflow (DEM → Bekker)
└── tests/                 # pytest unit tests for kernels
```

## Install (dev)

```bash
cd extensions/dfh
pip install -e .[dev]
```

## Use from HumanoidVerse

```bash
python humanoidverse/train_agent.py \
  +curriculum=stage1_dfh \
  +robot=hunter/hunter \
  +simulator=isaacsim \
  +exp=locomotion_soil \
  +algo=ppo_soil \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  +rewards=dfh/reward_hunter_dfh \
  +terrain=dfh/terrain_dfh_stage1_easy \
  +domain_rand=dfh/DR_dfh_soft \
  num_envs=2048 headless=True \
  project_name=DFH experiment_name=Hunter_DFH_S1_Easy
```

## Status

**v0.1 (current):** API + config skeletons. Calibration workflow doc.
**v0.2:** Bekker sink kernel + plastic heightfield write-back wired to IsaacSim.
**v0.3:** Slip-sinkage + bulldozing + anisotropy.
**v0.4:** Calibrated against Project Chrono DEM virtual bevameter.
**v1.0:** Ablations + paper.

## Reference

- Choi et al. 2023, *Learning quadrupedal locomotion on deformable terrain*, Sci. Robotics. [10.1126/scirobotics.ade2256]
- Hu, Li, Unjhawala 2023, *Calibration of an expeditious terramechanics model using a virtual bevameter*. [10.1002/rob.22276]
- Buse, Pignède, Barthelmes 2023, *Modelica Library for Contact Dynamics + Terramechanics*. [10.3384/ecp204433]
- Karpman, Kövecses, Holz 2020, *DEM for wheel-soil interaction*. [10.1016/j.jterra.2020.06.002]
- Alvarado et al. 2022, *Real-Time Locomotion on Soft Grounds with Dynamic Footprints*. [10.3389/frvir.2022.801856]
