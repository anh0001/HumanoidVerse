# Literature Landscape — Hunter Biped on Tilled Soil (IsaacSim, RL)

**Date:** 2026-05-01
**Source:** scite (4 queries, ~50 papers triaged, ~23 directly relevant)
**Repo state assumed:** rigid heightfield with `perlin`/`furrows` geometry, friction patches, contact_stiffness as compliance proxy. No deformation, no sinkage dynamics, no terramechanics.

## 1. Sub-directions

| # | Direction | Key Anchor |
|---|-----------|------------|
| A | Quad RL on physically-deformable granular media (parametric soil sim) | Choi 2023, Sci. Robotics |
| B | Blind/proprioceptive transfer from rigid sim → real soft ground | ANYmal Lee 2020; RMA Kumar 2021; Margolis 2024 |
| C | Perceptive locomotion w/ heightmap + history encoder | Miki 2022; He 2025 |
| D | Biped RL on compliant + uneven (rigid) terrain | **Singh 2024 (Humanoids) — closest published biped analog** |
| E | Terramechanics models (Bekker/Wong/SCM/DEM) for foot-soil contact | Karpman 2020; Hu 2023; Buse 2023 |
| F | Real-time soft-ground footprint deformation (animation) | Alvarado 2022 (Frontiers VR) |
| G | DEM/SPH high-fidelity offline, parametric model online (calibration loop) | Hu 2023; Zhang 2024; Little 2023 |
| H | Sim-to-real survey identifying contact fidelity as biped bottleneck | Bao 2025 |

## 2. Top Anchor Papers

1. **Choi et al. 2023** — *Learning quadrupedal locomotion on deformable terrain.* Sci. Robotics 8(74). [10.1126/scirobotics.ade2256] — Parametric granular sim + adaptive controller, Raibo on beach sand at 3 m/s. **The benchmark to beat / extend to bipeds.**
2. **Singh, Morisawa, Benallegue 2024** — *Robust Humanoid Walking on Compliant and Uneven Terrain with DRL.* IEEE Humanoids. [10.1109/humanoids58906.2024.10769793] / [arxiv 2504.13619] — Closest biped+compliant+RL; compliance via contact-stiffness DR not granular. Not IsaacSim.
3. **Lee et al. 2020** — *Learning quadrupedal locomotion over challenging terrain.* Sci. Robotics 5(47). [10.1126/scirobotics.abc5986] — ANYmal blind transfer to mud/snow/moss; baseline for "rigid-only training generalizes."
4. **Kumar et al. 2021** — *RMA: Rapid Motor Adaptation.* RSS. [10.15607/rss.2021.xvii.011] — Privileged-encoder pattern for online terrain inference.
5. **Miki et al. 2022** — *Robust perceptive locomotion in the wild.* Sci. Robotics 7(62). [10.1126/scirobotics.abk2822] — Heightmap+proprio attention encoder; deformable noise on heightmap as DR proxy.
6. **Margolis et al. 2024** — *Rapid locomotion via RL.* IJRR 43(4). [10.1177/02783649231224053] — Notes rigid sims leave deformable OOD; online sysid + adaptive vel curriculum.
7. **Bao, Peng, Zhou 2025** — *Sim-to-Real in DRL for Bipedal Locomotion (survey).* arXiv 2511.06465. [10.48550/arxiv.2511.06465] — Names contact modeling + solver fidelity as the central biped sim2real bottleneck.
8. **Chang et al. 2021** — *Learning Terrain Dynamics: GP modeling + optimal control on granular media.* IEEE TCST 29(4). [10.1109/tcst.2020.3009636] — Critiques spring-damper soil; GP-fits ground reaction on GM for 1-D hopper.
9. **Alvarado et al. 2022** — *Real-Time Locomotion on Soft Grounds with Dynamic Footprints.* Frontiers VR. [10.3389/frvir.2022.801856] — Bekker-style heightfield deformation in real-time (kinematic chars). **Cheap soft-soil model adaptable to IsaacSim heightfields.**
10. **Karpman, Kövecses, Holz 2020** — *DEM for wheel-soil interaction.* J. Terramechanics. [10.1016/j.jterra.2020.06.002] — Bekker vs DEM vs P² survey; flags Bekker breakdown at small loads/radii (humanoid foot regime).
11. **Hu, Li, Unjhawala 2023** — *Calibration of expeditious terramechanics via virtual bevameter + Bayesian inference.* J. Field Robotics. [10.1002/rob.22276] — DEM→SCM calibration workflow; **template for offline-fidelity → online-fast pattern.**
12. **Buse, Pignède, Barthelmes 2023** — *Modelica Library for Contact Dynamics + Terramechanics.* [10.3384/ecp204433] — Open Bekker-Wong+Hertz multibody plug-in (MMX rover); blueprint for IsaacSim integration.
13. **Little, Coker, Thornton 2023** — *Complex Terramechanics for Mobility, Part 2.* ASME IDETC. [10.1115/detc2023-117161] — DEM-fit Bekker params for cheap online use.
14. **Howard, Kannemeyer, Dolcetti 2022** — *Evolutionary terrain generation for curriculum RL.* arXiv 2203.15172. [10.48550/arxiv.2203.15172] — Bipedal walker + PPO + MAP-Elites heightfield curricula. Direct curriculum reference.
15. **He, Zhang, Jenelten 2025** — *Attention-based map encoding for generalized legged loco.* Sci. Robotics 10(105). [10.1126/scirobotics.adv3604] — Quadruped + 23-DoF humanoid; map encoder architecture choices.
16. **Ha et al. 2025** — *Learning-based legged loco SOTA.* IJRR 44(8). [10.1177/02783649241312698] — Recent survey for situating Hunter project.
17. **Kadokawa et al. 2023** — *Cyclic policy distillation: sim-to-real with DR.* RAS 165. [10.1016/j.robot.2023.104425] — Particle-method excavation + progressive-resolution distillation.
18. **Collins et al. 2021** — *Review of physics simulators.* IEEE Access 9. [10.1109/access.2021.3068769] — Catalogues Project Chrono's deformable/granular/FEM as IsaacSim alternative.

## 3. Recurring Open Problems

- **No biped RL on physically-deformable agricultural soil exists.** Choi 2023 = quad. Singh 2024 = compliant rigid, not granular. Niche genuinely open.
- **Bekker breaks at humanoid-foot scale.** Small contact patch + intermittent loading + heel-strike transients are out of model regime (Karpman 2020).
- **DEM/SPH too slow for 2048-env GPU RL.** Only scalable pattern is offline-DEM → calibrated-online-SCM (Hu 2023, Zhang 2024). No published wiring into IsaacSim.
- **Tilled-soil-specific phenomena unstudied.** Slip-sinkage coupling, bulldozing of furrow ridges, anisotropic friction along/across furrows — absent everywhere.
- **DR for compliance is just contact-stiffness/restitution noise** (what this repo already does); not validated against real soil deformation data.
- **Tactile/proprioceptive soil-type ID is implicit.** No dedicated benchmark for "infer soil class from gait."

## 4. Concurrent Work Risk (2024-2026)

| Paper | Overlap | Notes |
|-------|---------|-------|
| Singh 2024 (Humanoids) | **Medium** | Biped+RL+compliant uneven, but contact-stiffness DR not granular soil; not IsaacSim |
| Choi 2023 (Sci. Robotics) | Medium | Quadruped not biped; in-house sim not IsaacSim |
| He 2025 (Sci. Robotics) | Low | Humanoid trained but on rigid sparse footholds, not soil |
| Bao 2025 survey | None | Survey, not competing system |
| Xu 2025 G1 fall-safety | Low | Humanoid+RL but fall recovery not soil |

**Verdict:** "Hunter biped + tilled soil + IsaacSim + RL" niche is unoccupied.

## 5. Open Codebases / Sim Tools

- **Project Chrono** — built-in deformable/granular/FEM. Candidate for ground-truth sim.
- **IsaacGym/IsaacLab/legged_gym** — dominant RL stack, rigid-only. What this repo extends.
- **Modelica terramechanics library** (Buse 2023) — open Bekker-Wong plug-in.
- **Raisim, MuJoCo** — rigid only.
- **FTR-Bench** (Zhang 2025 https://github.com/nubot-nudt/FTR-Benchmark) — IsaacLab obstacle bench, model for RL terrain benches.
- **No open biped+deformable-soil codebase exists** — release would fill a real gap.
