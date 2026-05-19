# Final Proposal — Deformable Furrowed Heightfield (DFH) for Hunter on Tilled Soil

**Date:** 2026-05-01
**Repo:** HumanoidVerse (Hunter only, IsaacSim only)

## Problem Anchor (frozen)

Train Hunter humanoid policies in IsaacSim that **transfer to real tilled agricultural soil** — soil that visibly deforms under foot, has anisotropic shear (along/across furrows), and exhibits slip-sinkage coupling. Current rigid-heightfield contact in `humanoidverse/simulator/isaacsim/isaacsim.py` cannot represent any of these phenomena, leaving real soil out-of-distribution for any policy trained here.

## Method Thesis (one sentence)

Add a GPU-resident **plastic heightfield contact layer** to IsaacSim that updates surface elevation per simulation step using Bekker–Wong pressure-sinkage, Janosi–Hanamoto shear, furrow-anisotropic friction, and slip-sinkage coupling — wired as a drop-in replacement for the existing rigid `furrows`/`perlin` heightfield without changing the rest of the training stack.

## Dominant Contribution

First open biped + IsaacLab pipeline trained on a physically-deformable agricultural soil model. Choi 2023 covers quadruped on sand in a closed sim; Singh 2024 covers biped on rigid compliant terrain; nobody bridges biped × deformable agricultural soil × IsaacSim.

## Components

### C1 — Plastic Heightfield Buffer
- `H_plastic[i,j]`: signed elevation delta from base terrain. Initialized 0.
- Persists across steps. Reset on env reset.
- Implementation: torch tensor on GPU, shape `[num_envs, grid_h, grid_w]`. Updated in custom Warp kernel called from `isaacsim.py` per-physics-step hook.

### C2 — Bekker–Wong Sinkage Update
For each contact patch detected by IsaacSim contact reporter:
```
p = F_normal / A_contact      # contact pressure (Pa)
z_target = (p / k)^(1/n)      # Bekker pressure-sinkage, k = (kc/b + kφ)
H_plastic[i,j] = min(H_plastic[i,j], -z_target)   # plastic, no rebound
```
Params per cell: `(kc, kφ, n, c, φ)` from `terrain.dfh_params` config.

### C3 — Slip-Sinkage Coupling (Wong–Reece)
```
z_slip = z_target * (1 + α_s * |v_tangential| / |v_normal+ε|)
```
α_s in [0.05, 0.5] tunable. Captures the empirical extra sinking when feet drag.

### C4 — Bulldozing
When foot moves laterally on deformed cells, displaced volume redistributes:
```
ΔV = (z_old - z_new) * cell_area
H_plastic[neighbors] += ΔV / N_neighbors / cell_area
```
Conserves soil mass within local 3×3 stencil. Caps to prevent runaway ridges.

### C5 — Anisotropic Furrow Friction
Each cell stores furrow direction θ_f (from furrow generator at `isaacsim.py:442`).
```
α = angle(v_tangential, θ_f)
μ_eff = μ_∥ * cos²(α) + μ_⊥ * sin²(α)
```
Apply via PhysX material override per contact, or as additional tangential force injection if material API too coarse.

### C6 — Configs + Curriculum
- `terrain/terrain_dfh_stage1_easy.yaml` — soft Bekker params, low α_s, low anisotropy, shallow furrows. (Reuses Stage 1 furrow geometry.)
- `terrain/terrain_dfh_stage2_medium.yaml` — moderate.
- `terrain/terrain_dfh_stage3_full.yaml` — wet/loose params + full anisotropy.
- `domain_rand/DR_dfh_*.yaml` — sample (kc, kφ, n, c, φ, α_s, μ_∥, μ_⊥) ranges per regime.
- `curriculum/stage{1,2}_dfh.yaml` — promote on the existing `mean_episode_length ≥ 16s` + `term_reward ≥ -0.05/s` criteria from CLAUDE.md.

### C7 — Reward Tweaks (`rewards/loco/reward_hunter_dfh.yaml`)
Extend `reward_hunter_soil_locomotion.yaml` with:
- `penalty_sinkage_excess` (cap mean foot sinkage)
- `reward_foot_exit_velocity` (encourage clean lift-off, reduce energy lost to bulldozing)
- `desired_base_height` adapted dynamically to expected sinkage (already at 0.56m for rigid soil — drop to 0.52m for DFH stage 3)

### C8 — Observations
Reuse `obs/leggedloco_obs_singlestep_withlinvel` unchanged for the **base policy**. For the optional Idea 2 (RMA soil-ID) add `obs/leggedloco_obs_history` later — out of scope for v1.

## What's Out (deliberately)

- No new sim engine (no Chrono, no MuJoCo MPM)
- No DEM ground truth (treat as future work; calibrate Bekker params from published soil mech tables initially)
- No real-robot bevameter
- No exteroceptive sensing
- No multi-robot — Hunter only
- No `g1`/`h1`/`isaacgym`/`genesis` paths touched

## Risks + Mitigations

| Risk | Mitigation |
|------|-----------|
| GPU-Bekker too slow at 2048 envs | Coarsen heightfield to 0.20m grid; only update cells under active contacts (typically <100 cells/env/step) |
| Plastic heightfield breaks PhysX collision shape caching | Update collision mesh every K steps not every step; staleness empirically OK if K·dt < 50ms |
| Anisotropic μ via PhysX material API too coarse | Inject tangential force in contact callback instead |
| Hunter URDF foot collider point/sphere → tiny patch area → unphysical pressures | Switch to capsule/box foot collider; verify in `+robot=hunter/hunter` config |
| Policy exploits plastic deformation by carving trenches | Cap `H_plastic` floor at -0.10m per cell, re-equilibrate on env reset |
| Calibration vs real soil unverified | v1 cites published soil mech tables; v2 calibrates against Project Chrono DEM offline (kept out of v1 scope) |

## Success Criteria

**Primary (technical):**
1. DFH layer runs at ≥40 Hz physics step with `num_envs=2048` on a single A100/H100.
2. Sinkage profile vs published bevameter data within ±20% on 3 reference soil regimes (rigid/moderate/wet).
3. Hunter policy trained on DFH-stage3 achieves `mean_episode_length ≥ 18s` per CLAUDE.md furrow criteria.

**Secondary (research):**
4. DFH-trained policy outperforms rigid-trained policy on DFH-stage3 eval by ≥3× episode length (catastrophic-failure test from CLAUDE.md).
5. Cost-of-transport on DFH within 30% of rigid → soft-soil-aware gait emerged.

## Reviewer Score (self-est, NeurIPS reviewer lens)

**8/10** — clean problem, repo-grounded, novelty defensible, scope honest. Two main weaknesses: (a) no real-robot validation in v1 (paper would need to frame as "simulation-side modeling contribution + sim-to-sim transfer baseline"), (b) Bekker model breakdown at humanoid foot scale (Karpman 2020) is an honest caveat that needs a discussion section.
