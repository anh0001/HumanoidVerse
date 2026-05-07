# Experiment Plan — DFH for Hunter on Tilled Soil

**Date:** 2026-05-01
**Compute assumption:** single A100/H100 unless noted

## Phased Implementation

### Phase 0 — Foot Collider Audit (0.5 day)

- Inspect `humanoidverse/config/robot/hunter/hunter.yaml` foot collider geometry
- If point/sphere → switch to capsule/box of contact area ~0.014 m² (Hunter foot)
- Smoke test with `num_envs=16 headless=False +curriculum=stage0_plane`
- **Gate:** rigid-baseline gait unchanged after collider swap

### Phase 1 — Plastic Heightfield Buffer + Bekker Sink (3 days)

- Add `humanoidverse/simulator/isaacsim/dfh.py` with `DFHTerrainLayer` class
- Allocate `H_plastic[num_envs, H, W]` torch tensor
- Hook into per-step callback in `isaacsim.py:_pre_physics_step` (find correct hook)
- Implement Bekker pressure-sinkage update from `IsaacSim` contact reporter output
- Push `H_plastic` deltas back into PhysX heightfield each K=4 steps
- **Gate:** standing Hunter sinks 2-6mm into stage 1 soil, matches preset doc string

### Phase 2 — Slip-Sinkage + Bulldozing (2 days)

- Add C3, C4 to kernel
- Visual check in `headless=False`: dragging foot leaves a trench
- **Gate:** trench depth scales monotonically with `α_s`; bulldozed ridges visible

### Phase 3 — Anisotropic Friction (1 day)

- Store furrow direction `θ_f` per cell during furrow terrain generation (`isaacsim.py:442` extension)
- Per-contact tangential force injection with direction-dependent μ
- **Gate:** Hunter slides faster across furrows than along, observable in `headless=False`

### Phase 4 — Configs + Curriculum (1 day)

- Write `terrain/terrain_dfh_stage{1,2,3}_*.yaml`
- Write `domain_rand/DR_dfh_{soft,moderate,wet}.yaml`
- Write `curriculum/stage{1,2}_dfh.yaml`
- Write `rewards/loco/reward_hunter_dfh.yaml`
- **Gate:** dry run `--dry-run` for each stage passes Hydra composition

### Phase 5 — Train Block A: DFH-Stage1 Easy (~12-24 GPU-hr)

```bash
python humanoidverse/train_agent.py \
  +curriculum=stage1_dfh +robot=hunter/hunter +simulator=isaacsim \
  +exp=locomotion_soil +algo=ppo_soil \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  +rewards=loco/reward_hunter_dfh \
  +terrain=terrain_dfh_stage1_easy \
  +domain_rand=DR_dfh_soft \
  +checkpoint=logs/CurriculumStage0/latest/model.pt \
  num_envs=2048 headless=True \
  project_name=DFH experiment_name=Hunter_DFH_S1_Easy
```
**Promotion criteria (CLAUDE.md):** mean_episode_length ≥ 16s, term reward ≥ -0.05/s

### Phase 6 — Train Block B: DFH-Stage2 Medium (~12-24 GPU-hr)

Continue from Stage 1 checkpoint. Switch terrain + DR to `_medium`. Same gating.

### Phase 7 — Train Block C: DFH-Stage3 Full (~24-48 GPU-hr)

Continue from Stage 2. `terrain_dfh_stage3_full` + `DR_dfh_wet`. Target: episode length ~18-20s.

## Ablation Matrix (Phase 8 — paper-critical)

Each ablation: train-from-Stage0 → eval on terrain_dfh_stage3_full + DR_dfh_wet, 100 envs × 100 episodes via `sample_eps.py`.

| # | Ablation | Hypothesis | Metric |
|---|----------|------------|--------|
| A1 | Rigid-only training (existing furrows pipeline) → DFH eval | catastrophic failure | episode length ≪ DFH-trained |
| A2 | DFH w/o slip-sinkage (α_s=0) | slip-sinkage matters for tilled soil | DFH-trained advantage shrinks |
| A3 | DFH w/o anisotropy (μ_∥=μ_⊥) | anisotropy matters | episode length drops |
| A4 | DFH w/o bulldozing | bulldozing matters less than sinkage | smaller drop than A2 |
| A5 | Coarser heightfield (0.20m vs 0.10m) | speed/quality tradeoff | speed +X%, quality -Y% |
| A6 | Static `desired_base_height=0.6` (no soft-soil adapt) | dynamic height matters | early termination spike |

## Validation

| # | Test | Baseline | Pass criterion |
|---|------|----------|---------------|
| V1 | Sinkage realism | published bevameter sink curves (cite Hu 2023, Lim 2021) | within ±20% |
| V2 | DFH speed | rigid heightfield FPS | DFH ≥ 0.6× rigid at num_envs=2048 |
| V3 | Generalization | DFH-stage3 trained, eval on rigid | should still walk (≥80% rigid policy ep length) |
| V4 | Cost of transport | rigid-trained baseline | DFH-trained within 30% on DFH-stage3 |
| V5 | Catastrophic failure baseline | rigid-trained → DFH-stage3 | confirms gap (target: ≥3× ep length advantage for DFH-trained) |

## Run Order (first 5 commands to launch)

1. **Phase 0 collider audit** (cheap, blocking)
2. **Phase 1 standing-sink smoke test** (`num_envs=16 headless=False`)
3. **Phase 5 Stage 1 train** (long; kicks off after Phase 1-4 done)
4. **Phase 6 Stage 2** (after Stage 1 promotes)
5. **Phase 7 Stage 3** (after Stage 2 promotes)

Ablations A1-A6 launched in parallel after Phase 7 model exists.

## Compute Budget Estimate

| Block | GPU-hr |
|-------|--------|
| Phase 5 Stage 1 | 12-24 |
| Phase 6 Stage 2 | 12-24 |
| Phase 7 Stage 3 | 24-48 |
| Ablations A1-A6 (each ~Stage 3 cost) | 144-288 |
| Eval sweeps (sample_eps) | ~10 |
| **Total** | **~200-400 GPU-hr** |

Distill recipe: only run A1 + A3 + A5 if budget tight (paper-critical contrast set).

## Open Questions / Required Decisions

1. **Foot collider geometry** — capsule vs box? confirm Hunter foot CAD area
2. **Heightfield grid** — current `horizontal_scale=0.10`. Stays for stage 1; consider 0.05 for stage 3 detail
3. **Real soil calibration** — start from published tables or push to v2?
4. **Open-source release** — DFH layer as standalone IsaacLab extension or stays in HumanoidVerse?

## Tracker

See `refine-logs/EXPERIMENT_TRACKER.md` (to be created when first run launches).
