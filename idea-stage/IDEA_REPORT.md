# Idea Discovery Report — Hunter Biped on Tilled Soil (IsaacSim)

**Date:** 2026-05-01
**Direction:** Modify HumanoidVerse so Hunter robot trains/walks on realistic tilled soil in IsaacSim
**Pipeline:** scite lit survey → idea brainstorm → filter → novelty check → critical review → refinement
**Pilots:** SKIPPED per user (paper-only validation)

---

## Executive Summary

Repo currently models tilled soil as a **rigid heightfield** (perlin/furrows geometry + friction patches + static `contact_stiffness`). No soil deformation, no sinkage dynamics, no slip-sinkage coupling, no anisotropic furrow behavior. This is exactly what makes Hunter policies trained here brittle in real tilled fields.

Lit confirms the niche is open: no published RL paper trains a biped on physically-deformable agricultural soil. Closest references are Choi 2023 (quadruped on sand, custom sim) and Singh 2024 (biped on rigid uneven w/ contact-stiffness DR, not IsaacSim). The dominant scalable pattern in terramechanics is **offline DEM/SPH → online calibrated parametric model** (Hu 2023, Buse 2023, Alvarado 2022). Nobody has wired this into IsaacSim.

**🏆 Recommended path:** ship a **Deformable Furrowed Heightfield (DFH)** contact layer for IsaacSim — GPU-resident plastic heightfield with Bekker-Wong pressure-sinkage + Janosi-Hanamoto shear + furrow-anisotropic friction + slip-sinkage coupling. Train Hunter on it via the existing 3-stage curriculum. Backup idea: privileged-encoder soil-ID head (RMA-style) on top of DFH for online soil adaptation.

---

## Ranked Ideas

### 🏆 Idea 1 — Deformable Furrowed Heightfield (DFH) — RECOMMENDED

**One-liner:** Replace IsaacSim's rigid heightfield contact with a GPU-resident plastic heightfield that deforms under foot load, models slip-sinkage coupling, and exposes furrow anisotropy.

**Mechanism:**
- Per-step heightfield update: contact patches push grid cells down by `Δz = f(p, φ, c, μ)` (Bekker pressure-sinkage); plastic deformation persists.
- Bulldozing: lateral cell mass redistributes to adjacent cells when foot drags.
- Slip-sinkage: tangential slip velocity increases sink depth this step (Wong-Reece).
- Anisotropic friction: μ_along_furrow ≠ μ_across_furrow, depth-of-step modulates.
- All on GPU as Warp/torch kernels operating on the existing `mesh_type: heightfield` buffer at `humanoidverse/simulator/isaacsim/isaacsim.py:426-450`.

**Why it works:**
- Single-file plumbing inside `isaacsim.py`. No new sim engine. Stays IsaacSim-only.
- Reuses existing `terrain_furrows_stage{1,2,3}` configs as initial geometry.
- Falls back to rigid (deformation off) for ablation and Stage 0 plane.

**Novelty:** HIGH. No biped-on-deformable-soil RL paper. Choi 2023 is closest but quad + custom sim, no public IsaacSim port.

**Differentiation vs Choi 2023:**
- Biped (Hunter) not quad
- Furrowed agricultural soil + anisotropy, not isotropic sand
- IsaacLab-native (open release fills gap noted by survey papers)
- Calibratable from open bevameter/DEM data (Hu 2023 workflow)

**Reviewer score (self-est):** 8/10. Risk: GPU-Bekker speed at 2048 envs unknown; fall back to coarser heightfield grid + cached deformation kernels.

**Pilot signal:** N/A (skipped per user). Paper-only validation: lit gap + repo gap match cleanly.

---

### Idea 2 — Soil-ID Adaptive Policy (RMA-for-Biped) — BACKUP / COMPOSABLE

**One-liner:** Privileged-encoder student-teacher: teacher sees true soil params (cohesion c, friction angle φ, water content w), student infers them online from proprio history and adapts gait.

**Mechanism:**
- Stage A teacher: actor-critic conditioned on `[c, φ, μ_static, μ_dynamic, contact_stiffness]` privileged vector.
- Stage B student: distill via DAgger/RMA. Student input = proprio history (50 steps of joint states + IMU + contact forces). Output = action + soil-param embedding.
- Reuse existing `+algo=ppo_soil` and `obs/leggedloco_obs_singlestep_withlinvel`. Add encoder head in `agents/modules/ppo_modules.py`.

**Composes with Idea 1.** Adds online soil adaptation on top of DFH.

**Novelty:** MEDIUM. RMA pattern known; biped+soil application novel.

**Reviewer score:** 7/10. Risk: identifiability of soil params from biped proprio is empirically open.

---

### Idea 3 — Anisotropic Furrow Contact (cheap quick-win) — COMPOSABLE

**One-liner:** Extend `patchy_friction` to direction-dependent shear + sinkage along furrow orientation. No new physics engine.

**Mechanism:**
- Per-cell store furrow direction θ; at contact compute `μ_eff = μ⊥·cos²(α) + μ∥·sin²(α)` where α = stride heading vs furrow.
- Couple to per-cell sinkage depth (deeper in trough cells, ridges = higher).

**Subset of Idea 1.** Ship as Phase 1 of DFH if full DFH delayed.

**Novelty:** LOW-MEDIUM (anisotropy known in terramechanics, not in biped RL configs).

**Reviewer score:** 6/10 standalone, 8/10 as DFH staging.

---

### Idea 4 — DEM-Calibrated Parametric Foot-Soil Wrench

**One-liner:** Run offline Project Chrono DEM for Hunter foot on tilled soil samples (varying c, φ, w). Fit Bekker+Janosi parameters via Bayesian inference (Hu 2023 virtual bevameter). Apply as augmented contact wrench in IsaacSim.

**Why deprioritized:** Requires standing up Chrono pipeline + multi-day DEM runs. Out of scope for IsaacSim-only restriction unless explicitly approved.

**Reviewer score:** 7/10 as future work; 4/10 as immediate target given scope lock.

---

### Idea 5 — Energy/COT-Aware Reward for Soft Soil

**One-liner:** Add cost-of-transport, foot-exit-velocity, and contact-dwell penalties — proxies for energy lost to plastic deformation.

**Reviewer score:** 5/10 standalone (reward shaping). Bundle into Idea 1 reward set.

---

### Idea 6 — Furrow-Aware Look-Ahead Exteroception

**One-liner:** Augment proprio with low-res local sinkage map; let policy plan stride to land on ridges not troughs (biomechanics motivation: Darici & Kuo 2023).

**Reviewer score:** 6/10. Hunter sensor budget unclear; may need synthetic exteroceptive obs.

---

## Eliminated Ideas

| # | Idea | Reason |
|---|------|--------|
| E1 | Two-tier sim distillation (Choi-engine teacher → DFH student) | Needs second sim engine; out of scope |
| E2 | DEM-validated automated curriculum | Same — Chrono dep |
| E3 | Ankle-mounted tactile pseudo-sensor | Requires new sensor modeling; fold into Idea 2 instead |
| E4 | Real bevameter calibration | No real bevameter access stated; future work |
| E5 | Power-throttling adaptive-frequency policy | Subsumed by Idea 5 reward shaping |

---

## Refined Plan

See `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/EXPERIMENT_PLAN.md`.

**Top idea:** Idea 1 (DFH) shipped first. Idea 2 (RMA Soil-ID) layered on top once DFH stable. Idea 3 ships as Stage 1 of DFH rollout.

---

## Next Steps

- [ ] Implement DFH GPU kernel in `humanoidverse/simulator/isaacsim/isaacsim.py` (extend furrow handler at L442)
- [ ] Add `terrain_dfh_stage{1,2,3}` configs (subset of existing furrow stages w/ deformation toggles)
- [ ] Add `domain_rand=DR_dfh_{rigid,moderate,wet}` for Bekker param ranges
- [ ] Train Hunter via `+curriculum=stage1_dfh +simulator=isaacsim +robot=hunter/hunter`
- [ ] Validate vs rigid baseline on (a) sinkage realism, (b) policy COT, (c) zero-shot transfer to harder DFH params
- [ ] (Phase 2) Add RMA soil-ID head, retrain
- [ ] (Phase 3) Open-source DFH terrain layer as IsaacLab extension
