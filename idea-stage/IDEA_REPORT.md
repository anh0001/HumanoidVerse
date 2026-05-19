# Idea Discovery Report — Drastically Improving Hunter Locomotion (IsaacSim / PPO)

**Date:** 2026-05-19
**Direction:** Drastically improve Hunter biped locomotion performance (episode length, command tracking, terrain robustness) in HumanoidVerse. Pivot away from the DFH soil-realism line (confirmed null, RIGID 27.0 > DFH 26.2).
**Pipeline:** scite lit survey → idea brainstorm/filter → (next) novelty check → critical review → refinement
**Lit landscape:** `idea-stage/LITERATURE_LANDSCAPE.md`. Prior soil report archived at `IDEA_REPORT.soil-archive.md`.

## Executive Summary

The current Hunter pipeline is a **single-step MLP PPO with hand-tuned reward and no temporal memory, no privileged teacher, no periodic/symmetry structure**. The literature is unusually consistent that exactly these three absent ingredients are the highest-yield levers for blind biped locomotion. Recommended path: a **composable two-part stack** — (1) a privileged teacher + proprioceptive-history estimator trained with **Regularized Online Adaptation (ROA)** as the dominant contribution, multiplied by (2) a cheap **periodic-clock + bilateral-symmetry reward**. Both are IsaacSim-only, Hunter-only, orthogonal to and reusable by the existing terrain curriculum, and grounded in repo files that already exist (`leggedloco_obs_history_*`, asymmetric actor/critic obs, `ppo_modules.py`).

## Ranked Ideas

### 🏆 Idea 1 — Privileged Teacher + Regularized Online Adaptation (ROA) — RECOMMENDED (dominant)
**One-liner:** Add a privileged environment-latent encoder (critic-side: terrain/friction/mass/push params already available in sim) and a proprioceptive-history estimator trained *jointly* with the policy via ROA's single-stage regularized loss, so a blind Hunter infers terrain/dynamics online and adapts gait.
**Mechanism:** `PPOActor`/`PPOCritic` in `ppo_modules.py` gain a latent `z` head; teacher conditions on privileged vector, student estimator (TCN/GRU over the existing `short_history`, extended to ~50 steps of dof_pos/vel/actions/IMU) regresses `ẑ`; ROA regularizer aligns `ẑ→z` while policy trains on `ẑ` (no 3-phase RMA pipeline).
**Evidence:** Cheng 2023 ROA: *"≈30% improvement in the student policy ... order of magnitude better z regression"* vs RMA; A-RMA biped-proven on Cassie; Radosavovic 2024 history-conditioning dominates MLP.
**Repo fit:** asymmetric obs already exists (critic sees `base_lin_vel`); `leggedloco_obs_history_wolinvel.yaml` provides the history scaffold; single new module + loss term.
**Novelty (prelim):** RMA/ROA known on quadrupeds; **ROA on a HumanoidVerse Hunter biped across a soil/furrows curriculum in IsaacSim is unaddressed.** Needs Phase 3 deep check.
**Reviewer self-est:** 8/10. Risk: latent identifiability from biped proprio (mitigate: privileged-dropout, info bottleneck).

### 🥈 Idea 2 — Periodic-Clock + Bilateral-Symmetry Reward — RECOMMENDED (cheap multiplier)
**One-liner:** Add a Siekmann-style phase clock (swing/stance von-Mises reward) and a left/right mirror-symmetry penalty to `reward_hunter_locomotion.yaml` / `locomotion.py`.
**Evidence:** Mou 2023 — plain PPO without periodic reward *"fails periodic characteristics ... can lead to falls"*; symmetry → faster convergence; Wang 2022 — phase structure cut velocity-tracking error 60%.
**Repo fit:** pure reward addition next to existing `_reward_feet_air_time`; no architecture change. ~1 day.
**Novelty:** LOW standalone (known terms) but high *yield* and composes with Idea 1; ship as the Idea-1 reward set.
**Reviewer self-est:** 6/10 standalone, 8/10 as the Idea-1 companion.

### Idea 3 — Long Proprio-History Policy (GRU/TCN) — COMPOSABLE / enabler
Upgrade actor obs from single-step to a learned recurrent/TCN encoder over ~1–2 s history (Radosavovic 2024). This is effectively the *student backbone* of Idea 1; ship as Idea 1's architecture, or as a standalone ablation isolating "memory vs. teacher."
**Reviewer self-est:** 7/10; mostly an ablation arm of Idea 1.

### Idea 4 — AMP Style Reward from a Reference Gait
Replace the hand-tuned reward soup with an Adversarial Motion Prior discriminator trained on a short reference Hunter walk (or retargeted mocap). Lower cost-of-transport, natural gait (Escontrela 2022).
**Why deprioritized:** needs a reference trajectory source for Hunter; higher integration risk than Ideas 1–2; strong as a follow-up paper, not the first drastic win.
**Reviewer self-est:** 7/10 (future), 5/10 now given no curated reference set.

### Idea 5 — Navigation/Return Reward + Exploration on Furrows
Reframe furrows from velocity-tracking to a navigation/return objective with an exploration bonus + generalist→specialist finetune (Zhang 2024 IROS, ≥2.5 m/s on risky terrain).
**Reviewer self-est:** 6/10. Composes with Idea 1 for the furrows stage specifically.

### Idea 6 — Symmetry Data Augmentation in the PPO Buffer
Mirror left/right transitions when filling the rollout buffer (cheap sample-efficiency regularizer). Bundle into Idea 2.
**Reviewer self-est:** 5/10 standalone.

## Eliminated

| # | Idea | Reason |
|---|------|--------|
| E1 | Continue DFH soil-realism | Confirmed null after 4 review passes; reviewer recommends pivot |
| E2 | New sim engine / DEM | Out of scope (IsaacSim-only lock) |
| E3 | Exteroceptive height-map policy | Hunter sensor budget unclear; blind-proprio is the higher-evidence niche |
| E4 | Bigger MLP / hyperparam sweep | Literature: architecture *class* (memory/teacher) >> width; low ceiling |

## Recommended path

**Idea 1 (ROA teacher-student) + Idea 2 (periodic-symmetry reward) as one composable stack.** Idea 3 = Idea 1's backbone / ablation arm. Idea 5 = furrows-stage add-on. This maximizes expected drastic gain (memory + online adaptation + gait structure) while staying single-file-ish, IsaacSim-only, Hunter-only, and curriculum-compatible.

## Phase 3 — Novelty (scite)
No concurrent work pre-empts the specific stack. RMA/ROA/periodic-reward are individually known; **ROA + periodic-symmetry on a HumanoidVerse Hunter biped in IsaacSim across a soil/furrows curriculum, framed against a deformable-soil null, is unaddressed.** Honest framing: contribution is **integration + controlled empirical study**, not a new algorithm.

## Phase 4 — External critical review (Codex gpt-5.5, thread 019e3e82)
**Score 5.5/10. ALMOST (internal performance push); NO (publication) unless reframed + ablated.**

Key verdicts:
- "Drastic" gain is plausible **only if the current baseline is fragile** (falls often under soil/furrow/DR). If baseline already completes most episodes, expect 10–30% incremental. **Must measure baseline fragility first.**
- Novelty as "we combined these" is weak engineering. **Strongest reframe (adopted):** the DFH null is *evidence* that explicit soil modeling is not the bottleneck → test the sharper hypothesis *"robust blind biped locomotion on structured/deformable terrain is bottlenecked by online system identification, not terrain-model fidelity."* The null becomes a baseline arm, not an embarrassment.
- Highest-risk assumption: **short proprioceptive history contains enough signal for Hunter to infer terrain/dynamics early enough to act on it** (bipeds have less contact redundancy than quadrupeds).
- Mandatory 10-arm ablation matrix; ≥3 seeds; held-out terrain/friction/mass/push eval; matched env-steps & param counts.
- Minimum bar for "drastic": ≥2× episode length OR ≥50% fall-rate reduction on held-out soil/furrow, ≥25–30% tracking-error reduction under DR, beats history-only AND periodic-only ablations, latent demonstrably terrain-dependent and used.

## Refined deliverables
- `refine-logs/FINAL_PROPOSAL.md` (reframed: adaptation-vs-fidelity, DFH-null as arm)
- `refine-logs/EXPERIMENT_PLAN.md` (cheap falsification gate → ablation matrix)

## §0 GATE RESULT (executed 2026-05-19) — ✅ PASS, both halves
See `refine-logs/S0_RESULTS.md`.
- **§0-A fragility:** baseline 0 falls on plane (401-step episodes) vs **100% falls in <0.45 s** on soil_moderate/challenging + furrows_s2/s3 under DR. Drastic headroom is real and genuine (0→1 gap).
- **§0-B latent-identifiability:** an 8-step (~0.16 s) blind proprio window separates safe-vs-killer terrain at **99.9%**, 4-way regime at 80%+. ROA's core assumption empirically holds.
- En route, fixed 2 pre-existing repo bugs: `domain_rand_base.yaml` invalid-float typo; soil-terrain 1×1-tile capacity.

## Next steps
- [x] §0 falsification gate — PASSED → ROA is the right dominant idea (not the history-only fallback)
- [x] Implement Idea 2 (periodic-clock + symmetry reward) — DONE, smoke-validated (109 iters, no NaN; `rew_gait_phase`/`rew_penalty_gait_asymmetry` log correctly). `_reward_gait_phase` + `_reward_penalty_gait_asymmetry` in `locomotion.py`; soft scales + `gait_*` params in `reward_hunter_locomotion.yaml` (set both scales to 0 for the no-gait ablation arm)
- [x] Implement Idea 1 (privileged teacher + ROA estimator) — DONE, smoke-validated (462 iters, no NaN; `[ROA] latent_dim=16 actor_obs=234 critic_obs=237`). `PPOActorROA` in `ppo_modules.py`; `PPOROA(PPO)` in `agents/ppo/ppo_roa.py` (single-stage, symmetric reg, isolated subclass so baseline arms untouched); `+algo=ppo_roa` config; inference uses only actor_obs (eval-compatible). Also fixed broken `leggedloco_obs_history_wolinvel.yaml` (missing `short_history` scales) — needed for history ablation arms too
- [ ] Run §2 ablation matrix (A0–A10, seeds 1–3) per `refine-logs/EXPERIMENT_PLAN.md`
- [ ] `/run-experiment` for the matrix → `/auto-review-loop` to the claim bar
