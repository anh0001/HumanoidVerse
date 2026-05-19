# Final Proposal — Implicit Online Adaptation for Blind Hunter Locomotion (IsaacSim)

**Date:** 2026-05-19
**Repo:** HumanoidVerse (Hunter only, IsaacSim only, PPO)
**Reviewer-anchored:** Codex gpt-5.5 thread 019e3e82 (5.5/10 → reframed). Prior soil proposal archived `FINAL_PROPOSAL.soil-archive.md`.

## Problem Anchor (frozen)

The current Hunter controller is a single-step MLP PPO with no memory, no privileged teacher, and no gait structure. The prior DFH line proved that **explicit deformable-soil modeling does not improve performance** (clean null: rigid-trained ≥ soil-modeled). This reframes the question: *for blind Hunter locomotion on structured/deformable terrain, is the bottleneck simulator/terrain fidelity, or the policy's inability to infer terrain & dynamics online?*

## Hypothesis (the research story)

> **Robust blind biped locomotion on structured/deformable terrain is bottlenecked by online system identification, not by terrain-model fidelity.** Implicit online adaptation (privileged-teacher + proprioceptive-history estimator via Regularized Online Adaptation) plus minimal gait regularization will drastically outperform both the explicit-soil-modeling line and the current single-step-MLP PPO baseline, under matched compute and curriculum.

The DFH null is no longer an embarrassment — it is the **explicit-modeling control arm**.

## Method

**M1 — Privileged teacher.** Asymmetric actor-critic (already partially present): critic + a teacher latent head consume a sim-privileged vector `e` = {terrain class, friction, restitution, added base mass, push impulse, soil/DFH params}. Teacher policy `π_T(·|o, z)`, `z = E_priv(e)`, `z ∈ R^{8..16}` with an information bottleneck.

**M2 — Proprioceptive-history estimator.** TCN (RMA/ROA-style 1-D CNN) or GRU over ~50-step history of {dof_pos, dof_vel, actions, base_ang_vel, projected_gravity}. Reuses `leggedloco_obs_history_wolinvel.yaml` scaffold (extend length 5→~50). Produces `ẑ`.

**M3 — Regularized Online Adaptation (ROA, Cheng 2023).** Single-stage joint training: policy trains on `ẑ`; regularizer `L_ROA = ‖sg[z] − ẑ‖²` (+ optional symmetric term) aligns student↔teacher without RMA's 3 phases. No DAgger, no separate distillation run.

**M4 — Periodic-symmetry reward (cheap multiplier).** Add to `reward_hunter_locomotion.yaml` / `locomotion.py`:
- Siekmann-style phase clock: per-foot von-Mises swing/stance reward, phase `φ ∈ [0,1)` advanced by a learned or fixed period; **soft weight** so it cannot override terrain adaptation (reviewer risk #4).
- Bilateral symmetry penalty on the mirror-map of (obs, action); **soft weight** (uneven terrain may need asymmetry).

## What's Out (deliberately)
- No new sim engine; no DEM; no real robot (sim-only, framed as a controlled IsaacSim study).
- No g1/h1/isaacgym/genesis. Hunter + IsaacSim only.
- No exteroception — the entire point is blind proprioceptive adaptation.
- Periodic/symmetry weights kept soft and ablated, never hard constraints.

## Risks → Mitigations (reviewer-aligned)

| Risk | Mitigation |
|---|---|
| **Latent unidentifiable from biped proprio** (highest risk) | Run the <2 GPU-h falsification gate (EXPERIMENT_PLAN §0) BEFORE building ROA. Kill/retarget if it fails. |
| Baseline already strong → gains only incremental | Measure baseline fragility first (fall rate / ep_len on soil+furrows under DR); only claim "drastic" if baseline is fragile and gap closes per the bar. |
| ROA collapse (student ignores teacher) | Info-bottleneck on `z`, privileged-dropout, monitor `‖z−ẑ‖` and z-usage (zero-`ẑ` ablation must hurt). |
| Clock reward fights terrain curriculum | Soft weight + ablation arm with clock off; allow period to adapt. |
| Symmetry penalty wrong on uneven terrain | Soft weight + per-stage schedule; ablate. |
| Confounded attribution | Mandatory 10-arm matrix isolating memory vs teacher vs clock vs symmetry. |

## Success Criteria (minimum bar for "drastic", reviewer-set)
1. ≥2× episode length **OR** ≥50% fall-rate reduction on **held-out** soil/furrow terrain vs current baseline.
2. ≥25–30% command-tracking-error reduction under held-out friction/mass/push DR.
3. Full stack beats **history-only** AND **periodic-only** ablations (attribution proven).
4. Learned `ẑ` is demonstrably terrain/dynamics-dependent and causally used (zero/shuffle-`ẑ` degrades).
5. No major energy / unnatural-gait regression vs baseline.
6. Beats the DFH explicit-modeling arm on the same held-out eval.

## Reviewer Score (post-reframe self-est)
7/10 as a controlled empirical study with the reframe + full ablation matrix + falsification gate. Was 5.5/10 as "combine known methods." Remaining ceiling risk: latent identifiability (gated cheaply) and sim-only external validity (scope honestly).
