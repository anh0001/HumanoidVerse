# Fuzzy-Soil-on-Furrows — Final Consolidated Replication Report

**Date:** 2026-05-30 (final). Robot: Hunter. Sim: IsaacSim / IsaacLab 0.30.7 (PhysX).
Paper under test: `docs/wcci2026_hunter.pdf` (material curriculum + stance-gated
slip reward + contact-force cap, per-episode sampling of friction μ and
compliant-contact kₙ/cₙ, plus a fuzzy soil-state difficulty index).
Paper headline: >95% reduction in stance-phase slip.

## One-line verdict

**Replicated:** the friction curriculum's survival benefit, and the fuzzy index as a
1-D (friction) difficulty descriptor. **Not replicated / not showable in this sim:**
the headline slip-reduction claim and the support (stiffness) axis of the fuzzy model
— both blocked by the same root cause: PhysX lacks tangential soil-deformation physics.

## Plain-language summary (the "fuzzy" picture)

The paper describes soil with two fuzzy dials: **traction** (grip, set by friction μ)
and **support** (firm-vs-soft, set by contact stiffness kₙ — how much a foot sinks in
and is held). Its difficulty index combines them as `min(traction, support)`.

- We could move the **traction** dial freely. Training the robot across grippy→slippery
  soil (the friction curriculum) **worked strongly**: ~8.85× longer survival than the
  v7 baseline, far above the measured noise floor. **This replicates.**
- We could **not** truly move the **support** dial. PhysX compliant contact softens the
  *vertical* landing force but adds **no lateral grip** — a foot does not sink into the
  furrow wall and resist sideways slide. The paper's slip reduction depends on exactly
  that sinkage-grip. **So the slip claim does not replicate** — and we showed it's not
  the reward gate, not eval noise, and not (within the paper's stiffness range) contact
  compliance. It's a **simulator-fidelity gap for tangential soil deformation**, not a
  parameter we forgot to set.

## Evidence chain (each step gated the next)

| # | Test | Result |
|---|------|--------|
| A/B | Paper recipe vs v7 control, 3 seeds | Survival **8.85× ep_len**, 63% fewer falls/m; slip **3.6× worse**, not better → PARTIAL |
| C | Hard-gate ablation (isolate soft slip gate) | Gate **INERT** — ep_len 455 vs 452; survival win is the **friction curriculum**, not the gate |
| noise | Same checkpoint, 5 repeated evals | Eval noise **~8% CV ep_len, ~4–10% slip**; one swing hit 23%. 1-eval diffs untrustworthy; training-seed SD (~82) ≫ eval SD (~34) |
| D | Global compliant contact, paper's softest kₙ=50 kN/m (5-eval confirm) | Slip **298.1 ± 28.3** vs rigid **284.1 ± 13.1**, Welch t=1.01 → **WASH** (single eval of 259 was a low-tail draw) |
| audit | Our slip metric vs paper's | **Same definition** (stance-phase tangential foot slip, m/100 m, Fₙ≈1 N gate); only world-horizontal vs surface-tangent projection differs (~1% on shallow furrows) → non-replication is **not a metric artifact** |

| Set A | Fuzzy index d vs measured difficulty, μ sweep 0.35–0.85, 3 evals/μ (18 evals) | d tracks difficulty (Spearman ρ≈0.96–0.97, 4/4 metrics) **but adds nothing over raw μ** — the stiffness axis is inert, so d collapses to a monotone transform of friction |

**Built-in positive control (Arm D vs Set A):** the friction (Set A) and compliance
(Arm D) results together form a clean control — varying **friction** moves slip strongly
and monotonically (ρ≈0.97), while varying **compliance** does nothing (Welch t=1.01). The
simulator *can* express traction-driven slip but *cannot* express deformation-driven slip,
exactly the fidelity gap that explains the non-replication. A dedicated synthetic
lateral-friction control was judged unnecessary (and PhysX has no clean anisotropic-friction
knob); this contrast supplies the evidence.

Per the pre-registered Stage-0 rule (`armC_hardgate_preregister.md`,
`compliant_contact_feasibility.md`): since the *softest* global compliance is already a
wash, per-env kₙ/cₙ randomization (which only samples *stiffer* values) cannot rescue
the slip claim → the ~1–2 day per-env build is **not justified**. Not built.

## What we conclude

1. **The recipe's survival benefit transfers** and is robust (8.85× ep_len ≫ 8% noise).
   The driver is the **friction curriculum**; the soft slip gate and the contact-force
   cap are inert on this stack.
2. **The slip-reduction claim does not replicate** in HumanoidVerse/Hunter/IsaacSim-PhysX.
   Ruled out: reward gating (C), eval noise (sweep), contact compliance in the paper's
   range (D), metric mismatch (audit).
3. **Most likely cause:** PhysX rigid/compliant-*normal* contact does not model the
   tangential soil deformation / sinkage-grip the paper's mechanism relies on.
4. **The fuzzy difficulty index is a valid 1-D descriptor, not a validated 2-D model.**
   It orders soil by measured difficulty on the friction axis (ρ≈0.97), but a
   d-vs-raw-μ baseline shows it adds no information over friction alone, because the
   support/stiffness axis has no physical effect in PhysX. The fuzzy *structure*
   (min-rule, Easy/Mod/Hard binning) is unvalidated here — it collapses to a monotone
   transform of μ. Validating it needs a sim where both fuzzy inputs are physical.

## Threats to validity (do not overclaim)

- **This is non-replication of the slip *mechanism* in our simulator, NOT a disproof of
  the paper on real/deformable soil.** Foreground this.
- Absolute slip magnitudes are not comparable across robots/sims; only the **relative**
  within-setup A-vs-B comparison is meaningful — and it showed no slip benefit.
- 1 training seed for arms C and D (eval-noise controlled, but training-seed variation
  not). The decision is robust because effect sizes are either huge (curriculum) or
  null (gate, compliance), but a 3-seed confirmation would tighten the null arms.
- Other unreplicated paper deviations: PPOROA+privileged teacher (paper used vanilla
  PPO), PPO clip ε=0.1 vs 0.2, fuzzy interpretation layer logging-only (zero policy
  effect). None expected to affect the slip mechanism.

## If pursued further (not recommended now)

The only path that could change the conclusion is adding **tangential deformation
physics** PhysX lacks — e.g. a custom tangential-resistance/sinkage contact term, or a
Bekker-style soil model. (A synthetic lateral-friction *positive control* was considered
and rejected — see "Recommended next steps"; the Arm-D-vs-Set-A contrast already supplies
that evidence, and PhysX has no clean anisotropic-friction knob.) Neither rescues the
paper recipe as-is.

## Recommended next steps (per Codex, 2026-05-30)

- **STOP experimentation. This report is the deliverable.** The causal story is complete
  and verified.
- **NOT recommended:** the synthetic lateral-friction positive control (low evidential
  weight, no clean PhysX knob; the Arm-D-vs-Set-A contrast already covers it).
- **Future project (not a next step):** if the group wants to study soft-soil locomotion
  seriously, implement tangential soil-deformation physics (Bekker/terramechanics or a
  custom tangential contact term). That would let both the slip claim *and* the 2-D fuzzy
  model be tested — framed as a simulator extension, not a continuation of this study.
- Driving the fuzzy index (adaptive curriculum / reward shaping) is premature while the
  stiffness axis is inert — it would only test friction-aware shaping, not the fuzzy model.

## Process note

Two intermediate result files were corrupted/fabricated mid-session and retracted; every
number in this report was re-run and verified directly against source eval logs (raw →
aggregate → analysis chains checked cell-by-cell). Invalid runs are archived under
`logs/FuzzySoilFurrows/_invalid_noDR_sweep/` and `_singleshot_n1/` and excluded.

## Artifacts

`docs/experiments/`: `fuzzy_soil_furrows_plan.md`, `fuzzy_soil_furrows_result.md`,
`armC_hardgate_preregister.md`, `compliant_contact_feasibility.md`,
`fuzzy_setA_result.md`, this file.
Data: `logs/FuzzySoilFurrows/results.csv`, `eval_noise.csv`,
`armC_analysis_seed1.txt`, `armD_compliance_seed1.txt`,
`fuzzy_sweep.csv`, `fuzzy_sweep_repeats_raw.csv`, `fuzzy_analysis.txt`,
`fuzzy_validation.png`.
