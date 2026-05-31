# Fuzzy-Soil-on-Furrows — Consolidated Final Conclusion (Set A + Set B)

**Date:** 2026-06-01. Robot: Hunter. Sim: IsaacSim / IsaacLab 0.30.7 (PhysX).
Paper under test: `docs/wcci2026_hunter.pdf` (material curriculum + stance-gated slip
reward + contact-force cap + per-episode μ/kₙ sampling + a fuzzy Mamdani soil-difficulty
index). This document consolidates two studies into one paper-ready result; see
`fuzzy_soil_conclusion.md` (Set A detail) and `fuzzy_setB_result.md` (Set B detail).

## One-paragraph result

The recipe's **survival benefit replicates** (friction curriculum, ~8.85× episode
length over the v7 baseline). The paper's **headline slip-reduction does not replicate**,
and the **fuzzy soil model confers no demonstrated value** — neither as a passive
difficulty *label* (Set A) nor as an active curriculum-pacing *controller* (Set B).
All three negatives share **one root cause**: PhysX models only normal-direction
compliant contact, so the soil **support/stiffness axis (kₙ) is physically inert**,
and the 2-D fuzzy "soil state" collapses to a **1-D friction descriptor**. On that 1-D
axis the fuzzy machinery has no extra dimension to exploit, so it cannot beat a crisp
threshold. This is a **simulator-fidelity gap for tangential soil deformation**, not a
parameter we failed to set.

## Set A — fuzzy index as a descriptor (label)

| Test | Result |
|---|---|
| A/B paper recipe vs v7 control (3 seeds) | Survival **8.85× ep_len**, 63% fewer falls/m; slip **3.6× worse**, not better |
| Arm C hard-gate ablation | Slip gate **INERT** (ep_len 455 vs 452) — survival win is the friction curriculum |
| Eval noise (5 repeats) | **~8% CV ep_len**; 1-eval diffs untrustworthy |
| Arm D global compliant contact (softest kₙ=50 kN/m) | Slip **298±28 vs rigid 284±13, Welch t=1.01 → WASH** |
| Metric audit | Same slip definition as paper → non-replication not a metric artifact |
| Fuzzy d vs μ sweep (18 evals) | d tracks difficulty (Spearman ρ≈0.97, 4/4 metrics) **but adds nothing over raw μ** |

**Set A conclusion:** the fuzzy index is a valid *ordinal 1-D friction descriptor* but
adds no information beyond μ, because the stiffness axis is inert. The 2-D fuzzy model
and its support dimension are unvalidatable in this sim.

## Set B — fuzzy index as a curriculum-pacing controller (function)

Promoted the fuzzy machinery from a label to an active controller and tested the
non-negotiable: does fuzzy pacing beat a **crisp** baseline? Block-chained global
ground-friction curriculum, 3 arms × 3 seeds = 9 PPO-ROA runs (~24 h), matched compute,
μ-ladder {0.80,0.65,0.50,0.35} applied at construction (the Set-A-validated knob).
Arms differ only in the next-μ rule (calibrated identically from fixed block-0
mastery0=0.3875 → T=0.29): **fixed** (preset), **crisp** (hard threshold), **fuzzy**
(smooth accumulated Mamdani score).

**Held-out robustness AUC** (trapezoid of mean_ep_len/1000 over μ sweep {0.35..0.85},
2 reps × 100 eps; mean ± sd over 3 seeds):

| arm | AUC | per-seed | blocks→hardest |
|---|---|---|---|
| fixed | 0.643 ± 0.081 | 0.597, 0.596, 0.737 | 6 |
| crisp | 0.680 ± 0.075 | 0.706, 0.739, 0.596 | 3 |
| fuzzy | 0.724 ± 0.152 | 0.805, 0.818, 0.549 | 4 |

Contrasts (Welch t, n=3): **fuzzy vs crisp diff 0.044, t=0.45 → not significant**;
adaptive(crisp+fuzzy) vs fixed 0.702 vs 0.643, t=0.91 → not significant.

**Set B conclusion:** the curriculum mechanism works (arms paced genuinely differently;
training survival fell monotonically as μ hardened, confirming a real difficulty axis),
but **fuzzy ≈ crisp** (clean scoped negative on the fuzzy-specific claim), and
**adaptive vs fixed is only a non-significant trend** (fuzzy's higher mean is inflated by
one low-outlier seed, 0.549; its sd is 2× the others). No fuzzy-specific value, exactly
as Set A predicted.

## Unified conclusions

1. **Replicates:** the friction-curriculum survival benefit (robust, ≫ noise).
2. **Does not replicate:** the >95% slip-reduction headline (ruled out gate, eval noise,
   compliance, metric mismatch).
3. **No fuzzy-specific value, twice over:** the fuzzy index adds nothing over raw μ as a
   label (Set A) and its smooth pacing does not beat crisp thresholds as a controller
   (Set B).
4. **Single root cause:** PhysX normal-only compliant contact → kₙ inert → "soil" is 1-D
   friction → the 2-D fuzzy model has no second axis to add information on.
5. **What would change the verdict:** a simulator with tangential soil-deformation
   physics (Bekker/terramechanics or a custom tangential contact term) so both fuzzy
   inputs are physically meaningful. Framed as a sim extension, not a continuation.

## Threats to validity (do not overclaim)

- This is non-replication of the slip *mechanism* in **our simulator**, NOT a disproof of
  the paper on real/deformable soil. Foreground this.
- Set B: n=3 seeds, 2 eval reps; **no contrast is statistically significant** — treat
  AUC gaps as trends. Carries the ~8% eval noise. Adaptive/fuzzy behavior depends on the
  T=0.29 calibration (pre-registered from block-0, applied identically to both arms; a
  T-sweep is a deliberately-deferred appendix, not done post-hoc).
- Unreplicated paper deviations (both sets): PPOROA + privileged teacher (paper used
  vanilla PPO), clip ε=0.1 vs 0.2, fuzzy layer logging-only in Set A. None expected to
  affect the slip mechanism or the 1-D collapse.

## Artifacts

Set A: `fuzzy_soil_conclusion.md`, `fuzzy_setA_result.md`, `armC_hardgate_preregister.md`,
`compliant_contact_feasibility.md`; scripts `scripts/paper_fuzzy_soil/`; data
`docs/experiments/data/` (results.csv, eval_noise.csv, fuzzy_sweep*.csv, fuzzy_analysis.txt,
charts).
Set B: `fuzzy_setB_plan.md`, `fuzzy_setB_result.md`; scripts `scripts/setB_fuzzy_curriculum/`;
data `docs/experiments/data/setB/` (setB_analysis.txt, calib.env, 9 eval CSVs, decisions,
setB_summary.png). Raw run logs under `logs/FuzzySoilFurrowsSetB/` (gitignored).
