# Fuzzy Set-B — Adaptive-vs-Fixed Friction Curriculum (Result)

**Date:** 2026-05-31. Branch: `feature/fuzzy-adaptive-curriculum`. Robot: Hunter.
Sim: IsaacSim / IsaacLab (PhysX). Design + scope: `fuzzy_setB_plan.md`. Builds on
`fuzzy_soil_conclusion.md` (Set A). All numbers below are read from
`logs/FuzzySoilFurrowsSetB/setB_analysis.txt` and the per-run manifests — none typed
from memory.

## Question

Promote the paper's fuzzy machinery from a passive soil-difficulty *label* to an
active curriculum-pacing *function*, and test the user's non-negotiable: does fuzzy
pacing beat a **crisp** baseline? (Fuzzy-vs-fuzzy proves nothing.)

## Method (block-chained global friction curriculum)

3 arms × 3 seeds = **9 PPO-ROA runs**, each an 8-block checkpoint-chained curriculum
(752 iters total, matched), warm-started from the same checkpoint, optimizer resumed.
The curriculum drives the **global ground-traction μ** at construction (Set A's
proven knob; per-env / live-mutation paths are infeasible/unproven here — see plan).
Arms differ ONLY in the next-μ rule on a shared μ-ladder {0.80,0.65,0.50,0.35} and a
shared performance signal (mastery = mean ep_len / 1000):

- **fixed** — preset schedule, advance every 2 blocks (μ-path `[.8,.8,.65,.65,.5,.5,.35,.35]`).
- **crisp** — hard gate: advance iff mastery ≥ T (front-loads difficulty; μ-path `[.8,.65,.5,.35,.35,...]`).
- **fuzzy** — smooth accumulation of a Mamdani advance-score (intermediate pacing; μ-path `[.8,.8,.65,.5,.35,...]`).

Constants **calibrated from data, not guessed**: T and the fuzzy breakpoints were set
from the fixed arm's block-0 (μ=0.80) mean ep_len averaged over its 3 seeds
(mastery0 = 0.3875 → **T = 0.29**, fuzzy breaks 0.17/0.29/0.41; `calib.env`). Identical
constants for crisp & fuzzy.

**Held-out metric:** robustness AUC = trapezoid of mean(ep_len)/1000 over a μ sweep
{0.35,0.45,0.55,0.65,0.75,0.85} (2 reps × 100 eps each), normalized by μ-range.
Higher = more robust across difficulties. Secondary: blocks-to-hardest-rung (pacing).

## Results (held-out robustness AUC, mean ± sd over 3 seeds)

| arm   | AUC mean | AUC sd | per-seed AUC          | blocks→hardest (pacing) |
|-------|----------|--------|-----------------------|--------------------------|
| fixed | 0.6433   | 0.081  | 0.597, 0.596, 0.737   | 6  (slowest)             |
| crisp | 0.6800   | 0.075  | 0.706, 0.739, 0.596   | 3  (fastest)             |
| fuzzy | 0.7240   | 0.152  | 0.805, 0.818, 0.549   | 4  (intermediate)        |

**Contrasts (Welch t; n=3, small):**
- **adaptive (crisp+fuzzy) vs fixed:** 0.702 vs 0.643, **t = 0.91** → not significant.
- **fuzzy vs crisp:** 0.724 vs 0.680, diff = 0.044, **t = 0.45** → not significant.

## Verdict (honest, scoped)

1. **Fuzzy ≈ crisp — the fuzzy-specific claim returns a clean (scoped) NEGATIVE.**
   Fuzzy pacing is statistically indistinguishable from crisp-threshold pacing
   (diff 0.044, t=0.45). This is exactly the Set A prediction: with the support axis
   (kₙ) physically inert in PhysX, difficulty collapses to ~1-D friction, so the fuzzy
   machinery (smooth interpolation over Easy/Mod/Hard bins) has no extra dimension to
   exploit and adds no measurable value over hard thresholds. **The non-negotiable
   fuzzy-vs-crisp test is satisfied and fuzzy does not win.**

2. **Adaptive vs fixed — a non-significant trend, NOT demonstrated.** Both adaptive
   arms have higher mean AUC than fixed (0.70 vs 0.64) and reach hard soil sooner
   (3–4 blocks vs 6), but at n=3 with this variance the difference is within noise
   (t=0.91). Fuzzy's higher mean (0.724) is inflated by 2 strong seeds (0.805, 0.818)
   against 1 low outlier (0.549) — its sd (0.152) is twice the others'. **We do not
   claim adaptive beats fixed.** Any hint of benefit would be *friction-aware*
   curriculum pacing, not *fuzzy* reasoning.

3. **The curriculum mechanism works as intended.** The three arms paced genuinely
   differently (decision trails + μ-paths confirm: crisp front-loads, fuzzy is
   smoother, fixed is even), and survival fell monotonically as μ hardened during
   training — so the global construction-time μ knob is a real, working difficulty axis.

## Threats to validity (do not overclaim)

- **n=3 seeds, 2 eval reps; nothing is statistically significant.** The honest
  headline is *no arm is distinguishable from another*. Treat AUC differences as
  trends, not effects.
- AUC carries the known ~8% eval noise (Set A); 2 reps × 6 μ averages it only partly.
- **Threshold sensitivity:** the adaptive/fuzzy behavior depends on T (=0.29 here,
  calibrated from block-0). A different T (e.g. steady-state-based) would make crisp
  stall on hard soil. We pre-registered the block-0 rule and applied it identically
  to both adaptive arms; a T-sweep is out of scope.
- Same unreplicated paper deviations as Set A (PPOROA+teacher, clip ε=0.1, etc.).
- Optimizer state resumes across blocks; restart effects are shared by all arms and
  cancel in the contrast (do not compare these AUCs to an uninterrupted single run).

## Bottom line

Set B promoted the fuzzy index to a curriculum-pacing function and tested it against a
crisp baseline as required. **Result: fuzzy ≈ crisp (no fuzzy-specific value), and
adaptive-vs-fixed is an underpowered non-significant trend.** This is consistent with
and reinforces the Set A conclusion: in PhysX, where the soil support axis is inert,
the fuzzy model collapses to a 1-D friction descriptor and confers no advantage over
crisp thresholds — whether used as a label (Set A) or as an active controller (Set B).
A genuine fuzzy-specific result would require a simulator with real tangential
soil-deformation physics so that both fuzzy inputs are physically meaningful.

## Artifacts

Code: `scripts/setB_fuzzy_curriculum/` (`curriculum_controller.py`, `run_block_chain.py`,
`run_arm.sh`, `run_all_seeds.sh`, `run_phase2.sh`, `eval_heldout.sh`, `analyze_setB.py`).
Data: `logs/FuzzySoilFurrowsSetB/` — `setB_analysis.txt`, `calib.env`,
`eval/setB_<arm>_seed<seed>.csv` (9 held-out sweeps),
`setB_<arm>_seed<seed>/{decisions.csv,manifest.json}` (μ-paths + pacing).
Campaign wall-clock ≈ 23.7 h (9 chained runs + 9 held-out sweeps, one GPU).
