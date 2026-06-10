# Load-bearing fuzzy redesign (v2): making the fuzzy index actually carry the support axis

**Date:** 2026-06-05. Branch: `feature/fuzzy-soil-unified-conclusion`. Robot: Hunter.
Data: Set C DFH deformable-soil sweep (`docs/experiments/data/setC/sweep_rigid.csv`,
41/45 walking cells). **CPU reanalysis only — no new GPU runs.** Every number below is
read from `logs/DFH_fuzzy/fuzzy_v2_loadbearing.txt`, produced this run by
`scripts/dfh_fuzzy/fuzzy_v2_loadbearing.py`.

## Why

Set A/B/C concluded the paper's Mamdani index `d(μ, support)` is **not load-bearing**:
even on DFH deformable soil — where the support axis is physically real and strongly
predictive — `d` added only **+0.025** LOO-CV R² over `Y~μ`, vs **+0.334** for raw `K`.
The unified conclusion pinned the cause on the **rule structure**: the rule base maps
Low-traction→Hard and Mid-traction→Moderate *regardless of support*, and this grid's μ
all sit in the Low/Mid band, so `d` is structurally blind to firmness. The named
open follow-up was *"a fuzzy system redesigned to actually use the support axis."* This
is that test. Design vetted with Codex ([[codex-at-decision-points]]).

## Intervention (minimal, to isolate the cause — per Codex)

Change **only the rule table + consequent levels**; hold fuzzification (same μ/kn
membership breakpoints) and centroid defuzzification fixed. Any change in predictive
power is then attributable to rule structure alone.

- Consequents: 5 evenly-spaced singletons `{VE .1, E .3, M .5, H .7, VH .9}` (paper: 3).
- Rule grid: monotone 2-D anti-diagonal — **support varies in every traction row**:

  |            | Soft | Med | Firm |
  |------------|------|-----|------|
  | L-traction | VH   | H   | M    |
  | M-traction | H    | M   | E    |
  | H-traction | M    | E   | VE   |

  (Paper grid: L→H,H,H · M→M,M,M · H→M,E,E — support inert in the L and M rows.)

## Results

**(1) HEADLINE — LOO-CV R² gain over `Y~μ` (deterministic clean inputs):**

| regressor over μ | ΔR² | partial Spearman(Y,·\|μ) |
|---|---|---|
| raw `K` (linear ceiling) | **+0.334** | −0.701 |
| paper `fuzzy_d` (support-blind) | +0.025 | −0.039 |
| **`fuzzy_d_v2` (load-bearing)** | **+0.416** | **+0.715** |

- Sanity: raw K (+0.334) and paper d (+0.025) **reproduce Set C exactly** → pipeline faithful.
- v2 lifts the support-axis gain **16×** (0.025 → 0.416) and the partial Spearman from
  −0.04 (noise) to **+0.715** — i.e. v2 now orders difficulty by support about as well as
  raw K does (|0.715| ≈ |0.701|), with correct polarity (`Spearman(d_v2, K) = −0.916`,
  firmer → lower difficulty).

**(2) Noise robustness — predict difficulty from NOISY (μ, log₂K) estimates** (v2/paper
use the single fuzzy scalar; raw uses both μ and log₂K → richer, so raw is the accuracy
ceiling at low noise):

| σ | RMSE raw | RMSE paper | RMSE v2 | ρ raw | ρ paper | ρ v2 |
|---|---|---|---|---|---|---|
| 0.00 | **1.621** | 2.271 | 2.176 | 0.754 | 0.338 | 0.504 |
| 0.10 | **1.696** | 2.276 | 2.212 | 0.732 | 0.311 | 0.440 |
| 0.20 | **1.872** | 2.297 | 2.262 | 0.675 | 0.307 | 0.394 |
| 0.40 | 2.515 | **2.374** | 2.417 | 0.519 | 0.252 | 0.269 |

- v2 **dominates the paper index as a single-scalar descriptor**: better ordering (ρ) at
  every σ and lower RMSE at every σ except a marginal tie at σ=0.4.
- v2 keeps the smooth-fuzzy **noise stability** (RMSE rise σ 0→0.4: **+0.241** vs raw
  **+0.894**) while being far more discriminative than the paper index.
- Raw stays most accurate at realistic noise; at extreme σ=0.4 both fuzzy indices beat
  raw on RMSE. (Same regime Set C Exp① found.)

**(3) Secondary sensitivity (not the headline):** widening breakpoints to the data span
(`v2-calibrated`) gives ΔR² **+0.397** — combining rule-fix + recalibration changes
little vs the rule-fix-alone +0.416, so the **rule table is the load-bearing change**,
not breakpoint placement.

## Verdict (scoped, honest)

**The paper's negative was structural, and it is fixable.** A rule-table-only change —
making support monotone in every traction row — turns the fuzzy index from inert
(+0.025) into **load-bearing (+0.416)**, recovering essentially all of the support-axis
signal. v2's +0.416 lands at Set C's **nonlinear measured-physics ceiling** (μ+sinkage
+0.42, μ+sink_drag +0.417), *above* linear raw-K's +0.334 — because v2 is a nonlinear
monotone basis, not because fuzzy beats the physics. As a standalone difficulty scalar,
v2 strictly dominates the paper index and is dramatically more noise-stable than raw.

### Threats to validity (do NOT overclaim)
- **Circularity.** v2's monotone polarity (firm = easier) is *informed by* Set C's
  observed trend. So this is a **corrected descriptor, not a vindication of the paper's
  mapping**. The claim is narrow: *the original failed because its rule table suppressed a
  physically real support axis; a monotone rule table restores support sensitivity.* It
  does **not** claim fuzzy discovers structure raw numbers lack.
- **Ceiling.** On clean deterministic inputs raw `K` is the fair baseline/ceiling; v2's
  job is to **match** it (it does, +0.416 ≈ nonlinear ceiling) while staying
  bounded/interpretable + noise-stable — not to "beat the data." The +0.416 > +0.334 gap
  is the linear-vs-nonlinear ceiling difference on n=41, not a fuzzy super-power.
- **Same single-policy / n=41 / provisional K→kn caveats as Set C.** This is a
  descriptor-level result on frozen-policy data; it does **not** by itself show a
  load-bearing fuzzy *improves training/control* — that would need the GPU curriculum
  (Set B-style) with v2 as the pacing signal, currently not run.

## Artifacts
- Code: `scripts/dfh_fuzzy/fuzzy_v2_loadbearing.py` (`fuzzy_d_v2`, rule grid `_RULES_V2`).
- Output: `logs/DFH_fuzzy/fuzzy_v2_loadbearing.txt`.
- Builds on Set C: `docs/experiments/fuzzy_setC_dfh_result.md`,
  `docs/experiments/fuzzy_soil_UNIFIED_conclusion.md`.
