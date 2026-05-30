# Fuzzy Set-A — Difficulty-Index Descriptor Validation (Result)

**Date:** 2026-05-30. Question: does the paper's fuzzy soil-difficulty index *d*
(§III-C, Mamdani; Easy/Moderate/Hard, d∈[0.2,0.8], higher=harder) faithfully
describe measured task difficulty? The layer is logging-only (no policy effect),
so "works" = d orders soil conditions by real difficulty.

## Method

Fixed policy (Arm B seed1) evaluated across a **friction sweep** μ∈{0.35..0.85},
**full DR_paper_S3 held constant** (so the policy stays in-distribution; an earlier
NO-DR run collapsed it to ~33-step episodes and was discarded), rigid contact.
**3 evals × 100 eps per μ** (18 evals), distinct seeds, to average the ~8% eval
noise. All metrics parsed by script from each eval's own RESULTS block; means
aggregated automatically. Raw: `fuzzy_sweep_repeats_raw.csv`; means: `fuzzy_sweep.csv`;
analysis: `fuzzy_analysis.txt`; figure: `fuzzy_validation.png`.

## Aggregated results (mean of 3 evals/μ)

| μ | fuzzy d | regime | slip/100m | falls/100m | ep_len | vel_err |
|---|---|---|---|---|---|---|
| 0.35 | 0.80 | Hard | 389.99 | 62.16 | 433.97 | 0.353 |
| 0.45 | 0.80 | Hard | 298.96 | 57.48 | 398.03 | 0.281 |
| 0.55 | 0.60 | Moderate | 243.83 | 45.47 | 458.13 | 0.272 |
| 0.65 | 0.40 | Moderate | 213.15 | 41.73 | 481.40 | 0.272 |
| 0.75 | 0.20 | Easy | 175.78 | 33.73 | 561.93 | 0.261 |
| 0.85 | 0.20 | Easy | 157.93 | 32.57 | 570.27 | 0.261* |

(*vel_err 0.85 = 0.270 in agg CSV; table mirrors analysis rounding.)

## Findings (from fuzzy_analysis.txt)

**Monotonicity vs d (Spearman):** slip ρ=+0.971, falls ρ=+0.971, ep_len ρ=−0.971,
vel_err ρ=+0.955 — **4/4 metrics monotone, correct sign.** Regime separation is
clean: Easy→Hard slip 167→344, ep_len 566→416.

**Added value vs raw μ (circularity check):** |ρ| for d vs |ρ| for raw μ —
slip 0.971/1.000, falls 0.971/1.000, ep_len 0.971/0.943, vel_err 0.955/0.928.
**d adds no information beyond μ**; if anything raw μ is a marginally better
predictor (d ties μ=0.35≡0.45 at d=0.80 and μ=0.75≡0.85 at d=0.20, losing
resolution).

## Verdict (honest, scoped per Codex)

**d works as an ordinal difficulty descriptor on the friction axis** — it correctly
orders soil conditions by measured difficulty, robustly across 4 metrics with n=3
averaging. **But this is a limited claim:** because the stiffness axis kₙ is
physically inert in PhysX (proven by Arm D — compliant normal contact gives no
lateral grip), d collapses to a monotone transform of μ over this sweep. So a
positive result mainly restates "lower friction = harder," and the fuzzy
binning/min-rule adds nothing over raw μ here.

**It does NOT validate:** the 2D fuzzy model, the support (stiffness) dimension, or
any added value of the fuzzy structure over a plain friction scalar.

A genuine "the fuzzy model works" claim would need a sim where *both* fuzzy inputs
are physically meaningful (i.e. the missing tangential-deformation physics), or an
external difficulty signal not reducible to μ. That is out of scope here.
See `compliant_contact_feasibility.md`, `fuzzy_soil_conclusion.md`.
