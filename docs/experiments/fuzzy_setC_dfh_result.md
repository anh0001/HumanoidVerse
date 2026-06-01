# Fuzzy Set-C — DFH Deformable Soil: does the fuzzy index add value when BOTH axes are physical? (Result)

**Date:** 2026-06-01. Branch: `feat/dfh-fuzzy-descriptor` (off `feat/isaacsim-agri-field`).
Robot: Hunter. Sim: IsaacSim + **DFH** (Deformable Furrowed Heightfield: Bekker–Wong
pressure-sinkage, Janosi–Hanamoto shear, slip-sinkage, anisotropic friction). Builds on
Set A/B (`fuzzy_soil_final_conclusion.md`). All numbers read from the sweep CSVs this run.

## Question

Set A/B concluded the paper's fuzzy soil index adds nothing over raw μ because rigid
PhysX makes the **support/stiffness axis physically inert** (1-D collapse). Set C uses
DFH — a deformable-soil layer that *does* model sinkage/shear — to ask the question the
earlier sims couldn't: **when BOTH fuzzy inputs (traction μ and support stiffness) are
physically active, does the paper's fuzzy Mamdani index add predictive value over raw
traction/support descriptors?**

Method: frozen-policy descriptor sweep (no training) — eval a walker across a 2-D soil
grid (traction μ × Bekker support scale K), measure task difficulty + DFH force
diagnostics, test whether fuzzy `d` beats raw μ(+support). Codex protocol with a
pre-registered support-axis go/no-go gate.

## Getting a fair probe (two course corrections)

1. **Floor saturation (first sweep, K∈[0.8,1.25]).** `dfh_max_sink_m = 0.0499` for all
   27 cells — the foot bottomed out at the `sinkage_floor_m=-0.05` cap for every K, so
   support was artificially flat (gate failed: Δsink 0.5 mm). Fix: widen K firmer.
2. **Firmer sweep (K∈{1,2,4,8,16}) desaturated** sinkage (0.049 m → 0.008 m), but the
   mild-soil walker is **OOD on firm soil** (trained at K≈1): it walked worse, not
   easier, and 19/45 cells failed the walking filter, concentrated at high K — so
   difficulty was "distance from training soil," not soil difficulty.
3. **Coverage screen** (4 policies × grid, 1 seed) found a policy that walks across the
   whole K range: **`rigid_walker` 14/15** (per-K 3,3,3,2,3). `mildsoil` 7/15 (fails
   high K); `dfh_chain` finals 0/15 walking (**balancers** — survive, don't locomote,
   excluded). Clean descriptor test then run with `rigid_walker`, seeds {1,2,3}, 41/45
   walking, drops scattered (not clustered).

## Results (rigid_walker, 41 walking cells)

**Support-axis gate (μ=0.56, K=1 soft vs K=8 firm) — PASSES on physics AND behavior:**
- sinkage 0.0494 → 0.0228 m, **Δ26 mm (+116%)**, CI excl 0 (≥5 mm) ✓
- sink-drag 44.6 → 20.3 N, **Δ24 N (+120%)**, CI excl 0 (≥10 N) ✓
- vel_err 0.106 → 0.116, Δ=−0.0095, **95% CI excl 0** ✓
- aggregate difficulty Y vs K: **Spearman −0.665** (firmer = easier; correct soil direction)
- **Shadow control** (force-coupling OFF): vel_err **flat** across K (Δ 0.009, CI incl 0)
  while sinkage state still varies → the behavioral K-effect is caused by **DFH forces**.

**Fuzzy go/no-go (LOO-CV R² gain over Y~μ; partial Spearman | μ):**

| regressor over μ | CV-R² | gain | partial Spearman(Y,·|μ) |
|---|---|---|---|
| μ (baseline) | +0.012 | — | — |
| μ + K (raw support) | +0.346 | **+0.334** | **−0.701** |
| μ + sinkage (measured) | +0.427 | **+0.415** | — |
| μ + sink_drag (measured) | +0.428 | **+0.417** | — |
| μ + **fuzzy d** | +0.036 | **+0.025** | **−0.039** |

**Robustness:** all-45-cells with a failure penalty → μ+K Δ+0.216, μ+fuzzy_d Δ−0.041
(exclusions don't create the result). Permutation test: ΔR²(μ+K) − ΔR²(μ+fuzzy_d) =
+0.309, **p=0.001**.

## Verdict (scoped, per Codex)

> **DFH restores the missing physical support axis:** Bekker stiffness measurably
> changes sinkage, sink-drag, and fixed-policy locomotion difficulty (gate passes;
> shadow control confirms it's the forces). This is exactly what rigid PhysX could not
> express (Set A/B). **However, the paper's Mamdani fuzzy index adds no predictive value
> over raw traction/support descriptors** (CV-R² gain +0.025 vs +0.33–0.42 for raw/
> measured support; permutation p=0.001). The negative result is therefore **not** due
> to the simulator suppressing support physics — it is due to the **fuzzy index/rule
> structure discarding useful support information** (it collapses (μ, support) into a
> coarse Easy/Mod/Hard scalar; at mid-traction the rule base ignores support entirely).

**This sharpens Set A/B.** There, fuzzy failed because support was physically inert
(nothing to capture). Here, support is demonstrably active and strongly predictive, and
the paper's fuzzy mapping *still* fails to capture it — pinning the failure on the model
structure, not the sim.

## Scope & threats to validity (do NOT overclaim)

- This is a finding against **this paper's specific fuzzy mapping**, NOT against 2-D
  terrain descriptors in general. A different/finer fuzzy or learned 2-D descriptor
  could still help — untested.
- One frozen policy (`rigid_walker`, trained on rigid soil → mildly OOD at the soft
  end; mirror of mildsoil). It walks 41/45 across the range with scattered drops, so the
  support axis is fairly probed, but a domain-randomized DFH walker would remove the
  single-policy caveat (judged not worth the GPU; the rule-base structural argument +
  permutation result make a flipped verdict implausible).
- n=41 cells, 1 seed-set; K→kn support mapping is provisional (but fuzzy_d's failure is
  robust to it — d barely varies with support at mid-traction by rule-base construction).
- DFH Bekker params are uncalibrated defaults ("loose dry sand"); this is a deformable-
  soil *surrogate*, not calibrated real soil.

## Artifacts

Code: `scripts/dfh_fuzzy/` (`run_dfh_fuzzy_sweep.py`, `analyze_dfh_fuzzy.py`,
`run_coverage_screen.sh`, `screen_coverage.py`). DFH eval instrumentation added to
`humanoidverse/sample_eps.py` (`DFH_METRIC` prints). 13 DFH config symlinks repointed
relative. Data: `logs/DFH_fuzzy/` — `sweep_main.csv` (saturated pilot), `sweep_firm.csv`,
`sweep_rigid.csv` (clean), `sweep_shadow.csv`, `screen_*.csv`, `coverage_screen.txt`,
`dfh_fuzzy_analysis_rigid.txt`. Policies from HuggingFace `anhrisn/hunter-dfh-locomotion`.
