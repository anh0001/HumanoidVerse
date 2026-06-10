# Load-bearing fuzzy (v2) as a TRAINING curriculum — plan + GPU-gate decision

**Date:** 2026-06-05. Branch: `feature/fuzzy-soil-unified-conclusion`. Robot: Hunter.
Builds on the load-bearing descriptor result (`fuzzy_v2_loadbearing_result.md`) and the
Set B/C curriculum machinery. Decisions vetted with Codex ([[codex-at-decision-points]]);
every number is read from the proxy output files produced this run.

## Question

The descriptor result showed `fuzzy_d_v2` is load-bearing for *prediction* (support-axis
LOO-CV ΔR² 0.025 → 0.416). The open question (named in the unified conclusion) is whether
a load-bearing fuzzy improves *training/control* — i.e. used as a curriculum signal, does
it beat raw and the paper index? Set B answered "fuzzy ≈ crisp" but on PhysX where the
support axis was inert; the fair test needs DFH (active support) **and** a cheap proxy to
gate the ~24h GPU run (established protocol: proxy gates GPU).

## Proposed GPU experiment (documented; NOT launched — see gate decision)

DFH 2-D difficulty-targeting curriculum (Set B orchestrator on DFH soil, Set C Exp②
promoted to a real run):
- **Soil ladder:** 2-D grid (traction μ × Bekker K), ordered easy→hard.
- **Arms (differ ONLY in the difficulty signal the scheduler advances on):**
  `fixed` (1-D schedule) · `paper-fuzzy` · **`v2`** (load-bearing) · `raw` [μ, log₂K].
- **Compute:** 4 arms × 3 seeds = 12 chained DFH-PPO runs, matched iterations.
- **Primary metric:** held-out robustness AUC over a 2-D DFH soil sweep.
- **Known feasibility risks (pre-flight before any full run):** DFH `dfh_chain` finals are
  *balancers, not walkers* (memory) — DFH walker-curriculum training may be unstable;
  DFH is costlier per step than PhysX.

## GPU gate (cheap CPU proxy) — RAN, with Codex's fairness fix

Closed-loop difficulty-tracking curriculum over the measured DFH grid (15 cells, ladder
sorted by true difficulty); arms differ only in the difficulty-estimate mapping. Scheduler
advances/holds/regresses on the estimate. Two scheduler interfaces:

1. **magnitude** (first cut): threshold the OLS difficulty estimate in absolute units.
   `scripts/dfh_fuzzy/proxy_curriculum_v2.py --mode magnitude`
   → output `logs/DFH_fuzzy/proxy_curriculum_v2.txt`.
2. **percentile** (Codex's fairness fix): threshold the estimate's RANK in each arm's own
   distribution — removes the 1-scalar magnitude offset, testing purely whether the arm's
   *ordering* induces a good traversal under noise. This is the deciding gate.
   `--mode percentile` → output `logs/DFH_fuzzy/proxy_curriculum_v2_rank.txt`.

**Why the magnitude cut was unfair (Codex):** it tested "can one uncalibrated fuzzy scalar
replace a 2-D raw model at a shared threshold," not "does the estimate induce a good
curriculum." v2 even camped at σ=0 — a representational-capacity artifact (a single scalar
can't reproduce the 2-D difficulty surface), not v2's information content.

### Pre-registered pass condition (Codex, written BEFORE the percentile run)
At σ=0.3 **and** 0.4, v2 must: (1) beat paper-fuzzy on productive fraction AND maxY;
(2) not camp (maxY > 0); (3) be within 0.10 productive fraction of raw OR oracle;
(4) have ≥20% fewer reversals than raw.

### Percentile-mode results (productive↑, reversals↓, maxY = camping check)

| σ | oracle prod | raw prod / rev | paper-fuzzy prod / maxY | **v2 prod / maxY / rev** |
|---|---|---|---|---|
| 0.3 | 0.625 | 0.494 / 1.98 | 0.011 / −1.42 | **0.351 / +0.18 / 4.15** |
| 0.4 | 0.625 | 0.434 / 2.76 | 0.034 / −1.22 | **0.343 / +0.25 / 4.19** |

- **v2 fixes the camping** that sank paper-fuzzy: beats_paper ✓ and maxY > 0 ✓ at both σ.
- **But fails the pass condition at both σ:** (3) productive 0.351 vs raw 0.494 — outside
  the 0.10 bar at σ=0.3; (4) v2 has **more** reversals than raw (4.15 vs 1.98 → −110%),
  so the stability edge it showed in magnitude mode **reverses** under the fair interface
  (compressing 2-D difficulty to one scalar bunches cells at similar percentiles → noise
  flips the decision more often).

## DECISION: GPU NO-GO

**The pre-registered gate FAILS at both σ=0.3 and 0.4.** Per the pre-registration
discipline and Codex's "if it fails, run raw," the 12-run DFH curriculum is **not
justified**. Honest reading: the load-bearing redesign is real and valuable **as a
descriptor** (and it does fix paper-fuzzy's camping), but **as a single-scalar curriculum
decision signal it does not beat raw [μ, log₂K]** even under the interface designed to be
fair to it — and it loses the noise-stability that was fuzzy's only prior edge. This is
consistent with the whole study: the honest ceiling for clean/structured inputs is raw;
fuzzy's job is to *match* it interpretably, not to beat it as a controller.

### What would change this (not done)
- Use v2 as a **2-input** scheduler feature `[μ, d_v2]` (its proven-strong form) rather
  than a 1-scalar magnitude — but that is no longer "the fuzzy index as a difficulty
  scalar," it's a learned 2-D controller (different claim).
- **Codex's standing threat to validity for any eventual GPU claim:** the proxy uses the
  offline 15-cell true-difficulty ladder as substrate; a credible training claim needs v2
  to improve learning under *policy-induced* terrain interaction, not just offline
  ordering. Until a proxy reflecting that passes, GPU is unjustified.

## Addendum — the 2-input steelman `[μ, d_v2]` (the one untested path)

The 1-scalar tests fail because one fuzzy number can't reproduce the 2-D difficulty
surface. Final test: use fuzzy as a 2-INPUT scheduler feature `[μ, d_v2]` (its proven
descriptor form). Pre-registered (before running): "worth GPU" iff at σ=0.3 AND 0.4,
`v2_2in` matches raw accuracy (prod ≥ raw−0.03), doesn't camp (maxY>0), has **≥20% fewer
reversals than raw** (the stability payoff), and beats the 1-scalar v2. Match-only ⇒
"equivalent to raw"; worse ⇒ fail. (`scripts/dfh_fuzzy/proxy_curriculum_2input.py` →
`logs/DFH_fuzzy/proxy_curriculum_2input.txt`.)

| σ | oracle prod | raw prod / rev | v2_1in (scalar) prod / maxY | **v2_2in `[μ,d_v2]` prod / maxY / rev** |
|---|---|---|---|---|
| 0.0 | 0.625 | 0.625 / 0.00 | 0.000 / −1.43 (camps) | **0.625 / 0.90 / 0.00** (= raw) |
| 0.3 | 0.625 | 0.494 / 1.98 | 0.351 / +0.18 | **0.472 / 0.87 / 1.67** |
| 0.4 | 0.625 | 0.434 / 2.76 | 0.343 / +0.25 | **0.411 / 0.77 / 2.27** |

- The 2-input form **fully fixes** the scalar's camping (σ=0: 0.625/0.90, identical to raw).
- It **matches** raw accuracy at every σ (within −0.03) with a **16–18% reversal
  reduction — just under the pre-registered 20%** stability bar.
- **Pre-registered verdict at both σ: EQUIV (not WIN).**

**Closing decision:** `[μ, d_v2]` carries essentially the **same information as raw
`[μ, log₂K]`** — it ties, it does not beat, and the marginal stability edge misses the
pre-registered bar. Fuzzy's value is **interpretability/bounded-monotone structure, not
curriculum performance**, even in its strongest 2-input form. **GPU remains NO-GO.** This
closes the "does load-bearing fuzzy improve training?" question: **no** — raw is the
ceiling and fuzzy at best matches it.

**Codex final concurrence (independent check at the decision point):** agrees NO-GO;
holding the pre-registered 20% reversal line was correct; EQUIV is correctly not a WIN
(a ~17% reversal reduction at matched accuracy is below the bar and not GPU-worthy). The
question is **effectively closed** — an honest flip would need a *new* pre-registered
claim with stability as the *primary* target, or a setting where raw `[μ,log₂K]` is
unavailable/fails; otherwise more runs are "confirmation, not discovery." Endorsed
headline: **"Fuzzy soil difficulty is a strong interpretable descriptor of terrain
hardness, but it does not improve curriculum performance over the raw soil parameters for
Hunter locomotion."**

## Artifacts
- `scripts/dfh_fuzzy/proxy_curriculum_v2.py` (`--mode magnitude|percentile`, v2 1-scalar arm + gate).
- `scripts/dfh_fuzzy/proxy_curriculum_2input.py` (2-input `[μ,d_v2]` steelman + pre-reg gate).
- `logs/DFH_fuzzy/proxy_curriculum_v2.txt` (magnitude), `..._v2_rank.txt` (percentile),
  `..._2input.txt` (2-input/closing).
- Descriptor result: `docs/experiments/fuzzy_v2_loadbearing_result.md`.
