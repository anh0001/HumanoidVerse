# Fuzzy-Soil Investigation — Unified Conclusion (Set A + B + C + noise probes)

**Date:** 2026-06-02. Robot: Hunter. Sim: IsaacSim / IsaacLab (PhysX rigid; DFH
deformable). Paper under test: `docs/wcci2026_hunter.pdf` — a fuzzy Mamdani "soil-state"
difficulty index over two inputs (traction = friction μ, support = contact stiffness),
plus a material curriculum and stance-gated slip reward.

This ties together three studies + two follow-up probes into one answer. Every number
below is drawn from the per-study result docs / analysis files (no values from memory).

## The one-line answer

**The paper's fuzzy soil index does not add demonstrable value over the raw soil
numbers in any setting we could construct** — and we ruled out the obvious excuse (a
too-crude simulator) by reproducing the result in a deformable-soil sim where the second
axis is physically real. The failure is **structural** (the fuzzy rule base discards the
support information), not a simulator artifact. Scope: this is about *this paper's
specific Mamdani mapping*, **not** a claim that 2-D or learned terrain descriptors are
useless.

**Constructive coda (Set D):** we then *fixed* the structural failure — a rule-table
redesign makes the index load-bearing as a **descriptor** (support-axis CV-R² gain
0.025 → 0.416) — and asked whether that fix improves **training**. It does not: the fixed
index carries the same information as the raw soil numbers, so as a curriculum signal it
**ties raw and never beats it**. Net: fuzzy is a good interpretable *descriptor*, not a
source of *performance*.

## The studies (each closes the previous one's loophole)

| Study | Sim | Fuzzy used as | Support axis | Verdict |
|---|---|---|---|---|
| **A** | rigid PhysX | passive **label** | physically **inert** (kₙ no lateral grip) | d tracks difficulty (ρ≈0.97) but adds **nothing over raw μ** |
| **B** | rigid PhysX | active **curriculum** controller | inert | fuzzy ≈ crisp ≈ fixed (Welch t=0.45, n=3) |
| **C** | **DFH deformable** | **descriptor** | physically **active & predictive** | support adds CV-R² **+0.33** over μ; **fuzzy still +0.025** (p=0.001) |
| ① | DFH data | descriptor under **noisy sensing** | active | fuzzy **noise-stable** but "stable not accurate"; wins only at σ=0.4 |
| ② | DFH data (proxy) | **decision** signal under noise | active | fuzzy stability does **not** help decisions — it **camps at easy soil** |
| **D-desc** | DFH data | **REDESIGNED** load-bearing descriptor | active | rule-table fix lifts CV-R² gain **0.025 → 0.416** — failure was structural & **fixable** |
| **D-curr** | DFH data (proxy) | redesigned index as **curriculum** signal | active | even fixed, `[μ,d_v2]` **ties raw, never beats** (EQUIV) → GPU **no-go** |

### Set A — fuzzy as a label (rigid)
Recipe's **survival benefit replicates** (~8.85× episode length vs v7 baseline, driven by
the friction curriculum). The headline **slip-reduction does not replicate** (ruled out
reward gate, eval noise, compliant contact, metric mismatch). The fuzzy index orders soil
correctly on the μ axis (Spearman ρ≈0.97) **but adds no information over raw μ**, because
PhysX compliant contact is normal-only → the **support/stiffness axis is physically
inert** → the 2-D model collapses to 1-D friction. *(`fuzzy_soil_conclusion.md`.)*

### Set B — fuzzy as a curriculum controller (rigid)
Promoted the index from label to an active pacing function (fixed / crisp-threshold /
fuzzy-accumulator), 3 arms × 3 seeds = 9 PPO-ROA runs, held-out robustness AUC.
**Fuzzy ≈ crisp** (0.724 vs 0.680, Welch t=0.45 — not significant); adaptive-vs-fixed only
a non-significant trend. Same root cause: with support inert, fuzzy has no second axis to
exploit. *(`fuzzy_setB_result.md`.)*

### Set C — fuzzy where BOTH axes are physical (DFH deformable soil)
DFH (Bekker pressure-sinkage + Janosi–Hanamoto shear + slip-sinkage + anisotropic
friction) is the deformable-soil sim Set A/B said was needed. Frozen-policy 2-D descriptor
sweep (traction μ × Bekker stiffness K), pre-registered support-axis gate + force-off
shadow control + policy coverage screen.
- **Support axis restored** — gate passes: at μ=0.56, K=1→8 sinkage 0.049→0.023 m
  (Δ26 mm), sink-drag 44.6→20.3 N (Δ24 N), vel_err CI excl 0; shadow (forces off) flat.
- **Support is strongly predictive** of difficulty: LOO-CV R² gain over μ = **+0.33**
  (raw K), **+0.42** (measured sinkage / sink-drag); partial Spearman(Y,K|μ) = −0.70.
- **The fuzzy index STILL adds ~nothing**: CV-R² gain **+0.025**, partial Spearman(Y,d|μ)
  = −0.04; permutation p=0.001 that support beats fuzzy_d. **So the failure is the fuzzy
  model's structure** (its rule base collapses (μ,support) into a coarse Easy/Mod/Hard
  scalar and ignores support at mid-traction), not the simulator. *(`fuzzy_setC_dfh_result.md`.)*

### Probes ① & ② — giving fuzzy its best shot (its theoretical strengths)
- **① noisy soil sensing:** fuzzy's smooth bins are noise-robust (RMSE rise σ0→0.4:
  raw +0.90 vs fuzzy +0.12) and overtake raw at extreme noise (σ=0.4) — but win by being
  *stable, not accurate* (raw still orders better, Spearman 0.52 vs 0.24).
- **② decisions under noise (cheap proxy, gated GPU):** fuzzy has 67–81% fewer curriculum
  reversals but reaches the productive difficulty band only 9–15% vs raw 46–52% — it
  **camps at easy soil** (even at σ=0). Stability of an uninformative signal → GPU
  experiment not justified, not run.

### Set D — fixing the structure, then re-testing for a training gain
Set C pinned the failure on the rule base, so we **redesigned** it: a minimal,
rule-table-only change (monotone 2-D anti-diagonal grid, 5-level consequents; same
fuzzification/defuzzification) so support carries weight in *every* traction row.
- **D-descriptor (load-bearing — it works):** support-axis LOO-CV R² gain jumps
  **0.025 → 0.416** (≈ Set C's nonlinear measured-physics ceiling +0.42; raw K +0.33),
  partial Spearman(Y,d|μ) −0.04 → **+0.72**. Confirms the negative was **structural and
  fixable**, not the sim. *Honest scope:* the monotone polarity is informed by Set C's
  observed trend → a corrected descriptor, **not** a vindication of the paper's mapping;
  on clean inputs raw K is the ceiling, which v2 matches, not beats.
  *(`fuzzy_v2_loadbearing_result.md`.)*
- **D-curriculum (no training gain — Codex-gated):** does the fixed index make a better
  curriculum? As a 1-scalar signal it **fails** the fair percentile-space proxy (trails
  raw, loses its stability edge). In its strongest 2-input form `[μ, d_v2]` it **fully
  fixes the camping and matches raw accuracy**, but only ~17% fewer reversals (under the
  pre-registered 20% bar) → **EQUIV, not WIN**. `[μ, d_v2]` carries the **same
  information** as raw `[μ, log₂K]`. **GPU no-go** (Codex-confirmed; ~24h saved).
  *(`fuzzy_v2_curriculum_plan.md`.)*

## What this means

1. **Replicates:** the friction-curriculum survival benefit (robust).
2. **Does not replicate:** the >95% slip-reduction headline.
3. **No fuzzy-specific value, four ways:** as a label (A), a curriculum (B), a descriptor
   with a live support axis (C), and under the noisy-sensing regime fuzzy was designed for
   (①/②). Its only edge is noise-stability, bought by discarding resolution.
4. **Root cause is structural, not the sim.** Set C is the key: even where sinkage/
   stiffness genuinely move difficulty, the paper's Mamdani mapping fails to capture it.
5. **Structural failure is fixable — but the fix buys interpretability, not performance
   (Set D).** A redesigned rule base makes the index a *strong* descriptor (gain 0.416),
   yet even then it only *matches* the raw soil numbers as a training/curriculum signal —
   it never beats them. Raw `[μ, log₂K]` is the ceiling fuzzy can reach, not exceed.

## What would change the verdict

- ~~A **fuzzy system redesigned to actually use the support axis**~~ — **DONE (Set D):** the
  redesign works as a descriptor (gain 0.416) but ties raw as a training signal. Confirms
  the failure was structural & fixable, and that the fix adds interpretability, not
  performance. The remaining honest flip would need a **new** pre-registered claim with
  *noise-stability as the primary target*, or a setting where raw `[μ, log₂K]` is
  unavailable/fails (e.g. a real robot with only a coarse soil label, no precise sensors)
  — Codex's standing note.
- A **domain-randomized DFH walker** (removes Set C's single-policy / OOD caveat) — judged
  not worth the GPU given the structural argument + permutation result.
- DFH Bekker params are **uncalibrated** defaults; Set C is a deformable-soil *surrogate*,
  not calibrated real soil. This is non-replication of the *mechanism in simulation*, NOT
  a disproof of the paper on real deformable soil.

## Map of artifacts (across branches)

- **Set A:** `feature/fuzzy-soil` — `fuzzy_soil_conclusion.md`, `fuzzy_setA_result.md`,
  `scripts/paper_fuzzy_soil/`.
- **Set B:** `feature/fuzzy-adaptive-curriculum` — `fuzzy_setB_result.md`,
  `fuzzy_soil_final_conclusion.md` (A+B), `scripts/setB_fuzzy_curriculum/`.
- **Set C + ①/②:** `feat/dfh-fuzzy-descriptor` — `fuzzy_setC_dfh_result.md`,
  `scripts/dfh_fuzzy/`, data under `docs/experiments/data/setC/`. DFH extension lives on
  `feat/isaacsim-agri-field`; models on HuggingFace `anhrisn/hunter-dfh-locomotion`.
- **Set D (this branch):** `fuzzy_v2_loadbearing_result.md` (descriptor),
  `fuzzy_v2_curriculum_plan.md` (curriculum no-go); scripts
  `scripts/dfh_fuzzy/fuzzy_v2_loadbearing.py`, `proxy_curriculum_v2.py`,
  `proxy_curriculum_2input.py`; outputs under `logs/DFH_fuzzy/`.
- **This unified doc:** `feature/fuzzy-soil-unified-conclusion`.
