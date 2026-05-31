# Fuzzy Set-B — Adaptive-vs-Fixed Friction Curriculum (Plan)

**Date:** 2026-05-30. Branch: `feature/fuzzy-adaptive-curriculum`. Robot: Hunter.
Sim: IsaacSim / IsaacLab (PhysX). Builds on `fuzzy_soil_conclusion.md` (Set A).

## Question

Set A showed the paper's fuzzy soil-difficulty index `d` is a valid 1-D friction
descriptor but adds nothing over raw μ, because the support/stiffness axis (kₙ) is
physically inert in PhysX. Set B asks the next question: **promote the fuzzy
machinery from a passive label to an active curriculum-pacing function, and test
whether it beats a crisp baseline** — the user's non-negotiable ("fuzzy must
outperform crisp thresholds; fuzzy-vs-fuzzy proves nothing").

## What this CAN and CANNOT honestly claim (scope, per Codex)

- **CAN test:** a *difficulty-aware (adaptive) global ground-friction curriculum*
  vs a *fixed-linear schedule*, and whether *fuzzy* (smooth) pacing beats *crisp*
  (hard-threshold) pacing.
- **CANNOT test:** "fuzzy soil reasoning improves learning." With kₙ inert, the
  difficulty axis is effectively 1-D friction, so the fuzzy index reduces to a
  transform of μ. **Expected honest outcome: fuzzy ≈ crisp.** The real upside is
  whether *adaptive* beats *fixed*.
- This is **not** per-env soil randomization and **not** per-rollout live mutation.
  It is a *staged global ground-friction curriculum by checkpoint chaining*.

## Why block-chained construction-time μ (not live mutation)

Code facts that fixed the design:
- The repo's `randomize_friction` is wired to `mdp.randomize_joint_parameters`
  (JOINT/motor friction, `mode="startup"`) — **not** ground traction, not per-episode.
- Ground **traction** μ (the axis Set A validated, slip-vs-μ ρ≈0.97) is set ONCE at
  sim construction in `RigidBodyMaterialCfg.static/dynamic_friction` from the terrain
  YAML (`isaacsim.py:503-504`). The furrow ground is a **single shared prim**, so
  per-env μ is not a drop-in.
- So the curriculum drives the **global** ground μ, changed **between training
  blocks** via the construction-time override `++terrain.static_friction=μ
  ++terrain.dynamic_friction=μ` — the *proven* knob (Set A swept exactly this).
  Live mid-sim material mutation is unproven in this stack and carries the same
  fidelity risk that sank the compliance axis; we avoid it.

Optimizer/RNG state is resumed across blocks (`++algo.config.load_optimizer=True`,
ppo.py:169) for continuity; block 0 warms from the shared checkpoint without
optimizer load. Restart transients are shared by all three arms, so they cancel in
the contrast (do NOT compare these to an uninterrupted single-process baseline).

## Design

- **Ladder (EASY→HARD):** μ ∈ {0.80, 0.65, 0.50, 0.35} (static = dynamic = μ).
- **Blocks:** 8 × 94 iters = **752 total iters** (≈ Arm B's 750), matched across all
  arms/seeds. Warm-start: `FixedStageFv7/model_4250.pt` (same for all).
- **Per-block signal:** `Train/mean_episode_length` (read from the block's
  TensorBoard log). `mastery m = mean_ep_len / 1000` (cap = 20s × 50Hz, from config).
- **Three arms (identical ladder / signal / total iters / DR / reward / obs / env —
  differ ONLY in next-μ choice):**
  - **fixed** — performance-independent; advance one rung every 2 blocks (linear).
  - **crisp** — hard all-or-nothing gate: advance one rung iff `m ≥ T` (T=0.50).
  - **fuzzy** — smooth proportional pacing: Mamdani over `m` (paper's ramp/tri/ramp
    memberships, min-AND, max-agg, singleton defuzz) → advance score `a∈[0,1]`;
    accumulate `progress += a`, advance one rung when `progress ≥ 1` (carry remainder).
    Same 1-rung/block cap as crisp.
- **Why fuzzy = accumulation, not a threshold on `a`:** thresholding a defuzzified
  scalar on a 1-D signal is just a relabeled crisp threshold (tautological). The
  accumulator is the genuine "smooth interpolation vs hard bins" contrast — fuzzy
  CAN differ (it creeps forward at medium mastery where crisp stalls). Whether that
  helps is the empirical question. `crisp_threshold` and the fuzzy breakpoints are
  calibrated to the achievable mastery range (confirmed in the pre-flight, NOT
  guessed from memory).

## Compute

3 arms × 3 seeds = **9 chained runs**, each ≈ one Arm-B-chain's total iterations.
Pre-flight (cheap functional smoke) gates the 9 runs.

## Pre-flight gate (functional, minutes — not the live-mutation probe Codex first
specified; dropped because Set A already proved construction-time μ physics)

A 2-block smoke (`SMOKE=1 BLOCKS=2 ITERS=6 NUM_ENVS=256`) must show:
1. block 1 loads block 0's checkpoint (chaining works);
2. `++terrain.static_friction` override actually applies per block;
3. the orchestrator reads the block metric and the controller picks the next μ;
4. held-out eval can load a final checkpoint and sweep fixed μ.
If the pre-flight fails, fix tooling before spending the 9 runs.

## Metrics (held-out)

- **Primary:** robustness AUC over a held-out fixed-μ sweep ({0.35..0.85}) at matched
  compute — mean survival/robustness across difficulties (higher = better).
- **Secondary:** steps/blocks to reach a held-out robustness threshold (convergence).
- **Decision trail:** per-arm `decisions.csv` (mastery, advance, μ path per block).

## Expected outcome & honest verdict template

> A global adaptive friction curriculum {does / does not} beat a fixed-linear
> schedule at matched compute. Fuzzy (smooth) pacing {≈ / >} crisp (hard-threshold)
> pacing — and since PhysX makes the support axis inert (difficulty ≈ 1-D friction),
> fuzzy is **not** expected to add measurable value over crisp. Any adaptive>fixed
> effect is *friction-aware*, not *fuzzy-specific*.

## Artifacts

`scripts/setB_fuzzy_curriculum/`: `curriculum_controller.py` (3 controllers),
`run_block_chain.py` (orchestrator), `run_arm.sh` (env wrapper), `eval_heldout.sh`,
`analyze_setB.py`. Data under `logs/FuzzySoilFurrowsSetB/setB_<arm>_seed<seed>/`.
