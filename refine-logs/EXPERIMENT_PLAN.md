# Experiment Plan — Implicit Online Adaptation for Blind Hunter Locomotion

**Date:** 2026-05-19  ·  Prior soil plan archived `EXPERIMENT_PLAN.soil-archive.md`
**Gate-first design:** cheapest falsification before any architecture work.

## §0 — Falsification Gate (MUST run first, <2 GPU-h, no GPU train of ROA)

**Purpose:** test the single highest-risk assumption — *can a 50-step proprioceptive history infer control-relevant terrain/dynamics early?*

1. Reuse an existing trained Hunter checkpoint (`logs/CurriculumStage2/model_910050.pt`) OR a quick privileged-conditioned rollout collector. Collect rollouts under randomized {friction, added mass, push, terrain class soil/furrow} — log per-step 50-window proprio history + ground-truth privileged params.
2. Train a small supervised TCN/GRU: history → predict privileged params / discretized bins. (~20–40 min on the contended GPU; CPU-feasible fallback.)
3. **PASS** if: (a) prediction ≫ chance on friction class + soil/rigid + disturbance regime; (b) accuracy rises within the **first few stance transitions**, not post-fall; (c) prediction quality correlates with control-relevant outcomes.
4. **FAIL** → ROA is not the right dominant idea. Fallback: demote to history-only policy + periodic-symmetry reward (Idea 3 + Idea 2), drop the privileged teacher. Re-review.

**Also in §0 (cheap, no GPU): baseline fragility measurement.** Run `sample_eps.py` with current baseline on soil + furrows under DR; record fall rate, mean ep_len, tracking error. Defines whether "drastic" headroom even exists.

## §1 — Cheap multiplier first (Idea 2, ~1 day, ~3–4 GPU-h)

Implement periodic-clock + symmetry reward (soft weights). Train 3 seeds, current obs/arch, plane→soil curriculum, ~3000 iters each (or `Sim-to-Real-in-15min`-style fast recipe for pilots). Compare vs baseline. Fast signal; ships independently if positive.

## §2 — Ablation Matrix (reviewer-mandated, the core result)

Same curriculum, same total env-steps, same DR distribution, matched param counts where possible, **seeds {1,2,3}** (extend to 5 for submission), exclude seed 0 per existing protocol.

| # | Arm | Isolates |
|---|---|---|
| A0 | single-step MLP PPO (current) | baseline |
| A1 | + asymmetric critic only | critic info |
| A2 | + 5-step history | short memory |
| A3 | + 50-step TCN/GRU history, no teacher | **memory alone** |
| A4 | privileged teacher / oracle (upper bound) | ceiling |
| A5 | ROA student (M1–M3) | **adaptation** |
| A6 | + periodic-clock only | clock alone |
| A7 | + symmetry only | symmetry alone |
| A8 | + clock + symmetry | gait reg combined |
| A9 | **Full: ROA + clock + symmetry** | proposed stack |
| A10 | DFH explicit-soil arm (reuse Pass-4 ckpts) | fidelity-vs-adaptation |

**Eval:** held-out terrain/friction/mass/push (NOT training curriculum) + in-distribution. Metrics: mean ep_len, fall rate, lin/ang tracking error, cost-of-transport, gait regularity, `‖z−ẑ‖`, z-causality (zero/shuffle-`ẑ`).

## §3 — Claim Gate (minimum bar)
Claim "drastic improvement" only if A9 vs A0: ≥2× ep_len OR ≥50% fewer falls on held-out soil/furrow; ≥25–30% tracking-error cut under DR; **A9 > A3 and A9 > A8** (memory-only and gait-only do not explain it); `ẑ` causal; no energy/gait regression; A9 ≥ A10 (beats explicit soil modeling).

## Run order & budget
1. §0 gate + fragility (<2 GPU-h) — **decision point**.
2. §1 periodic-symmetry pilot (~3–4 GPU-h).
3. §2 matrix: ~10 arms × 3 seeds. Stage pilots at reduced iters; promote on signal. Est. ~30–50 GPU-h total — sequence after the GPU frees (currently externally contended at 76%).
4. `/run-experiment` to deploy; `/auto-review-loop` to iterate to the bar.

## Tracker
`refine-logs/EXPERIMENT_TRACKER.md` to be created at first launch (arm × seed × status × metrics).
