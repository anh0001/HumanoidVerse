# Arm C (hard-gate ablation) — Pre-registration

**Status:** PRE-REGISTERED before results. Launched 2026-05-29 12:03 local; 1-seed
directional run, full chain ~1h49m + ~15m eval/analysis → expected ~14:10 local.
This file locks the decision language before the numbers are visible.

## Question

Arm B (paper recipe) applied the friction curriculum **and** the soft slip gate
(`kind=soft`, β=4, F_th=1N) together, so its 8.85× ep_len gain and 3.6× slip
*worsening* (vs Arm A) cannot be credit-assigned. Arm C re-runs the **exact Arm B
recipe with the single change `slip_gate.kind=hard`** to isolate the gate.

## Setup (only difference from Arm B = gate kind)

- Warm-start: `FixedStageFv7/model_4250.pt` (same as Arm B).
- Chain: S2a (DR_paper_S2a) → S2b (DR_paper_S2b) → S3 (DR_paper_S3), 250 iters each,
  hard gate active in **all three** stages (FULL warm-start — avoids the
  curriculum-history confound a branch-from-S2b would carry; Codex call).
- Reward weights (w_slip=0.5, w_force=0.05, F_max=150N), terrain, PPO hyperparams:
  identical to Arm B.
- Eval: `terrain_furrows_with_maize + DR_paper_S3`, 100 eps, command [0.3,0,0].
- Arm B seed-1 is **re-evaluated** with the new per-episode dump so the matched-
  distance comparison uses fresh, comparable per-episode data.

## Metric note (pre-registered)

The headline `slip_per_100m` confounds gate-effect with survival: an arm that walks
farther accumulates more tangential foot travel. The **confound-controlled metric is
the OLS slope of slip_distance ~ distance** (marginal slip per additional meter),
plus matched-distance bins. Within-episode windowing (first 0.72 m) needs per-step
logging and is deferred to a 3-seed confirmation if Arm C is consequential.

## Pre-registered decision rule

Compare Arm C against Arm B seed-1 (re-eval) on the confound-controlled metrics.

- **GATE IS HARMFUL** — Arm C keeps Arm B's ep_len (within ~1 SE, i.e. ep_len_C ≳ 400
  steps) **AND** Arm C's marginal slip/m slope is materially lower than Arm B's
  (≤ 0.8× B's slope). → The paper's soft gate was actively adding slip; recommend
  hard gate, and the survival win is attributable to the curriculum.
- **GATE IS INERT** — Arm C matches Arm B on both ep_len and slope (slope within
  ±20%). → The soft gate did nothing; the curriculum is the whole story. The slip
  non-replication is then NOT a gating artifact → the missing k_n/c_n compliant-
  contact physics moves to the front as the next investment.
- **GATE HELPS SURVIVAL** — Arm C ep_len materially below Arm B (< 400 steps). → The
  soft gate was contributing to survival; do not discard it.
- **INCONCLUSIVE** — eval crashed, per-episode data missing, or ep_len_C between the
  bands with no clear slope difference. → 3-seed run needed before any call.

## What this run CANNOT decide (pre-stated)

- It cannot make the paper's absolute slip claim (~11 /100m) replicate — that is the
  k_n/c_n compliant-contact hypothesis, untouched here.
- 1 seed gives a direction, not significance. A consequential result is confirmed
  with 3 seeds + per-step slip logging before it enters any writeup.
