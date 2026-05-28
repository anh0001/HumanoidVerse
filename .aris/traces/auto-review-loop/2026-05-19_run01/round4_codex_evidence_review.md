# Auto-Review-Loop trace — Pass 4 Round 4 (EVIDENCE REVIEW, FINAL/MAX)
- date: 2026-05-19
- skill: auto-review-loop
- reviewer: codex gpt-5.5 xhigh
- threadId: 019e3cf3-5e9a-73e2-805f-453cf1a98399 (FRESH; prior 019e38d8 lost to server restart during the 22h sweep — 2nd thread loss this project)
- difficulty: medium
- scores: design 5.5/10, evidence 2/10, **overall 3/10**
- verdict: **not ready (claim)**; headline claim **NOT supported** — clean null/negative
- loop status: TERMINATED at MAX_ROUNDS (4/4) without positive assessment

## Evidence submitted (ultra, seed 1 only, 50 ep/eval, 2048 envs)
- paired improvement DFH@DFH−RIGID@DFH = **−0.8** (26.2 vs 27.0), norm −2.8%
- FULL−SHUFFLED = +2.3 (26.2 vs 23.9) — moot since FULL ≯ RIGID
- dose match: FULL 4.2%wt / SHUF 4.5%wt (SHUF slightly heavier + more clipped)
- cap saturation WORSE than calib: eval stance_p95 ≈85–92% of 500N cap, clip 17–31% (calib was 0.55%); tot/robot ~4.1%wt (below 5–10% ULTRA target)
- SCOPE SHORTFALL disclosed: SEEDS=1 only (sweep ran SEEDS=1, not {1,2}) — below R3 ≥2-seed design-gate bar

## Reviewer rulings
- Score design 5.5 / evidence 2 / overall 3/10. Verdict: not ready.
- Headline claim NOT supported — clean null/negative on primary metric.
- Minimum honest next step: treat as negative/null Pass-4 result; PIVOT the claim
  unless a targeted repair re-run is done.
- Repair acceptance bar (if re-run): (a) lower `sinkage_drag_k` to de-saturate cap
  (stance p95 <70–75% of cap, clip <5%); (b) seeds 1–3 for all 3 conditions;
  (c) DFH_FULL beats RIGID on DFH eval in ≥2/3 seeds (pref. all 3), mean paired
  improvement positive, no outlier dependence; (d) DFH_FULL beats SHUFFLED under
  matched drag; (e) cap not dominating impulse; (f) training/eval curves to rule
  out asymmetric undertraining at 2000 iters.
- Honest framing if not re-run: "At the tested ultra single-seed setting, DFH did
  not improve deformable-terrain episode length over rigid training; spatial
  coupling may affect behavior vs shuffled control but did not yield the claimed
  generalization benefit."
- Pipeline-fix validity: sample_eps TB writer = legitimate instrumentation repair
  (mirrors ppo.py tags, same infos sources, stale events removed) — NOT a validity
  threat. num_envs=2048 eval = moderate external-validity limitation (eval layout
  == train layout weakens broad "generalization" reading) but does NOT explain
  away the null (negative even under DFH-favorable setup). Do not overclaim either way.

(Full verbatim response embedded in review-stage/AUTO_REVIEW.md Round 4 section.)
