# Experiment Tracker — §2 Ablation Matrix

**Launched:** 2026-05-19 (core wave, bg PID 263226, commit 07e3814) · `logs/AblationMatrix_wave.log`
**Verified training:** A0_baseline_s1 reached iter 8/3000 at 2048 envs, no capacity/runtime error (env_spacing=2.5 fix applied + verified).
**Train:** furrows_s2 + YES_domain_rand, 3000 iters, 2048 envs, seeds {1,2,3}
**Held-out eval:** furrows_s3 (+DR) and soil_challenging (+DR) via sample_eps, 256 envs × 64 ep
**Calibration:** PASSED — all 5 arms compose + run train→eval end-to-end (20-iter dry, ALL ARMS COMPLETE).

| Arm | algo | obs | gait | isolates | status |
|-----|------|-----|------|----------|--------|
| A0_baseline | ppo | singlestep_withlinvel | off | baseline | running (queued) |
| A3_hist50 | ppo | hist50_wolinvel | off | memory alone | queued |
| A5_roa | ppo_roa | history_wolinvel | off | adaptation alone | queued |
| A8_gait | ppo | singlestep_withlinvel | on | gait-reg alone | queued |
| A9_full | ppo_roa | history_wolinvel | on | proposed stack | queued |

Sequenced (GPU externally contended): ~15 train runs + 30 evals. Est. ~30–60 GPU-h.

## Claim gate (EXPERIMENT_PLAN §3) — fill from eval logs
Headline "drastic" holds iff A9 vs A0 on held-out: **≥2× ep_len OR ≥50% fewer falls**,
**A9 > A3** (memory-only doesn't explain it), **A9 > A8** (gait-only doesn't), no energy/gait regression.

| Arm | seed | furrows_s3 falls | furrows_s3 ep_len | soil_chal falls | soil_chal ep_len |
|-----|------|------------------|-------------------|-----------------|------------------|
| _to be filled by analyze script after wave_ | | | | | |

Baseline reference (§0-A, model_910050): furrows_s3 64/64 falls, ~20 steps; soil_chal 64/64, ~20 steps; plane 0/16, 401 steps.
