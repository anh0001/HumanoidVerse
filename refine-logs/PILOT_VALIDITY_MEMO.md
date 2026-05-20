# Warm-start Pilot — Validity Memo (Codex Option B)

**Date:** 2026-05-20  ·  Codex thread 019e4348 (continuation of 019e4287)

## Verdict: **PILOT VALID** ✅

No structural bug. A0_s1's flat-collapse is policy-collapse-on-killer-terrain, not a reward/obs/command channel defect. Pilot warm-start is producing non-degenerate plane-pretrained policies as designed.

## Evidence

### A0_s1 autopsy (3000 iters from-scratch, furrows_s2+DR)
All 3000 iter blocks parsed from `logs/AblationMatrix/A0_baseline_s1/train.log`:

| iter | ep_len | mean_reward | tracking_lin_vel | penalty_ang_vel_xy |
|------|--------|-------------|------------------|-------------------|
| 0 | 23.95 | −5.44 | 0.0011 | −0.107 |
| 100 | 30.61 | −4.34 | 0.0041 | −0.157 |
| 1000 | 27.91 | −3.78 | 0.0041 | −0.138 |
| 2999 | 29.28 | −3.45 | 0.0038 | −0.128 |

- ep_len trajectory: **first=23.95 / latest=29.28 / min=23.95 / max=37.25** across all 3000 iters.
- Policy climbed from 24→30 steps in 100 iters, then **plateaued for 2900 more iters**. Never escaped fall-in-0.6s regime.
- Reward signs/scales sensible, action_clip_frac=0 (no saturation), penalty_ang_vel_xy modestly improved (-0.16→-0.13).
- **Diagnosis:** from-scratch furrows_s2+YES_DR is too hard within 3000 iters at this network/lr. Not a bug, just unlearnable budget. Validates Codex's reframe.

### Pilot poison check
Diff of `config.yaml` between A0_s1 and pilot's A0_baseline_P:
- **reward_scales: identical** (all 18 keys match, including gait scales correctly 0 for OFF arms).
- **command ranges: identical** (lin_vel_x/y ∈ [−1,1], yaw ∈ [−1,1], heading ∈ [−π,π]).
- **DR diffs: intentional and correct** — pilot Stage P disables all DR (NO_domain_rand) for plane warm-up; A0_s1 had DR on because it was from-scratch furrows. Stage F (queued) will re-enable DR via `+domain_rand=YES_domain_rand` matching A0_s1.

### Live signal — pilot Stage P A0_baseline at iter 340/500
| iter | ep_len | mean_reward | tracking_lin_vel |
|------|--------|-------------|------------------|
| 0 | 23.68 | −8.89 | 0.0016 |
| 85 | 38.76 | −1.98 | 0.0059 |
| 170 | 138.12 (~2.8 s) | −1.15 | 0.0255 |
| 255 | **1001** (full 20 s cap) | +3.06 | 0.2411 |
| 340 | 995.64 (sustained) | +7.24 | 0.2249 |

→ Plane warm-up learns 0→walking in ~250 iters. The warm-loaded Stage F has a real chance to show arm separation on furrows_s2+DR. **This is the exact "sane plane learning signal" Codex's falsification check required.**

## Action

Per Codex's falsification path: pivot from B (validity audit) to **A (claim-gate analyzer)**. Next deliverable: parser that consumes `logs/WarmstartPilot/pilot_summary.csv` when the pilot finishes (~17 h from now), applies the 4-branch decision rule (1: relaunch matrix / 2: build curriculum / 3: re-scope / 4: refine), prints the verdict and the ready-to-fire next-step commands.

## Falsification (would invalidate this verdict)
- If pilot Stage F shows the same flat 24→30 ep_len curve as A0_s1 despite warm-start → not "killer terrain too hard from scratch" but a deeper bug.
- If pilot Stage P A0 _stops_ improving and falls back below ep_len 100 by iter 500 → unstable training (LR/entropy/PPO clip issue).
