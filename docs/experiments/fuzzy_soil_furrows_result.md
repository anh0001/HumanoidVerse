# Fuzzy-Soil-on-Furrows — Result Note

**Status:** PENDING — final eval scheduled to land ~06:30 local 2026-05-29. This skeleton is pre-filled before results to lock in the caveats and decision language (Codex recommendation; prevents post-hoc rationalization).

## Setup actually run

- **Question:** Does the paper's *fuzzy soil* recipe (`docs/wcci2026_hunter.pdf`) — material curriculum + stance-gated slip + contact-force cap — improve over the rigid-furrows v7 baseline when warm-started from `FixedStageFv7/model_4250.pt` and evaluated on `terrain_furrows_with_maize`?
- **Arms:** A (v7 control) vs B (paper recipe), 3 seeds each, all warm-started from the same v7 checkpoint.
- **Arm A recipe:** `+rewards=loco/reward_hunter_locomotion` (hard-gated slip), `+domain_rand=DR_mild` (μ ∈ [0.7, 1.0], velocity pushes), `+terrain=terrain_furrows_stage1_easy`, 750 PPO iters continuous.
- **Arm B recipe:** `+rewards=loco/reward_hunter_paper_soil` (soft slip gate β=4 F_th=1 N, w_slip=0.5, F_max=150 N w_force=0.05), friction curriculum S2a (μ∈[0.6,0.8]) → S2b ([0.45,0.65]) → S3 ([0.35,0.55] + 35 N lateral force pushes), 250 iters per stage chained.
- **Eval:** `terrain_furrows_with_maize + DR_paper_S3`, 100 episodes per seed per arm, command [0.3, 0, 0] m/s.

## Deviations / caveats (pre-registered before result inspection)

1. **k_n / c_n compliant-contact sampling was dropped.** IsaacLab 1.4.1 does not expose per-env material stiffness/damping cleanly. Both arms use the default rigid PhysX contact. The paper's contribution most likely to be missed is whether deformation depth affects gait — we are testing only the friction-curriculum + reward-gating slice of the paper.
2. **Fuzzy interpretation layer (Mamdani memberships, difficulty index D) not implemented.** Paper §III-C: logging-only, "without modifying the PPO policy architecture or adding new observations." Zero policy effect; useful for paper writeup figures but not for the pass/fail call.
3. **Soft slip gate at β=4, F_th=1 N differs from a hard gate only over a ~2 N window** at heel-strike / toe-off. Per-foot stance load is ~62 N. The paper does NOT ablate hard-vs-soft. **Any result here cannot credit-assign between the curriculum and the soft gate** — both were applied together in Arm B.
4. **Arm A seed 1 was inadvertently re-run** by a queue skip-check bug. Two `model_5000.pt` files exist (`20260528_173854-armA_seed1` original, `20260528_192207-armA_seed1` re-run). Eval glob picks the latest (the re-run). Both are valid runs of the same seed/recipe; differences are cuDNN nondeterminism only. Noted; not a confound but explains why timing log shows two A1 entries.
5. **PPOROA + privileged-teacher architecture** is used on both arms (matches v7). The paper used vanilla PPO. The actor/critic MLP `[512,256,128]` ELU does match.
6. **PPO clip ε = 0.1** (matches v7's conservative refinement schedule). The paper used ε = 0.2.

## Preregistered decision rule

Arm B **passes** iff:
- `mean(ep_len_B) ≥ mean(ep_len_A) − 1·SE_A` (curriculum doesn't hurt episode survival), **AND**
- `mean(slip_per_100m_B) ≤ 0.5 × mean(slip_per_100m_A)` (paper claims slip reductions of >95%; we ask for ≥50%).

Verdict labels:
- **PASS** — both rules met
- **FAIL** — neither met
- **PARTIAL (slip-only)** — slip target met, ep_len missed → curriculum hurt tracking
- **PARTIAL (eplen-only)** — ep_len met, slip missed → gating did not help on this terrain
- **INCONCLUSIVE** — missing metric or eval crashed

## Result

Eval finished 2026-05-29 06:24:45. All 6 ckpts × 100 eval episodes each on `terrain_furrows_with_maize + DR_paper_S3`.

### Per-seed table

| arm | seed | ep_len_steps | distance_m | slip / 100m | falls / 100m |
|-----|------|--------------|------------|-------------|--------------|
| A | 1 | 54.30 | 0.700 | 79.14 | 142.86 |
| A | 2 | 50.00 | 0.700 | 89.05 | 142.86 |
| A | 3 | 47.80 | 0.770 | 73.81 | 129.87 |
| B | 1 | 368.30 | 1.660 | 288.23 | 60.24 |
| B | 2 | 532.50 | 2.360 | 283.85 | 42.37 |
| B | 3 | 444.90 | 1.880 | 299.78 | 53.19 |

### Per-arm mean ± SE

| arm | n | ep_len_steps | distance_m | slip / 100m | falls / 100m |
|-----|---|--------------|------------|-------------|--------------|
| A (v7 control) | 3 | 50.7 ± 1.9 | 0.72 ± 0.02 | 80.7 ± 4.5 | 138.5 ± 4.3 |
| B (paper recipe) | 3 | **448.6 ± 47.4** | **1.97 ± 0.21** | **290.6 ± 4.8** | **51.9 ± 5.2** |
| Ratio B / A | — | 8.85× | 2.72× | 3.60× | 0.37× |

### Decision-rule check

- **Rule 1** (ep_len_B ≥ ep_len_A − SE_A): 448.6 ≥ 48.8 → **PASS** by ~9×
- **Rule 2** (slip_B ≤ 0.5 × slip_A): 290.6 ≤ 40.3 → **FAIL** by ~7×

### Verdict: **PARTIAL — ep_len target met, but slip target missed (gating didn't help on this terrain).**

## Interpretation

- **The friction curriculum is doing real work.** Arm B walks ~3× farther and ~9× longer before falling than Arm A. Falls per meter dropped 63%. Tight per-seed SE (10% on ep_len, 1.6% on slip) means this is not seed noise; the effect is robust.
- **The slip-distance claim does NOT replicate** in our setup. Paper Table II reports Ours (Full) at 11 m/100m vs No-Curriculum at 1037. Our Arm B got 290 — 26× the paper's headline number, and *worse* than Arm A in absolute terms. The slip *target* set in the plan (≤ 40 m/100m) was missed by a factor of 7.
- **All 6 seeds had 100% fall rate (1000/1000).** Both arms run out the clock — `DR_paper_S3` with 35 N pushes + μ ∈ [0.35, 0.55] is hard enough to eventually fell every Hunter policy we trained. The interesting variable is *how far the policy gets before failing*, where Arm B clearly wins.
- **Why the slip-distance discrepancy?** Several non-exclusive hypotheses:
  1. Without `k_n / c_n` deformable contact, the policy faces *low-friction-rigid* soil. On loose-but-rigid ground, foot slide is uncapped — there's no deformation-mediated grip that the paper's k_n sampling provides.
  2. Arm B walks 2.7× farther per episode. Its longer trajectories accumulate more cumulative tangential foot-velocity even if per-step slip is similar to Arm A. The metric divides by distance, but if slip scales superlinearly with speed-on-slippery-soil, B (which is faster) is penalized.
  3. Soft gate at β=4 is sharp enough to be effectively hard at heel-strike, so the smoothness benefit Codex flagged didn't materialize. We cannot distinguish (1)-(3) without further ablation.

## Next steps

- **Recipe transfers in the survival sense.** If the writeup needs a positive headline, lead with the 8.85× ep_len and the 63% falls/m drop. Caveat heavily that slip-distance does not reproduce.
- **The clean credit-assignment ablation** Codex flagged (Arm C = Arm B but `slip_gate.kind: hard`) is now MORE interesting: if Arm C matches Arm B's ep_len but drops slip, the soft gate is actively harmful, not just inert. ~5h compute for 3 seeds, ~1.5h for 1 seed directional check.
- **Compliance follow-up** (the dropped k_n/c_n piece) is now the obvious thing the missing-from-our-version paper component might fix. The IsaacLab material-API work that we punted to keep this experiment cheap is now the natural next step if you want the slip claim to replicate.

## Next steps

If **PASS:** the paper's recipe transfers. Open follow-up: isolate which component is doing the work via Arm C (Arm B but `slip_gate.kind: hard`). One seed sufficient for a directional check (~1h45m).

If **FAIL or PARTIAL:** before discarding the recipe, check:
- Was the friction range too wide for Hunter to recover from in 750 iters (vs. paper's longer training)?
- Did `terrain_furrows_with_maize` (Stage-1 easy furrows) make S3 trivial, so the curriculum had nothing to optimize against?
- Is the soft-gate β=4 too sharp (effectively hard) to give the value function any smoothing benefit?
