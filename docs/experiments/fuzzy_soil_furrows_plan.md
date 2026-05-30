# Fuzzy-Soil-on-Furrows Transfer Test

**Question.** Does the paper's *fuzzy-soil* recipe (`docs/wcci2026_hunter.pdf`) — material curriculum + stance-gated slip + contact-force cap — improve over the rigid-furrows v7 baseline when warm-started from `FixedStageFv7/model_4250.pt` and evaluated on `terrain_furrows_with_maize`?

**Scope.** Coarse A/B, not a full ablation. Two arms × three seeds.

## Scope reduction vs. the paper

| Paper element | This experiment | Reason |
|---|---|---|
| Per-episode `k_n, c_n` (compliant contact) sampling | **dropped** — friction-only curriculum | IsaacLab 1.4.1 does not expose per-env material stiffness/damping cleanly; a USD-material mutation hook is 1–2 days of risky dev. The friction curriculum + slip-gate + force-cap are the parts that move paper Table II's slip-distance numbers most. |
| Fuzzy interpretation layer (Mamdani memberships, difficulty index `D`) | not implemented | Logging-only in the paper; out of scope for "does it work". |
| Vanilla PPO | replaced by **PPOROA** | Matches v7. We keep PPOROA on both arms so the comparison isolates the soil-recipe change. |
| PPO clip `ε = 0.2` | `ε = 0.1` | Matches v7's conservative refinement schedule (Codex prescription). |

Everything else (network `[512,256,128]` ELU, `γ=0.99`, `λ=0.95`, `T=24`, `E=5`, lr `2.5e-4`) already matches the paper.

## Arms

| | **Arm A — v7 baseline** | **Arm B — paper recipe** |
|---|---|---|
| Warm-start | `FixedStageFv7/.../model_4250.pt` | same |
| Algo | PPOROA | PPOROA |
| Reward | `loco/reward_hunter_locomotion.yaml` (hard slip gate) | `loco/reward_hunter_paper_soil.yaml` (soft slip gate β=4, F_th=1 N; force cap F_max=150 N w_force=0.05; w_slip=0.5) |
| DR | `DR_mild` (μ ∈ [0.7, 1.0], velocity push) | **chain:** `DR_paper_S2a` (μ∈[0.6,0.8]) → `DR_paper_S2b` (μ∈[0.45,0.65]) → `DR_paper_S3` (μ∈[0.35,0.55] + 35 N lateral force pushes) |
| Terrain | `terrain_furrows_stage1_easy` | same |
| Iters | 750 (continuous) | 250 × 3 stages (checkpoint-chained) |
| Seeds | {1, 2, 3} | {1, 2, 3} |

## Files added

| Path | Purpose |
|---|---|
| `humanoidverse/envs/legged_base_task/legged_robot_base.py` (edit at L963) | `_reward_penalty_slippage` honors `rewards.slip_gate.kind ∈ {hard, soft}`. Default `hard` ⇒ v7-compatible; `soft` ⇒ paper Eq. 4 + tangential velocity (world-z normal). |
| `humanoidverse/config/rewards/loco/reward_hunter_paper_soil.yaml` | Arm B reward: `slip_gate.kind: soft`, w_slip=0.5, F_max=150 N, w_force=0.05. |
| `humanoidverse/config/domain_rand/DR_paper_S2a.yaml` | μ ∈ [0.6, 0.8], no pushes. |
| `humanoidverse/config/domain_rand/DR_paper_S2b.yaml` | μ ∈ [0.45, 0.65], no pushes. |
| `humanoidverse/config/domain_rand/DR_paper_S3.yaml` | μ ∈ [0.35, 0.55], `push_force_N: 35`. |
| `scripts/paper_fuzzy_soil/train_armA_v7_baseline.sh` | One run per seed. |
| `scripts/paper_fuzzy_soil/train_armB_paper_chain.sh` | S2a→S2b→S3 chain per seed (auto-resolves latest checkpoint between stages). |
| `scripts/paper_fuzzy_soil/eval_arms.sh` | Evaluates all seeds × both arms on `terrain_furrows_with_maize` + `DR_paper_S3` (paper's hardest condition), 100 eps per seed (paper protocol). Appends to `logs/FuzzySoilFurrows/results.csv`. |

## Reproduction

```bash
# Arm A (v7 control), 3 seeds, sequential
for s in 1 2 3; do SEED=$s scripts/paper_fuzzy_soil/train_armA_v7_baseline.sh; done

# Arm B (paper recipe), 3 seeds, sequential
for s in 1 2 3; do SEED=$s scripts/paper_fuzzy_soil/train_armB_paper_chain.sh; done

# Joint evaluation
scripts/paper_fuzzy_soil/eval_arms.sh
```

## Primary metrics (paper Table II columns)

Reported per-arm as mean ± std over seeds; from `logs/FuzzySoilFurrows/results.csv` and per-seed eval logs:

- **Slip distance** (m / 100 m) — paper Eq. 6 numerator integrated over evaluation.
- **Fall rate** (falls / 100 m) — terminations triggered by contact / orientation.
- **Average episode length** (steps).
- **Velocity tracking error** (m/s) — paper §IV.
- **σ_pitch** (deg) — pitch variability (optional, paper Table II).

## Pre-flight audit (must pass before claiming success)

1. **Soft gate fires correctly.** Add a one-line debug print of `mu_stance` mean (per-foot) once per N iters; expect ~0 during swing, ~1 during stance. The "configured but inert" failure mode is the dominant silent risk.
2. **Force-cap penalty active.** Confirm `penalty_feet_contact_forces` reward shows nonzero values when `F_max=150 N` is exceeded (it will be — Hunter's 12.7 kg static load is ~62 N per foot, dynamic peaks reach 200–300 N during heel-strike on furrows).
3. **Friction range honored.** Sample 10 envs at episode-start and log realized `static_friction`; verify it matches the stage's `friction_range`.

## Decision rule

Arm B "works on furrows" iff, on the held-out `terrain_furrows_with_maize + DR_paper_S3` eval:

- mean episode length (Arm B) ≥ mean episode length (Arm A) − 1·SE, **and**
- slip distance (Arm B) ≤ 0.5 × slip distance (Arm A).

If both pass, the paper's recipe transfers. If only the slip target passes (Arm B walks shorter but slips less), it's a partial success — the curriculum is hurting raw tracking but the gating is doing its job. If neither, the recipe does not transfer cleanly on this furrow geometry, and the next step is to either (a) widen the curriculum (more iters per stage), or (b) revisit the dropped compliance term.
