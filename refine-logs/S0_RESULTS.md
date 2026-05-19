# §0 Gate Results

**Date:** 2026-05-19  ·  Plan: `refine-logs/EXPERIMENT_PLAN.md` §0

## §0-A — Baseline fragility (Half A)

**Baseline:** `logs/CurriculumStage2/model_910050.pt` (clean curriculum Stage2, trained Aug/Sep 2025).
**Eval:** `sample_eps.py`, num_envs=256, 64 episodes, eval_command=[0.3,0,0], headless.

| Condition | Terrain | DR | Falls | Avg ep len (steps) | Dist (m) |
|---|---|---|---|---|---|
| soil_moderate | terrain_soil_moderate_tilled (6×6) | DR_soil_moderate | 64/64 | 16.0 | 0.69 |
| soil_challenging | terrain_soil_challenging_wet (6×6) | DR_soil_challenging | 64/64 | 19.7 | 1.13 |
| furrows_s2 | terrain_furrows_stage2_medium | YES_domain_rand | 64/64 | 22.3 | 0.73 |
| furrows_s3 | terrain_furrows_stage3_full | YES_domain_rand | 64/64 | 20.4 | 0.76 |

### Decisive sanity control — RESOLVED ✅
Same checkpoint, `terrain_locomotion_plane` + `NO_domain_rand`, 8 s cap:

| Condition | Falls | Avg ep len | Dist |
|---|---|---|---|
| **plane, NO DR (sanity)** | **0 / 16** | **401 steps (full 8 s cap)** | 2.01 m |

The baseline walks flawlessly on plane (0 falls, runs to cap) → **eval harness is valid, NOT a checkpoint mismatch**. Therefore the soil/furrows 100%-collapse is **GENUINE catastrophic fragility**.

### §0-A verdict: **PASS — drastic headroom confirmed and genuine**
Baseline is stable on flat ground but **100%-fatal within ~0.3–0.45 s on soil/furrows under DR**. This is a 0→1 gap, not a marginal one. Reviewer gating-question #1 ("drastic gain only plausible if baseline is fragile") is satisfied decisively. Proceed to §0-B latent-identifiability probe, then the ablation matrix.

## Repo bugs fixed en route (pre-existing)
1. `humanoidverse/config/domain_rand/domain_rand_base.yaml:17-18` — `x/y : [-0.1., 0.1.]` (invalid double-dot floats parsed as strings → `torch.tensor` crash). Fixed to `[-0.1, 0.1]`. Affected any DR inheriting base with `randomize_base_com:True`.
2. Soil terrains are 1×1-tile (capacity 9 envs); launcher now overrides `++terrain.num_rows/num_cols=6` for soil.

## §0-B — Latent-identifiability probe ✅ PASS

**The actual falsification gate** (reviewer's single highest-risk assumption: can a short proprioceptive history infer terrain/dynamics *early*, pre-fall?).

Built: `humanoidverse/collect_probe_data.py` (records the exact policy `actor_obs` stream + episode age + fall flag + privileged com/push under each regime), `scripts/s0_gate/train_probe.py` (TCN over in-episode windows; windows strictly within one episode since the deployment history buffer restarts each episode). 4 regimes × 128 envs × 250 steps, obs_dim=42.

| Window L | 4-way regime acc (chance 0.25) | safe(plane)-vs-killer acc |
|---|---|---|
| 8 (~0.16 s) | 0.807 | **0.999** |
| 12 | 0.835 | 0.998 |
| 16 | 0.877 | 1.000 |

`early(age≤8) acc` for L≥12 is `nan` = empty bucket (a length-L window cannot end before episode-age L); L=8 early bucket = 0.670 (≫ chance). Report: `refine-logs/S0B_PROBE_REPORT.md`.

**Verdict: PASS.** Within ~0.16 s of blind proprioception the terrain regime — critically, safe-vs-killer — is near-perfectly legible. The ROA premise (privileged latent is recoverable from proprio history early enough to act) is **empirically supported**, not assumed.

## §0 GATE: PASS (both halves)
- §0-A: drastic headroom is real and genuine (0→1 gap on soil/furrows).
- §0-B: the dominant idea's core assumption holds.
→ Green-light to build the ROA + periodic-symmetry stack and run the §2 ablation matrix. ROA is the right dominant idea (not the history-only fallback).

## Artifacts
- `scripts/s0_gate/{run_fragility.sh,run_probe.sh,train_probe.py}`
- `humanoidverse/collect_probe_data.py`
- `refine-logs/S0B_PROBE_REPORT.md`, `logs/S0Gate/**`
- Repo bug fixes (pre-existing): `domain_rand_base.yaml` invalid floats; soil-terrain tile capacity workaround.
GPU budget: ≈ well under the <2 GPU-h §0 cap (GPU was externally contended throughout).
