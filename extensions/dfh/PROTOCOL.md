# DFH Evaluation Protocol (Pass 3 onward)

## Seed Exclusion Rule (pre-declared)

A run is classified as **simulator-init failure** and excluded from analysis iff:
- `Train/mean_episode_length` at iter ≥ 1000 is `< 100` AND
- `Episode/rew_termination` mean over last 20 iters is `≤ -3.5` (near the -4 max-penalty floor).

Empirical observation: seed=0 collapses universally across DFH_FULL / SHADOW / RIGID / SHUFFLED with ep_len ~45 at 5000 iters and term -4.0 — this is an IsaacSim env-spawn condition, not a DFH-attributable failure.

**Going forward:** use seeds `{1, 2, 3, 4, 5}`. Seed 0 is excluded by the rule above.

## Paired Cross-Eval Protocol

For each tuned drag setting, train three policies per seed `s ∈ {1, 2}`:
- `RIGID_s`     — rigid furrowed heightfield, matched friction (PhysX only)
- `SHADOW_s`    — DFH state computed but force_coupling.enabled=False
- `DFH_FULL_s`  — DFH state + force coupling active

Then evaluate every checkpoint on:
- `eval_rigid`  — rigid furrows, no DFH forces
- `eval_dfh`    — same DFH config as the corresponding train condition

**Primary metric:** paired regret
```
regret_s = ep_len(DFH_FULL_s @ eval_dfh) - ep_len(RIGID_s @ eval_dfh)
```
Reported per seed; aggregated as mean ± seed-level std.

**Force budget** logged during training and eval:
- `dfh_mean_applied_drag_n`, `dfh_mean_normal_force_n`
- `dfh_applied_drag_pct_weight = applied_drag_n / (mass × g)`
- `dfh_mean_sink_m`, `dfh_max_sink_m`
- `dfh_drag_clipped_frac` (fraction of contacts hitting `max_drag_force_n` cap)

## Success Bar
- Mean applied drag in `[30, 60] N` (5–10% Hunter body weight) on DFH_FULL.
- SHADOW applied drag = 0 (sanity).
- `regret_s > 0` for both s∈{1,2} (DFH-trained beats rigid-trained on DFH eval).
- Mean regret ≥ 10% of RIGID baseline ep_len.

## Anti-Confound: Reward Shaping
`penalty_sinkage_excess` weight held at 0 during the primary sweep so the regret signal is attributable to **dynamics**, not reward shaping. Reward sweep `{0, -1, -2}` runs only after dynamics regret is established.
