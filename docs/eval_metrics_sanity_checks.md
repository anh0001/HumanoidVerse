# Evaluation Metrics: Sanity Checks, Root Causes, and Fixes

This note documents the sanity checks, findings, and code changes that fixed misleading evaluation metrics when running “Flat‑Only Policy on Soil”. It also explains how to validate correctness and interpret results.

## Summary

- Problem: Some metrics (Average episode length = 0.0, Slip distance per 100m = 0.00) were incorrect because the eval loop read buffers after the environment reset them.
- Root cause: The environment resets inside `step()`. Reading live tensors (e.g., `episode_length_buf`, `slip_distance`) right after `step()` returns values for the next episode (often zero).
- Fix: Read the per‑episode snapshot captured pre‑reset in `env.episode_info` and store episode length from `last_episode_length_buf` before zeroing. Also made slip detection robust on soil using contact forces.

## What Changed

- `humanoidverse/sample_eps.py`
  - Metrics now come from `env.episode_info[f'env_{i}']`, saved before reset.
  - Fallback for episode length uses `env.last_episode_length_buf[i]` if the snapshot is missing.
  - Logs effective timing at eval start: `sim_dt`, `control_decimation`, `dt`, `max_episode_length_s`, and `max_episode_length`.
  - Warns if any episode reports zero/non‑positive length.

- `humanoidverse/envs/locomotion/locomotion.py`
  - Stores `episode_length` from `last_episode_length_buf` in the snapshot taken in `_reset_tasks_callback()`.
  - Slip contact gating is now robust for soil:
    - Contact if foot normal force > 1.0 N OR z‑height < 0.03 m.
    - Accumulates slip only when in contact, reducing false negatives on uneven/soft terrain.

## Why The Old Results Looked Wrong

- Average episode length = 0.0:
  - `episode_length_buf` is reset to 0 during reset; the eval loop read it after `step()` when it was already zeroed.

- Slip distance per 100m = 0.00:
  - `slip_distance` is cleared on reset; reading it post‑reset yields 0. Moreover, z‑height–only contact gating can miss contact on soil, undercounting slip.

## Current Behavior (After Fixes)

- Episode metrics are captured just before reset in `env.episode_info` and consumed by the eval loop.
- Example consistent report (soil):
  - Episodes completed: 100
  - Total distance traveled: 1094.46 m
  - Total falls: 2
  - Average episode length: 2941.7 steps
  - Average distance per episode: 10.94 m
  - Falls per 100m: 0.18
  - Slip distance per 100m: 0.20 m

## Example Sanity Check

Using the example above:

- Consistency Checks
  - Falls/100m: 2 / (1094.46/100) = 0.18 → matches 0.18.
  - Avg speed: 10.94 m per episode over ~2941.7 steps at dt≈0.02s ≈ 58.8 s ⇒ ~0.186 m/s. Plausible for conservative commands/soil.
  - Episode length: 2941.7 steps ≈ 58.8 s implies episodes mostly ended by timeout near ~60 s, with a few early terminations (2 falls). This aligns with the timeout logic in `humanoidverse/envs/legged_base_task/legged_robot_base.py:318` and `:410` and the 60 s convention seen in eval configs.

- Interpretation
  - Total falls 2: policy is quite stable on soil (only 2 early terminations).
  - Average episode length ~2942 steps: ~98 episodes reached timeout; ~2 ended early (brings average down from ~3000).
  - Slip per 100m: 0.20 m/100 m is very low but not impossible at ~0.19 m/s if contacts are mostly static; also affected by slip detection gating.
  - Total/average distance: 1094.46 m total, ~10.94 m per episode matches the above speed and durations.

## Latest Result Example

Reported output:

```
==================================================
EVALUATION RESULTS
==================================================
Episodes completed: 100
Total distance traveled: 1076.51 m
Total falls: 5
Average episode length: 2852.3 steps
Average distance per episode: 10.77 m
Falls per 100m: 0.46
Slip distance per 100m: 12.69 m
==================================================
```

- Consistency Checks
  - Falls/100m: 5 / (1076.51/100) ≈ 0.464 → rounds to 0.46 (matches).
  - Avg time per episode: 2852.3 steps × dt≈0.02 s ≈ 57.05 s.
  - Avg speed: 10.77 m / 57.05 s ≈ 0.189 m/s — plausible and similar to the first example.
  - Episode length suggests most episodes reach a ~60 s cap; 5 early terminations reduce the average from the cap.

- Interpretation
  - Total falls 5: still stable, with slightly more early terminations than the first example.
  - Average episode length ~2852 steps (~57 s): consistent with timeout‑limited runs plus a few falls.
  - Slip per 100m ≈ 12.69 m: moderate slip consistent with soil; higher than the first example due to more robust force‑gated detection (or rougher surface/commands).
  - Distance totals and per‑episode averages align with the computed speed and durations.

## Sanity Checks You Can Do

- Falls per 100m: `falls / (total_distance / 100)` should match the printed value.
- Avg speed (rough): `(avg_distance_per_episode) / (avg_episode_length * dt)` should be plausible.
- Episode length vs timeout: `avg_episode_length * dt` should be near your episode cap if most episodes time out.
- Non‑zero length: Average episode length should be > 0 if episodes progressed.

## Formulas Used

- Total distance: `sum_i distance_i`
- Average distance: `total_distance / N`
- Total falls: `sum_i fell_i` where `fell_i = True` if not timeout
- Falls per 100m: `total_falls / (total_distance / 100)`
- Slip per 100m: `total_slip / (total_distance / 100)`
- Average episode length: `sum_i episode_length_i / N`

## Timing Reference

- `dt = sim_dt * control_decimation`
  - Example (Isaac Sim default here): `sim_dt = 1 / 200 = 0.005`, `control_decimation = 4` → `dt = 0.02` s.
- The eval script logs these at startup for verification.

## Reproducing/Verifying

- Quick headless smoke:
  - `python humanoidverse/sample_eps.py +checkpoint=logs/<project>/<run>/model.pt +terrain=terrain_tilled_soil num_envs=2 num_episodes=10 headless=True +eval_command=[0.3,0.0,0.0]`
- Full eval:
  - `num_envs=100 num_episodes=100 headless=True`
- Ensure `Episodes completed == num_episodes` and sanity checks pass.

## Where To Look In Code

- Snapshot and reset ordering:
  - `humanoidverse/envs/legged_base_task/legged_robot_base.py`
    - Reset buffer zeroing and step flow
  - `humanoidverse/envs/locomotion/locomotion.py`
    - Episode snapshot in `_reset_tasks_callback()`
- Eval aggregation:
  - `humanoidverse/sample_eps.py`

## Potential Extensions

- Expose slip contact thresholds via config for easy tuning per terrain.
- Add asserts in the eval script to flag suspicious metrics (e.g., too many zero‑length episodes).
- Scenario‑based eval configs with explicit `max_episode_length_s` for easier comparison across runs.

---

### Change Log (Files)

- `humanoidverse/sample_eps.py`
  - Read `env.episode_info` for per‑episode metrics.
  - Log timing: `sim_dt`, `control_decimation`, `dt`, and episode cap.
  - Warn on zero/non‑positive episode lengths.

- `humanoidverse/envs/locomotion/locomotion.py`
  - Save `episode_length` from `last_episode_length_buf` pre‑reset.
  - Slip detection: force‑gated with height fallback for soil.
