# Furrows Curriculum Schedule (Staged Plan)

This playbook outlines a staged curriculum to train Hunter on furrowed soil. It complements, and does not replace, `docs/furrows_training_troubleshooting.md`.

## Goals
- Stabilize on shallow, aligned furrows (Stage 1),
- Consolidate on moderate furrows with mild yaw jitter (Stage 2),
- Reach full groove depth and orientation variability (Stage 3).

The plan uses dedicated terrain presets and a lightweight in‑run reward curriculum already wired into `locomotion_soil.yaml`.

## Stage Presets
- Stage 1 (Easy): `+terrain=terrain_furrows_stage1_easy`
  - depth `[0.10, 0.15]`, spacing `[2.5, 3.0]`, orientation `[0, 0]`, crest `0.05`.
- Stage 2 (Medium): `+terrain=terrain_furrows_stage2_medium`
  - depth `[0.15, 0.20]`, spacing `[2.0, 2.5]`, orientation `[-3, 3]`, crest `0.03`.
- Stage 3 (Full): `+terrain=terrain_furrows_stage3_full`
  - depth `[0.20, 0.30]`, spacing `[1.5, 2.0]`, orientation `[-5, 5]`, crest `0.0`.

Files live under `humanoidverse/config/terrain/`.

## Core Training Command (template)
```
python humanoidverse/train_agent.py \
  +robot=hunter/hunter +simulator=isaacsim \
  +exp=locomotion_soil +algo=ppo_soil \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  +rewards=loco/reward_hunter_soil_locomotion \
  +terrain=<STAGE_TERRAIN> \
  num_envs=2048 headless=True \
  project_name=SoilFurrows experiment_name=<RUN_NAME>
```

## Recommended Progression
1) Stage 1 Easy
```
+terrain=terrain_furrows_stage1_easy
```
Promote when all hold for a full eval window (e.g., 3–5 checkpoints):
- mean episode length ≥ 16 s,
- Episode/rew_termination ≥ −0.05/s,
- tracking_lin_vel ≥ 0.05/s.

2) Stage 2 Medium
```
+terrain=terrain_furrows_stage2_medium \
+checkpoint=logs/SoilFurrows/latest/model.pt
```
Promote when:
- mean episode length ≥ 18 s,
- Episode/rew_termination ≥ −0.03/s,
- feet_air_time shows non‑zero spikes and tracking improves.

3) Stage 3 Full
```
+terrain=terrain_furrows_stage3_full \
+checkpoint=logs/SoilFurrows/latest/model.pt
```
Target:
- mean episode length ~ 19–20 s,
- stable tracking, small orientation/height penalties.

## Single‑Run Alternative (mixed difficulty)
Use row‑curriculum in one run:
```
+terrain=terrain_tilled_soil_furrows
```
We enabled `curriculum: True` and `num_rows: 5` so envs sample multiple difficulty levels concurrently.

## In‑Run Reward Curriculum (no terrain rebuild)
Already enabled in `locomotion_soil.yaml` under `env.config.furrow_curriculum`. It progressively:
- lowers `feet_height_target` (0.12 → 0.06),
- re‑enables `penalty_feet_height` (0.0 → −1.0),
- restores `termination` (−80 → −200).

This keeps early phases forgiving while converging to the final objective.

## Metrics To Watch (TensorBoard)
- `Episode/rew_termination` (closer to 0 is better),
- `Episode/rew_tracking_lin_vel`, `Episode/rew_tracking_ang_vel`,
- `Episode/rew_feet_air_time` (should become non‑zero),
- `Env/action_clip_frac` (should remain low),
- `Train/mean_episode_length`.

## Helpful CLI Overrides (quick nudges)
- Ease depth/jitter temporarily:
```
+terrain.terrain_kwargs.depth_range_m=[0.10,0.15] \
+terrain.terrain_kwargs.orientation_deg=[0.0,0.0]
```
- Encourage foot clearance:
```
+rewards.feet_height_target=0.12 +rewards.reward_scales.penalty_feet_height=0.0
```
- Reduce termination dominance during a cold start:
```
+rewards.reward_scales.termination=-80
```

## Smoke Test
Run with fewer envs to verify wiring:
```
num_envs=64 +env.config.locomotion_command_resampling_time=4.0 headless=True
```

## Notes
- The reward/env/robot defaults have already been tuned in this repo to favor early survival and foot clearance on furrows.
- See `docs/furrows_training_troubleshooting.md` for fixes if episodes cap at ~10 s and rewards are flat.

