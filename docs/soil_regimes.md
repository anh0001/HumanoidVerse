Soil Regimes and Disturbances

This repo now includes three soil presets and matching domain randomization profiles that map to the regimes you described. They work with both training and evaluation via simple Hydra overrides.

Regimes
- Rigid Reference: `+terrain=terrain_soil_rigid_reference +domain_rand=DR_soil_rigid`
- Moderate Tilled: `+terrain=terrain_soil_moderate_tilled +domain_rand=DR_soil_moderate`
- Challenging Wet/Loose: `+terrain=terrain_soil_challenging_wet +domain_rand=DR_soil_challenging`

Key properties
- Friction ranges (μ):
  - Rigid: [0.7, 0.9]
  - Moderate: [0.5, 0.7]
  - Challenging: [0.3, 0.5]
- Contact stiffness k_contact (N/m):
  - Rigid: 1e6
  - Moderate: [1e4, 5e4] (nominal 3e4)
  - Challenging: [1e3, 5e3] (nominal 3e3)

Disturbances
- Lateral pushes set by force (not arbitrary velocity): `push_force_N=50`, with duration sampled in `[0.12, 0.20]` s at random intervals. Internally, this is converted to a velocity impulse: Δv = F·Δt / m.
- Patchy low-friction zones are parameterized in the terrain configs for Isaac Sim; when available, thin static patches (2×2 m at μ=0.2) can be spawned to cover 10–30% of each env.

Evaluation
- Use the provided `humanoidverse/sample_eps.py` to gather episode-level metrics (distance, falls/100 m, slip/100 m). For quick smoke runs:
  - Rigid: `python humanoidverse/sample_eps.py +simulator=isaacsim +terrain=terrain_soil_rigid_reference +domain_rand=DR_soil_rigid num_envs=100 num_episodes=100 headless=True`
  - Moderate: same but with `terrain_soil_moderate_tilled` and `DR_soil_moderate`
  - Challenging: same but with `terrain_soil_challenging_wet` and `DR_soil_challenging`

Notes
- Furrows: In Isaac Gym, furrows come from `humanoidverse/utils/terrain.py`. In Isaac Sim, real furrows are now supported via a custom height‑field generator (`humanoidverse/simulator/isaacsim/furrow_terrain.py`) that maps `terrain_kwargs.type=furrows` to parallel grooves with configurable depth/spacing/orientation. Optional red patches still indicate low‑friction zones when `terrain.patchy_friction.enabled=True`.
- If total robot mass isn’t available from the simulator, the push logic falls back to a conservative 60 kg to avoid unrealistic impulses. You can specify `+robot.mass_kg=XX` to calibrate.
