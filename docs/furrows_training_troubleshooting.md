# Furrows Training Troubleshooting Guide

This document explains why `terrain_tilled_soil` works but `terrain_tilled_soil_furrows` breaks during training, and provides minimal, high-leverage changes to make furrows trainable.

## Root Cause Analysis

### Terrain Complexity Mismatch

**Flat soil preset** has micro-roughness (≈15 mm) with no large obstacles, so low step height and conservative commands suffice for successful locomotion.

**Furrows introduce significant challenges:**
- 0.20–0.30 m deep troughs
- Steep sidewalls requiring higher foot clearance
- Pitch/roll transients from terrain changes
- Humanoid must lift feet much higher than flat terrain training

### Current Configuration Issues

#### 1. Low Foot-Clearance Penalty
- **Location**: `reward_hunter_soil_locomotion.yaml:74–79, 51–55`; `locomotion.py:240–244`
- **Issue**: `feet_height_target: 0.02` with penalty beyond ±0.02 m tolerance punishes >4 cm foot lift
- **Problem**: Furrows require 12+ cm clearance, but policy is penalized for this behavior

#### 2. Command Zeroing
- **Location**: `locomotion.py:111–115`; `locomotion_soil.yaml:22–25`
- **Issue**: Commands with magnitude ≤ 0.2 are zeroed, but command range is only ±0.3
- **Problem**: Most commands become zero → no forward motion → no gait development

#### 3. Termination Dominance
- **Location**: `legged_robot_base.py:133–141, 654–656, 411–413`
- **Issue**: Termination penalty of -200 scaled by dt=0.02 ⇒ -4 per fall (logged as -0.2/s)
- **Problem**: Positive reward terms (~+0.01/s) cannot compete with termination penalties

## Solution: Staged Approach

### Stage 1: Terrain Easing for Furrows

Reduce terrain difficulty while policy learns foot-clearance behavior:

**File**: `terrain_tilled_soil_furrows.yaml:62–65`
```yaml
terrain_kwargs:
  depth_range_m: [0.10, 0.15]  # Reduced from [0.20, 0.30]
  spacing_range_m: [2.0, 2.5]  # Gentler slopes
  orientation_deg: [0.0, 0.0]  # Run along furrows initially
  crest_offset_m: 0.05  # Optional: smoother transitions
```

### Stage 2: Reward Shaping Alignment

#### Allow Higher Foot Swing
**File**: `reward_hunter_soil_locomotion.yaml:51–55`
```yaml
feet_height_target: 0.12  # Increased from 0.02
penalty_feet_height: 0.0  # Disable penalty initially
```

#### Increase Tracking Signal Strength
**File**: `reward_hunter_soil_locomotion.yaml:28–31`
```yaml
tracking_lin_vel: 2.0  # Increased from default
tracking_ang_vel: 1.5  # Increased from default
```

#### Reduce Termination Impact
**File**: `reward_hunter_soil_locomotion.yaml:57`
```yaml
termination: -80.0  # Reduced from -200.0
```

### Stage 3: Command and Environment Tuning

#### Reduce Command Zeroing
**File**: `locomotion.py:111–115`
```python
if torch.norm(command) <= 0.05:  # Reduced from 0.2
    command[:] = 0.0
```

#### Increase Command Variety
**File**: `locomotion_soil.yaml:22–26`
```yaml
lin_vel_x: [-0.4, 0.4]  # Slightly wider range
ang_vel_yaw: [-0.4, 0.4]  # Slightly wider range
locomotion_command_resampling_time: 6.0  # Increased from 15.0
```

#### Tolerate Trough Stepping
**File**: `locomotion_soil.yaml:37`
```yaml
termination_min_base_height: 0.35  # Slightly lower threshold
```

### Stage 4: Actuation Headroom

**File**: `hunter.yaml:106–110`
```yaml
action_scale: 0.35  # Increased from 0.25 for better foot lift
```

## Quick CLI Overrides (No File Edits Required)

### Terrain Easing
```bash
+terrain.terrain_kwargs.depth_range_m=[0.10,0.15] \
+terrain.terrain_kwargs.spacing_range_m=[2.0,2.5] \
+terrain.terrain_kwargs.orientation_deg=[0.0,0.0] \
+terrain.terrain_kwargs.crest_offset_m=0.05
```

### Reward Modifications
```bash
+rewards.reward_scales.penalty_feet_height=0.0 \
+rewards.feet_height_target=0.12 \
+rewards.reward_scales.tracking_lin_vel=2.0 \
+rewards.reward_scales.tracking_ang_vel=1.5 \
+rewards.reward_scales.termination=-80
```

### Environment Settings
```bash
+env.config.locomotion_command_resampling_time=6.0 \
+env.config.termination_scales.termination_min_base_height=0.35
```

### Robot Configuration
```bash
+robot.control.action_scale=0.35
```

## Progressive Training Strategy

1. **Stage 1**: Use shallow, aligned furrows (depth 0.10-0.15m, orientation 0°)
2. **Stage 2**: Gradually increase depth (0.15→0.20→0.30m)
3. **Stage 3**: Introduce spacing variation (2.5→2.0→1.5m)
4. **Stage 4**: Add orientation randomization (0°→±5°→±15°)
5. **Stage 5**: Re-enable foot height penalties with higher targets

## Why This Works

- **Shallower, aligned furrows** remove "wall" obstacles while policy learns foot lifting
- **Removing foot-height penalty** stops punishing required clearance behavior
- **More frequent, non-zero commands** create step/contact events for reward learning
- **Reduced termination magnitude** prevents single large penalty from overwhelming exploration
- **Larger action scale** provides controller headroom for higher foot trajectories

## Monitoring Success

Look for these indicators during training:
- Increasing `feet_air_time` rewards
- Decreasing termination frequency
- Higher `feet_height_target` achievement without penalties
- Stable forward velocity tracking on furrow terrain

## Common Pitfalls

1. **Too aggressive terrain progression** - increase difficulty gradually
2. **Keeping foot height penalties** - disable initially, re-enable carefully
3. **Insufficient action scale** - humanoid needs torque headroom for clearance
4. **Command starvation** - ensure commands aren't being zeroed excessively

## Applied Changes Summary

Applied the furrow-stability bundle. Summary of focused changes:

### Terrain easing

**humanoidverse/config/terrain/terrain_tilled_soil_furrows.yaml**
- `depth_range_m: [0.10, 0.15]` (was [0.2, 0.3])
- `spacing_range_m: [2.0, 2.5]` (was [1.5, 2.0])
- `orientation_deg: [0.0, 0.0]` (start aligned)
- `crest_offset_m: 0.05` (gentler troughs)

**Rationale**: reduce sidewall/trapping while the policy learns foot clearance.

### Rewards tuning

**humanoidverse/config/rewards/loco/reward_hunter_soil_locomotion.yaml**
- `tracking_lin_vel: 2.0, tracking_ang_vel: 1.5` (stronger shaping)
- `penalty_feet_height: 0.0` (disable during Stage‑1 for higher swing)
- `termination: -80.0` (reduce dominance; still penalizes falls)
- `feet_height_target: 0.12` (12 cm swing clearance)

**Rationale**: favor moving + clearing ridges over early falling.

### Env adjustments

**humanoidverse/config/env/locomotion_soil.yaml**
- Command ranges widened to ±0.4 m/s, ±0.4 rad/s
- `locomotion_command_resampling_time: 6.0` (more variety within short episodes)
- `termination_min_base_height: 0.35` (tolerate trough dips)

**Rationale**: more motion signal and fewer premature resets.

### Locomotion logic

**humanoidverse/envs/locomotion/locomotion.py**
- Lowered "zero small commands" threshold from 0.2 to 0.05

**Rationale**: stop zeroing most samples; let envs move and generate contacts.

### Robot actuation

**humanoidverse/config/robot/hunter/hunter.yaml**
- `action_scale: 0.35` (was 0.25)

**Rationale**: more headroom for lifting feet on furrows.

## What to expect

- Longer episode lengths (fewer falls)
- Non‑zero feet_air_time
- Higher tracking terms

If stable after some iterations, you can gradually restore furrow depth/orientation and re‑enable penalty_feet_height or raise termination again.

## Lightweight In-Run Curriculum Implementation

Added a lightweight, in-run curriculum and made furrows trainable. Here's what I changed and how it works.

### Terrain (Furrows)

**humanoidverse/config/terrain/terrain_tilled_soil_furrows.yaml**

Eased initial geometry so the policy can learn foot clearance:
- `depth_range_m: [0.10, 0.15]` (was [0.2, 0.3])
- `spacing_range_m: [2.0, 2.5]` (was [1.5, 2.0])
- `orientation_deg: [0.0, 0.0]` (start aligned; add jitter later)
- `crest_offset_m: 0.05` (lift ridges to reduce early trapping)

Enabled row-based curriculum (mixture of difficulties across envs):
- `curriculum: True, num_rows: 5, num_cols: 1`

**Why**: shallower, aligned furrows with a slight crest lift reduce early falls; enabling curriculum spreads difficulty across envs to build robustness.

### Rewards (soil locomotion)

**humanoidverse/config/rewards/loco/reward_hunter_soil_locomotion.yaml**

Boosted tracking so positives can compete with termination:
- `tracking_lin_vel: 2.0, tracking_ang_vel: 1.5`

Temporarily disabled swing height penalty (we allow high steps initially):
- `penalty_feet_height: 0.0`

Reduced termination severity (still penalizes falls):
- `termination: -80.0`

Raised swing-clearance target for furrows:
- `feet_height_target: 0.12`

### Env (commands, terminations, curriculum schedule)

**humanoidverse/config/env/locomotion_soil.yaml**

Widened command ranges and increased variety:
- `lin_vel_x/y: [-0.4, 0.4], ang_vel_yaw: [-0.4, 0.4]`
- `locomotion_command_resampling_time: 6.0`

Tolerate trough dips:
- `termination_min_base_height: 0.35`

Added a lightweight time-based curriculum (no terrain rebuild) that ramps swing-height penalty and termination over time (env steps):
- `furrow_curriculum.enabled: True`

**Phases:**
- step 0: feet_height_target 0.12, penalty_feet_height 0.0, termination -80
- step 50k: feet_height_target 0.10, penalty_feet_height -0.5, termination -100
- step 150k: feet_height_target 0.08, penalty_feet_height -1.0, termination -150
- step 300k: feet_height_target 0.06, penalty_feet_height -1.0, termination -200

### Locomotion logic

**humanoidverse/envs/locomotion/locomotion.py**

Lowered "zero small commands" threshold so more envs actually move:
- From 0.2 → 0.05

### Robot actuation

**humanoidverse/config/robot/hunter/hunter.yaml**
- Increased `action_scale: 0.35` (more headroom to lift feet on furrows)

### In-run curriculum implementation

**humanoidverse/envs/legged_base_task/legged_robot_base.py**

Added a tiny scheduler that applies `env.config.furrow_curriculum` phases over time:
- Updates `rewards.feet_height_target`
- Updates `reward_scales.penalty_feet_height` and `reward_scales.termination` on the fly (handles dt scaling)
- Tracks current phase in `Env/furrow_curr_phase` via `to_log`

## Notes and expectations

This is deliberately "lightweight": we don't rebuild terrain during training. Instead:

- The terrain YAML now uses row‑based curriculum so your 2048 envs sample a spread of difficulties.
- The in‑run curriculum ramps the key reward targets/scales so behavior evolves from high‑clearance survival to efficient stepping.

As episode lengths stabilize and tracking improves, you can gradually:
- Tighten `orientation_deg` from `[0.0, 0.0]` to `[-5.0, 5.0]`.
- Increase `depth_range_m` toward `[0.20, 0.30]`.
- Re‑enable the full `termination: -200`, as our schedule already does at 300k steps.