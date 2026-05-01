# HumanoidVerse AI Coding Agent Instructions

## Project Overview
HumanoidVerse is a humanoid robot training framework using IsaacSim/IsaacLab with a 3-stage curriculum learning approach. The codebase implements progressive training from plane terrain → soft soil → full randomization for robust locomotion policies.

## Core Architecture & Data Flow

### Hydra Configuration System
All training uses **Hydra overrides** with compositional configs:
```bash
# Training pattern: mandatory components via + prefix
python humanoidverse/train_agent.py +simulator=isaacsim +exp=locomotion +robot=hunter/hunter +rewards=loco/reward_hunter_locomotion +terrain=terrain_locomotion_plane +obs=loco/leggedloco_obs_singlestep_withlinvel
```

Config hierarchy: `humanoidverse/config/` with subdirs:
- `curriculum/` - 3-stage training progression (stage0_plane.yaml, stage1_soft_soil.yaml, stage2_full_rand.yaml)
- `robot/` - Robot configurations (hunter/, h1/)
- `terrain/` - Environment terrains (plane, soil variants, randomized)
- `domain_rand/` - Randomization parameters (friction, mass, PD gains, external forces)
- `exp/` - Experiment types (locomotion, locomotion_soil, locomotion_soil_advanced)
- `rewards/`, `obs/`, `algo/` - Modular components

### Simulator Abstraction
Multi-simulator support via `BaseSimulator` pattern:
- `IsaacSim` (primary) - NVIDIA Isaac Sim with Lab extension
- `IsaacGym` - Legacy GPU simulation
- `Genesis` - Emerging physics engine

Entry points handle simulator-specific initialization:
```python
# train_agent.py detects simulator and initializes AppLauncher for IsaacSim
simulator_type = config.simulator['_target_'].split('.')[-1]
if simulator_type == 'IsaacSim':
    from omni.isaac.lab.app import AppLauncher
```

### Domain Randomization Architecture
Episodic randomization via `_episodic_domain_randomization()` in `LeggedRobotBase`:
- PD gain scaling (`_kp_scale`, `_kd_scale`)
- Control delay (`action_queue`, `action_delay_idx`)
- Mass/friction/terrain properties randomization
- External perturbations (push forces with intervals)

## Critical Development Workflows

### Environment Setup
```bash
conda activate isaaclab  # Must use isaaclab conda environment
python -m pip install -e .  # Editable install
```

### Training Commands
```bash
# Smoke test (quick validation)
num_envs=2 headless=True

# Full training progression
# Stage 0: +curriculum=stage0_plane num_envs=4096
# Stage 1: +curriculum=stage1_soft_soil +checkpoint=logs/Stage0/model.pt  
# Stage 2: +curriculum=stage2_full_rand +checkpoint=logs/Stage1/model.pt
```

### Evaluation Patterns
```bash
# Basic eval
python humanoidverse/eval_agent.py +checkpoint=logs/project/run/model.pt

# Scenario testing
python humanoidverse/sample_eps.py +checkpoint=model.pt +terrain=terrain_soil_rigid_reference +eval_command=[0.2,0.0,0.0] num_envs=100 num_episodes=100

# Perturbation testing  
+domain_rand.push_robots=True +domain_rand.max_push_vel_xy=0.7
```

## Project-Specific Patterns

### File Organization
- Source: `humanoidverse/` (agents, envs, simulator, utils, config)
- Assets: `humanoidverse/data/robots/` (USD/URDF)
- Logs: `logs/` (auto-created, gitignored) with structure `logs/<project>/<timestamp>-<experiment>-<task>-<robot>/`
- Tests: `tests/test_*.py` using pytest

### Configuration Conventions
- Use `++` for nested overrides: `++simulator.config.scene.env_spacing=4.0`
- Load checkpoints: `+checkpoint=logs/path/model_iteration.pt`
- Environment naming: `num_envs` for parallel environments, `headless=True` for server runs
- Logging: `project_name` and `experiment_name` create organized directory structure

### Domain Randomization Levels
- `NO_domain_rand` - Disable all randomization
- `DR_soil_moderate` - Medium soil conditions  
- `DR_soil_challenging` - Wet/loose soil with full randomization
- Push forces configured via `push_interval_s=[min,max]` and `max_push_vel_xy`

### Curriculum Learning Integration
Configs define progressive difficulty:
```yaml
# stage0_plane.yaml - Foundation with high friction, no randomization
# stage1_soft_soil.yaml - Soil terrain introduction  
# stage2_full_rand.yaml - Full randomization + domain randomization
```

## Key Integration Points

### IsaacSim/IsaacLab Integration
- `AppLauncher` handles headless/GUI modes and argument parsing
- `EventManager` applies domain randomization at startup/reset
- `SimulationContext` manages physics stepping and rendering

### Logging & Monitoring
- TensorBoard: `tensorboard --bind_all --port=7777 --logdir logs`
- Wandb: Add `+opt=wandb` to training commands
- Config persistence: Auto-saved to `logs/<experiment>/config.yaml`

### Multi-Environment Execution
- Background training: `nohup ... > logfile.log 2>&1 &`
- Display forwarding: `DISPLAY=:77 XAUTHORITY=$HOME/.Xauthority xterm -hold -e "..."`
- VS Code tasks provide pre-configured training/eval scenarios

## Common Debugging Patterns
- Reduce environments for testing: `num_envs=16` instead of 4096
- Enable visualization: `headless=False` 
- Check domain randomization: Examine `domain_rand` config values
- Verify checkpoint loading: Check `logs/` directory structure and model file existence
- Soil simulation issues: Adjust `env_spacing=4.0` for soil physics stability