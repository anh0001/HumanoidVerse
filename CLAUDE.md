# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope (LOCKED)

- **Robot:** Hunter only. Always pass `+robot=hunter/hunter`. Do not touch or recommend `g1` / `h1` configs even though they exist on disk — they are unmaintained.
- **Simulator:** IsaacSim / IsaacLab only. Always pass `+simulator=isaacsim`. The `isaacgym` and `genesis` backends under `humanoidverse/simulator/` are legacy and out of scope.
- **Paper search:** Use the `scite` MCP tool for any literature lookup, citation verification, or related-work search. Prefer it over arXiv-only or web search when academic grounding is needed.

## Common Development Commands

### Training Commands
```bash
# Train a locomotion policy
python humanoidverse/train_agent.py \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_hunter_locomotion \
+robot=hunter/hunter \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=HumanoidLocomotion \
experiment_name=Hunter_Locomotion \
headless=True

# Train with wandb logging
python humanoidverse/train_agent.py \
[...same parameters...] \
+opt=wandb
```

### Evaluation Commands
```bash
# Evaluate a trained model
python humanoidverse/eval_agent.py +checkpoint=logs/BaseModel/model_157400.pt

# Evaluate with specific overrides
python humanoidverse/eval_agent.py +checkpoint=logs/xxx/../xx.pt \
+domain_rand.push_robots=True
```

### Curriculum Training Commands
```bash
# Stage 0: Plane terrain (foundation)
python humanoidverse/train_agent.py +curriculum=stage0_plane +robot=hunter/hunter

# Stage 1: Soft soil terrain
python humanoidverse/train_agent.py +curriculum=stage1_soft_soil +robot=hunter/hunter

# Stage 2: Full randomization
python humanoidverse/train_agent.py +curriculum=stage2_full_rand +robot=hunter/hunter
```

### IsaacLab Setup Commands
```bash
# Navigate to IsaacLab directory and activate conda environment
cd IsaacLab
conda activate isaaclab

# Install extensions and dependencies
./isaaclab.sh --install

# Run tests
./isaaclab.sh --test

# Format code
./isaaclab.sh --format

# Build documentation
./isaaclab.sh --docs
```

### Environment Setup
```bash
# Install HumanoidVerse dependencies
pip install -e .

# Verify IsaacSim integration
python -c "import isaacsim; print('IsaacSim imported successfully')"
```

## Architecture Overview

HumanoidVerse is an IsaacSim-based framework for Hunter robot sim-to-real learning with modular architecture:

### Core Components

**1. Simulator** (`humanoidverse/simulator/`)
- `isaacsim/`: IsaacSim/IsaacLab simulator integration (primary and recommended)
- `isaacgym/`, `genesis/`: Legacy simulators (deprecated, not actively maintained)

**2. Environments** (`humanoidverse/envs/`)
- `base_task/`: Base environment abstractions
- `legged_base_task/`: Base classes for legged robot tasks
- `locomotion/`: Locomotion-specific environments
- `env_utils/`: Utilities for command generation, terrain, visualization

**3. Agents** (`humanoidverse/agents/`)
- `base_algo/`: Base algorithm abstraction
- `ppo/`: PPO (Proximal Policy Optimization) implementation
- `modules/`: Network architectures and data utilities
- `callbacks/`: Training callbacks and analysis tools

**4. Configuration System** (`humanoidverse/config/`)
- Hydra-based configuration management with hierarchical configs:
  - `algo/`: Algorithm configurations (PPO variants)
  - `curriculum/`: Multi-stage training curricula
  - `domain_rand/`: Domain randomization settings
  - `env/`: Environment configurations
  - `robot/`: Robot-specific configurations
  - `simulator/`: Simulator-specific settings
  - `terrain/`: Terrain generation configs

### Key Design Patterns

**IsaacSim Simulator**: All training and evaluation uses IsaacSim/IsaacLab for physics simulation

**Hydra Configuration**: All components configured via YAML files with command-line overrides using `+config_group=config_name`

**Progressive Curriculum Training**: Three-stage curriculum (plane → soil → full randomization) for robust policy learning

**Hunter Robot**: Framework optimized for Hunter humanoid robot locomotion

### Training Pipeline

1. **Environment Creation**: Instantiate simulator and task environment
2. **Algorithm Setup**: Initialize PPO agent with actor-critic networks  
3. **Training Loop**: Collect experience, update policy, log metrics
4. **Evaluation**: Test trained policy with keyboard control support
5. **Export**: Save models as PyTorch (.pt) and ONNX (.onnx) formats

### File Structure Logic

- `train_agent.py`: Main training entry point
- `eval_agent.py`: Policy evaluation with interactive control
- `sample_eps.py`: Episode sampling for systematic metrics collection
- `logs/`: Training outputs, checkpoints, and TensorBoard logs
- `logs_eval/`: Evaluation results and renderings
- `IsaacLab/`: IsaacLab framework (git submodule/download)

### Environment Variables

Required environment variables (set in your conda environment):
- `ISAACLAB_PATH`: Path to IsaacLab directory
- `CARB_APP_PATH`: Isaac Sim kit path
- `EXP_PATH`: Isaac Sim apps path
- `ISAAC_PATH`: Isaac Sim installation path

### Dependencies

Core Python packages (from pyproject.toml):
- `hydra-core>=1.2.0`: Configuration management
- `numpy==1.26.4`: Numerical computing
- `wandb`: Experiment tracking
- `tensorboard`: Logging and visualization
- `onnx`, `onnxruntime`: Model export/inference
- `torch`: Deep learning framework (from Isaac Sim/Lab)
- `rich`, `termcolor`, `loguru`: Enhanced console output and logging
- `matplotlib`, `plotly`: Visualization and plotting
- `meshcat`: 3D visualization
- `pynput`: Keyboard input handling for evaluation
- `scipy`: Scientific computing utilities
- `opencv-python`: Computer vision operations

### Testing and Validation

Tests are primarily handled through IsaacLab's testing framework:

```bash
# Run IsaacLab tests (from IsaacLab directory)
./isaaclab.sh --test

# Run specific test categories
./isaaclab.sh --test -k "test_environments"
```

For HumanoidVerse-specific testing, verify installation:
```bash
# Test core imports
python -c "import humanoidverse; print('HumanoidVerse imported successfully')"

# Verify configuration loading
python -c "from hydra import compose, initialize; initialize(config_path='humanoidverse/config'); print('Config system working')"

# Smoke test with minimal environments
python humanoidverse/train_agent.py +curriculum=stage0_plane +robot=hunter/hunter num_envs=2 headless=True --dry-run
```

### Evaluation and Analysis Tools

**Episode Sampling for Metrics Collection**:
```bash
# Systematic evaluation with metrics tracking
python humanoidverse/sample_eps.py \
+checkpoint=logs/project/model.pt \
+terrain=terrain_soil_rigid_reference \
+eval_command=[0.2,0.0,0.0] \
num_envs=100 \
num_episodes=100 \
headless=True

# Soil regime testing (rigid, moderate, challenging)
python humanoidverse/sample_eps.py \
+terrain=terrain_soil_moderate_tilled \
+domain_rand=DR_soil_moderate \
+exp=locomotion \
num_envs=100 \
num_episodes=100 \
headless=True
```

**Baseline Experiment Validation**:
```bash
# No-curriculum baseline (should show poor performance)
python humanoidverse/train_agent.py +curriculum=stage2_full_rand +robot=hunter/hunter

# Flat-ground model on soil (should show catastrophic failure)
python humanoidverse/sample_eps.py +checkpoint=logs/plane_model/model.pt +terrain=terrain_tilled_soil

# Small network experiment (should show degraded performance)
python humanoidverse/train_agent.py +curriculum=stage0_plane +robot=hunter/hunter algo=ppo_small
```

### Soil Terrain Training

**Soil Regimes**: Three soil difficulty levels with matching domain randomization:

```bash
# Rigid reference soil (easiest - high friction, stiff contact)
+terrain=terrain_soil_rigid_reference +domain_rand=DR_soil_rigid

# Moderate tilled soil (medium - moderate friction/compliance)
+terrain=terrain_soil_moderate_tilled +domain_rand=DR_soil_moderate

# Challenging wet/loose soil (hardest - low friction, soft contact)
+terrain=terrain_soil_challenging_wet +domain_rand=DR_soil_challenging
```

**Furrows Curriculum**: Staged training for furrowed terrain navigation:

```bash
# Stage 1: Easy furrows (shallow depth, aligned)
python humanoidverse/train_agent.py \
+robot=hunter/hunter \
+exp=locomotion_soil +algo=ppo_soil \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+rewards=loco/reward_hunter_soil_locomotion \
+terrain=terrain_furrows_stage1_easy \
num_envs=2048 headless=True \
project_name=SoilFurrows experiment_name=Hunter_Furrows_S1_Easy

# Stage 2: Medium furrows (deeper, mild yaw jitter)
+terrain=terrain_furrows_stage2_medium \
+checkpoint=logs/SoilFurrows/latest/model.pt

# Stage 3: Full furrows (full depth/orientation variability)
+terrain=terrain_furrows_stage3_full \
+checkpoint=logs/SoilFurrows/latest/model.pt
```

**Furrow Promotion Criteria**:
- Stage 1 → 2: Mean episode length ≥16s, termination reward ≥-0.05/s
- Stage 2 → 3: Mean episode length ≥18s, termination reward ≥-0.03/s, non-zero feet_air_time
- Stage 3 target: Episode length ~19-20s, stable tracking

### Debugging and Development Tips

**Configuration Debugging**:
```bash
# List all available configurations for a group
python humanoidverse/train_agent.py --help | grep -A 20 "robot:"

# Validate config without training
python humanoidverse/train_agent.py --cfg job --resolve

# Check Hydra composition
python humanoidverse/train_agent.py +curriculum=stage0_plane +robot=hunter/hunter --cfg job
```

**Development Patterns**:
```bash
# Quick test with minimal setup
num_envs=16 headless=False  # For visual debugging
num_envs=4096 headless=True # For full training

# Background training with logging
nohup python humanoidverse/train_agent.py [...args...] > training.log 2>&1 &

# Display forwarding for remote training with visualization
DISPLAY=:77 XAUTHORITY=$HOME/.Xauthority xterm -hold -e "training_command"
```

**Hydra Override Patterns**:
```bash
# Use + for config group selection
+curriculum=stage0_plane +robot=hunter/hunter

# Use ++ for nested parameter overrides (if needed)
++env.config.max_episode_length_s=60

# Checkpoint loading
+checkpoint=logs/CurriculumStage0/latest/model.pt
```

**Common Issues**:
- **CUDA/GPU errors**: Ensure NVIDIA drivers are up-to-date and Isaac Sim has GPU access
- **Import errors**: Verify IsaacLab path is set: `echo $ISAACLAB_PATH`
- **Config not found**: Check config group names match directory structure in `humanoidverse/config/`
- **Memory issues**: Reduce `num_envs` parameter for debugging on smaller GPUs
- **Soil simulation instability**: Terrain configs handle spacing automatically; check terrain YAML files
- **Checkpoint loading failures**: Verify model file exists and path is correct
- **Furrows early termination**: Episodes capping at ~10s indicate need for easier terrain or adjusted reward curriculum
- **Flat rewards on furrows**: Check `feet_height_target` and `penalty_feet_height` scaling in reward config

**Log Analysis**:
- Training logs: `logs/<project_name>/<timestamp>-<experiment_name>/`
- TensorBoard: `tensorboard --bind_all --port=7777 --logdir logs/`
- Wandb dashboard: Check wandb.ai for experiment tracking
- VS Code tasks: Use `.vscode/tasks.json` for pre-configured commands

**Key Training Metrics to Monitor**:
- `Train/mean_episode_length`: Should increase over training
- `Episode/rew_termination`: Closer to 0 is better (indicates fewer early terminations)
- `Episode/rew_tracking_lin_vel`, `Episode/rew_tracking_ang_vel`: Command tracking performance
- `Episode/rew_feet_air_time`: Should be non-zero for proper gait (especially on furrows)
- `Env/action_clip_frac`: Should remain low (high values indicate policy saturation)

### VS Code Integration

The repository includes pre-configured VS Code tasks for common workflows (`.vscode/tasks.json`):

**Training Tasks**:
- Curriculum stages (test and full training modes)
- Baseline experiments (no-curriculum, small network)
- Specialized soil training scenarios
- Furrows curriculum (Stage 1 Easy, Stage 2 Medium, Stage 3 Full - both test and full training)

**Evaluation Tasks**:
- Model evaluation with perturbations
- Terrain-specific testing (rigid, moderate, challenging soil)
- Diagnostic scenarios (standing, walking with no domain randomization)

**Monitoring Tasks**:
- TensorBoard launch
- Background evaluation with logging

Access via VS Code Command Palette → "Tasks: Run Task" or Ctrl+Shift+P.

**Note**: Tasks use background execution with `nohup` for full training, and `xterm` with `DISPLAY` forwarding for visual debugging.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 69 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/anhar/codes/Auto-claude-code-research-in-sleep`).
Update with: `bash /home/anhar/codes/Auto-claude-code-research-in-sleep/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
