# Hunter Locomotion Evaluation System

This document describes the structured evaluation scenarios implemented for the Hunter robot locomotion testing.

## Overview

The evaluation system provides systematic testing of different locomotion capabilities with predefined scenarios that maintain constant commands throughout the evaluation period. This allows for consistent and reproducible testing of:

- Forward/backward locomotion
- Lateral movement (left/right)
- Rotational movement (clockwise/counter-clockwise)
- Static balance
- Combined movements (diagonal)
- Manual control for interactive testing

## Quick Start

### Using the Script (Recommended)

```bash
# Forward locomotion test
./evaluate_hunter.sh forward

# Manual control mode (use WASD keys)
./evaluate_hunter.sh manual

# Test with custom checkpoint
./evaluate_hunter.sh forward --checkpoint path/to/your/model.pt

# Run in headless mode
./evaluate_hunter.sh static --headless
```

### Using VS Code Tasks

1. Open Command Palette (Ctrl+Shift+P)
2. Type "Tasks: Run Task"
3. Select from available evaluation scenarios:
   - "Evaluate Hunter - Forward Locomotion Scenario"
   - "Evaluate Hunter - Backward Locomotion Scenario"
   - "Evaluate Hunter - Left Lateral Locomotion Scenario"
   - "Evaluate Hunter - Right Lateral Locomotion Scenario"
   - "Evaluate Hunter - Clockwise Rotation Scenario"
   - "Evaluate Hunter - Counter-Clockwise Rotation Scenario"
   - "Evaluate Hunter - Static Balance Scenario"
   - "Evaluate Hunter - Manual Control Mode"

### Using Command Line Directly

```bash
# Activate environment
source ~/miniconda3/bin/activate isaaclab

# Run specific scenario
python humanoidverse/eval_agent.py \
    +checkpoint=logs/FullTrainingHunter/20250715_134736-FullTrainingHunter-locomotion-hunter/model_83000.pt \
    +simulator=isaacsim \
    +exp=locomotion \
    +env.config.max_episode_length_s=60 \
    +env.config.locomotion_command_ranges.lin_vel_x=[0.5,0.5] \
    +env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0] \
    +env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0] \
    +env.config.locomotion_command_resampling_time=1000.0
```

## Available Scenarios

### 1. Forward Locomotion (`scenario_forward`)
- **Purpose**: Evaluate forward walking at constant velocity
- **Command**: Linear velocity X = 0.5 m/s
- **Duration**: 60 seconds
- **Use Case**: Basic forward walking capability assessment

### 2. Backward Locomotion (`scenario_backward`)
- **Purpose**: Evaluate backward walking at constant velocity
- **Command**: Linear velocity X = -0.5 m/s
- **Duration**: 60 seconds
- **Use Case**: Reverse walking capability assessment

### 3. Left Lateral Movement (`scenario_left`)
- **Purpose**: Evaluate left lateral movement
- **Command**: Linear velocity Y = 0.5 m/s
- **Duration**: 60 seconds
- **Use Case**: Lateral mobility assessment

### 4. Right Lateral Movement (`scenario_right`)
- **Purpose**: Evaluate right lateral movement
- **Command**: Linear velocity Y = -0.5 m/s
- **Duration**: 60 seconds
- **Use Case**: Lateral mobility assessment

### 5. Clockwise Rotation (`scenario_rotate_cw`)
- **Purpose**: Evaluate rotation in place
- **Command**: Angular velocity = -0.5 rad/s
- **Duration**: 60 seconds
- **Use Case**: Turning and rotational stability

### 6. Counter-Clockwise Rotation (`scenario_rotate_ccw`)
- **Purpose**: Evaluate rotation in place
- **Command**: Angular velocity = 0.5 rad/s
- **Duration**: 60 seconds
- **Use Case**: Turning and rotational stability

### 7. Static Balance (`scenario_static`)
- **Purpose**: Evaluate standing balance
- **Command**: All velocities = 0
- **Duration**: 60 seconds
- **Use Case**: Basic stability and balance assessment

### 8. Diagonal Movements
- `scenario_forward_left`: Forward + left movement
- `scenario_forward_right`: Forward + right movement
- **Use Case**: Combined movement capability testing

## Interactive Manual Control

During any evaluation, you can use keyboard controls for manual testing:

### Movement Controls
- **W**: Move forward
- **S**: Move backward  
- **A**: Move left
- **D**: Move right
- **Q**: Rotate counter-clockwise
- **E**: Rotate clockwise
- **X**: Stop all movement

### Force Controls
- **1**: Increase left hand upward force
- **2**: Decrease left hand upward force
- **3**: Increase right hand upward force
- **4**: Decrease right hand upward force

### Task Controls
- **N**: Move to next task (if applicable)

## Configuration Files

### Main Config: `humanoidverse/config/eval/eval_locomotion_scenarios.yaml`

This file defines all evaluation scenarios with their specific parameters:

```yaml
scenarios:
  scenario_forward:
    name: "Forward Locomotion"
    description: "Evaluate forward walking at constant velocity"
    eval_overrides:
      env:
        config:
          max_episode_length_s: 60
          locomotion_command_ranges:
            lin_vel_x: [0.5, 0.5]  # Constant forward velocity
            lin_vel_y: [0.0, 0.0]  # No lateral movement
            ang_vel_yaw: [0.0, 0.0]  # No rotation
            heading: [0.0, 0.0]  # Fixed heading
          locomotion_command_resampling_time: 1000.0
```

### Tasks Configuration: `.vscode/tasks.json`

Contains VS Code task definitions for each scenario, allowing easy execution from the VS Code interface.

## Customization

### Adding New Scenarios

1. Edit `humanoidverse/config/eval/eval_locomotion_scenarios.yaml`
2. Add a new scenario under the `scenarios` section:

```yaml
scenario_custom:
  name: "Custom Scenario"
  description: "Your custom evaluation scenario"
  eval_overrides:
    env:
      config:
        max_episode_length_s: 60
        locomotion_command_ranges:
          lin_vel_x: [0.3, 0.3]  # Custom velocities
          lin_vel_y: [0.2, 0.2]
          ang_vel_yaw: [0.1, 0.1]
          heading: [0.0, 0.0]
        locomotion_command_resampling_time: 1000.0
```

3. Add corresponding VS Code task in `.vscode/tasks.json`
4. Update the evaluation script `evaluate_hunter.sh` if needed

### Modifying Evaluation Parameters

Key parameters you can adjust in scenarios:

- `max_episode_length_s`: Duration of evaluation
- `locomotion_command_ranges`: Target velocities (min, max ranges)
- `locomotion_command_resampling_time`: How often commands change (set high for constant commands)

### Metrics and Logging

The system automatically logs:
- Linear velocity tracking
- Angular velocity tracking  
- Base height and orientation
- Joint positions and velocities
- Contact forces
- Energy consumption

Logs are saved to `logs_eval/locomotion_scenarios/[timestamp]/`

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Update the checkpoint path in tasks or use `--checkpoint` flag
2. **Environment activation fails**: Ensure isaaclab conda environment is properly installed
3. **Display issues**: For headless operation, add `headless=True` parameter

### Getting Help

```bash
# Show available options
./evaluate_hunter.sh --help

# Check if environment is working
source ~/miniconda3/bin/activate isaaclab
python -c "import humanoidverse; print('Setup OK')"
```

## Integration with Existing Workflow

This evaluation system integrates seamlessly with the existing HumanoidVerse framework:

- Uses existing configuration system (Hydra)
- Compatible with all simulators (IsaacSim, IsaacGym, Genesis)
- Maintains existing keyboard control functionality
- Preserves policy export capabilities
- Works with existing logging and metrics systems

The structured scenarios provide systematic evaluation while the manual control mode offers the flexibility for interactive testing that was available in the original IsaacGym implementation.
