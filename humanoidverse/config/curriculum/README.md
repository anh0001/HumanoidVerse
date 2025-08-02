# Hunter Locomotion Training Curriculum

This directory contains three-stage curriculum presets for Hunter robot locomotion training, designed to progressively advance from simple plane terrain to complex soil-based locomotion with full domain randomization.

## Curriculum Stages

### Stage 0: Static Plane (`stage0_plane.yaml`)
- **Purpose**: Foundation stage for initial policy learning
- **Terrain**: Simple plane with high friction
- **Randomization**: None (completely deterministic)
- **Use case**: Loading pretrained model_157400.pt and basic locomotion learning

### Stage 1: Soft Soil (`stage1_soft_soil.yaml`)
- **Purpose**: Introduction to soil physics without additional complexity
- **Terrain**: Tilled soil with realistic material properties
- **Randomization**: None (consistent soil conditions)
- **Use case**: Adapting plane-trained policy to soil dynamics

### Stage 2: Full Randomization (`stage2_full_rand.yaml`)
- **Purpose**: Maximum robustness training for real-world deployment
- **Terrain**: Variable soil with property randomization
- **Randomization**: Full domain randomization including pushes, parameter variations
- **Use case**: Final training stage for robust soil locomotion

## Usage with Hydra Overrides

Use these presets by adding the curriculum override to your training command:

```bash
# Stage 0: Start with plane terrain (loading pretrained model)
python scripts/train.py +curriculum=stage0_plane +robot=hunter/hunter

# Stage 1: Move to soil physics
python scripts/train.py +curriculum=stage1_soft_soil +robot=hunter/hunter

# Stage 2: Full randomization training
python scripts/train.py +curriculum=stage2_full_rand +robot=hunter/hunter
```

## Progressive Training Workflow

1. **Stage 0**: Load your existing `model_157400.pt` checkpoint and verify locomotion on plane terrain
2. **Stage 1**: Fine-tune the plane-trained policy on soil terrain without randomization
3. **Stage 2**: Train with full randomization to achieve robust soil locomotion capabilities

## Configuration Override Details

Each stage preset overrides:
- Terrain configuration (plane → soil)
- Domain randomization settings (none → soil-specific)
- Material properties and environmental variations

The presets are designed to work with the existing Hunter robot configuration and can be combined with other Hydra overrides as needed.

## Target: terrain_tilled_soil.yaml Compatibility

All soil-based stages (Stage 1 and 2) are designed to work toward the goal of robust locomotion on `terrain_tilled_soil.yaml`, providing a structured path from basic plane locomotion to advanced soil terrain navigation.