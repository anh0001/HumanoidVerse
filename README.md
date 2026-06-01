<h1 align="center"> Hunter Robot Training in IsaacSim </h1>

<div align="center">

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html)

[![IsaacLab](https://img.shields.io/badge/IsaacLab-1.4.1-b.svg)](https://isaac-sim.github.io/IsaacLab/)

[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

</div>

# Hunter Robot Locomotion Training

> **Scope (locked):** This repository targets the **Hunter humanoid robot only** and the **NVIDIA IsaacSim / IsaacLab** simulator only. Other robot configs under `humanoidverse/config/robot/` (`g1`, `h1`, …) and other simulator backends (`isaacgym`, `genesis`) are legacy and not supported by this pipeline. Do not introduce or test against them.

This repository provides a complete training pipeline for the Hunter humanoid robot using IsaacSim/IsaacLab. The framework implements a progressive 3-stage curriculum learning approach that enables robust locomotion policies across diverse terrains, from simple planes to challenging soil conditions.</div>

## Pretrained Models

Trained Hunter locomotion checkpoints are released on the Hugging Face Hub:

<div align="center">
  <a href="https://huggingface.co/anhrisn/hunter-dfh-locomotion">
    <img src="assets/hunter_agri_field.png" width="520px"/>
  </a>
  <br/>
  <em>Hunter walking through a maize row-crop field in IsaacSim (DFH deformable, furrowed soil).</em>
</div>

**Model repo:** [`anhrisn/hunter-dfh-locomotion`](https://huggingface.co/anhrisn/hunter-dfh-locomotion) (public, MIT)

The release covers locomotion on **deformable, furrowed agricultural terrain** modelled with the DFH (Deformable Furrowed Heightfield: Bekker terramechanics + anisotropic friction + sink-coupled lateral drag):

| Group | What | Deploy? |
|---|---|---|
| `walkers/mildsoil_walker/model_5050.pt` | True forward walker on mild DFH soil (~0.12 BW drag) | **Recommended deployment target** |
| `walkers/v7_baseline/model_4250.pt` | Flat-ground walker; warm-start root; used in the agri-field render above | Baseline |
| `walkers/rigid_walker/model_5050.pt` | Walker on rigid furrows (no soil drag) | Reference |
| `dfh_chain/` (12 stages) | Drag-survival curriculum (robust balancers, not walkers) | Research / warm-start only |

Each checkpoint folder on the Hub ships its training-time `config.yaml`, so it loads directly:

```bash
python humanoidverse/eval_agent.py +checkpoint=<downloaded>/walkers/mildsoil_walker/model_5050.pt
```

## Features
- **3-Stage Curriculum Learning**: Progressive training from plane terrain → soft soil → full randomization
- **Hunter Robot Support**: Optimized for Hunter humanoid robot with complete locomotion pipeline
- **IsaacSim/IsaacLab Integration**: Built on NVIDIA's Isaac simulation platform
- **Robust Evaluation**: Multiple terrain scenarios including soil conditions and external perturbations
- **Complete Training Pipeline**: From basic locomotion to advanced terrain navigation


# Environment Setup

## Prerequisites
- NVIDIA GPU with CUDA support
- Linux Ubuntu 22.04 or later
- Miniconda or Anaconda
- IsaacSim 4.2.0
- IsaacLab 1.4.1

## Quick Setup
```bash
# 1. Activate IsaacLab conda environment
source ~/miniconda3/bin/activate isaaclab

# 2. Install HumanoidVerse dependencies
pip install -e .

# 3. Verify IsaacSim integration
python -c "import isaacsim; print('IsaacSim imported successfully')"
```

For detailed installation instructions, please refer to the [Installation Guide](docs/installation_guide.md).



# Curriculum Training Pipeline

The Hunter robot training follows a 3-stage curriculum learning approach for robust locomotion across diverse terrains.

## Stage 0: Foundation Training (Plane Terrain)
Build basic locomotion skills on flat terrain:

```bash
# Test training (small scale)
python humanoidverse/train_agent.py \
+curriculum=stage0_plane \
+robot=hunter/hunter \
+simulator=isaacsim \
+exp=locomotion \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=16 \
project_name=CurriculumStage0Test \
experiment_name=Hunter_Stage0_Plane \
headless=False

# Full training
python humanoidverse/train_agent.py \
+curriculum=stage0_plane \
+robot=hunter/hunter \
+simulator=isaacsim \
+exp=locomotion \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=CurriculumStage0 \
experiment_name=Hunter_Stage0_Plane \
headless=True
```

## Stage 1: Soil Adaptation Training
Continue from Stage 0 model with soft soil terrain:

```bash
# Continue from Stage 0 (recommended)
python humanoidverse/train_agent.py \
+curriculum=stage1_soft_soil \
+robot=hunter/hunter \
+simulator=isaacsim \
+exp=locomotion_soil \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
++simulator.config.scene.env_spacing=4.0 \
+checkpoint=logs/CurriculumStage0/latest/model.pt \
num_envs=2048 \
project_name=CurriculumStage1Continue \
experiment_name=Hunter_Stage1_FromStage0 \
headless=True
```

## Stage 2: Full Randomization Training
Final stage with complete domain randomization:

```bash
# Continue from Stage 1 (recommended)
python humanoidverse/train_agent.py \
+curriculum=stage2_full_rand \
+robot=hunter/hunter \
+simulator=isaacsim \
+exp=locomotion_soil_advanced \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
++simulator.config.scene.env_spacing=4.0 \
+checkpoint=logs/CurriculumStage1/latest/model.pt \
num_envs=2048 \
project_name=CurriculumStage2Continue \
experiment_name=Hunter_Stage2_FromStage1 \
headless=True
```

# Evaluation Scenarios

After training, evaluate your models across different scenarios to assess robustness and performance.

## Basic Policy Evaluation
```bash
# Evaluate any trained model
python humanoidverse/eval_agent.py +checkpoint=logs/your_project/your_experiment/model_xxxx.pt
```

## Stage-Specific Evaluation
Evaluate models on their respective terrains:

```bash
# Stage 0: Plane terrain evaluation
python humanoidverse/eval_agent.py \
+curriculum=stage0_plane \
+robot=hunter/hunter \
+checkpoint=logs/CurriculumStage0/latest/model.pt \
+simulator=isaacsim \
+exp=locomotion \
++env.config.max_episode_length_s=10000 \
++env.config.locomotion_command_resampling_time=10000

# Stage 1: Soil terrain evaluation
python humanoidverse/eval_agent.py \
+curriculum=stage1_soft_soil \
+robot=hunter/hunter \
+checkpoint=logs/CurriculumStage1/latest/model.pt \
+simulator=isaacsim \
+exp=locomotion \
++env.config.max_episode_length_s=10000 \
++env.config.locomotion_command_resampling_time=10000

# Stage 2: Full randomization evaluation
python humanoidverse/eval_agent.py \
+curriculum=stage2_full_rand \
+robot=hunter/hunter \
+checkpoint=logs/CurriculumStage2/latest/model.pt \
+simulator=isaacsim \
+exp=locomotion \
++simulator.config.scene.env_spacing=4.0 \
++env.config.max_episode_length_s=10000 \
++env.config.locomotion_command_resampling_time=10000
```

## Robustness Testing

### External Perturbations
Test locomotion stability with lateral pushes:
```bash
python humanoidverse/eval_agent.py \
+checkpoint=logs/your_model/model.pt \
+simulator=isaacsim \
+domain_rand.push_robots=True \
+domain_rand.max_push_vel_xy=0.7 \
+domain_rand.push_interval_s=[3,8]
```

### Forward Locomotion Scenario
Test sustained forward walking:
```bash
python humanoidverse/eval_agent.py \
+checkpoint=logs/your_model/model.pt \
+simulator=isaacsim \
+exp=locomotion \
++env.config.max_episode_length_s=60 \
+env.config.locomotion_command_ranges.lin_vel_x=[0.5,0.5] \
+env.config.locomotion_command_ranges.lin_vel_y=[0.0,0.0] \
+env.config.locomotion_command_ranges.ang_vel_yaw=[0.0,0.0] \
++env.config.locomotion_command_resampling_time=1000.0
```

### Soil Terrain Evaluation
Test on different soil conditions:
```bash
# Rigid soil (easier)
python humanoidverse/sample_eps.py \
+simulator=isaacsim \
+terrain=terrain_soil_rigid_reference \
+domain_rand=DR_soil_rigid \
+exp=locomotion \
num_envs=100 \
num_episodes=100 \
headless=True

# Moderate tilled soil (medium)
python humanoidverse/sample_eps.py \
+simulator=isaacsim \
+terrain=terrain_soil_moderate_tilled \
+domain_rand=DR_soil_moderate \
+exp=locomotion \
num_envs=100 \
num_episodes=100 \
headless=True

# Challenging wet/loose soil (hard)
python humanoidverse/sample_eps.py \
+simulator=isaacsim \
+terrain=terrain_soil_challenging_wet \
+domain_rand=DR_soil_challenging \
+exp=locomotion \
num_envs=100 \
num_episodes=100 \
headless=True
```

# Monitoring Training Progress

## TensorBoard Logging
Monitor training progress in real-time:
```bash
tensorboard --bind_all --port=7777 --logdir logs
```

## Training Logs
- Training logs: `logs/<project_name>/<timestamp>-<experiment_name>/`
- Model checkpoints: `logs/<project_name>/<timestamp>-<experiment_name>/model_<iteration>.pt`
- Configuration files: `logs/<project_name>/<timestamp>-<experiment_name>/config.yaml`

## Wandb Integration (Optional)
Add `+opt=wandb` to any training command for advanced experiment tracking:
```bash
python humanoidverse/train_agent.py \
+curriculum=stage0_plane \
+robot=hunter/hunter \
... \
+opt=wandb
```

# Training Results

After completing the 3-stage curriculum (approximately 3000-5000 epochs total), the Hunter robot achieves robust locomotion across diverse terrains:

<div align="center">
  <img src="assets/isaacsim_isaacsim.gif" width="800px"/>
</div>

# References and Acknowledgements

This Hunter robot training pipeline is built upon HumanoidVerse, a multi-simulator framework for humanoid robot learning. Key inspirations:

- **[Legged Gym](https://github.com/leggedrobotics/legged_gym)**: Foundation for locomotion training and domain randomization
- **[ProtoMotions](https://github.com/NVlabs/ProtoMotions)**: Hydra configuration management and codebase structure  
- **[RSL RL](https://github.com/leggedrobotics/rsl_rl)**: PPO algorithm implementation reference

**LeCAR Lab Contributors**: [Gao Jiawei](https://gao-jiawei.com/), [Tairan He](https://tairanhe.com/), [Wenli Xiao](https://wenlixiao-cs.github.io/), [Yuanhang Zhang](https://hang0610.github.io/), [Zi Wang](https://www.linkedin.com/in/zi-wang-b675aa236/), and the full [LeCAR Lab](https://lecar-lab.github.io/) team.

Special thanks to [Guanya Shi](https://www.gshi.me/) for project guidance and support.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
