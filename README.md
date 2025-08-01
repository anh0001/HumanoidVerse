<h1 align="center"> HumanoidVerse: A Multi-Simulator Framework for 
    
Humanoid Robot Sim-to-Real Learning. </h1>

<div align="center">
<p align="center">
    <img src="assets/humanoidverse-logo-crop-png.png"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
</p>


[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html)

[![IsaacLab](https://img.shields.io/badge/IsaacLab-1.4.1-b.svg)](https://isaac-sim.github.io/IsaacLab/)

[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

</div>

# What is HumanoidVerse?
HumanoidVerse supports multiple simulators and tasks for humanoid robot sim-to-real learning. A key design logic is the separation and modularization of simulators, tasks, and algorithms, allowing for conviniently switching between simulators and tasks, and develop new ones with minimal efforts.

We compared the scope of HumanoidVerse with other sim-to-real frameworks and summarized the differences in supported features in the following table:

<div align="center">

| Framework | Multi Simulators | Sim2Sim & Sim2Real |
| --- | --- | --- |
| HumanoidVerse | :white_check_mark: | :white_check_mark: |
| [Mujoco Playground](https://playground.mujoco.org/#) | :x: | :white_check_mark: |
| [ProtoMotions](https://github.com/NVlabs/ProtoMotions) | :white_check_mark: | :x: |
| [Humanoid Gym](https://github.com/roboterax/humanoid-gym) | :x: | :white_check_mark: |
| [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) | :x: | :white_check_mark: |
| [Legged Gym](https://github.com/leggedrobotics/legged_gym) | :x: | :x: |

</div>

## TODO
- [x] Support for IsaacSim simulator.
- [x] Support for multiple embodiments: (Currently) Unitree Humanoid H1-10DoF, H1-19DoF, G1-12DoF, G1-23DoF.
- [ ] Sim-to-Sim and Sim-to-Real pipelines.
- [ ] Motion tracking tasks.

# News

- 2025-02-04: :tada: Initial Public Release! We have released the locomotion training pipeline for humanoid robots in IsaacSim.


# Installation

For detailed installation instructions, please refer to the [Installation Guide](docs/installation_guide.md).



# Training & Evaluation
We support training & evaluating in IsaacSim simulator.
## Policy Training
To train your policy, follow this command format:
```bash
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=<task_name> \
+domain_rand=<domain_randomization> \
+rewards=<reward_function> \
+robot=<robot_name> \
+terrain=<terrain_name> \
+obs=<observation_name> \
num_envs=<num_envs> \
project_name=<project_name> \
experiment_name=<experiment_name> \
headless=<headless_mode>
```
<details>
<summary>(Optional) By default, the training process is logged by tensorboard. You can also use `wandb` for logging.</summary>

If you want to use `wandb` for logging, you can add `+opt=wandb` in the command.

```bash
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_hunter_locomotion \
+robot=hunter/hunter \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=HumanoidLocomotion \
experiment_name=Hunter_loco_IsaacSim \
headless=True \
+opt=wandb
```
</details>

## Policy Evaluation

After running the training command, you can find the checkpoints and log files in the `logs/<project_name>/<timestamp>-_<experiment_name>-<exp_type>-<robot_type>` directory.

To evaluate the policy, follow this command format:

```bash
python humanoidverse/eval_agent.py +checkpoint=logs/xxx/../xx.pt
```

`logs/xxx/../xx.pt` is the relative path to the checkpoint file. You only need to run this command, our script will automatically find and load the training config.

<details>
<summary>If you want to override some of the training config, you can use `+` to override the configs.</summary>

```bash
python humanoidverse/eval_agent.py +checkpoint=logs/xxx/../xx.pt \
+domain_rand.push_robots=True \
+simulator=isaacsim
```
</details>

# Start Training Your Humanoids!

Here is the command to train & evaluate the locomotion policy on Hunter Humanoid Robot in IsaacSim.

## IsaacSim
<details>
<summary>Training Command</summary>

```bash
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_hunter_locomotion \
+robot=hunter/hunter \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=HumanoidLocomotion \
experiment_name=Hunter_loco_IsaacSim \
headless=True
```
</details>

After around 3000 epochs, evaluating in IsaacSim:

<div align="center">
  <img src="assets/isaacsim_isaacsim.gif" width="800px"/>
</div>

# References and Acknowledgements

This project is inspired by the following projects:

- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) inspired us to use `hydra` for configuration management and influenced the overall structure of the codebase.
- [Legged Gym](https://github.com/leggedrobotics/legged_gym) provided the reference code for training locomotion tasks, handling domain randomizations, and designing reward functions. The starting point of our codebase is `git clone git@github.com:leggedrobotics/legged_gym.git`.
- [RSL RL](https://github.com/leggedrobotics/rsl_rl) provided an example for the implementation of the PPO algorithm.

This project is made possible thanks to our amazing team members at [LeCAR Lab](https://lecar-lab.github.io/):
- [Gao Jiawei](https://gao-jiawei.com/) led the development of this project, designed the overall architecture, and implemented the core components, including the simulators, robots, tasks, and the training and evaluation framework. 
- [Tairan He](https://tairanhe.com/) implemented the design of domain randomizations, integrated the IsaacSim simulator (together with Zi Wang), and helped significantly with debugging in the early stages of the project.
- [Wenli Xiao](https://wenlixiao-cs.github.io/) implemented the design of the observation dictionary, proprioception configuration, and history handlers, designed the actor-critic network architecture in PPO, and also helped greatly with debugging in the early stages of the project.
- [Yuanhang Zhang](https://hang0610.github.io/) contributed significantly to the sim-to-sim and sim-to-real pipelines and helped with debugging our PPO implementation.
- [Zi Wang](https://www.linkedin.com/in/zi-wang-b675aa236/) integrated the IsaacSim simulator into HumanoidVerse.
- [Ziyan Xiong](https://ziyanx02.github.io/) integrated the Genesis simulator into HumanoidVerse.
- [Haotian Lin](https://www.linkedin.com/in/haotian-lin-9b29b7324/) implemented MPPI in HumanoidVerse (to be released soon).
- [Zeji Yi](https://iscoyizj.github.io/) and [Chaoyi Pan](https://panchaoyi.com/) provided crucial help with our sim-to-sim pipeline in the early stages of the project.

Special thanks to [Guanya Shi](https://www.gshi.me/) for his invaluable support and unwavering guidance throughout the project.

# Citation
Please use the following bibtex if you find this repo helpful and would like to cite:

```bibtex
@misc{HumanoidVerse,
  author = {CMU LeCAR Lab},
  title = {HumanoidVerse: A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LeCAR-Lab/HumanoidVerse}},
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
