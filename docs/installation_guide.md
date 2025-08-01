# IsaacSim 4.2 and IsaacLab 1.4.1 Installation Guide

This guide provides step-by-step instructions for installing IsaacSim 4.2 and IsaacLab 1.4.1 for use with HumanoidVerse.

## Prerequisites

Before starting the installation, ensure you have:

- **Operating System**: Ubuntu 22.04
- **GPU**: NVIDIA GPU with compute capability 7.0+ (RTX 20 series or newer recommended)
- **NVIDIA Drivers**: Version 525.60.11 or newer
- **Python**: 3.10
- **Conda/Miniconda**: For environment management
- **Git**: For cloning repositories
- **Sufficient Storage**: At least 15GB free space

### Verify Prerequisites

```bash
# Check NVIDIA driver
nvidia-smi

# Check Python version
python3 --version

# Check conda
conda --version

# Check Git
git --version
```

## Step 1: Download and Install IsaacSim 4.2

### Download IsaacSim Binary

1. Visit the [NVIDIA Isaac Sim download page](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html)
2. Download the **Isaac Sim 4.2.0** Linux binary zip file

### Install IsaacSim

```bash
# Create installation directory in home folder
mkdir -p ~/isaacsim_4.2
cd ~/isaacsim_4.2

# Extract the downloaded zip file (adjust path as needed)
unzip ~/Downloads/isaac-sim-4.2.0-linux-x86_64-release.zip

# Make the setup script executable
chmod +x ./setup_conda_env.sh
```

### Verify IsaacSim Installation

```bash
# Test IsaacSim installation
cd ~/isaacsim_4.2
./isaac-sim.sh --help

# If successful, you should see help output
```

## Step 2: Download and Setup IsaacLab 1.4.1

### Navigate to Project Root

```bash
# Navigate to your HumanoidVerse project root directory
cd /path/to/HumanoidVerse
```

### Download IsaacLab

```bash
# Download IsaacLab 1.4.1 release
wget https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v1.4.1.tar.gz

# Extract the archive
tar -xzf v1.4.1.tar.gz

# Rename the folder for consistency
mv IsaacLab-1.4.1 IsaacLab

# Remove the downloaded archive
rm v1.4.1.tar.gz
```

## Step 3: Create Symbolic Link

Create a symbolic link from IsaacLab to your IsaacSim installation:

```bash
# Navigate to IsaacLab directory
cd IsaacLab

# Create symbolic link to IsaacSim installation
ln -s ~/isaacsim_4.2/isaac-sim _isaac_sim

# Verify the symbolic link
ls -la _isaac_sim
# Should show: _isaac_sim -> /home/username/isaacsim_4.2
```

## Step 4: Create Conda Environment

### Setup IsaacLab Conda Environment

```bash
# Make sure you're in the IsaacLab directory
cd IsaacLab

# Create conda environment using IsaacLab's script
./isaaclab.sh --conda

# This will:
# 1. Create a new conda environment named 'isaaclab'
# 2. Install required Python packages
# 3. Setup the environment for IsaacLab

# Source conda profile first
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate isaaclab
```

## Step 5: Install IsaacLab

### Run Installation Script

```bash
# Make sure you're in the IsaacLab directory and conda environment is activated
cd IsaacLab
conda activate isaaclab

# Run the installation script
./isaaclab.sh --install

# This will:
# 1. Install IsaacLab Python package
# 2. Setup necessary dependencies
# 3. Configure the environment
```

## Step 6: Install HumanoidVerse Dependencies

### Install HumanoidVerse for IsaacSim

```bash
# Navigate back to HumanoidVerse root
cd /path/to/HumanoidVerse

# Ensure isaaclab conda environment is activated
conda activate isaaclab

# Install HumanoidVerse
pip install -e .
```

## Step 7: Verification

### Test IsaacSim Integration

```bash
# Test IsaacSim standalone
cd ~/isaacsim_4.2
./isaac-sim.sh

# Should launch IsaacSim GUI (if display is available)
# Press Ctrl+C to exit
```

### Test IsaacLab

```bash
# Navigate to IsaacLab
cd /path/to/HumanoidVerse/IsaacLab

# Activate environment
conda activate isaaclab

# Run a basic IsaacLab test
python source/standalone/tutorials/00_sim/create_empty.py

# Should run without errors
```

### Test HumanoidVerse with IsaacSim

```bash
# Navigate to HumanoidVerse root
cd /path/to/HumanoidVerse

# Activate environment
conda activate isaaclab

# Test IsaacSim integration with HumanoidVerse
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_hunter_locomotion \
+robot=hunter/hunter \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=1 \
project_name=TESTInstallation \
experiment_name=Hunter_loco_IsaacSim \
headless=False

# Should launch without errors
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing NVIDIA Drivers

```bash
# Update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525
sudo reboot
```

#### 2. IsaacSim Not Found

```bash
# Check if symbolic link is correct
cd IsaacLab
ls -la _isaac_sim

# Recreate symbolic link if needed
rm _isaac_sim
ln -s ~/isaacsim_4.2/isaac-sim _isaac_sim
```

#### 3. Conda Environment Issues

```bash
# Remove and recreate environment
conda deactivate
conda env remove -n isaaclab
cd IsaacLab
./isaaclab.sh --conda
```

#### 4. Permission Issues

```bash
# Fix permissions for IsaacSim
chmod +x ~/isaacsim_4.2/isaac-sim/isaac-sim.sh
chmod +x ~/isaacsim_4.2/isaac-sim/python.sh
```

#### 5. Library Path Issues

Add to your `~/.bashrc`:

```bash
# Add IsaacSim to library path
export LD_LIBRARY_PATH=~/isaacsim_4.2/kit/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/isaacsim_4.2/kit/bin:$LD_LIBRARY_PATH
```

#### 6. Isaac Sim 4.2 Python Environment Integration

If you encounter `ModuleNotFoundError: No module named 'omni.isaac.kit'` when running tasks, you need to integrate Isaac Sim's Python environment with your conda environment. Create conda activation/deactivation scripts:

```bash
# Create conda environment directories
mkdir -p ~/miniconda3/envs/isaaclab/etc/conda/activate.d ~/miniconda3/envs/isaaclab/etc/conda/deactivate.d

# Create activation script
cat > ~/miniconda3/envs/isaaclab/etc/conda/activate.d/isaac_sim.sh << 'EOF'
#!/bin/bash
# Save current environment variables
export ISAAC_SIM_PYTHONPATH_BACKUP="$PYTHONPATH"
export ISAAC_SIM_LD_LIBRARY_PATH_BACKUP="$LD_LIBRARY_PATH"

# Source Isaac Sim environment
source ~/isaacsim_4.2/setup_python_env.sh
EOF

# Create deactivation script
cat > ~/miniconda3/envs/isaaclab/etc/conda/deactivate.d/isaac_sim.sh << 'EOF'
#!/bin/bash
# Restore original environment variables
if [ -n "$ISAAC_SIM_PYTHONPATH_BACKUP" ]; then
    export PYTHONPATH="$ISAAC_SIM_PYTHONPATH_BACKUP"
    unset ISAAC_SIM_PYTHONPATH_BACKUP
fi

if [ -n "$ISAAC_SIM_LD_LIBRARY_PATH_BACKUP" ]; then
    export LD_LIBRARY_PATH="$ISAAC_SIM_LD_LIBRARY_PATH_BACKUP"
    unset ISAAC_SIM_LD_LIBRARY_PATH_BACKUP
fi
EOF

# Make scripts executable
chmod +x ~/miniconda3/envs/isaaclab/etc/conda/activate.d/isaac_sim.sh
chmod +x ~/miniconda3/envs/isaaclab/etc/conda/deactivate.d/isaac_sim.sh
```

After setting this up, deactivate and reactivate your conda environment:
```bash
conda deactivate
conda activate isaaclab
```

This automatically sources Isaac Sim's Python environment when the conda environment is activated, eliminating the need to modify VSCode tasks or manually source setup scripts.

#### 7. EXP_PATH KeyError

If you encounter `KeyError: 'EXP_PATH'` when running training tasks, this means the Isaac Sim environment variables are not properly set in your conda environment. Add the missing environment variables to your conda activation script:

```bash
# Add Isaac Sim environment variables to conda activation script
cat >> ~/miniconda3/envs/isaaclab/etc/conda/activate.d/setenv.sh << 'EOF'

# for Isaac Sim
export CARB_APP_PATH=~/isaacsim_4.2/kit
export EXP_PATH=~/isaacsim_4.2/apps
export ISAAC_PATH=~/isaacsim_4.2
EOF

# Add corresponding unset commands to deactivation script
cat >> ~/miniconda3/envs/isaaclab/etc/conda/deactivate.d/unsetenv.sh << 'EOF'
unset CARB_APP_PATH
unset EXP_PATH
unset ISAAC_PATH
EOF
```

**Note:** Replace `~/isaacsim_4.2` with your actual Isaac Sim installation path.

After adding these variables, reactivate your conda environment:
```bash
conda deactivate
conda activate isaaclab
```

## Environment Variables

Add these to your `~/.bashrc` for convenience:

```bash
# IsaacSim paths
export ISAAC_SIM_PATH=~/isaacsim_4.2
export ISAAC_LAB_PATH=/path/to/HumanoidVerse/IsaacLab

# Python path for IsaacLab
export PYTHONPATH=$ISAAC_LAB_PATH/source:$PYTHONPATH

# Conda environment auto-activation (optional)
# conda activate isaaclab
```

Don't forget to reload your bash configuration:

```bash
source ~/.bashrc
```

## Next Steps

After successful installation:

1. **Read the HumanoidVerse Documentation**: Familiarize yourself with the project structure
2. **Run Examples**: Try the provided examples in the HumanoidVerse repository
3. **Configure Your Robot**: Set up your specific robot configuration
4. **Start Training**: Begin training your humanoid robot policies

## Additional Resources

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [NVIDIA Omniverse Forums](https://forums.developer.nvidia.com/c/omniverse/)

---

**Note**: This installation process requires significant computational resources and may take 30-60 minutes depending on your system and internet connection.