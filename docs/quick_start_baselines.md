# Quick Start: Baseline Experiments

This is a condensed guide for quickly running all baseline experiments and collecting results.

## Prerequisites Checklist

- [ ] HumanoidVerse environment installed and working
- [ ] `isaaclab` conda environment activated  
- [ ] IsaacSim properly configured
- [ ] At least 16GB VRAM available for training
- [ ] 4-8 hours available for training experiments

## Step-by-Step Execution

### Step 1: Train Stage-0 Baseline (if needed)

First, ensure you have a flat-ground trained model:

```bash
# Check if you have a Stage-0 model
ls logs/*/model_*.pt

# If not, train one (2-3 hours):
python humanoidverse/train_agent.py +curriculum=stage0_plane +robot=hunter/hunter \
  +simulator=isaacsim +exp=locomotion num_envs=4096 headless=true \
  project_name=Stage0Baseline experiment_name=FlatGroundPolicy
```

### Step 2: Run All Training Experiments

Run these in parallel or sequentially:

```bash
# Experiment 1: No-curriculum baseline (6-8 hours)
python humanoidverse/train_agent.py +exp=locomotion_no_curriculum \
  seed=42 headless=true num_envs=2048 \
  project_name=BaselineExperiments experiment_name=NoCurriculumBaseline &

# Experiment 2: Small network curriculum (8-10 hours)  
python humanoidverse/train_agent.py +exp=locomotion_terrain +algo=ppo_small \
  seed=42 headless=true num_envs=2048 \
  project_name=BaselineExperiments experiment_name=SmallNetworkBaseline &

# Wait for completion
wait
```

### Step 3: Collect All Evaluation Data

Create and run the evaluation script:

```bash
#!/bin/bash
# File: evaluate_baselines.sh

echo "Starting baseline evaluations..."

# Create results directory
mkdir -p results/baseline_experiments
cd results/baseline_experiments

# 1. Flat-ground policy on soil (Experiment 2)
echo "Evaluating flat-ground policy on soil..."
python ../../humanoidverse/sample_eps.py \
  +checkpoint=../../logs/Stage0Baseline/latest/model.pt \
  +terrain=terrain_tilled_soil \
  num_envs=100 num_episodes=100 headless=true \
  > flat_on_soil_results.txt 2>&1

# 2. No-curriculum baseline on soil
echo "Evaluating no-curriculum policy..."  
python ../../humanoidverse/sample_eps.py \
  +checkpoint=../../logs/BaselineExperiments/*/NoCurriculumBaseline*/latest/model.pt \
  +terrain=terrain_tilled_soil \
  num_envs=100 num_episodes=100 headless=true \
  > no_curriculum_results.txt 2>&1

# 3. Small network curriculum on soil
echo "Evaluating small network policy..."
python ../../humanoidverse/sample_eps.py \
  +checkpoint=../../logs/BaselineExperiments/*/SmallNetworkBaseline*/latest/model.pt \
  +terrain=terrain_tilled_soil \
  num_envs=100 num_episodes=100 headless=true \
  > small_network_results.txt 2>&1

# 4. Full curriculum baseline (for comparison - if available)
if ls ../../logs/*/FullCurriculum*/model_*.pt 2>/dev/null; then
    echo "Evaluating full curriculum policy..."
    python ../../humanoidverse/sample_eps.py \
      +checkpoint=../../logs/*/FullCurriculum*/latest/model.pt \
      +terrain=terrain_tilled_soil \
      num_envs=100 num_episodes=100 headless=true \
      > full_curriculum_results.txt 2>&1
fi

echo "All evaluations complete!"
echo "Results saved in: $(pwd)"
ls -la *.txt
```

Make it executable and run:

```bash
chmod +x evaluate_baselines.sh
./evaluate_baselines.sh
```

## Expected Results Format

Each evaluation will produce output like:

```
========================================
EVALUATION RESULTS
========================================
Episodes completed: 100
Total distance traveled: 45.67 m
Total falls: 78
Average episode length: 89.3 steps
Average distance per episode: 0.46 m
Falls per 100m: 170.82
Slip distance per 100m: 45.23 m
========================================
```

## Results Interpretation Table

| Experiment | Expected Falls/100m | Expected Slip/100m | Interpretation |
|------------|--------------------|--------------------|----------------|
| **Flat-Ground on Soil** | 400-800 | 50-100 | ❌ Catastrophic failure |
| **No-Curriculum** | 100-300 | 25-50 | ❌ Poor performance |  
| **Small Network** | 20-40 | 10-20 | ⚠️ Degraded but functional |
| **Full Curriculum** | 5-20 | 5-15 | ✅ Good performance |

## Quick Results Summary Script

Create a summary script:

```bash
#!/bin/bash
# File: summarize_results.sh

echo "BASELINE EXPERIMENTS SUMMARY"
echo "=================================="

for file in results/baseline_experiments/*_results.txt; do
    if [ -f "$file" ]; then
        experiment=$(basename "$file" _results.txt)
        echo
        echo "📊 $experiment:"
        
        # Extract key metrics
        falls_per_100m=$(grep "Falls per 100m:" "$file" | awk '{print $4}')
        slip_per_100m=$(grep "Slip distance per 100m:" "$file" | awk '{print $5}')
        avg_distance=$(grep "Average distance per episode:" "$file" | awk '{print $5}')
        
        echo "   Falls/100m: $falls_per_100m"
        echo "   Slip/100m: $slip_per_100m"  
        echo "   Avg Distance: $avg_distance"
        
        # Performance assessment
        if (( $(echo "$falls_per_100m > 100" | bc -l) )); then
            echo "   Status: ❌ Poor"
        elif (( $(echo "$falls_per_100m > 30" | bc -l) )); then  
            echo "   Status: ⚠️ Degraded"
        else
            echo "   Status: ✅ Good"
        fi
    fi
done

echo
echo "=================================="
echo "Analysis complete!"
```

Run with: `bash summarize_results.sh`

## Troubleshooting Quick Fixes

### Common Issues:

**Error: "Could not find config path"**
```bash
# Fix: Specify config explicitly
python humanoidverse/sample_eps.py \
  +checkpoint=path/to/model.pt \
  +simulator=isaacsim +exp=locomotion +terrain=terrain_tilled_soil \
  num_envs=100 num_episodes=100 headless=true
```

**Error: "CUDA out of memory"**
```bash
# Fix: Reduce num_envs
num_envs=64  # Instead of 100
```

**Error: "Module not found"**
```bash
# Fix: Ensure correct working directory
cd /home/anhar/codes/HumanoidVerse
source ~/miniconda3/bin/activate isaaclab
```

## Complete Timeline

- **Setup & Stage-0**: 30 minutes + 2-3 hours training
- **Baseline Training**: 6-10 hours (can run overnight)
- **Evaluations**: 30-45 minutes total
- **Analysis**: 15 minutes

**Total**: ~8-14 hours (mostly automated)

## VS Code Integration

Use the pre-configured tasks:

1. Open Command Palette (`Ctrl+Shift+P`)
2. Type "Tasks: Run Task"
3. Select:
   - "Train No-Curriculum Baseline (Stage-2 only)"  
   - "Train Smaller-Net 3-Stage Curriculum"
   - "Evaluate Flat-Only Policy on Soil"

## Success Criteria

✅ **Experiment successful if**:
- Flat-ground policy: >100 falls/100m on soil
- No-curriculum: >50 falls/100m  
- Small network: 15-40 falls/100m
- All evaluations complete without errors
- Results show clear performance hierarchy

The experiments validate curriculum learning necessity and demonstrate quantifiable improvements in locomotion robustness.