# Metrics Tracking System Guide

This guide explains how the metrics tracking system works and how to interpret the collected data for curriculum learning evaluation.

## Overview

The metrics tracking system automatically collects performance data during both training and evaluation to quantitatively assess the effectiveness of curriculum learning approaches.

## Tracked Metrics

### 1. Distance Traveled
**What it measures**: Total forward progress (in meters) before episode termination

**How it's calculated**:
```python
start_pos = robot_position_at_episode_start[:2]  # x,y coordinates
end_pos = robot_position_at_termination[:2]
distance = ||end_pos - start_pos||  # Euclidean distance
```

**Interpretation**:
- **High distance (>10m)**: Robot walks successfully for extended periods
- **Medium distance (1-10m)**: Some forward progress before falling
- **Low distance (<1m)**: Immediate failure, cannot establish stable gait

### 2. Slip Distance
**What it measures**: Cumulative foot sliding during ground contact (in meters)

**How it's calculated**:
```python
# For each foot in contact with ground:
if foot_height < 0.02m:  # Contact threshold
    if previous_contact_position exists:
        slip += ||current_position - previous_position||
    update previous_contact_position
```

**Interpretation**:
- **Low slip (<5m/100m)**: Good traction, stable foot placement
- **Medium slip (5-20m/100m)**: Some sliding, manageable for locomotion
- **High slip (>20m/100m)**: Poor surface interaction, unstable gait

### 3. Fall Detection
**What it measures**: Whether episode ended due to robot fall vs timeout

**How it's calculated**:
```python
fell = not timeout_flag  # Episode terminated early = fall
# timeout_flag = (episode_length >= max_episode_length)
```

**Interpretation**:
- **Fall = True**: Robot lost stability and terminated early
- **Fall = False**: Robot maintained stability until timeout

### 4. Episode Length
**What it measures**: Number of simulation steps before termination

**Interpretation**:
- **Long episodes (>500 steps)**: Stable locomotion
- **Short episodes (<100 steps)**: Quick failures

## Implementation Details

### Environment Integration

The tracking system is integrated into the `LeggedRobotLocomotion` environment:

```python
class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        # ... existing code ...
        
        # Initialize tracking buffers
        self.start_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.slip_distance = torch.zeros(self.num_envs, device=self.device)  
        self.last_foot_pos = [[None, None] for _ in range(self.num_envs)]
    
    def _post_physics_step(self):
        # Update slip tracking every simulation step
        # ... slip calculation code ...
    
    def _reset_tasks_callback(self, env_ids):
        # Collect metrics before reset
        # Store in self.episode_info dictionary
        # Reset tracking buffers for new episodes
```

### Data Collection Flow

1. **Episode Start**: Record robot starting position
2. **During Episode**: Accumulate slip distance each physics step
3. **Episode End**: Calculate final metrics and store
4. **Reset**: Clear buffers for next episode

### Foot Contact Detection

Critical for accurate slip measurement:

```python
# Ground contact threshold
CONTACT_THRESHOLD = 0.02  # 2cm above ground = in contact

left_contact = left_foot_pos[:, 2] < CONTACT_THRESHOLD
right_contact = right_foot_pos[:, 2] < CONTACT_THRESHOLD
```

**Tuning Notes**:
- **Too low (0.001m)**: Misses soft contacts on deformable terrain
- **Too high (0.05m)**: False positives when foot hovers above ground
- **Recommended**: 0.02m works well for most terrains

## Data Access Methods

### Method 1: sample_eps.py (Recommended)

For systematic evaluation with detailed metrics:

```bash
python humanoidverse/sample_eps.py \
  +checkpoint=path/to/model.pt \
  +terrain=terrain_type \
  num_envs=100 num_episodes=100 headless=true
```

**Output Format**:
```
========================================
EVALUATION RESULTS  
========================================
Episodes completed: 100
Total distance traveled: 245.67 m
Total falls: 23
Average episode length: 342.5 steps
Average distance per episode: 2.46 m
Falls per 100m: 9.36
Slip distance per 100m: 12.84 m
========================================
```

### Method 2: Environment Episode Info

Access during training via environment callbacks:

```python
# In training loop
obs, rewards, dones, infos = env.step(actions)

# Access episode info for completed episodes
done_indices = torch.nonzero(dones).flatten()
for env_idx in done_indices:
    if hasattr(env, 'episode_info'):
        episode_data = env.episode_info.get(f'env_{env_idx}', {})
        distance = episode_data.get('distance', 0.0)
        slip = episode_data.get('slip_distance', 0.0)
        fell = episode_data.get('fell', False)
```

### Method 3: TensorBoard Logging

For real-time monitoring during training:

```python
# In reward computation callback
if self.log_dict is not None:
    self.log_dict['episode_distance'] = current_episode_distances
    self.log_dict['slip_rate'] = current_slip_distances
    self.log_dict['fall_rate'] = current_fall_rates
```

## Validation and Debugging

### Sanity Checks

1. **Distance Consistency**: 
   ```python
   assert distance_traveled >= 0, "Distance cannot be negative"
   assert distance_traveled < 1000, "Unrealistic distance (check units)"
   ```

2. **Slip Bounds**:
   ```python
   assert slip_distance >= 0, "Slip cannot be negative" 
   assert slip_distance < distance_traveled * 10, "Slip too high vs distance"
   ```

3. **Episode Length**:
   ```python
   assert episode_length > 0, "Episodes must have positive length"
   assert fell == (episode_length < max_episode_length), "Fall logic error"
   ```

### Common Issues

**Issue**: All slip distances are zero
- **Cause**: Contact detection threshold too strict
- **Fix**: Increase `CONTACT_THRESHOLD` to 0.03-0.05m

**Issue**: Unrealistically high slip values  
- **Cause**: Foot position noise or coordinate system issues
- **Fix**: Add position smoothing or check coordinate transforms

**Issue**: No falls detected when robot clearly falls
- **Cause**: Termination conditions not properly configured  
- **Fix**: Verify `terminate_after_contacts_on` and height thresholds

**Issue**: Distance is always very small
- **Cause**: Robot spinning in place vs moving forward
- **Fix**: Check command generation and reward function alignment

### Debug Visualization

Add debugging prints to track metrics in real-time:

```python
def _reset_tasks_callback(self, env_ids):
    for env_id in env_ids:
        env_id_int = int(env_id.item())
        if hasattr(self, 'start_pos'):
            distance = # ... calculate distance ...
            slip = # ... get slip distance ...
            
            print(f"Env {env_id_int}: Distance={distance:.2f}m, "
                  f"Slip={slip:.2f}m, Fell={fell}")
```

## Performance Baselines

### Curriculum Learning Expectations

Based on Hunter robot locomotion experiments:

**Stage 0 (Flat Ground)**:
- Falls/100m: 0-5 (excellent stability)
- Slip/100m: 0-3m (good traction)
- Avg distance/episode: >10m

**Stage 1 (Soft Soil)**:
- Falls/100m: 5-15 (good stability)  
- Slip/100m: 5-10m (some sliding expected)
- Avg distance/episode: 5-15m

**Stage 2 (Full Randomization)**:
- Falls/100m: 10-30 (acceptable for challenging terrain)
- Slip/100m: 8-20m (realistic for varied surfaces)
- Avg distance/episode: 3-10m

### Failure Mode Detection

**Catastrophic Failure**:
- Falls/100m: >100
- Avg distance/episode: <0.5m
- Episode length: <50 steps

**Marginal Performance**:
- Falls/100m: 30-100  
- Slip/100m: >30m
- Inconsistent episode lengths

**Success Indicators**:
- Falls/100m: <30
- Stable slip rates across episodes
- Episode lengths approaching timeout

## Data Export and Analysis

### CSV Export Format

For statistical analysis:

```python
import csv

def export_episode_data(episode_infos, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'distance', 'slip_distance', 'fell', 'episode_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, info in enumerate(episode_infos):
            writer.writerow({
                'episode': i,
                'distance': info['distance'], 
                'slip_distance': info['slip_distance'],
                'fell': info['fell'],
                'episode_length': info['episode_length']
            })
```

### Statistical Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_metrics(episode_data):
    distances = [ep['distance'] for ep in episode_data]
    slips = [ep['slip_distance'] for ep in episode_data] 
    falls = [ep['fell'] for ep in episode_data]
    
    # Aggregate metrics
    total_distance = sum(distances)
    total_falls = sum(falls)
    falls_per_100m = (total_falls / (total_distance/100)) if total_distance > 0 else 0
    
    # Distribution analysis
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(distances, bins=20)
    plt.xlabel('Distance Traveled (m)')
    plt.ylabel('Frequency')
    plt.title('Episode Distance Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(slips, bins=20)  
    plt.xlabel('Slip Distance (m)')
    plt.ylabel('Frequency')
    plt.title('Slip Distance Distribution')
    
    plt.subplot(1, 3, 3)
    plt.bar(['Completed', 'Fell'], [len(episode_data) - sum(falls), sum(falls)])
    plt.ylabel('Episodes')
    plt.title('Episode Outcomes')
    
    plt.tight_layout()
    plt.savefig('metrics_analysis.png')
    
    return {
        'falls_per_100m': falls_per_100m,
        'avg_distance': np.mean(distances),
        'avg_slip': np.mean(slips),
        'success_rate': 1 - (sum(falls) / len(episode_data))
    }
```

This metrics tracking system provides the quantitative foundation needed to validate curriculum learning effectiveness and support publication-quality results.