# HumanoidVerse Documentation

This directory contains comprehensive documentation for running baseline experiments and understanding the metrics tracking system.

## Documentation Overview

### 📋 [baseline_experiments.md](./baseline_experiments.md)
**Comprehensive experiment guide**
- Detailed explanation of all three baseline experiments
- Step-by-step commands for each experiment
- Expected results and interpretation
- Troubleshooting and validation criteria
- Publication-ready results format

### 📊 [metrics_tracking_guide.md](./metrics_tracking_guide.md) 
**Deep dive into metrics system**
- Detailed explanation of all tracked metrics
- Implementation details and code examples
- Debugging and validation techniques
- Data export and statistical analysis methods
- Performance baselines and benchmarks

### 🚀 [quick_start_baselines.md](./quick_start_baselines.md)
**Fast-track experiment execution**
- Condensed step-by-step instructions
- Ready-to-use bash scripts
- Quick results interpretation
- Troubleshooting common issues
- Complete timeline and VS Code integration

## Quick Navigation

### For First-Time Users
1. Start with **[quick_start_baselines.md](./quick_start_baselines.md)** for immediate execution
2. Refer to **[baseline_experiments.md](./baseline_experiments.md)** for detailed understanding
3. Use **[metrics_tracking_guide.md](./metrics_tracking_guide.md)** for analysis and debugging

### For Researchers  
1. **[baseline_experiments.md](./baseline_experiments.md)** → Understand experimental design
2. **[metrics_tracking_guide.md](./metrics_tracking_guide.md)** → Validate data collection
3. **[quick_start_baselines.md](./quick_start_baselines.md)** → Batch execution scripts

### For Developers
1. **[metrics_tracking_guide.md](./metrics_tracking_guide.md)** → Implementation details
2. **[baseline_experiments.md](./baseline_experiments.md)** → Configuration options
3. **[quick_start_baselines.md](./quick_start_baselines.md)** → Integration examples

## Experiment Summary

The baseline experiments validate curriculum learning effectiveness through three key comparisons:

| Experiment | Purpose | Expected Result |
|------------|---------|-----------------|
| **No-Curriculum** | Show curriculum necessity | Poor performance (>100 falls/100m) |
| **Flat-Ground on Soil** | Show transfer limitations | Catastrophic failure (>400 falls/100m) |
| **Small Network** | Show capacity importance | Degraded performance (20-40 falls/100m) |

## File Organization

```
docs/
├── README_docs.md              # This overview file
├── baseline_experiments.md     # Comprehensive experiment guide  
├── metrics_tracking_guide.md   # Detailed metrics documentation
├── quick_start_baselines.md    # Fast execution guide
└── installation_guide.md       # Environment setup (existing)
```

## Key Metrics Explained

- **Falls per 100m**: Primary stability metric (lower = better)
- **Slip distance per 100m**: Traction effectiveness (lower = better)  
- **Average episode length**: Locomotion duration (higher = better)
- **Success rate**: Percentage of episodes without falls

## Getting Help

### Common Use Cases

**"I want to run experiments quickly"**
→ Use [quick_start_baselines.md](./quick_start_baselines.md)

**"I need to understand the results"**  
→ See interpretation sections in [baseline_experiments.md](./baseline_experiments.md)

**"My metrics look wrong"**
→ Debug using [metrics_tracking_guide.md](./metrics_tracking_guide.md)

**"I want to modify the experiments"**
→ Study implementation in [metrics_tracking_guide.md](./metrics_tracking_guide.md)

### Support Resources

1. **Configuration issues**: Check VS Code tasks in `.vscode/tasks.json`
2. **Environment problems**: Refer to `installation_guide.md`
3. **Training failures**: Check logs in `logs/` directory
4. **Metric anomalies**: Use debug prints in metrics guide

## Expected Workflow

### Phase 1: Setup (30 minutes)
- Review [quick_start_baselines.md](./quick_start_baselines.md) prerequisites
- Ensure Stage-0 model is available
- Prepare execution environment

### Phase 2: Training (8-12 hours, mostly automated)  
- Launch no-curriculum and small-network experiments
- Monitor progress via TensorBoard or logs
- Wait for completion

### Phase 3: Evaluation (45 minutes)
- Run evaluation scripts on all trained models
- Collect metrics using `sample_eps.py`
- Generate summary results

### Phase 4: Analysis (30 minutes)
- Interpret results using baseline tables
- Validate against expected performance ranges
- Generate publication-ready figures

## Integration with Main Repository

These experiments integrate seamlessly with the existing HumanoidVerse structure:

- **Configs**: New experiment configs in `humanoidverse/config/`
- **VS Code**: Updated tasks in `.vscode/tasks.json`  
- **Environment**: Enhanced metrics in `humanoidverse/envs/locomotion/`
- **Evaluation**: New `sample_eps.py` script for systematic testing

The implementation maintains backward compatibility while adding comprehensive baseline experiment capabilities.