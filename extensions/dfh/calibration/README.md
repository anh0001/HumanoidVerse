# DFH Calibration — Virtual Bevameter Workflow (Hu 2023)

Goal: fit the DFH `BekkerParams` (kc, k_phi, n, cohesion, friction_angle) to a higher-fidelity ground-truth source so the cheap GPU model in `ext_dfh.kernels` matches measurable soil behaviour. We mirror the workflow of:

> Hu, Li, Unjhawala (2023). *Calibration of an expeditious terramechanics model using a higher-fidelity model, Bayesian inference, and a virtual bevameter test.* J. Field Robotics. https://doi.org/10.1002/rob.22276

## Calibration Loop

```
                                     ┌────────────────────────────────┐
                                     │ Reference soil regime           │
                                     │   (rigid / moderate / wet)      │
                                     └────────────────────────────────┘
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                │                                                                    │
                ▼                                                                    ▼
┌──────────────────────────────┐                                  ┌────────────────────────────┐
│ Project Chrono DEM           │                                  │ ext_dfh GPU Bekker layer   │
│   - particle soil bed        │                                  │   - candidate (kc, kphi,n) │
│   - virtual bevameter plate  │  ────────────────────────────►   │   - simulate same plate    │
│   - sweep load 0.5 -> 50 kPa │       compare sink curves        │     loading curve          │
└──────────────────────────────┘                                  └────────────────────────────┘
                │                                                                    │
                └────────────► Bayesian fit (e.g., emcee / Pyro NUTS) ◄──────────────┘
                                              │
                                              ▼
                              posterior over (kc, kphi, n, c, phi)
                                              │
                                              ▼
                          terrain_dfh_stage{1,2,3}*.yaml params
```

## Steps

### 1. Generate DEM ground truth (offline, one-time)

Use Project Chrono `chrono_granular` or `pyChrono`:

```python
# scripts/calibration/run_chrono_bevameter.py  (TODO v0.4)
# - build a 1m x 1m x 0.3m soil bed of ~50k particles
# - regimes: rigid (mu=0.7, c=20kPa), moderate (mu=0.55, c=10kPa), wet (mu=0.4, c=5kPa)
# - drive a 0.10 x 0.0345 m flat plate (Hunter foot footprint) downward
# - log force vs sinkage at 5 N steps from 5 N to 500 N (covers 1.5 - 145 kPa)
# - save: data/calibration/chrono_bevameter_<regime>.npz
```

### 2. Bayesian fit (offline, one-time per regime)

```python
# scripts/calibration/fit_bekker.py  (TODO v0.4)
# - load chrono_bevameter_<regime>.npz
# - prior:  kc ~ LogNormal(ln(1e3), 0.7); kphi ~ LogNormal(ln(5e5), 0.7); n ~ Uniform(0.7, 1.3)
# - likelihood:  z_dfh(p) ~ Normal(z_chrono(p), 0.005 m)
# - sampler: emcee or Pyro/NUTS, 2000 warmup, 5000 samples
# - report posterior mean + 95% CI, save to:
#     extensions/dfh/configs/terrain/terrain_dfh_stage{1,2,3}_*.yaml
```

### 3. Validation (offline)

Cross-check sinkage at unseen pressures (V1 in `EXPERIMENT_PLAN.md`):

```
sinkage_pred  vs  sinkage_chrono   should agree within +/- 20%
```

### 4. (v2) Real-soil cross-check

When real bevameter or in-field foot-pressure data becomes available, repeat step 2 against measured curves.

## Why Project Chrono not IsaacSim DEM

IsaacSim has no native DEM/granular solver. Chrono's granular module is mature, validated against soil mechanics literature (Karpman 2020), and runs offline so it does not violate the "IsaacSim-only at runtime" scope lock. Chrono is only used for **calibration data generation**, not in any training loop.

## Bibliography

- Hu, W., Li, P., Unjhawala, H. M. (2023). *Calibration of an expeditious terramechanics model using a higher-fidelity model, Bayesian inference, and a virtual bevameter test.* J. Field Robotics. [10.1002/rob.22276](https://doi.org/10.1002/rob.22276)
- Zhang, Y., Dai, J., Hu, W. (2024). *Using high-fidelity discrete element simulation to calibrate an expeditious terramechanics model in a multibody dynamics framework.* [10.21203/rs.3.rs-4792928/v1](https://doi.org/10.21203/rs.3.rs-4792928/v1)
- Karpman, E., Kövecses, J., Holz, D. (2020). *Discrete element modelling for wheel-soil interaction.* J. Terramechanics. [10.1016/j.jterra.2020.06.002](https://doi.org/10.1016/j.jterra.2020.06.002)
- Buse, F., Pignède, A., Barthelmes, S. (2023). *A Modelica Library to Add Contact Dynamics and Terramechanics to Multi-Body Mechanics.* [10.3384/ecp204433](https://doi.org/10.3384/ecp204433)
