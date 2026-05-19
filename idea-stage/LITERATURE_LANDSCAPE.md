# Literature Landscape — Drastically Improving Hunter Biped Locomotion (IsaacSim / PPO)

**Date:** 2026-05-19
**Direction:** Performance-pivot. Prior DFH soil-realism line is a confirmed null (RIGID 27.0 > DFH 26.2 ep_len). New objective: drastically improve raw locomotion performance (episode length, command tracking, terrain robustness) of the Hunter PPO pipeline.
**Source:** scite MCP (project-mandated). Prior soil landscape archived at `LITERATURE_LANDSCAPE.soil-archive.md`.

## Current pipeline gap (repo-grounded)

- Obs = `leggedloco_obs_singlestep_withlinvel` — **single-step, no temporal memory**.
- Policy = plain MLP actor-critic PPO (`ppo` / `ppo_soil`). **No privileged teacher, no latent env encoder.**
- Reward = hand-tuned `reward_hunter_*` — **no periodic/clock or symmetry structure**.
- Curriculum = staged terrain (plane→soil→furrows) with velocity-tracking reward.
- These are exactly the three levers the literature most consistently credits with large gains.

## High-leverage levers (ranked by evidence strength)

### L1 — History-conditioned policy + privileged teacher → online adaptation
- **Radosavovic et al. 2024, *Science Robotics*** (10.1126/scirobotics.adi9579): causal **transformer over proprioceptive obs-action history**; zero-shot real-world walking, robust to disturbances, **emergent context-dependent gait change per terrain**. History models dominate MLP.
- **Kumar et al. 2021 RMA** (10.15607/rss.2021.xvii.011): privileged env encoder → latent `z`; CNN adaptation module regresses `ẑ` from proprio history, adapts in fractions of a second; 70–80% blind success on stairs/rubble.
- **Kumar et al. 2022 A-RMA** (10.48550/arxiv.2205.15299): **bipedal** (Cassie); PPO re-finetune on estimated `ẑ` → beats RL and model-based baselines.
- **Cheng et al. 2023 ROA** (10.48550/arxiv.2303.11330): **Regularized Online Adaptation** — single-stage joint teacher+estimator training replaces RMA's 3 phases. Verbatim: *"significant improvement in the student policy (about 30% ...) compared to RMA, achieving an order of magnitude improvement in regressing to z."*

### L2 — Periodic / symmetry reward structure
- **Mou et al. 2023, *Biomimetics*** (10.3390/biomimetics8080616): periodic gait objective + curriculum. Verbatim: *"PPO, which does not utilize the periodic reward ... exhibits significant variations in foot height and fails to meet the periodic characteristics ... can lead to falls."*
- **Wang et al. 2023, *Sensors*** (10.3390/s23041873): Siekmann-style clock reward → emergent periodic+symmetric gait "without any prior information."
- Wang/Wei/Xie 2022 (10.3390/mi13101688): phase-gated heuristic+RL → **60% less velocity-tracking error**, faster convergence.

### L3 — Motion priors as style reward
- **Escontrela et al. 2022 AMP-for-robots** (10.48550/arxiv.2203.15103): style reward from seconds of mocap replaces complex reward; **lower cost-of-transport**, natural gait, better transfer.
- Vollenweider et al. 2022 Multi-AMP (10.48550/arxiv.2203.14912): multiple switchable styles, no perf loss.

### L4 — Task framing + exploration on hard terrain
- **Zhang et al. 2024 IROS** (10.1109/iros58592.2024.10801909): replace velocity-tracking with **navigation-style reward** + generalist→specialist finetune + exploration → ≥2.5 m/s on stepping stones/beams. Directly relevant to furrows.

### L5 — Sample-efficient recipe
- Seo et al. 2025, *Sim-to-Real Humanoid Locomotion in 15 Minutes* (10.48550/arxiv.2512.01996): fast PPO recipe — useful for cheap pilots/ablations.

## Recurring themes / open problems

1. **Architecture ≈ as important as algorithm.** Memory/history conditioning is the most consistently credited upgrade over MLP for blind biped robustness.
2. **Imperfect latent estimation is the dominant bottleneck;** ROA's single-stage regularized training is the best-performing, simplest fix (≈30% over RMA).
3. **Periodic/symmetry priors** convert irregular MLP gaits into stable periodic ones and accelerate convergence — cheap, high-yield, absent in this repo.
4. **DR scope is task-specific** — more ≠ better; adaptive DR is open.
5. **Bipeds need motion bootstrapping then weaning;** from-scratch gives unnatural gaits.
6. **Velocity-tracking reward over-constrains on hard terrain;** navigation framing + exploration unlocks agility.

## Open niche for this repo
No HumanoidVerse/Hunter result combines (a) proprioceptive-history teacher-student with **regularized online adaptation** and (b) **periodic-symmetry reward** in IsaacSim. That stack is the highest-evidence path to a *drastic* jump over the current single-step-MLP PPO baseline, and it is orthogonal to and reusable by the existing terrain curriculum.
