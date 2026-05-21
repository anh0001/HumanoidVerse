# Pilot Verdict (Codex 4-branch decision rule)

**Source:** `logs/WarmstartPilot/pilot_summary.csv`  ·  **Generated:** by `apply_pilot_decision.py`

## Held-out results (last ckpt per arm × condition)

| arm | iter | furrows_s3 ep_len | furrows_s3 fall_rate | soil_chal ep_len | soil_chal fall_rate |
|---|---|---|---|---|---|
| A0_baseline | 1500 | 23.0 | 1.000 | 19.7 | 1.000 |
| A3_hist50 | 1500 | 24.6 | 1.000 | 20.8 | 1.000 |
| A8_gait | 1500 | 22.0 | 1.000 | 20.2 | 1.000 |
| A9_full | 3500 | 25.5 | 1.000 | 24.1 | 1.000 |

## Verdict

**Branch 2** — ALL ARMS FAIL on held-out -> terrain too hard even after plane warm-start. Build staged terrain curriculum before any matrix.

### Evidence
- `A0 baseline:  ep_len=21.4  fall_rate=1.000`
- `A3 hist50:    ep_len=22.7  fall_rate=1.000`
- `A8 gait:      ep_len=21.1  fall_rate=1.000`
- `A9 full:      ep_len=24.8  fall_rate=1.000`

### Thresholds applied
- nonzero locomotion: ep_len ≥ 200 steps AND fall_rate < 0.8
- near-tie (Branch 3/4): |Δ ep_len| / A9 ≤ 0.2
- A9 ≫ baselines (Branch 1): A9 ≥ (1 + 0.3) × each baseline

### Ready-to-fire next step
```bash
# Branch 2: build staged terrain curriculum BEFORE any matrix.
# Stage 0: plane (already validated by pilot Stage P).
# Stage 1: shallow furrows / low DR / large env_spacing.
# Stage 2: medium furrows (current furrows_s2).
# Stage 3: full furrows + DR (current target).
# Promotion criteria: ep_len threshold per stage from CLAUDE.md.
# New configs needed under humanoidverse/config/curriculum/.
```
