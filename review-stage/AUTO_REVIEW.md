# AUTO_REVIEW — DFH terrain support (Hunter / IsaacSim)

Goal: make HumanoidVerse training work on DFH terrain — closed-loop sinkage / slip-sinkage / bulldozing / anisotropic friction.

Reviewer: Codex MCP (xhigh).
Loop start: 2026-05-09.

---

## Round 1 (2026-05-09)

### Assessment (Summary)
- Score: 2/10
- Verdict: NO
- ThreadId: `019e099b-1a76-7e21-ad48-ff1c5bf1026c`

### Top weaknesses (ranked)
1. `world_xy_to_cell` ignores env origin + mesh origin; `.long()` truncates not floors.
2. Per-env state collapses via `amin(dim=0)` → shared global terrain noise.
3. Anisotropic friction (`friction_at`) is dead code; PhysX uses scalar μ.
4. PhysX collision writeback unproven — `refresh_physics_objects` swallowed; no contact-height verification.
5. Slip-sinkage `|v_t|/(|v_n|+1e-3)` blows up in stance → instant floor saturation.
6. `DR_dfh_*.yaml` parameter ranges are config-only, never read.
7. Bekker `b = horizontal_scale_m` and `contact_patch_area_m2 = 3.45e-3` not calibrated to Hunter foot.
8. Furrow direction defaults to 0° everywhere; sampler in `furrow_terrain.py` not piped to DFH layer.
9. Reward (`reward_hunter_dfh.yaml`) lacks DFH-specific terms; no terrain-relative base height; no high-base termination.
10. Warm-start success is not evidence — falsification tests required.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 2/10. Verdict: NO.**

This is not ready for a top venue under the claim "DFH actually works and demonstrably matters." The code has a DFH buffer and a USD writeback path, but the closed-loop physical claim is not established, and several paths are either no-op, globally aliased across envs, or mapped in the wrong coordinate frame.

**Ranked Weaknesses**

1. **Coordinate mapping is fundamentally wrong.**
   `kernels.py:135` `world_xy_to_cell()` uses raw world XY, no env origin, no mesh lower-left origin, no local terrain offset, and `.long()` truncates instead of flooring. With 16x16 furrow terrain and env packing, most contacts can clamp to borders.
   Minimum fix: change `world_xy_to_cell(pos_xy_m, env_idx, env_origins_xy, terrain_origin_xy, grid_h, grid_w, scale)` and map `local_xy = pos_xy_m - env_origins_xy[env_idx] + [terrain_length/2, terrain_width/2]`; use `torch.floor(local_xy / scale).long()`. Wire env origins from `isaacsim.py:942` into the DFH adapter.

2. **Per-env DFH state collapses into one shared terrain.**
   `writeback.py:197` reduces `h_plastic` with `amin(dim=0)`. That destroys per-env causality and turns 2048 envs into global terrain noise. The test even enshrines this behavior at `test_writeback.py:93`.
   Minimum fix: remove cross-env min. Either create one mutable terrain prim per env, or write each env's `h_plastic[env]` only into the mesh region around that env origin. Change writer signature to `__call__(h_plastic, env_origins)` and precompute `vertex_env_idx, vertex_local_cell_i, vertex_local_cell_j`.

3. **Anisotropic friction is dead code.**
   `dfh_layer.py:123` defines `friction_at()`, but nothing calls it. PhysX uses the scalar material from `isaacsim.py:500`. The anisotropy claim is currently vacuous.
   Minimum fix: for DFH configs, set ground PhysX friction low/zero and apply a custom tangential contact force per active foot in `DFHIsaacSimAdapter.on_physics_step()`: decompose `v_t` along/across furrow, cap by `mu_dir * normal_force`, and apply through IsaacLab external-force APIs before `scene.write_data_to_sim()`.

4. **PhysX collision writeback is not proven to affect contacts.**
   `writeback.py:164` swallows refresh failure; `isaacsim.py:619` still logs `writeback=on`. A USD point mutation is not enough evidence that cooked PhysX collision changed.
   Minimum fix: make `_mark_collision_dirty()` return success/failure, expose a `dfh_writeback_collision_refresh_ok` metric, and add an IsaacSim probe test: deform one cell under a falling foot/box and assert contact height changes in PhysX.

5. **Slip-sinkage numerics explain instant trenching.**
   `kernels.py:47` uses `|v_t| / (|v_n| + 1e-3)`. In stance, `v_n≈0`, so modest tangential velocity multiplies sinkage by tens or hundreds and hits `sinkage_floor_m` immediately.
   Minimum fix: replace velocity ratio with bounded slip ratio, e.g. `slip = vt / clamp(vt + abs(vn), min=0.05)`, then `z_eff = z * (1 + alpha * clamp(slip, 0, slip_max))`.

6. **DFH domain randomization is config-only.**
   `DR_dfh_soft.yaml:10` defines `dfh_param_ranges`, but no code reads it. `DFHParams` is scalar/frozen at `config.py:37`.
   Minimum fix: add per-env parameter tensors in `DFHTerrainLayer`, implement `dfh_randomize(env_ids, ranges)`, and call it from `legged_robot_base.py:854`.

7. **Bekker/contact geometry is not calibrated.**
   `dfh_layer.py:84` uses a fixed area; `kernels.py:40` uses grid scale as Bekker width. Hunter ankle collision is a rotated box at `hunter.urdf:250`, so the current constant is at best a rough guess.
   Minimum fix: add explicit `contact_area_m2` and `bekker_width_m` from the actual foot collider/contact normal; do not pass `horizontal_scale_m` as Bekker plate width.

8. **Furrow direction is not connected to generated terrain.**
   `isaacsim.py:606` initializes all furrow directions to 0 deg, while `furrow_terrain.py:35` samples orientation jitter for stage 2/3. Even if friction were called, direction is wrong.
   Minimum fix: return/store per-tile sampled orientation from terrain generation and pass a per-cell `furrow_direction_deg` tensor into `DFHTerrainLayer.initialize()`.

9. **Reward/diagnostics allow hiding failure.**
   `reward_hunter_dfh.yaml:18` disables DFH-specific rewards; `legged_robot_base.py:352` logs absolute base height error, not terrain-relative height. No high-height/flying termination exists.
   Minimum fix: implement `_reward_penalty_sinkage_excess()` in `locomotion.py:197`, log relative base height, and terminate non-finite or absurd high base states.

10. **Warm-start success is not evidence.**
    The FromS3 checkpoint could be solving rigid furrows while DFH is ineffective or globally noisy. Current unit tests validate tensor updates, not causal robot-terrain coupling.

**Discriminating Falsification Test**

Run paired eval/training with identical seeds, commands, initial states, and the exact same rigid furrow mesh:
1. TRUE_DFH: corrected per-env DFH writeback + friction.
2. SHADOW_DFH: compute `h_plastic` but no PhysX/friction writeback.
3. SHUFFLED_DFH: write back another env's or spatially shifted `h_plastic`.
4. RIGID_FURROWS: DFH off, same geometry/material.

Required evidence: TRUE_DFH must differ significantly from SHADOW_DFH/RIGID_FURROWS in tracking, slip, stance timing, contact height; SHUFFLED_DFH should degrade or change behavior. Also log correlation between foot impulse and subsequent local terrain depression under the same foot. If performance is unchanged under shadow or shuffled DFH, the DFH claim is falsified.

</details>

### Plan for Round 1 fixes (Phase C)

Implement now (cheap, correctness-critical):
- F1: Fix `world_xy_to_cell` to take env origins + terrain origin; switch `.long()` → `.floor().long()`.
- F2: Plumb env origins through `HunterFootContactSource` / `DFHIsaacSimAdapter.on_physics_step`.
- F3: Bounded slip ratio in `apply_slip_sinkage`.
- F4: Switch DFH primary coupling from USD-mesh writeback to **per-env applied wrench** (sinkage drag + anisotropic tangential force) on the foot bodies. This sidesteps cross-env `amin` aliasing entirely; USD writeback remains as opt-in visual.
- F5: Per-env DFH parameter tensors + `dfh_randomize(env_ids, ranges)` reading `dfh_param_ranges`.
- F6: Add `_reward_penalty_sinkage_excess` and terrain-relative base height log; add high-base termination.
- F7: Connect furrow direction from `furrow_terrain.py` sampling into `DFHTerrainLayer.initialize`.

Defer (heavy):
- D1: True per-env collision-mesh writeback (per-env terrain prim or per-env vertex region).
- D2: IsaacSim Kit probe test for collision recook.
- D3: Falsification matrix (TRUE / SHADOW / SHUFFLED / RIGID) — design after F1–F7 land.

Status: continuing to round 2 after fixes.

### Actions taken (Phase C)

- F1 (coord mapping fix): `kernels.world_xy_to_cell` now accepts
  `env_origin_xy` + `extent_xy`, subtracts the env origin and centers the
  per-env tile, switched `.long()` → `.floor().long()`. Legacy origin-less
  call still works for unit tests.
- F2 (env-origin plumbing): `DFHIsaacSimAdapter.attach_env_origins_provider`
  + lazy lookup in `on_physics_step`; `DFHTerrainLayer.step` accepts
  `env_origins_xy` and indexes per-env. Wired in `isaacsim.py` to lambda
  reading `self.env_origins` after `create_envs`.
- F3 (slip-sinkage numerics): `apply_slip_sinkage` rewritten to bounded
  ratio `slip = vt / clamp(vt + |vn|, eps_floor)` with `eps_floor = max(eps, 1e-3)`
  and `slip_max = 1.0`. No more divide-by-eps blow-up.
- F4 (force-coupled DFH, opt-in writeback): primary closed-loop coupling is
  now per-env applied wrench: anisotropic drag
  `(mu_eff − baseline_mu) · |F_n|` plus sinkage drag
  `k_sink · |H_plastic| · |F_n|`, both opposing tangential motion, capped
  at `max_drag_force_n`. Visual USD writeback default-OFF
  (`terrain.dfh.writeback_enabled`); avoids cross-env amin aliasing.
  Routing via `Articulation.set_external_force_and_torque` on per-foot
  bodies through `build_force_apply_for_robot`.
- F5 (per-env DR): `DFHTerrainLayer.randomize_envs(env_ids, ranges)`
  populates per-env tensors for `bekker_kc/k_phi/n`, `mu_along/across`,
  `slip_sinkage_alpha`, `bulldoze_share`, indexed in `step` and
  `_friction_at`. `simulator.dfh_randomize` exposed; called from
  `legged_robot_base._reset_buffers_callback` reading
  `domain_rand.dfh_param_ranges` (already present in DR_dfh_*.yaml).
- F6 (reward + termination): `_reward_penalty_sinkage_excess` implemented
  in `locomotion.py` (per-foot quadratic excess above `sigma_m=0.04`).
  Reward scale enabled in `reward_hunter_dfh.yaml` at `−2.0`. New high-base
  termination + non-finite base termination in `legged_robot_base.py:474`
  (catches the prior FixB blow-up where `base_height_err_mean → 200`).
- Patch area corrected to `1.40e-2 m²` (Hunter foot ~140 cm²) in all three
  DFH stage YAMLs.
- Test fix (out-of-scope but pre-existing): mock `net_forces_w` sign
  corrected (now matches IsaacLab `+Z` ground reaction convention).

Tests: 27/27 pass after fixes (was 23/27 due to pre-existing mock bug).
Hydra compose smoke for `+terrain=terrain_dfh_stage1_easy +domain_rand=DR_dfh_soft +rewards=loco/reward_hunter_dfh`: all keys resolve, force_coupling present, DR ranges visible, reward scale + sigma propagated.

Deferred:
- D1 (true per-env collision-mesh writeback): heavy — needs per-env
  terrain prim or per-env vertex region. Force-coupled mode side-steps it
  for closed-loop coupling.
- D2 (IsaacSim Kit probe test): needs Kit running.
- D3 (falsification matrix TRUE/SHADOW/SHUFFLED/RIGID): design after R2 review.
- F7 (furrow direction wiring from `furrow_terrain.py`): defaults still 0°;
  anisotropy is non-zero through `mu_eff` lookup but direction uniform.

---

## Round 2 (2026-05-09)

### Assessment (Summary)
- Score: 4/10
- Verdict: NO

### Top weaknesses (ranked)
1. External force frame: ext_dfh applied world-frame forces; IsaacLab `set_external_force_and_torque` expects body-local.
2. Stale wrench: no-contact path returns without clearing force buffer.
3. DFH extent (12 m) doesn't match parent furrow tile (8 m).
4. Force coupling can only add drag, not lower friction; baseline_mu mismatched parent dynamic_friction.
5. Contact patch area 0.014 m² unsupported by URDF; rotated ankle-box bottom face is ≈ 1.75e-3 m².
6. Slip-sinkage is bounded surrogate (acceptable but document).

### Actions taken
- Force-frame fix: added `_quat_rotate_inverse_wxyz` and rotate accumulated world-frame forces to per-foot body frame in `build_force_apply_for_robot`.
- Stale-wrench fix: `DFHIsaacSimAdapter._clear_force_buffer()` invoked on no-contact and force-coupling-disabled branches.
- Extent fix: `terrain_dfh_stage{1,2,3}.terrain_length_m / terrain_width_m = 8.0`.
- Baseline_mu fix: stage 2/3 override parent `static_friction`/`dynamic_friction`; baseline_mu=parent dyn for all stages so anisotropy deltas are non-negative.
- Patch area: 1.75e-3 m² in all stages.
- Slip-sinkage docstring updated.

Tests: 27/27 pass. Hydra compose verified for all three stages.

---

## Round 3 (2026-05-09)

### Assessment (Summary)
- Score: 5/10
- Verdict: NO for submission, ALMOST for launching paired training

### Top weaknesses (ranked)
1. v0.2 is a force-coupled surrogate, not deformable contact.
2. Furrow direction still uniform per env.
3. env_spacing=5.0 default + 8 m tiles only fits 256 envs at 16×16.
4. Force applied at body frame, not contact point.
5. `query_sinkage_at_feet` still origin-less.
6. No unit tests for new critical paths.

### Actions taken
- Furrow direction wiring: per-env θ sampled from `terrain.terrain_kwargs.orientation_range_deg` and broadcast to grid; passed to `DFHTerrainLayer.initialize`.
- env_spacing override `2.5` added to `stage1_dfh.yaml` and `stage2_dfh.yaml` curriculum yamls.
- `DFHIsaacSimAdapter.query_sinkage_at_feet` now uses env_origins + extent_xy.
- Unit tests added: `test_quat_rotate_inverse_wxyz_known_rotations`, `test_adapter_clears_force_buffer_when_no_contacts`. 29/29 pass.
- Documented v0.2 framing as "force-coupled DFH surrogate"; full deformable contact deferred to v0.3.

---

## Round 4 (2026-05-09) — TERMINAL

### Assessment (Summary)
- Score: 6/10
- Verdict: ALMOST (stop condition met: score >= 6 AND verdict contains "almost")

### Top weaknesses (ranked)
1. Furrow direction is range-matched, not actual-tile-matched (statistically consistent for stages 2/3).
2. Contact-source exception path didn't clear wrench buffer (FIXED post-review as a trivial robustness fix).
3. No collision-height coupling — keep "force-coupled surrogate" framing in docs/paper.
4. Two DR keys unused: `cohesion_pa`, `friction_angle_deg` — remove from v0.2 DR claim or wire later.
5. Real blocker is training evidence, not code.

### Actions taken (post-R4)
- Trivial: exception-path now invokes `_clear_force_buffer()` before returning. 29/29 tests still pass.

### Reviewer Raw Response (R4)

<details>
<summary>Click to expand</summary>

**Score: 6/10. Verdict: ALMOST.**

For v0.2 as a force-coupled DFH surrogate, this is now structurally ready for a paired training launch. The closed loop is real enough that `terrain_dfh_stage1_easy + DR_dfh_soft + reward_hunter_dfh` should expose the policy to dynamics distinct from rigid furrows: per-env plastic state affects anisotropic and sinkage drag, force application is in the correct body frame, stale wrenches are cleared, and env-origin mapping is used consistently.

It is not submission-ready yet because the evidence is still missing, and because the implementation is a force surrogate rather than deformable collision contact.

Discriminating test: run matched seeds + geometry — `DFH_FULL`, `SHADOW_DFH` (update h_plastic, disable force + reward), `RIGID_MATCHED`, `SHUFFLED_DFH` (force on but spatially shifted depth_after / mu_eff). Convincing evidence = DFH_FULL separates from SHADOW_DFH / RIGID_MATCHED on tracking, slip, stance timing, drag diagnostics, learned gait; SHUFFLED_DFH degrades.

Smallest path to READY for v0.2: add the exception-path clear (DONE), launch paired smoke + training. If `DFH_FULL` beats / clearly differs from `SHADOW_DFH` / `RIGID_MATCHED` with nonzero logged `last_mean_drag_n`, `last_max_sink_m`, sinkage penalty, v0.2 becomes defensible.

</details>

---

## Method Description

DFH v0.2 (force-coupled surrogate) — closed-loop deformable-furrowed-heightfield extension on top of HumanoidVerse / IsaacLab Hunter locomotion.

The IsaacSim wrapper attaches a `DFHIsaacSimAdapter` per training run when `terrain.dfh_enabled=True`. The adapter owns a per-env plastic-deformation buffer `H_plastic[N_envs, grid_h, grid_w]` (signed metres, negative = sunk) on CUDA, plus per-env scalar parameter tensors for Bekker `kc/k_phi/n`, anisotropic friction `μ_along/μ_across`, slip-sinkage `α`, and bulldozing share, sampled at episode reset from `domain_rand.dfh_param_ranges`. Per-cell furrow direction is initialised by sampling one θ per env from `terrain.terrain_kwargs.orientation_range_deg`.

Each physics step (before `sim.step`):
1. `HunterFootContactSource` reads per-foot world-frame position, linear velocity, and `contact_sensor.data.net_forces_w`, filters by 5 N normal-force threshold, and emits a `FootContact` snapshot with `env_idx`, `foot_local_idx`, `pos_xy_m`, `normal_force_n`, `v_tangential_xy`, `v_normal_z`.
2. `DFHTerrainLayer.step(contacts, env_origins_xy)` maps contacts to per-env grid cells (env-origin centered, 8 m × 8 m extent, `floor` quantisation), computes Bekker pressure-sinkage with per-env `(kc, k_phi, n)` and a bounded slip-sinkage surrogate `slip = |v_t| / clamp(|v_t|+|v_n|, eps_floor) ∈ [0, 1]`, ratchets `H_plastic` toward the new floor, and bulldozes 4-neighbour cells with per-env share. Returns per-contact `(cell_i, cell_j, z_eff, depth_after, mu_eff)`.
3. `DFHIsaacSimAdapter._apply_force_coupling` computes per-foot tangential drag = `(mu_eff − baseline_mu).clamp_min(0) · |F_n|` (anisotropic friction recovery on top of PhysX scalar μ) plus sinkage drag = `k_sink · |H_plastic| · |F_n|`, both opposing tangential motion direction, capped at `max_drag_force_n`. Forces accumulate per `(env, foot)` in a `(N_envs, n_feet, 3)` world-frame buffer, then are rotated into each foot's body frame via `quat_rotate_inverse(body_quat_w, force_world)` and pushed through `Articulation.set_external_force_and_torque(forces_local, torques_local, body_ids=foot_idx)`. On no-contact / force-coupling-disabled / contact-source-exception paths, the buffer is explicitly zeroed so stale wrench from the prior step does not persist.

Training side: `LeggedRobotLocomotion._reward_penalty_sinkage_excess` reads `H_plastic` at each foot's per-env cell and penalises `sum_i max(|sink_i| − σ, 0)²` with `σ = 0.04 m`. `LeggedRobotBase` adds non-finite and high-base (>5 m) terminations to catch numeric blow-ups that the previous slip-sinkage divide-by-eps could trigger. Per-env DR is plumbed via `_reset_buffers_callback → simulator.dfh_randomize → DFHTerrainLayer.randomize_envs` reading `domain_rand.dfh_param_ranges`.

The visual USD heightfield writeback (Path A) is opt-in (`terrain.dfh.writeback_enabled=False` by default) so the cross-env `amin` reduction over a shared mesh does not contaminate per-env state. Per-env collision-mesh writeback is left as v0.3 work.

---

## Final Status

- Loop terminated at round 4/4 (max).
- Score progression: 2 → 4 → 5 → 6.
- Verdict progression: NO → NO → NO/ALMOST → ALMOST.
- All structural blockers from rounds 1–3 closed.
- Submission-ready requires the discriminating training matrix (DFH_FULL / SHADOW_DFH / RIGID_MATCHED / SHUFFLED_DFH) + paper-grade ablations; that's a separate experiment phase, not a code phase.
- Tests: 29/29 pass (`extensions/dfh/tests`).
- Hydra compose verified for all DFH stages with corrected extent / patch / baseline.

### Files changed (summary)
- extensions/dfh/source/ext_dfh/kernels.py — bounded slip-sinkage, env-origin-aware `world_xy_to_cell`.
- extensions/dfh/source/ext_dfh/dfh_layer.py — env-origin plumbing in `step`, per-env DR, `_friction_at` exposes `mu_eff`.
- extensions/dfh/source/ext_dfh/integration.py — env-origin provider, force-coupling, stale-wrench clearing, body-frame rotation, exception-path clear.
- extensions/dfh/configs/terrain/terrain_dfh_stage{1,2,3}*.yaml — 8 m extent, 1.75e-3 m² patch, baseline_mu aligned with PhysX μ, force_coupling block, writeback off.
- extensions/dfh/configs/rewards/reward_hunter_dfh.yaml — penalty_sinkage_excess scale -2.0.
- extensions/dfh/configs/curriculum/stage{1,2}_dfh.yaml — env_spacing 2.5.
- extensions/dfh/tests/test_integration.py — fixed pre-existing sign bug, added quat-rotate + stale-wrench tests.
- humanoidverse/simulator/isaacsim/isaacsim.py — DFH attach: env-origins provider, force-apply factory, per-env furrow-direction sampling, opt-in writeback.
- humanoidverse/envs/legged_base_task/legged_robot_base.py — `dfh_randomize` plumbing, non-finite + high-base termination.
- humanoidverse/envs/locomotion/locomotion.py — `_reward_penalty_sinkage_excess` implementation.

### Next steps for the user
1. Smoke launch: `+curriculum=stage1_dfh +robot=hunter/hunter +simulator=isaacsim +exp=locomotion_soil +algo=ppo_soil +obs=loco/leggedloco_obs_singlestep_withlinvel num_envs=64 headless=True project_name=DFHVerify experiment_name=Hunter_DFH_S1_R4` to confirm wiring + sane drag magnitudes (`adapter.last_mean_drag_n`, `adapter.last_max_sink_m`).
2. Paired-training matrix: `DFH_FULL` / `SHADOW_DFH` (force_coupling.enabled=False + penalty_sinkage_excess=0) / `RIGID_MATCHED` (terrain_furrows_stage1_easy + matched dynamic_friction) / `SHUFFLED_DFH` (env-shuffled `depth_after`/`mu_eff`).
3. If matrix shows DFH_FULL ≠ SHADOW_DFH ≠ RIGID_MATCHED with the right sign (more drag → slower stance / more conservative gait / non-trivial sinkage penalty), v0.2 is paper-defensible.

---

# Pass 2 — Improvement loop after smoke (2026-05-10)

ThreadId: `019e0ec8-5c1c-7581-8772-2e9efec9122d`. Score 6.5 → 7 → 7. Verdict ALMOST → ALMOST → ALMOST. 4 rounds, max reached. Plumbing for paired-evidence matrix complete; matrix not yet executed.

## Round 1 (Pass 2)
Reviewer scored 6/10 ALMOST after smoke results. Core gap: drag/sink stats not surfaced; sinkage penalty mathematically tiny (max episode penalty 8e-6 → reads -0.0000); SHUFFLED knob missing; seeding commented out; per-condition Hydra launcher unspecified.

## Round 2 (Pass 2)
Implemented:
- Drag/sink scalar split (`last_mean_aniso_drag_n`, `last_mean_sink_drag_n`, `last_drag_clipped_frac`, `last_contact_count`, `last_mean_normal_force_n`, `last_mean_sink_m`, `last_max_sink_m`).
- TB wiring in `legged_robot_base.py:350-413` → `Env/dfh_*` tags.
- `terrain.dfh.force_coupling.shuffle_env_ids` knob with adversarial permutation in `_apply_force_coupling`.

Reviewer flagged 3 issues: (a) SHADOW diagnostics blocked by force-coupling-disabled branch; (b) no-contact path leaves stale stats; (c) SHUFFLED permuted env_idx confounds direction/contact membership; should permute only DFH state inputs.

## Round 3 (Pass 2)
Implemented:
- Diagnostics split: `_update_diagnostics` runs unconditionally after `layer.step`; `_apply_force_coupling` only overwrites applied-drag stats.
- `_zero_diagnostics` on no-contact and exception paths.
- SHUFFLED narrowed: permutes only `mu_eff` and `depth_after`; `v_t`, `f_n`, `env_idx`, `foot_local_idx` stay bound to actual contact.
- RIGID logging completeness: else-branch zeros all `dfh_*` keys.
- Seeding wired in `train_agent.py:100`: `seeding(int(config.seed), torch_deterministic=...)` runs before env instantiation.
- 3 new unit tests: `test_diagnostics_populate_when_force_coupling_disabled`, `test_diagnostics_zero_on_no_contact`, `test_shuffle_env_ids_routes_drag_to_actual_contacts`. 32/32 pass.

Reviewer scored 7/10 ALMOST. Sketched launcher for paired matrix.

## Round 4 (Pass 2) — TERMINAL
Implemented:
- Wouldbe vs applied drag split: `last_mean_wouldbe_drag_n` (always populated by `_update_diagnostics`), `last_mean_applied_drag_n` (overwritten by `_apply_force_coupling` on success). `last_force_apply_ok` flag flips False on exception.
- TB tags: `Env/dfh_mean_wouldbe_drag_n`, `Env/dfh_mean_applied_drag_n`, `Env/dfh_force_apply_ok`. RIGID else-branch updated to zero them.
- Launch script `extensions/dfh/scripts/run_paired_evidence.sh`: 4 conditions × 3 seeds, 5000 PPO iters at num_envs=2048, sequential, ITERS/SEEDS/CONDITIONS overridable. IsaacSim env sourced. `experiment_dir`/`save_dir`/`hydra.run.dir` overrides ensure logs land at `logs/DFHEvidence/<condition>_s<seed>/`.

Score 7/10 ALMOST. Path to 8/10 READY = run dry (100 iters per condition seed=0) → confirm abort metrics → run full matrix (~25-30h wall-clock) → analysis script: paired deltas by seed for `Train/mean_episode_length`, `rew_tracking_lin_vel`, `rew_tracking_ang_vel`, `rew_termination`, `Env/dfh_*`. READY criterion: FULL has applied drag > 0; SHADOW has wouldbe > 0 + applied = 0; RIGID has all `dfh_*` = 0; FULL behavior differs consistently (3/3 seeds, |mean| > 2σ) from both controls; SHUFFLED degrades or shifts vs FULL.

### Files added/changed (Pass 2)
- extensions/dfh/source/ext_dfh/integration.py — diagnostics split, SHUFFLED knob, force_apply_ok flag, wouldbe vs applied stats.
- extensions/dfh/tests/test_integration.py — 3 new behavior tests.
- extensions/dfh/scripts/run_paired_evidence.sh — paired matrix launcher.
- humanoidverse/envs/legged_base_task/legged_robot_base.py — `Env/dfh_*` TB wiring + RIGID-zero schema.
- humanoidverse/train_agent.py — seeding uncommented.

### Score progression (both passes)
Pass 1: 2 → 4 → 5 → 6.
Pass 2: 6.5 → 7 → 7 → 7.

### Next user action
Dry-run the matrix:
```bash
ITERS=100 SEEDS=0 CONDITIONS="DFH_FULL SHADOW RIGID SHUFFLED" \
  bash extensions/dfh/scripts/run_paired_evidence.sh
```
Verify `Env/dfh_*` tags per condition match the abort checklist. Then run full matrix (drop the env-var overrides).




---

# Pass 3 — Improving DFH performance

## Round 1 (2026-05-12 11:08 JST)

### Assessment (Summary)
- Score: 4/10
- Verdict: No (not ready)
- Key criticisms:
  1. No DFH-specific policy advantage demonstrated (FULL ≈ SHADOW for seeds 1-2)
  2. Perturbation too small (~10 N drag = ~1-2% body weight; need 30-60 N)
  3. No cross-eval (rigid-trained on DFH vs DFH-trained on DFH)
  4. Seed 0 collapse needs pre-declared exclusion rule
  5. Mechanism under-instrumented (need mu_eff, force budget, slip, h_plastic logs)

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Recommends **A+B hybrid**: tune drag harder + add cross-evaluation. Sweep `sinkage_drag_k {4,20,40}`, `mu_across {0.65,0.85,1.05}`, target applied drag 30-60 N (5-10% body weight). Then cross-eval: train RIGID, SHADOW, DFH_FULL_tuned; evaluate each on RIGID and DFH eval; compute regret = score(DFH-trained on DFH) − score(RIGID-trained on DFH). Pair by seed (1-5, exclude seed 0). Re-frame claim to: "Force-coupled DFH produces policies with lower regret on deformable-terrain evaluation than rigid-trained baselines, while preserving stable IsaacSim training." Defer v0.3 mesh writeback.

</details>

### Actions Taken (this round)
- Bump drag knobs in `extensions/dfh/configs/terrain/terrain_dfh_stage{1,2,3}*.yaml`: sinkage_drag_k 4→20, mu_across +0.20, max_drag_force_n →300
- Write seed-exclusion protocol doc
- Build cross-eval harness script
- Extend TB logging (force budget pct of weight + h_plastic mean)
- Launch tuned sweep + cross-eval (background, ~12 GPU-hrs)

### Status
- Continuing to round 2 after sweep partial signal
- Difficulty: medium
- threadId: 019e19f1-a15d-72b0-a0f0-ef3e843cbe3d

## Round 2 (2026-05-12 11:18 JST)

### Assessment (Summary)
- Score: 5.5/10
- Verdict: Not ready
- Top criticisms:
  1. No tuned result yet (harness ready, not launched)
  2. Bumped knobs may be too strong (~20-25% body weight); need dose-response
  3. Only 2 seeds in launch script; need 1-5 for paper claim
  4. Training-regret alone insufficient; need cross-eval + force-budget logs
  5. Reward shaping deferred; don't mix into first tuned result

### Reviewer Recommendation
Dose-response sweep: mild (k=8, mu_across=0.70, cap=200) / medium (k=14, 0.78, 250) / strong (k=20, 0.85, 300). Pick weakest stable setting where mean drag ≥ 5% body weight AND DFH regret visible. Then 2000-iter cross-eval for seeds 1,2. If sign positive, expand to 5000 iters × seeds 1-5. Primary: raw paired regret; secondary: normalized regret = (DFH@DFH − RIGID@DFH) / RIGID@RIGID.

### Actions Taken
- Reverted stage1 yaml to MILD default (k=8, mu_across=0.70, cap=200)
- Extended launcher to dose-response: calibration phase (3 settings × seed 1 × 800 iters), then `STRENGTH=...` selects full sweep
- Cross-eval phase uses matching dose for eval terrain

### Status
- Continuing to round 3
- Difficulty: medium
- threadId: 019e19f1-a15d-72b0-a0f0-ef3e843cbe3d

## Round 3 (2026-05-12 11:32 JST)

### Assessment (Summary)
- Score: 6/10 (experiment design) / 5/10 (evidence)
- Verdict: Almost (for design) / No (for claim)
- STOP CONDITION MET: score ≥ 6 AND verdict contains "almost"

### Reviewer Raw Response
Design now sufficient at score 6. Path to "Ready":
- Calibration selects mild or medium (not only strong)
- Mean total per-robot drag in [30, 60] N (5-10% body weight)
- drag_clipped_frac < 0.1
- Both seeds positive regret; mean regret ≥ 10% RIGID@RIGID
- track_lv also improves
- Fall rate no worse for DFH@DFH vs RIGID@DFH
- No reward-shaping confounds in primary result
- SHADOW/RIGID sanity = zero applied DFH drag
- Extend to seeds 1-5 for venue submission

### Pre-stop Fix Applied
Reviewer flagged metric semantics: confirm numerator is total per-robot, not per-contact. Fixed:
- Added `last_total_drag_per_robot_n` (sum of contact f_mag / num_envs) to adapter
- Logged 3 new TB tags: `dfh_total_drag_per_robot_n`, `dfh_mean_drag_per_contact_n`, `dfh_applied_drag_pct_weight` (now uses total / weight)

### Status
- LOOP COMPLETE (score ≥ 6, verdict "Almost")
- Pending: launch `extensions/dfh/scripts/run_tuned_sweep_crosseval.sh` after existing matrix finishes (~16:00 JST today)
- Difficulty: medium
- threadId: 019e19f1-a15d-72b0-a0f0-ef3e843cbe3d

---

# Pass 4 — Re-improve scores after v0.2 matrix falsification

## Round 1 (2026-05-14 02:00 JST)

### Assessment (Summary)
- Score: 4/10
- Verdict: Not ready
- Key reviewer points:
  - v0.2-weak is now empirically null (SHUFFLED_s1 outperforms FULL_s1)
  - SHUFFLED becomes required negative control for tuned result
  - Training-time regret alone insufficient for 7+; cross-eval mandatory
  - Likely tuned failure mode: saturation/policy compensation (no terrain-memory exploitation)
  - **8/10 minimum experiment:** FULL + RIGID + SHUFFLED × seeds 1-3 trained at tuned strength, all evaluated on DFH_FULL eval, paired

### Actions Taken
- Phase A calibration launched (background; CALIB_mild_s1 ETA ~87min)
- Extended launcher to add SHUFFLED training in Phase B (negative control)
- Phase B now: FULL + RIGID + SHUFFLED × seeds {1,2}, 2000 iters
- Phase C cross-eval: 3 conditions × {rigid, dfh} eval per seed

### Status
- Continuing to round 2 after Phase A lands
- Difficulty: medium
- threadId: 019e252c-9dde-7f52-9f31-4588d7b33de0

## Round 2 (2026-05-14 20:00 JST)

### Calibration Results (800 iters, seed=1)
| Dose | ep_len | track_lv | term | total_drag/robot_N | pct_weight | mean_drag/contact_N | clip% | F_n_mean |
|------|--------|----------|------|---------------------|------------|----------------------|-------|----------|
| mild   | 576 | 0.86 | -1.89 | 10.6 | 1.7%  | 17.0 | 0.4% | 32.7 N |
| medium | 680 | 1.03 | -1.44 | 12.5 | 2.0%  | 23.2 | 0.4% | 28.8 N |
| strong | 563 | 0.84 | -1.90 | 21.5 | 3.5%  | 32.5 | 0.6% | 30.3 N |

### Assessment
- Score: 4.5/10
- Verdict: Not ready
- Diagnosis: `mean_normal_force_n` ≈ 30 N because contact threshold 5 N admits swing/transient contacts, diluting average. All doses below 5% body weight target.
- Reviewer accepts Option 2+3: extend to ULTRA + add stance-gated p95 metrics.

### Actions Taken
- Added adapter diagnostics: `last_stance_drag_contact_mean_n`, `last_stance_drag_contact_p95_n` (gated F_n > 100 N), `last_total_drag_per_robot_p95_n`, `last_stance_fraction`
- Logged TB tags: `dfh_stance_drag_contact_mean_n`, `dfh_stance_drag_contact_p95_n`, `dfh_total_drag_per_robot_p95_n`, `dfh_stance_fraction`
- Added ULTRA dose: k=40, mu_across=1.0, mu_along=0.60, cap=500
- Launched ULTRA calibration (background, ETA ~90min)
- ULTRA stability bar: ep_len≥500, track_lv≥0.75, term>-2.2, clip_frac<5%

### Status
- Continuing to round 3 after ULTRA lands
- Difficulty: medium
- threadId: 019e252c-9dde-7f52-9f31-4588d7b33de0

## Round 3 (2026-05-18) — DESIGN-GATE (Pass 4)

### Context
Power outage killed the prior Codex thread (019e252c) and interrupted the sweep
before Phase B. Calibration checkpoints survived. ULTRA calibration analysed,
strength=ultra selected, Phase B/C relaunched (~22 GPU-h, bg PID 111152).
Fresh Codex thread opened: **019e38d8-dc92-7f11-a23c-9ce338c358e6**.

### ULTRA calibration vs pre-declared bars (all pass)
ep_len 584 (≥500) · track_lv 0.865 (≥0.75) · term -1.80 (>-2.2) ·
clip 0.55% (<5%) · drag 6.7% wt (5-10%) · total/robot 40.7 N (30-60 N).
mild/medium/strong rejected (1.7/2.0/3.5% — below 5% target).

### Assessment (Summary)
- Score: **Design 6/10 conditional** (5/10 if cap saturation hidden);
  **Evidence 1/10** (decisive Phase B/C pending ~22 h).
- Verdict: **almost (design-gate pilot); not ready (claim)**.
- Reviewer decision: **GO on the running sweep — do NOT change dose midstream.**
- STOP CONDITION not honoured: design 6/"almost" is marginal but reviewer is
  explicit the *claim* needs pending evidence + seed 3 + analysis fixes.
  Evidence gate kept OPEN (anti-gaming); loop continues to R4 post-sweep.

### Top weaknesses (ranked) + minimum fix
1. **Cap saturation** — stance p95 = 500 N = cap, mean 353 N, but robot
   clip 0.55% / 40.7 N in band. Not auto-fatal; must report active/
   stance-conditioned clip & pre-cap percentiles. If fix needed: LOWER
   sinkage_drag_k (not raise cap). → analysis-side surfaced now; pre-cap
   percentile instrumentation deferred to next pass (needs re-run).
2. **2 seeds** — enough for 6/10 design-gate IF both agree & SHUFFLED
   degrades; NOT enough for the claim. Min next step: seed 3 (then 1-5).
3. **SHUFFLED must be dose-matched** to FULL except spatial coupling →
   added FULL-vs-SHUFFLED matched-dose diagnostic table.
4. **"regret" sign confusing** → renamed to paired_improvement, sign doc'd.
5. **2000 iters may undertrain asymmetrically** → flag training curves at
   evidence review; if still rising it's a calibration, not final, result.

### Actions Taken (Phase C — non-GPU, zero-risk to running sweep)
- Rewrote `extensions/dfh/scripts/analyze_regret.py`:
  - regret → **paired_improvement** with explicit sign documentation (W4).
  - FULL-vs-SHUFFLED matched-dose table: drag%wt, total N, p95, stance p95,
    clip%, stance_frac (W1, W3).
  - cap-saturation flag (stance_p95≥480 & clip<5%) (W1).
  - per-seed sign-agreement verdict: POSITIVE / NULL / INCONCLUSIVE (W2).
  - CSV widened to all diagnostic columns.
  - Smoke-tested on empty crosseval (graceful "sweep still running").
- Deferred (needs re-run, NOT done mid-sweep to preserve run-1-vs-rest
  consistency): adapter pre-cap requested-drag percentiles + capped-impulse
  fraction. Logged as the Pass-5 instrumentation gap.

### Results
- No evidence yet (Phase B run 1/6 DFH_FULL_ultra_s1 training).
- Analysis harness now hardened so the 22 h output is interpretable on
  arrival (cap confound + negative-control validity visible).

### Status
- Continuing to Round 4 (MAX) — the EVIDENCE review — after Phase B/C lands.
- Difficulty: medium
- threadId: 019e38d8-dc92-7f11-a23c-9ce338c358e6

## Round 4 (2026-05-19) — EVIDENCE REVIEW (FINAL / MAX_ROUNDS)

### Assessment (Summary)
- Score: **design 5.5/10, evidence 2/10, overall 3/10**
- Verdict: **not ready** (for the claim)
- Headline claim: **NOT supported** — clean null/negative on the primary metric
- Reviewer thread: prior 019e38d8 lost to server restart during the 22h sweep
  (2nd thread loss); fresh continuation thread **019e3cf3-5e9a-73e2-805f-453cf1a98399**
- Key points:
  - DFH_FULL does NOT beat RIGID_TUNED on DFH eval (26.2 < 27.0; paired
    improvement −0.8 / −2.8%). FULL>SHUFFLED (+2.3) cannot rescue the claim.
  - SCOPE SHORTFALL: SEEDS=1 only (sweep ran SEEDS=1, not {1,2}) — below the
    R3 ≥2-seed design-gate bar and the seed-3 claim bar, regardless of sign.
  - Cap saturation at eval is WORSE than ULTRA calibration: stance p95 ≈
    85–92% of the 500 N cap, clip 17–31% (calib 0.55%), tot/robot ~4.1% wt
    (below 5–10% ULTRA target). Eval operating point heavily cap-limited and
    materially harsher-clipped than the calibrated point.
  - Pipeline-fix validity: added `sample_eps.py` TB writer judged a legitimate
    instrumentation repair (mirrors `ppo.py` tags, same `infos` sources, stale
    events removed) — NOT a validity threat. `num_envs=2048` eval = moderate
    external-validity limitation but does NOT explain away the null (negative
    even under a DFH-favorable matched-layout setup).

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (codex gpt-5.5 xhigh, thread 019e3cf3)</summary>

**Final Ruling**

1. **Score: design 5.5/10, evidence 2/10. Overall: 3/10.**
   The design is close to the right experiment, but the executed evidence misses core acceptance conditions: single seed only, cap-heavy force regime, and negative paired improvement.

2. **Verdict for the claim: not ready.**

3. **Headline claim: not supported by this evidence.**
   On the primary metric, DFH_FULL does **not** beat RIGID_TUNED on DFH eval: `26.2 < 27.0`, paired improvement `-0.8` / `-2.8%`. That is a clean null/negative for the stated claim. FULL beating SHUFFLED is interesting, but it cannot rescue the claim because FULL does not beat RIGID, and SHUFFLED also appears slightly harsher and more clipped.

4. **Minimum honest next step:**
   Treat this as a **negative/null Pass-4 result**, not a positive. The paper/claim should pivot unless you run a targeted repair experiment.

   If re-running, I would require both:
   - **De-saturate the cap by lowering `sinkage_drag_k`**, not by raising the cap. Target: stance p95 comfortably below cap, e.g. `<70-75%` of 500 N, clip fraction preferably `<5%`, while mean drag remains in the intended physical range if possible.
   - **Run seeds 1-3 minimum** for DFH_FULL, RIGID_TUNED, SHUFFLED, with the same train/eval matrix.

   Acceptance bar:
   - DFH_FULL must beat RIGID_TUNED on DFH eval in at least **2/3 seeds**, preferably all 3.
   - Mean paired improvement across seeds must be positive with no obvious outlier-only dependence.
   - DFH_FULL must also beat SHUFFLED under matched drag statistics.
   - Cap saturation must not dominate the impulse.
   - If claiming mechanism, include training/eval curves showing the result is not just asymmetric undertraining at 2000 iters.

   Without that rerun, the honest framing is: **"At the tested ultra single-seed setting, DFH did not improve deformable-terrain episode length over rigid training; spatial coupling may affect behavior relative to shuffled control, but it did not yield the claimed generalization benefit."**

5. **Pipeline fixes validity threat?**
   The TensorBoard writer fix is not a major validity threat if it mirrors `ppo.py` tags from the same `infos['episode']` and `infos['to_log']` sources, and stale event files were removed. That sounds like a legitimate instrumentation repair.

   The `num_envs=2048` eval constraint is a moderate external-validity limitation, because eval terrain layout matching the training layout weakens a broad "generalization" interpretation. But since the result is negative even under that potentially favorable setup for DFH, it does not explain away the null. It mostly means: do not overclaim from these numbers either way.

</details>

### Actions Taken
- Recovered the dead Pass-4 ultra sweep (prior bg pid 111152 killed at SHUFFLED
  iter 973/2000). DFH_FULL_ultra_s1 + RIGID_TUNED_s1 already complete; retrained
  SHUFFLED_ultra_s1 fresh → 2000 via `recover_shuffled_phaseBC_ultra.sh`.
- Fixed 4 latent breaks in the never-validated Phase C→analyze_regret pipeline:
  (1) `+num_episodes` (not in base_eval struct); (2) added a config-gated
  TensorBoard `SummaryWriter` to `humanoidverse/sample_eps.py` emitting the same
  `Episode/*` + `Env/dfh_*` tags `ppo.py` logs (sourced from `infos['episode']`
  / `infos['to_log']`), via new `+eval_tb_dir`; (3) removed invalid struct keys
  (project_name/experiment_name/experiment_dir/save_dir), route via
  `+eval_tb_dir` + `eval_name`; (4) ran cross-eval at `num_envs=2048` because the
  packed per-env spawn-anchor terrain path is hard-coupled to the training env
  count. Same fix mirrored into `run_tuned_sweep_crosseval.sh`.
- Cleaned a stale-tfevents artifact (empty event files from crashed attempts
  produced one bogus all-zero row); re-ran `analyze_regret`. All 6 crosseval
  dirs now hold exactly 1 tfevents.

### Results
- ultra, seed 1, 50 ep/eval, 2048 envs:
  DFH@DFH 26.2 · RIGID@DFH 27.0 · SHUF@DFH 23.9 · improv −0.8 (norm −2.8%) ·
  FULL−SHUF +2.3. analyze_regret verdict: `NEGATIVE/NULL — claim not supported`.
- CSV: `logs/DFHTuned/crosseval/regret_ultra.csv`.

### Status
- **LOOP TERMINATED** at MAX_ROUNDS (Pass 4, Round 4/4) WITHOUT positive
  assessment. Final overall score 3/10, claim not supported.
- Difficulty: medium
- threadId: 019e3cf3-5e9a-73e2-805f-453cf1a98399

---

# Final Summary (Pass 4 close-out)

**Outcome:** The DFH generalization claim is **not supported** at the tested
ultra single-seed setting. This is a clean scientific **null/negative**:
RIGID-trained marginally outlasts DFH-trained on deformable-terrain eval
(27.0 vs 26.2). The spatial-memory negative control is internally consistent
(FULL > SHUFFLED, +2.3) but moot because FULL never exceeds RIGID.

**Remaining blockers (effort to convert null → decisive, per reviewer bar):**
1. Cap saturation — eval drag clips 17–31% and stance p95 ≈ 90% of the 500 N
   cap. Fix = LOWER `sinkage_drag_k` (do not raise cap) until stance p95 <70–75%
   of cap and clip <5%. Effort: 1 short re-calibration (~1–2 GPU-h) to pick k.
2. Statistical power — only seed 1. Need seeds 1–3 × {DFH_FULL, RIGID_TUNED,
   SHUFFLED} × 2000 iters + 6 cross-evals/seed. Effort: ~3× the current
   sweep (~30–40 GPU-h) at the de-saturated dose.
3. Undertraining check — pull late-iteration training curves to rule out
   asymmetric undertraining at 2000 iters (cheap, no GPU).

**Recommendation:** Do NOT proceed to paper writing on this result. Either
(a) **pivot/reframe** to an honest null finding ("DFH spatial coupling did not
yield a generalization benefit over rigid training at the tested setting; it
does alter behavior vs a shuffled control"), or (b) run the **targeted repair**
(de-saturated `sinkage_drag_k` + seeds 1–3) with the exact acceptance bar above
before any positive claim. Given two prior nulls in Pass 4 and a consistent
cap-saturation confound, a pivot to a negative-result / diagnostic contribution
is the lower-risk path; the repair is justified only if the de-saturated pilot
shows a positive seed-1 trend first.

## Method Description

DFH (Deformable-terrain Force-coupling Hypothesis) augments IsaacSim rigid-body
Hunter locomotion with a per-environment soil force-coupling layer: an
anisotropic sinkage-drag model applies horizontal resistive force at foot
contacts as a function of penetration depth and along/across friction
(`mu_along`, `mu_across`, `sinkage_drag_k`, capped at `max_drag_force_n`), with
per-env spatial coupling so terrain response is spatially consistent within an
episode. Training: PPO (`ppo_soil`) on `terrain_dfh_stage1_easy` with
`DR_dfh_soft`, 2000 iters, 2048 envs. Evaluation matrix: {DFH_FULL,
RIGID_TUNED, SHUFFLED (env-id-shuffled, dose-matched negative control)} ×
{rigid eval, dfh eval}, scored by mean episode length; the claim metric is the
paired improvement of DFH-trained over RIGID-trained on DFH eval. Cross-eval
metrics are logged to TensorBoard from `sample_eps.py` (mirroring the training
logger's `Episode/*` and `Env/dfh_*` tags) and aggregated by
`analyze_regret.py` into a paired-improvement + dose-match + cap-saturation
report. **Result at the tested ultra single-seed setting: null/negative.**
