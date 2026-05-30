# Compliant-Contact (k_n / c_n) Feasibility — Scoping Note

**Date:** 2026-05-29. No code changed; investigation only.
**Question:** Can the paper's "soft soil" knob — per-env contact stiffness k_n and
damping c_n — be set/randomized in this IsaacLab without a new physics backend?
This is the last standing hypothesis for why the slip claim didn't replicate (the
gate was ruled inert by Arm C; eval noise quantified at ~8% CV).

## Headline correction

The original result note said *"IsaacLab 1.4.1 does not expose per-env material
stiffness/damping cleanly."* Two corrections:
1. **The installed version is 0.30.7**, not 1.4.1 (`IsaacLab/source/extensions/omni.isaac.lab/config/extension.toml:4`).
2. The claim is **right about per-env randomization but wrong about exposure**: the
   compliant-contact fields ARE exposed in the config layer; only the *per-env
   randomization* path is missing.

## Findings

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Version | 0.30.7 | `config/extension.toml:4` |
| 2 | `RigidBodyMaterialCfg` has `compliant_contact_stiffness` + `compliant_contact_damping` | **EXPOSED** | `sim/spawners/materials/physics_materials_cfg.py:74,81`; flows to PhysX at `physics_materials.py:70` |
| 3 | `DeformableBodyMaterialCfg` (FEM) exists | Exists, not for terrain | `physics_materials_cfg.py:90-130`; FEM soft-bodies, terrain importer takes rigid only |
| 4 | Per-env DR for stiffness/damping | **MISSING** | `envs/mdp/events.py randomize_rigid_body_material` samples a `(num_buckets, 3)` tensor = (static_fric, dyn_fric, restitution) only; `set_material_properties()` tensor API has no stiffness/damping slot |
| 5 | HumanoidVerse terrain material build | friction/restitution only | `humanoidverse/simulator/isaacsim/isaacsim.py:500-506` (generator), `:524-530` (plane) — compliant fields never set |
| 6 | Terrain material scope | single shared material, all envs | `terrains/terrain_importer.py` spawns one material at `{prim}/physicsMaterial` |

## The reframe: two very different costs

The expensive thing the original note balked at is **per-env RANDOMIZED** k_n/c_n.
But the first scientific question is not "can we randomize compliance" — it is
**"does adding ANY contact compliance change foot slip at all?"** That needs only a
GLOBAL (all-env-shared) compliant value, which is trivially supported.

### Stage 0 — GLOBAL compliant contact (CHEAP, do first)

A shared non-zero `compliant_contact_stiffness`/`damping` on the terrain material is
a ~4-line edit at `isaacsim.py:500-506`, reading two new fields off `terrain_config`
exactly like `static_friction` already is:

```python
physics_material=sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode=_fric_mode,
    restitution_combine_mode=_rest_mode,
    static_friction=self.terrain_config.static_friction,
    dynamic_friction=self.terrain_config.dynamic_friction,
    restitution=getattr(self.terrain_config, "restitution", 0.0),
    compliant_contact_stiffness=getattr(self.terrain_config, "compliant_contact_stiffness", 0.0),
    compliant_contact_damping=getattr(self.terrain_config, "compliant_contact_damping", 0.0),
),
```

Plus the same two lines in the plane branch (`:524-530`) and the two fields added to
the furrows terrain YAMLs. **No event manager, no PhysX bindings, no per-env work.**

- **Effort:** code ~30-60 min.
- **Probe (free-ish):** eval an existing Arm B/C checkpoint on compliant-vs-rigid
  terrain (~20 min) to see if the contact model even shifts the slip mechanic.
  Caveat: a frozen policy on suddenly-compliant ground tests robustness, not learned
  behavior — a real read needs a short retrain (~2h/seed) with compliance on.
- **Decision value:** if global compliance does NOT move slip, per-env randomization
  (the expensive build) is pointless — STOP. If it does, the hypothesis is confirmed
  and per-env variation becomes worth costing.

### Stage 1 — PER-ENV randomized k_n/c_n (EXPENSIVE, only if Stage 0 positive)

Genuinely not supported by the existing API. Three sub-options, cheapest first:
1. **Per-sub-terrain baked compliance** (realistic): the terrain *generator* tiles a
   grid of sub-terrains; assign a different compliant value per sub-terrain row so
   envs spawned on different tiles see different compliance — a discrete approximation
   of per-env randomization. Requires teaching the terrain importer to spawn distinct
   materials per sub-terrain (it currently spawns one shared material). **~1-2 days.**
2. **Custom event term writing USD `/physicsMaterial` attributes** — blocked by the
   shared-material scope; would need per-env terrain prims, fighting PhysX instancing.
3. **New PhysX tensor bindings** for compliant contact (C++ extension). Highest effort.

## Bottom line

- **Per-env k_n/c_n: NOT feasible without custom work** (~1-2 days for the
  per-sub-terrain approximation; more for true per-env). Original note's spirit holds.
- **BUT global compliant contact is a ~4-line change** and answers the actual first
  question — does compliance affect slip at all — for ~1 hour of work, before any
  expensive build. This is the missed cheap path.

## Recommended next action

Stage 0: add the two compliant-contact fields, set a low global stiffness on the
furrows terrain, retrain one Arm-B-recipe seed (~2h) and eval slip vs the rigid
baseline. Only if slip moves materially do we cost out Stage 1 per-env randomization.
