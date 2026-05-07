"""IsaacSim integration shim for the DFH terrain layer.

This module concretely wires :class:`ext_dfh.DFHTerrainLayer` into the
HumanoidVerse IsaacSim wrapper.  Three lifecycle hooks are added inside
``humanoidverse/simulator/isaacsim/isaacsim.py``:

  * Construction          -> :func:`build_dfh_adapter`     (called once after terrain init)
  * Per physics step      -> :meth:`DFHIsaacSimAdapter.on_physics_step`
  * Per env reset         -> :meth:`DFHIsaacSimAdapter.on_reset`

A separate concrete :class:`HunterFootContactSource` reads the IsaacLab
contact sensor + body state for the Hunter feet and produces a
:class:`FootContact` snapshot the layer can consume.

The PhysX heightfield write-back (visible terrain deformation) is left as a
v0.3 TODO — for v0.2 the layer maintains the plastic buffer and applies
contact-wrench corrections via foot external forces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

import torch

from ext_dfh.config import DFHConfig
from ext_dfh.dfh_layer import DFHTerrainLayer, FootContact
from ext_dfh.kernels import world_xy_to_cell


# ----------------------------------------------------------------- contact src


class HunterFootContactSource:
    """Pulls per-step foot contacts for the Hunter humanoid from IsaacLab.

    Hunter contact bodies (per ``robot/hunter/hunter.yaml``):
      * ``left_ankle_link``
      * ``right_ankle_link``

    The IsaacSim wrapper exposes:
      * ``contact_sensor.data.net_forces_w``  — (N_envs, N_bodies, 3)
      * ``_robot.data.body_pos_w``            — (N_envs, N_bodies, 3)
      * ``_robot.data.body_lin_vel_w``        — (N_envs, N_bodies, 3)
    """

    def __init__(
        self,
        robot: Any,
        contact_sensor: Any,
        foot_body_names: List[str],
        normal_force_threshold_n: float = 5.0,
    ) -> None:
        self._robot = robot
        self._contact_sensor = contact_sensor
        self._foot_body_names = list(foot_body_names)
        self._normal_force_threshold = normal_force_threshold_n
        # Body indices resolved lazily on first __call__ (IsaacLab Articulation
        # is not safe to query until after the simulation has played one step).
        self._foot_body_idx_robot: Optional[List[int]] = None
        self._foot_body_idx_sensor: Optional[List[int]] = None

    def _resolve_body_indices(self) -> None:
        idx_robot: List[int] = []
        idx_sensor: List[int] = []
        for name in self._foot_body_names:
            ir, _ = self._robot.find_bodies(name)
            is_, _ = self._contact_sensor.find_bodies(name)
            if not ir or not is_:
                raise RuntimeError(f"Foot body '{name}' not found in robot/contact_sensor.")
            idx_robot.append(ir[0])
            idx_sensor.append(is_[0])
        self._foot_body_idx_robot = idx_robot
        self._foot_body_idx_sensor = idx_sensor

    def __call__(self) -> FootContact:
        if self._foot_body_idx_robot is None:
            self._resolve_body_indices()
        # Robot body pos/vel are world-frame; (N_envs, N_bodies, 3).
        body_pos = self._robot.data.body_pos_w
        body_lin_vel = self._robot.data.body_lin_vel_w
        contact_forces = self._contact_sensor.data.net_forces_w  # (N_envs, N_bodies, 3)

        # Stack across feet -> (N_envs, N_feet, 3).
        feet_pos = body_pos[:, self._foot_body_idx_robot, :]
        feet_vel = body_lin_vel[:, self._foot_body_idx_robot, :]
        feet_force = contact_forces[:, self._foot_body_idx_sensor, :]

        n_envs, n_feet, _ = feet_pos.shape
        # Flatten to (N_envs * N_feet, 3) for vector kernels.
        pos_flat = feet_pos.reshape(-1, 3)
        vel_flat = feet_vel.reshape(-1, 3)
        force_flat = feet_force.reshape(-1, 3)

        normal_force = force_flat[:, 2]  # IsaacLab net_forces_w on body is +Z under ground reaction (foot in contact)
        active = normal_force > self._normal_force_threshold

        if not active.any():
            empty_long = torch.empty(0, dtype=torch.long, device=pos_flat.device)
            empty_2 = torch.empty(0, 2, device=pos_flat.device)
            empty_1 = torch.empty(0, device=pos_flat.device)
            return FootContact(
                env_idx=empty_long,
                pos_xy_m=empty_2,
                normal_force_n=empty_1,
                v_tangential_xy=empty_2,
                v_normal_z=empty_1,
            )

        # Build env_idx for active flat entries.
        env_grid = torch.arange(n_envs, device=pos_flat.device).repeat_interleave(n_feet)

        return FootContact(
            env_idx=env_grid[active],
            pos_xy_m=pos_flat[active, :2],
            normal_force_n=normal_force[active],
            v_tangential_xy=vel_flat[active, :2],
            v_normal_z=vel_flat[active, 2],
        )


# ----------------------------------------------------------------- adapter


@dataclass
class IsaacSimContactSource:
    """Adapter contract used by :class:`DFHIsaacSimAdapter`."""

    get_contacts: Callable[[], FootContact]


class DFHIsaacSimAdapter:
    """Owns one :class:`DFHTerrainLayer` and drives it from IsaacSim hooks."""

    def __init__(
        self,
        num_envs: int,
        config: DFHConfig,
        contact_source: IsaacSimContactSource,
    ) -> None:
        self.layer = DFHTerrainLayer(num_envs=num_envs, config=config)
        self._contact_source = contact_source
        self._physx_writeback: Optional[Callable[[torch.Tensor], None]] = None

    # ----------------------------------------------------------- registration

    def attach_physx_writeback(self, writeback: Callable[[torch.Tensor], None]) -> None:
        """Register the function that pushes ``H_plastic`` into PhysX heightfield."""
        self._physx_writeback = writeback

    def initialize(self, furrow_direction_deg: Optional[torch.Tensor] = None) -> None:
        self.layer.initialize(furrow_direction_deg=furrow_direction_deg)

    # ----------------------------------------------------------- hook surface

    def on_physics_step(self) -> None:
        """Called from IsaacSim ``simulate_at_each_physics_step`` BEFORE ``sim.step``."""
        try:
            contacts = self._contact_source.get_contacts()
        except (AttributeError, RuntimeError) as e:
            # Articulation not fully initialised yet (typical for the first 1-2
            # physics steps).  Skip silently; the next step will succeed.
            logger.debug("DFH contact source not ready yet: %s", e)
            return
        if contacts.env_idx.numel() == 0:
            return
        self.layer.step(contacts)
        if self.layer.needs_physx_writeback() and self._physx_writeback is not None:
            self._physx_writeback(self.layer.h_plastic)

    def on_reset(self, env_ids: torch.Tensor) -> None:
        self.layer.reset(env_ids)

    # ----------------------------------------------------------- diagnostics

    def query_sinkage_at_feet(
        self, foot_pos_xy_m: torch.Tensor, env_idx: torch.Tensor
    ) -> torch.Tensor:
        """Convenience: get current plastic sinkage under a set of foot positions.

        Useful for reward shaping (``penalty_sinkage_excess``) or observations.
        """
        cell_i, cell_j = world_xy_to_cell(
            foot_pos_xy_m,
            self.layer.cfg.grid_h,
            self.layer.cfg.grid_w,
            self.layer.cfg.horizontal_scale_m,
        )
        return self.layer.h_plastic[env_idx, cell_i, cell_j]


# ----------------------------------------------------------------- factory


def build_dfh_adapter(
    num_envs: int,
    dfh_config_dict: dict,
    robot: Any,
    contact_sensor: Any,
    foot_body_names: List[str],
) -> DFHIsaacSimAdapter:
    """Build a fully wired :class:`DFHIsaacSimAdapter` for IsaacSim.

    ``dfh_config_dict`` is the resolved ``terrain.dfh`` block from Hydra.
    """
    cfg = _dfh_config_from_dict(dfh_config_dict)
    contact_src = HunterFootContactSource(
        robot=robot,
        contact_sensor=contact_sensor,
        foot_body_names=foot_body_names,
    )
    adapter = DFHIsaacSimAdapter(
        num_envs=num_envs,
        config=cfg,
        contact_source=IsaacSimContactSource(get_contacts=contact_src),
    )
    return adapter


def _dfh_config_from_dict(d: dict) -> DFHConfig:
    """Build a frozen :class:`DFHConfig` from a Hydra-resolved dict.

    Hydra delivers nested OmegaConf dicts; we materialise just what we need
    so the rest of the layer doesn't depend on omegaconf at import time.
    """
    from ext_dfh.config import BekkerParams, DFHParams, FrictionAnisotropy

    params_d = d.get("params", {}) or {}
    bek_d = params_d.get("bekker", {}) or {}
    aniso_d = params_d.get("anisotropy", {}) or {}

    bekker = BekkerParams(
        kc=float(bek_d.get("kc", 1.4e3)),
        k_phi=float(bek_d.get("k_phi", 8.2e5)),
        n=float(bek_d.get("n", 1.0)),
        cohesion=float(bek_d.get("cohesion", 1.0e4)),
        friction_angle_deg=float(bek_d.get("friction_angle_deg", 30.0)),
    )
    anisotropy = FrictionAnisotropy(
        mu_along=float(aniso_d.get("mu_along", 0.55)),
        mu_across=float(aniso_d.get("mu_across", 0.70)),
        enabled=bool(aniso_d.get("enabled", True)),
    )
    params = DFHParams(
        bekker=bekker,
        anisotropy=anisotropy,
        slip_sinkage_alpha=float(params_d.get("slip_sinkage_alpha", 0.20)),
        slip_sinkage_eps=float(params_d.get("slip_sinkage_eps", 1.0e-3)),
        bulldoze_share=float(params_d.get("bulldoze_share", 0.4)),
        sinkage_floor_m=float(params_d.get("sinkage_floor_m", -0.10)),
        contact_patch_area_m2=float(params_d.get("contact_patch_area_m2", 3.45e-3)),
    )
    return DFHConfig(
        horizontal_scale_m=float(d.get("horizontal_scale_m", 0.10)),
        vertical_scale_m=float(d.get("vertical_scale_m", 0.002)),
        terrain_length_m=float(d.get("terrain_length_m", 12.0)),
        terrain_width_m=float(d.get("terrain_width_m", 12.0)),
        physx_writeback_every_k_steps=int(d.get("physx_writeback_every_k_steps", 4)),
        device=str(d.get("device", "cuda")),
        params=params,
        store_furrow_direction=bool(d.get("store_furrow_direction", True)),
    )
