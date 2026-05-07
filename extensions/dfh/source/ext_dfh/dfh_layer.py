"""DFHTerrainLayer: public API for the deformable furrowed heightfield.

The layer owns the plastic-deformation buffer and exposes three lifecycle hooks
to be called from the IsaacSim wrapper:

  * :meth:`initialize` — once, after terrain mesh exists.
  * :meth:`step` — every physics step, given foot contact info.
  * :meth:`reset` — on env reset, restore buffer to zero for the given envs.

Heightfield write-back into PhysX is delegated to :mod:`ext_dfh.integration`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ext_dfh.config import DFHConfig
from ext_dfh.kernels import (
    anisotropic_friction_coefficient,
    apply_slip_sinkage,
    bekker_sinkage_target,
    bulldoze_neighbours,
    update_plastic_height,
    world_xy_to_cell,
)


@dataclass
class FootContact:
    """Per-foot contact snapshot from IsaacSim contact reporter."""

    env_idx: torch.Tensor       # (N,) long
    pos_xy_m: torch.Tensor      # (N, 2) world XY
    normal_force_n: torch.Tensor  # (N,) Newtons, positive = pressing down
    v_tangential_xy: torch.Tensor  # (N, 2) m/s
    v_normal_z: torch.Tensor    # (N,) m/s


class DFHTerrainLayer:
    """Plastic heightfield + Bekker sink + slip-sinkage + bulldozing + anisotropy."""

    def __init__(self, num_envs: int, config: DFHConfig) -> None:
        self.num_envs = num_envs
        self.cfg = config
        self.device = torch.device(config.device)
        self._h_plastic: Optional[torch.Tensor] = None
        self._furrow_dir_deg: Optional[torch.Tensor] = None
        self._step_counter = 0

    # ---------------------------------------------------------------- lifecycle

    def initialize(self, furrow_direction_deg: Optional[torch.Tensor] = None) -> None:
        """Allocate the plastic buffer and (optionally) per-cell furrow direction."""
        h, w = self.cfg.grid_h, self.cfg.grid_w
        self._h_plastic = torch.zeros(self.num_envs, h, w, device=self.device, dtype=torch.float32)
        if self.cfg.store_furrow_direction:
            if furrow_direction_deg is None:
                # Default: 0 deg (along +x). Override at terrain-gen time.
                self._furrow_dir_deg = torch.zeros(
                    self.num_envs, h, w, device=self.device, dtype=torch.float32
                )
            else:
                self._furrow_dir_deg = furrow_direction_deg.to(self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Zero the plastic buffer for the given envs."""
        if self._h_plastic is None:
            return
        self._h_plastic[env_ids] = 0.0

    def step(self, contacts: FootContact) -> None:
        """Update the plastic buffer from this step's foot contacts.

        Caller is responsible for filtering ``contacts`` to active foot links.
        """
        if self._h_plastic is None:
            raise RuntimeError("DFHTerrainLayer.initialize() must be called first.")

        params = self.cfg.params
        # Pressure from normal force + nominal foot patch area.
        pressure = contacts.normal_force_n / params.contact_patch_area_m2
        z_target = bekker_sinkage_target(pressure, params, self.cfg.horizontal_scale_m)
        z_eff = apply_slip_sinkage(
            z_target, contacts.v_tangential_xy, contacts.v_normal_z, params
        )

        cell_i, cell_j = world_xy_to_cell(
            contacts.pos_xy_m,
            self.cfg.grid_h,
            self.cfg.grid_w,
            self.cfg.horizontal_scale_m,
        )

        # Capture pre-update depth so we can compute delta for bulldozing.
        depth_before = self._h_plastic[contacts.env_idx, cell_i, cell_j]
        self._h_plastic = update_plastic_height(
            self._h_plastic, contacts.env_idx, cell_i, cell_j, z_eff, params
        )
        depth_after = self._h_plastic[contacts.env_idx, cell_i, cell_j]
        delta_z = (depth_before - depth_after).clamp(min=0.0)
        self._h_plastic = bulldoze_neighbours(
            self._h_plastic, contacts.env_idx, cell_i, cell_j, delta_z, params
        )

        self._step_counter += 1

    # ----------------------------------------------------------------- queries

    def needs_physx_writeback(self) -> bool:
        """True when the plastic buffer should be pushed into PhysX this step."""
        return self._step_counter % self.cfg.physx_writeback_every_k_steps == 0

    @property
    def h_plastic(self) -> torch.Tensor:
        """Current plastic deformation buffer (signed m, negative = sunk)."""
        if self._h_plastic is None:
            raise RuntimeError("Buffer not initialised.")
        return self._h_plastic

    def friction_at(
        self, velocity_xy: torch.Tensor, cell_i: torch.Tensor, cell_j: torch.Tensor,
        env_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Effective friction coefficient at given contacts (anisotropic)."""
        if self._furrow_dir_deg is None:
            return torch.full(
                velocity_xy.shape[:-1],
                self.cfg.params.anisotropy.mu_across,
                device=velocity_xy.device,
            )
        from ext_dfh.kernels import degrees_to_unit_xy

        dir_deg = self._furrow_dir_deg[env_idx, cell_i, cell_j]
        dir_xy = degrees_to_unit_xy(dir_deg)
        return anisotropic_friction_coefficient(velocity_xy, dir_xy, self.cfg.params)
