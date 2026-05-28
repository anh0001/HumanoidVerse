"""DFHTerrainLayer: public API for the deformable furrowed heightfield.

The layer owns the plastic-deformation buffer and exposes three lifecycle hooks
to be called from the IsaacSim wrapper:

  * :meth:`initialize` — once, after terrain mesh exists.
  * :meth:`step` — every physics step, given foot contact info.
  * :meth:`reset` — on env reset, restore buffer to zero for the given envs.

Heightfield write-back into PhysX is delegated to :mod:`ext_dfh.integration`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ext_dfh.config import DFHConfig
from ext_dfh.kernels import (
    anisotropic_friction_coefficient,
    apply_slip_sinkage,
    bekker_sinkage_target,
    bulldoze_neighbours,
    degrees_to_unit_xy,
    update_plastic_height,
    world_xy_to_cell,
)


@dataclass
class FootContact:
    """Per-foot contact snapshot from IsaacSim contact reporter."""

    env_idx: torch.Tensor       # (N,) long, env index per active contact
    pos_xy_m: torch.Tensor      # (N, 2) world XY
    normal_force_n: torch.Tensor  # (N,) Newtons, positive = pressing down
    v_tangential_xy: torch.Tensor  # (N, 2) m/s
    v_normal_z: torch.Tensor    # (N,) m/s
    # Foot index within the env's foot list (0..n_feet-1). When omitted (e.g.
    # legacy callers, unit tests) the value is left as zeros — force coupling
    # in the IsaacSim adapter will route every contact to foot 0 in that case.
    foot_local_idx: torch.Tensor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.foot_local_idx is None:
            self.foot_local_idx = torch.zeros_like(self.env_idx)


class DFHTerrainLayer:
    """Plastic heightfield + Bekker sink + slip-sinkage + bulldozing + anisotropy."""

    # Names of per-env scalar parameters DR may sample. See
    # ``randomize_envs`` for the wiring; values default to the scalar
    # in ``self.cfg.params`` when the per-env tensor is absent.
    _PER_ENV_PARAM_KEYS = (
        "bekker_kc",
        "bekker_k_phi",
        "bekker_n",
        "mu_along",
        "mu_across",
        "slip_sinkage_alpha",
        "bulldoze_share",
    )

    def __init__(self, num_envs: int, config: DFHConfig) -> None:
        self.num_envs = num_envs
        self.cfg = config
        self.device = torch.device(config.device)
        self._h_plastic: Optional[torch.Tensor] = None
        self._furrow_dir_deg: Optional[torch.Tensor] = None
        self._step_counter = 0
        # Per-env DR parameter tensors, populated lazily on first
        # ``randomize_envs`` call. When a key is missing, kernels fall
        # back to the scalar in ``self.cfg.params``.
        self._per_env: Dict[str, torch.Tensor] = {}
        self._extent_xy = (
            float(self.cfg.grid_h * self.cfg.horizontal_scale_m),
            float(self.cfg.grid_w * self.cfg.horizontal_scale_m),
        )

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

    def step(
        self,
        contacts: FootContact,
        env_origins_xy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Update the plastic buffer from this step's foot contacts.

        ``env_origins_xy`` (shape ``(num_envs, 2)``) is the world-frame XY of
        each env. When provided, contact positions are mapped to per-env grid
        cells centered on that origin. Without it, the legacy origin-less
        mapping is used (back-compat for unit tests).

        Returns a dict of per-contact tensors useful for force coupling:
            ``cell_i``, ``cell_j``, ``z_eff`` (sinkage applied),
            ``depth_after`` (plastic depth at the contact cell after update),
            ``mu_eff`` (anisotropic friction coefficient at the contact cell).
        """
        if self._h_plastic is None:
            raise RuntimeError("DFHTerrainLayer.initialize() must be called first.")

        params = self.cfg.params
        env_idx = contacts.env_idx

        # Per-env parameters (fall back to scalar config defaults if not set).
        kc = self._gather_per_env("bekker_kc", env_idx, params.bekker.kc)
        k_phi = self._gather_per_env("bekker_k_phi", env_idx, params.bekker.k_phi)
        n_exp = self._gather_per_env("bekker_n", env_idx, params.bekker.n)
        slip_alpha = self._gather_per_env("slip_sinkage_alpha", env_idx, params.slip_sinkage_alpha)
        bulldoze_share = self._gather_per_env("bulldoze_share", env_idx, params.bulldoze_share)

        # Bekker pressure-sinkage with per-env (kc, k_phi, n).
        pressure = contacts.normal_force_n / params.contact_patch_area_m2
        b = max(self.cfg.horizontal_scale_m, 1e-6)
        k_modulus = kc / b + k_phi
        p_clip = torch.clamp(pressure, min=0.0)
        z_target = torch.pow(p_clip / k_modulus, 1.0 / n_exp)

        z_eff = apply_slip_sinkage(
            z_target,
            contacts.v_tangential_xy,
            contacts.v_normal_z,
            params,
            slip_alpha=slip_alpha,
        )

        env_origin_for_contact: Optional[torch.Tensor] = None
        if env_origins_xy is not None:
            env_origin_for_contact = env_origins_xy[env_idx]
        cell_i, cell_j = world_xy_to_cell(
            contacts.pos_xy_m,
            self.cfg.grid_h,
            self.cfg.grid_w,
            self.cfg.horizontal_scale_m,
            env_origin_xy=env_origin_for_contact,
            extent_xy=self._extent_xy,
        )

        # Capture pre-update depth so we can compute delta for bulldozing.
        depth_before = self._h_plastic[env_idx, cell_i, cell_j]
        self._h_plastic = update_plastic_height(
            self._h_plastic, env_idx, cell_i, cell_j, z_eff, params
        )
        depth_after = self._h_plastic[env_idx, cell_i, cell_j]
        delta_z = (depth_before - depth_after).clamp(min=0.0)
        # Bulldoze with per-env share — re-implemented inline so per-env tensors
        # work cleanly without changing the kernel signature.
        share = bulldoze_share
        push = (share * delta_z / 4.0).clamp(min=0.0)
        gh, gw = self._h_plastic.shape[-2], self._h_plastic.shape[-1]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = (cell_i + di).clamp(0, gh - 1)
            nj = (cell_j + dj).clamp(0, gw - 1)
            current = self._h_plastic[env_idx, ni, nj]
            self._h_plastic[env_idx, ni, nj] = torch.minimum(
                current + push, torch.zeros_like(current)
            )

        # Anisotropic friction coefficient at each contact (used by the
        # IsaacSim adapter to apply a per-foot tangential drag force).
        mu_eff = self._friction_at(
            contacts.v_tangential_xy, env_idx, cell_i, cell_j
        )

        self._step_counter += 1
        return {
            "cell_i": cell_i,
            "cell_j": cell_j,
            "z_eff": z_eff,
            "depth_after": depth_after,
            "mu_eff": mu_eff,
        }

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
        """Public wrapper around the internal anisotropic friction lookup."""
        return self._friction_at(velocity_xy, env_idx, cell_i, cell_j)

    def _friction_at(
        self,
        velocity_xy: torch.Tensor,
        env_idx: torch.Tensor,
        cell_i: torch.Tensor,
        cell_j: torch.Tensor,
    ) -> torch.Tensor:
        """Per-contact anisotropic μ using per-env (μ_along, μ_across) and
        per-cell furrow direction (defaults to 0° when not initialised).
        """
        params = self.cfg.params
        mu_along = self._gather_per_env(
            "mu_along", env_idx, params.anisotropy.mu_along
        )
        mu_across = self._gather_per_env(
            "mu_across", env_idx, params.anisotropy.mu_across
        )
        if not params.anisotropy.enabled:
            return mu_across
        if self._furrow_dir_deg is None:
            dir_deg = torch.zeros(
                velocity_xy.shape[0], device=velocity_xy.device
            )
        else:
            dir_deg = self._furrow_dir_deg[env_idx, cell_i, cell_j]
        dir_xy = degrees_to_unit_xy(dir_deg)
        eps = 1e-6
        v_norm = torch.linalg.vector_norm(
            velocity_xy, dim=-1, keepdim=True
        ).clamp_min(eps)
        v_hat = velocity_xy / v_norm
        cos_a = (v_hat * dir_xy).sum(dim=-1).clamp(-1.0, 1.0)
        cos2 = cos_a * cos_a
        sin2 = 1.0 - cos2
        return mu_along * cos2 + mu_across * sin2

    # ----------------------------------------------------------------- DR

    def randomize_envs(
        self,
        env_ids: torch.Tensor,
        ranges: Dict[str, "tuple"],
    ) -> None:
        """Sample per-env DFH parameters uniformly inside ``ranges``.

        ``ranges`` maps DR keys (``bekker_kc``, ``bekker_k_phi``, ``bekker_n``,
        ``mu_along``, ``mu_across``, ``slip_sinkage_alpha``,
        ``bulldoze_share``, ``cohesion_pa``, ``friction_angle_deg``) to a
        ``[lo, hi]`` pair. Unknown keys are ignored. Must be called after
        :meth:`initialize`.
        """
        if self._h_plastic is None:
            return
        if env_ids is None or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(self.device).long()
        for key in self._PER_ENV_PARAM_KEYS:
            if key not in ranges:
                continue
            lo, hi = float(ranges[key][0]), float(ranges[key][1])
            tensor = self._per_env.get(key)
            if tensor is None:
                tensor = torch.full(
                    (self.num_envs,),
                    0.5 * (lo + hi),
                    device=self.device,
                    dtype=torch.float32,
                )
                self._per_env[key] = tensor
            tensor[env_ids] = (
                lo + (hi - lo) * torch.rand(env_ids.numel(), device=self.device)
            )

    def _gather_per_env(
        self,
        key: str,
        env_idx: torch.Tensor,
        scalar_default: float,
    ) -> torch.Tensor:
        tensor = self._per_env.get(key)
        if tensor is None:
            return torch.full(
                env_idx.shape,
                float(scalar_default),
                device=env_idx.device,
                dtype=torch.float32,
            )
        return tensor[env_idx.to(tensor.device)].to(env_idx.device)
