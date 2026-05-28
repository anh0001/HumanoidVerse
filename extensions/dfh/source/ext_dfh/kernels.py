"""GPU kernels for the DFH terrain layer.

All kernels operate on the per-env plastic-deformation buffer
``H_plastic[num_envs, grid_h, grid_w]`` (signed metres, negative = sunk).

Kernels are pure functions: they take the buffer + contact data and return the
updated buffer. The :class:`DFHTerrainLayer` owns the buffer and decides when to
call each kernel.

NOTE: v0.1 reference implementations are torch-native (vectorised, no Warp).
A Warp port is planned for v0.2 once the algebra is locked.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ext_dfh.config import DFHParams


def bekker_sinkage_target(
    contact_pressure_pa: torch.Tensor,
    params: DFHParams,
    cell_width_m: float,
) -> torch.Tensor:
    """Bekker pressure-sinkage:  z = (p / k)^(1/n)  with k = kc/b + k_phi.

    Args:
        contact_pressure_pa: per-contact pressure [N/m^2], shape ``(...,)``.
        params: DFH parameters.
        cell_width_m: characteristic contact width b [m] (use cell width).

    Returns:
        Sinkage depth (positive metres) with same shape as input.
    """
    bek = params.bekker
    k_modulus = bek.kc / max(cell_width_m, 1e-6) + bek.k_phi
    # Guard against negative pressures from solver noise.
    p = torch.clamp(contact_pressure_pa, min=0.0)
    z = torch.pow(p / k_modulus, 1.0 / bek.n)
    return z


def apply_slip_sinkage(
    z_target: torch.Tensor,
    v_tangential: torch.Tensor,
    v_normal: torch.Tensor,
    params: DFHParams,
    slip_alpha: Optional[torch.Tensor] = None,
    slip_max: float = 1.0,
) -> torch.Tensor:
    """Wong-Reece slip-sinkage coupling, bounded.

    Original form ``z * (1 + alpha * |v_t|/(|v_n|+eps))`` blows up in stance when
    ``v_n -> 0``. We use a bounded slip ratio:
        slip = |v_t| / max(|v_t| + |v_n|, eps_floor)
        z_eff = z * (1 + alpha * clamp(slip, 0, slip_max))
    which is in ``[0, 1]`` and is well-behaved at zero normal velocity.

    ``slip_alpha`` may be a per-contact tensor (used by per-env DR); when ``None``
    the scalar ``params.slip_sinkage_alpha`` is broadcast.
    """
    eps_floor = max(float(params.slip_sinkage_eps), 1e-3)
    vt = torch.linalg.vector_norm(v_tangential, dim=-1)
    vn = torch.abs(v_normal)
    denom = torch.clamp(vt + vn, min=eps_floor)
    slip = torch.clamp(vt / denom, min=0.0, max=slip_max)
    alpha = slip_alpha if slip_alpha is not None else params.slip_sinkage_alpha
    return z_target * (1.0 + alpha * slip)


def update_plastic_height(
    h_plastic: torch.Tensor,
    env_idx: torch.Tensor,
    cell_i: torch.Tensor,
    cell_j: torch.Tensor,
    z_eff: torch.Tensor,
    params: DFHParams,
) -> torch.Tensor:
    """Plastic update: H[env, i, j] = min(H[env, i, j], -z_eff).

    Plastic = ratchet only. Sinking depth never recovers within an episode.
    Floor capped by ``params.sinkage_floor_m``.
    """
    new_depth = torch.clamp(-z_eff, min=params.sinkage_floor_m)
    current = h_plastic[env_idx, cell_i, cell_j]
    h_plastic[env_idx, cell_i, cell_j] = torch.minimum(current, new_depth)
    return h_plastic


def bulldoze_neighbours(
    h_plastic: torch.Tensor,
    env_idx: torch.Tensor,
    cell_i: torch.Tensor,
    cell_j: torch.Tensor,
    delta_z: torch.Tensor,
    params: DFHParams,
) -> torch.Tensor:
    """Conserve soil volume by lifting 4-neighbour cells when centre cell sinks.

    delta_z is the *additional* sink in metres (positive). Each of 4 neighbours
    receives ``share * delta_z / 4`` upward push.
    """
    share = params.bulldoze_share
    push = (share * delta_z / 4.0).clamp(min=0.0)
    grid_h, grid_w = h_plastic.shape[-2], h_plastic.shape[-1]
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni = (cell_i + di).clamp(0, grid_h - 1)
        nj = (cell_j + dj).clamp(0, grid_w - 1)
        # Lift = move toward 0 from below.
        current = h_plastic[env_idx, ni, nj]
        h_plastic[env_idx, ni, nj] = torch.minimum(current + push, torch.zeros_like(current))
    return h_plastic


def anisotropic_friction_coefficient(
    velocity_xy: torch.Tensor,
    furrow_dir_xy: torch.Tensor,
    params: DFHParams,
) -> torch.Tensor:
    """Direction-dependent friction:  mu = mu_par * cos^2(a) + mu_perp * sin^2(a).

    Args:
        velocity_xy: tangential velocity vector at contact, shape (..., 2).
        furrow_dir_xy: unit furrow direction at the cell, shape (..., 2).

    Returns:
        Effective friction coefficient with shape ``velocity_xy.shape[:-1]``.
    """
    aniso = params.anisotropy
    if not aniso.enabled:
        return torch.full(velocity_xy.shape[:-1], aniso.mu_across, device=velocity_xy.device)

    eps = 1e-6
    v_norm = torch.linalg.vector_norm(velocity_xy, dim=-1, keepdim=True).clamp_min(eps)
    v_hat = velocity_xy / v_norm
    cos_a = (v_hat * furrow_dir_xy).sum(dim=-1).clamp(-1.0, 1.0)
    cos2 = cos_a * cos_a
    sin2 = 1.0 - cos2
    return aniso.mu_along * cos2 + aniso.mu_across * sin2


def world_xy_to_cell(
    pos_xy_m: torch.Tensor,
    grid_h: int,
    grid_w: int,
    horizontal_scale_m: float,
    env_origin_xy: Optional[torch.Tensor] = None,
    extent_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert world (x, y) metres to integer cell index in a per-env grid.

    The DFH grid is treated as a square tile of physical extent
    ``(extent_x, extent_y) = (grid_h * scale, grid_w * scale)`` centered on the
    env origin. ``pos_xy`` is shifted by the env origin and the half-extent so
    that the centre of the tile lands on cell ``(grid_h//2, grid_w//2)``.

    Args:
        pos_xy_m: (..., 2) world positions.
        grid_h, grid_w: grid resolution.
        horizontal_scale_m: cell width (m).
        env_origin_xy: (..., 2) world origin for each contact's env. When None,
            the legacy origin-less mapping is used (back-compat for tests).
        extent_xy: physical (length, width) of the tile (m). Defaults to
            ``(grid_h * scale, grid_w * scale)``.

    Returns:
        Tuple of long tensors ``(cell_i, cell_j)`` clamped to ``[0, grid-1]``.
    """
    if extent_xy is None:
        extent_xy = (grid_h * horizontal_scale_m, grid_w * horizontal_scale_m)
    if env_origin_xy is None:
        local_x = pos_xy_m[..., 0]
        local_y = pos_xy_m[..., 1]
    else:
        local_x = pos_xy_m[..., 0] - env_origin_xy[..., 0] + 0.5 * float(extent_xy[0])
        local_y = pos_xy_m[..., 1] - env_origin_xy[..., 1] + 0.5 * float(extent_xy[1])
    cell_i = torch.clamp(torch.floor(local_x / horizontal_scale_m).long(), 0, grid_h - 1)
    cell_j = torch.clamp(torch.floor(local_y / horizontal_scale_m).long(), 0, grid_w - 1)
    return cell_i, cell_j


def degrees_to_unit_xy(angle_deg: torch.Tensor) -> torch.Tensor:
    """Convert angle in degrees to a unit (x, y) direction vector."""
    angle_rad = angle_deg * (math.pi / 180.0)
    return torch.stack([torch.cos(angle_rad), torch.sin(angle_rad)], dim=-1)
