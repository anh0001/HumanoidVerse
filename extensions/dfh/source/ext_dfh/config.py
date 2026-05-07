"""Frozen dataclass configs for the DFH terrain layer.

Parameter groupings track the physical model:
  * Bekker pressure-sinkage  -> :class:`BekkerParams`
  * Friction anisotropy      -> :class:`FrictionAnisotropy`
  * Slip-sinkage + bulldozing knobs and runtime tuning -> :class:`DFHParams`
  * Top-level wiring (grid + update cadence + device) -> :class:`DFHConfig`
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class BekkerParams:
    """Bekker pressure-sinkage law:  p = (kc/b + k_phi) * z^n .

    Defaults from Wong's loose dry sand. Override per terrain regime via Hydra.
    """

    kc: float = 1.4e3       # cohesive modulus of deformation [N/m^(n+1)]
    k_phi: float = 8.2e5    # frictional modulus of deformation [N/m^(n+2)]
    n: float = 1.0          # sinkage exponent [-]
    cohesion: float = 1.0e4  # soil cohesion c [Pa]
    friction_angle_deg: float = 30.0  # internal friction angle phi [deg]


@dataclass(frozen=True)
class FrictionAnisotropy:
    """Direction-dependent friction along/across furrows."""

    mu_along: float = 0.55       # parallel to furrow direction
    mu_across: float = 0.70      # perpendicular to furrow direction
    enabled: bool = True


@dataclass(frozen=True)
class DFHParams:
    """Runtime parameters for the deformable layer (per-regime).

    Keep all numeric tuning here so YAML overrides via Hydra remain ergonomic.
    """

    bekker: BekkerParams = field(default_factory=BekkerParams)
    anisotropy: FrictionAnisotropy = field(default_factory=FrictionAnisotropy)

    # Wong-Reece slip-sinkage coupling.  z_eff = z * (1 + alpha_s * |v_t|/(|v_n|+eps))
    slip_sinkage_alpha: float = 0.20
    slip_sinkage_eps: float = 1.0e-3

    # Bulldozing redistribution: fraction of displaced volume sent to neighbour cells.
    bulldoze_share: float = 0.4

    # Plastic floor: cells cannot deform deeper than this (m, negative). Stops trench carving.
    sinkage_floor_m: float = -0.10

    # Foot contact patch area used to convert F_normal -> pressure (m^2).
    # Hunter ankle box collider area ~= 0.1001 * 0.0345 = 3.45e-3 m^2.
    contact_patch_area_m2: float = 3.45e-3


@dataclass(frozen=True)
class DFHConfig:
    """Top-level configuration for :class:`DFHTerrainLayer`."""

    # Heightfield grid (matches IsaacSim TerrainGenerator horizontal_scale).
    horizontal_scale_m: float = 0.10
    vertical_scale_m: float = 0.002

    # Terrain extent (per-env).
    terrain_length_m: float = 12.0
    terrain_width_m: float = 12.0

    # How often the deformed heightfield is pushed back into PhysX (in physics steps).
    # K=1 = every step (max fidelity, slowest). K=4 = every 4 steps (good default).
    physx_writeback_every_k_steps: int = 4

    # Device for the H_plastic buffer.
    device: str = "cuda"

    # Per-regime tuning.
    params: DFHParams = field(default_factory=DFHParams)

    # Furrow direction stored per cell? If False, ext only does sinkage/bulldozing
    # (no anisotropy).  Auto-True when anisotropy.enabled is True.
    store_furrow_direction: bool = True

    @property
    def grid_h(self) -> int:
        return int(self.terrain_length_m / self.horizontal_scale_m)

    @property
    def grid_w(self) -> int:
        return int(self.terrain_width_m / self.horizontal_scale_m)
