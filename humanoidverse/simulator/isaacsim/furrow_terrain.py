from __future__ import annotations

import numpy as np

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from omni.isaac.lab.terrains.height_field.utils import height_field_to_mesh


@height_field_to_mesh
def furrows_height_field(difficulty: float, cfg: "HfFurrowsTerrainCfg") -> np.ndarray:
    """Height-field generator for agricultural furrows.

    Produces parallel grooves with configurable depth, spacing, and orientation.
    Depth is applied downward (negative heights) so crests are near 0 and troughs reach ``-depth``.
    """
    # Resolve grid in pixels
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)

    # Coordinates in meters (centered at 0..size)
    x = np.arange(width_px, dtype=np.float64) * cfg.horizontal_scale
    y = np.arange(length_px, dtype=np.float64) * cfg.horizontal_scale
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Sample parameters within ranges (use difficulty to bias within range)
    # If the user runs curriculum, difficulty spans [0,1); otherwise it's uniform from generator.
    def lerp_range(rng: tuple[float, float]) -> float:
        lo, hi = float(rng[0]), float(rng[1])
        return lo + float(difficulty) * (hi - lo)

    depth_m = lerp_range(cfg.depth_range)
    spacing_m = lerp_range(cfg.spacing_range)
    # small random orientation within the allowed range (degrees)
    theta_deg = np.random.uniform(cfg.orientation_range_deg[0], cfg.orientation_range_deg[1])
    theta = np.deg2rad(theta_deg)

    # Rotate coordinates so furrows run along rotated x'
    xr = np.cos(theta) * xx + np.sin(theta) * yy

    # Triangular waveform in [-1, 1] along x'
    # tri(z) = 2*abs(frac(z) - 0.5) - 1
    # Use period = spacing_m
    phase = xr / max(1e-6, spacing_m)
    tri = 2.0 * np.abs(phase - np.floor(phase + 0.5)) - 1.0

    # Heights in meters: troughs at -depth_m, crests near 0, plus optional offset
    h_m = -0.5 * depth_m * (1.0 + tri) + float(getattr(cfg, "crest_offset_m", 0.0))

    # Convert to discrete height-field units (int16 steps of vertical_scale)
    hf_raw = np.rint(h_m / cfg.vertical_scale).astype(np.int16)
    return hf_raw


@configclass
class HfFurrowsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for agricultural furrows as a height-field terrain."""

    function = furrows_height_field

    # Depth of grooves (meters)
    depth_range: tuple[float, float] = (0.05, 0.15)
    # Spacing between adjacent furrow centers (meters)
    spacing_range: tuple[float, float] = (0.8, 1.2)
    # Orientation jitter in degrees around x-axis alignment
    orientation_range_deg: tuple[float, float] = (-10.0, 10.0)
    # Optional upward offset to lift ridges above the nominal plane (meters)
    crest_offset_m: float = 0.0
