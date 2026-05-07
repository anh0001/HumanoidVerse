"""Deformable Furrowed Heightfield (DFH) terrain layer for IsaacSim."""

from ext_dfh.config import BekkerParams, DFHConfig, DFHParams, FrictionAnisotropy
from ext_dfh.dfh_layer import DFHTerrainLayer
from ext_dfh.writeback import (
    BaselineMesh,
    PhysxHeightfieldWriteback,
    build_writeback,
    snapshot_baseline_from_trimesh,
)

__all__ = [
    "BaselineMesh",
    "BekkerParams",
    "DFHConfig",
    "DFHParams",
    "DFHTerrainLayer",
    "FrictionAnisotropy",
    "PhysxHeightfieldWriteback",
    "build_writeback",
    "snapshot_baseline_from_trimesh",
]

__version__ = "0.2.0"
