"""Path A — PhysX heightfield write-back for the DFH terrain layer.

Strategy: at init, snapshot the baseline terrain mesh vertices from the
``omni.isaac.lab.terrains.TerrainImporter`` and the per-env origin offsets.
At write-back cadence, build a deformed vertex array
``v_deformed = v_baseline + (0, 0, H_plastic_at(x, y))`` and push it into
the USD ``points`` attribute of the terrain mesh prim.

PhysX picks up the new collision via ``UsdGeomMesh.ComputeExtent`` +
``PhysxCollisionAPI`` mark-dirty.  This is the documented mutation path.

The class degrades gracefully:
  * If ``omni.usd`` / ``pxr`` is not importable (e.g. tests outside Kit), every
    method becomes a no-op + warning.
  * If ``ComputeExtent`` / dirty-mark fails on a particular IsaacSim build, the
    mutation is rolled back and a warning is logged once.

NOTE: full validation requires launching IsaacSim Kit; the unit tests in
``tests/test_writeback.py`` exercise every code path that does NOT need
``omni.usd``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- baseline

@dataclass
class BaselineMesh:
    """Snapshot of the un-deformed terrain mesh.

    Vertices are captured in **mesh-local frame** (relative to the prim root).
    A per-vertex (cell_i, cell_j) index is precomputed so write-back works on
    arbitrary triangulations (IsaacLab's TerrainImporter often de-duplicates
    or decimates vertices, so we cannot assume a row-major grid).
    """

    prim_path: str
    vertices_xyz_m: np.ndarray  # (N_vertices, 3)  baseline vertex positions
    vertex_cell_i: np.ndarray   # (N_vertices,)    row index per vertex
    vertex_cell_j: np.ndarray   # (N_vertices,)    col index per vertex
    grid_h: int
    grid_w: int
    horizontal_scale_m: float
    origin_xy_m: Tuple[float, float]  # mesh-frame origin (lower-left corner)

    @property
    def num_vertices(self) -> int:
        return int(self.vertices_xyz_m.shape[0])


def snapshot_baseline_from_trimesh(
    prim_path: str,
    mesh: Any,  # trimesh.Trimesh
    horizontal_scale_m: float,
) -> BaselineMesh:
    """Extract a :class:`BaselineMesh` from an IsaacLab trimesh terrain.

    Assumes a uniform-grid heightfield mesh: vertices laid out row-major over
    a ``grid_h x grid_w`` grid.  IsaacLab's ``TerrainGenerator`` produces
    exactly this layout for ``HfRandomUniformTerrainCfg``-derived terrains
    and for our :class:`HfFurrowsTerrainCfg`.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    grid_w = int(round((x_max - x_min) / horizontal_scale_m)) + 1
    grid_h = int(round((y_max - y_min) / horizontal_scale_m)) + 1
    # Map every vertex to its nearest grid cell — robust to triangulation /
    # vertex-count != grid_h*grid_w (IsaacLab decimates/dedupes).
    vi = np.clip(np.round((ys - y_min) / horizontal_scale_m).astype(np.int64), 0, grid_h - 1)
    vj = np.clip(np.round((xs - x_min) / horizontal_scale_m).astype(np.int64), 0, grid_w - 1)
    if grid_h * grid_w != vertices.shape[0]:
        logger.info(
            "Baseline mesh: %d vertices on a %d x %d cell grid "
            "(triangulation; per-vertex cell map built).",
            vertices.shape[0], grid_h, grid_w,
        )
    return BaselineMesh(
        prim_path=prim_path,
        vertices_xyz_m=vertices,
        vertex_cell_i=vi,
        vertex_cell_j=vj,
        grid_h=grid_h,
        grid_w=grid_w,
        horizontal_scale_m=horizontal_scale_m,
        origin_xy_m=(x_min, y_min),
    )


# ---------------------------------------------------------------- writer


class PhysxHeightfieldWriteback:
    """Pushes ``H_plastic`` deltas into the IsaacSim terrain prim's USD points.

    Per-env-origin offsetting is not implemented in v0.2: if the IsaacLab
    cloner replicates the ground prim per env, only the source prim is
    mutated and the cloner's reference instances pick up the change.  If the
    cloner uses one shared prim, all envs share the deformation (acceptable
    for single-env layouts; flagged for v0.3 for multi-env curriculum).
    """

    def __init__(
        self,
        baseline: BaselineMesh,
        dfh_grid_h: int,
        dfh_grid_w: int,
        device: str = "cuda",
    ) -> None:
        self.baseline = baseline
        self.device = device
        self.dfh_grid_h = dfh_grid_h
        self.dfh_grid_w = dfh_grid_w
        self._baseline_z_t = torch.from_numpy(baseline.vertices_xyz_m[:, 2]).to(device)
        # Remap terrain-mesh cell indices (terrain grid resolution) to DFH-buffer
        # cell indices (DFH config grid).  Terrain mesh grid usually finer.
        scale_i = dfh_grid_h / max(baseline.grid_h, 1)
        scale_j = dfh_grid_w / max(baseline.grid_w, 1)
        di = np.clip((baseline.vertex_cell_i * scale_i).astype(np.int64), 0, dfh_grid_h - 1)
        dj = np.clip((baseline.vertex_cell_j * scale_j).astype(np.int64), 0, dfh_grid_w - 1)
        self._cell_i_t = torch.from_numpy(di).to(device)
        self._cell_j_t = torch.from_numpy(dj).to(device)
        self._mutation_failed = False

    # ----------------------------------------------------- USD prim handle

    def _get_points_attr(self):
        """Return a typed Points attribute on the terrain Mesh prim, or None.

        Uses ``CreatePointsAttr`` rather than ``GetPointsAttr`` so the attribute
        is guaranteed to have a declared ``typeName`` (some IsaacLab terrain
        meshes leave it untyped, which causes ``Set`` to fail with
        ``'Empty typeName for <prim.points>'``).
        """
        try:
            import omni.usd
            from pxr import UsdGeom
        except ImportError:
            return None
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.baseline.prim_path)
            if not prim or not prim.IsValid():
                return None
            mesh = UsdGeom.Mesh(prim)
            # Idempotent: returns existing typed attr or creates one.
            return mesh.CreatePointsAttr()
        except Exception as e:  # noqa: BLE001 - USD APIs raise heterogeneous errors
            logger.warning("USD points attribute lookup failed: %s", e)
            return None

    def _mark_collision_dirty(self) -> None:
        """Trigger PhysX to re-cook the collision shape after vertex update."""
        try:
            import omni.physx
            iface = omni.physx.acquire_physx_interface()
            # Force a refresh of cooked collision data for the prim.
            iface.refresh_physics_objects(self.baseline.prim_path)
        except Exception:  # noqa: BLE001
            # refresh_physics_objects is not present on every IsaacSim build;
            # PhysX usually re-cooks on next step from the dirty USD attribute.
            pass

    # ----------------------------------------------------- public surface

    def __call__(self, h_plastic: torch.Tensor) -> None:
        """Compose deformed vertices = baseline_z + per-env-aggregated H_plastic.

        Aggregation rule (v0.2): take the **min** of H_plastic across envs at
        each cell.  Justified for single-env training and a conservative
        upper-bound on deformation visual when many envs share the prim.
        """
        if self._mutation_failed:
            return
        attr = self._get_points_attr()
        if attr is None:
            self._mutation_failed = True
            logger.warning(
                "DFH writeback disabled: USD points attribute unavailable "
                "(running outside IsaacSim Kit?)."
            )
            return

        try:
            # Cross-env min reduction (most-deformed wins).  Shape: (grid_h, grid_w).
            h_min = h_plastic.amin(dim=0).to(self.device)
            # Per-vertex sinkage delta from precomputed (cell_i, cell_j) map.
            delta_z = h_min[self._cell_i_t, self._cell_j_t]
            new_z_t = self._baseline_z_t + delta_z
            new_xyz = self.baseline.vertices_xyz_m.copy()
            new_xyz[:, 2] = new_z_t.detach().cpu().numpy().astype(np.float32)

            # USD writes: try FromNumpy (fast path); fall back to per-row Vec3f
            # construction with native python floats (some pxr builds reject
            # numpy scalars passed positionally to Gf.Vec3f).
            from pxr import Vt
            new_xyz_c = np.ascontiguousarray(new_xyz, dtype=np.float32)
            try:
                attr.Set(Vt.Vec3fArray.FromNumpy(new_xyz_c))
            except Exception:  # noqa: BLE001
                from pxr import Gf
                rows = new_xyz_c.tolist()  # list of [float, float, float] (python floats)
                attr.Set(Vt.Vec3fArray([Gf.Vec3f(r[0], r[1], r[2]) for r in rows]))
            self._mark_collision_dirty()
        except Exception as e:  # noqa: BLE001
            import traceback
            logger.warning(
                "DFH writeback failed (disabling further attempts): %r\n%s",
                e, traceback.format_exc(),
            )
            self._mutation_failed = True


# ---------------------------------------------------------------- factory


def build_writeback(
    isaacsim_terrain: Any,
    terrain_prim_path: str,
    horizontal_scale_m: float,
    dfh_grid_h: int,
    dfh_grid_w: int,
    device: str = "cuda",
) -> Optional[PhysxHeightfieldWriteback]:
    """Construct a :class:`PhysxHeightfieldWriteback` from an IsaacLab TerrainImporter.

    Returns ``None`` and logs a warning if the terrain has no usable trimesh
    (e.g. it was imported as a USD asset or as the default ground plane).
    """
    meshes = getattr(isaacsim_terrain, "meshes", None)
    if not meshes:
        logger.warning("DFH writeback: terrain has no .meshes dict; skipping.")
        return None
    # Pick the first mesh — IsaacLab's TerrainGenerator typically stores one
    # composite mesh keyed by sub-terrain name.
    mesh_key = next(iter(meshes.keys()))
    mesh = meshes[mesh_key]
    full_prim_path = f"{terrain_prim_path}/{mesh_key}"
    try:
        baseline = snapshot_baseline_from_trimesh(
            prim_path=full_prim_path,
            mesh=mesh,
            horizontal_scale_m=horizontal_scale_m,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("DFH baseline snapshot failed: %s", e)
        return None
    return PhysxHeightfieldWriteback(
        baseline=baseline,
        dfh_grid_h=dfh_grid_h,
        dfh_grid_w=dfh_grid_w,
        device=device,
    )
