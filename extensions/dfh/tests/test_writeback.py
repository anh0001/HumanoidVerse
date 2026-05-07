"""Tests for ext_dfh.writeback that do NOT require omni.usd / IsaacSim Kit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from ext_dfh.writeback import (
    BaselineMesh,
    PhysxHeightfieldWriteback,
    build_writeback,
    snapshot_baseline_from_trimesh,
)


# --------------------------------------------------------------------- mocks


@dataclass
class _MockTrimesh:
    vertices: np.ndarray
    faces: np.ndarray


def _make_grid_mesh(grid_h: int, grid_w: int, scale: float, z_const: float = 0.0) -> _MockTrimesh:
    """Build a row-major grid_h x grid_w vertex array (faces unused)."""
    xs = np.arange(grid_w) * scale
    ys = np.arange(grid_h) * scale
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    verts = np.stack([xv.ravel(), yv.ravel(), np.full(xv.size, z_const)], axis=1).astype(np.float32)
    faces = np.zeros((1, 3), dtype=np.int32)  # not used
    return _MockTrimesh(vertices=verts, faces=faces)


@dataclass
class _MockTerrainImporter:
    meshes: dict


# ------------------------------------------------------------------- tests


@pytest.mark.unit
def test_snapshot_baseline_recovers_grid_dims() -> None:
    mesh = _make_grid_mesh(grid_h=8, grid_w=12, scale=0.10)
    bl = snapshot_baseline_from_trimesh("/World/ground/flat", mesh, horizontal_scale_m=0.10)
    assert bl.grid_h == 8
    assert bl.grid_w == 12
    assert bl.num_vertices == 8 * 12
    assert bl.origin_xy_m == (0.0, 0.0)


@pytest.mark.unit
def test_snapshot_baseline_handles_decimated_mesh(caplog: pytest.LogCaptureFixture) -> None:
    """Real IsaacLab terrain dedupes/decimates vertices -- vertex_count <= grid_h*grid_w."""
    mesh = _make_grid_mesh(grid_h=8, grid_w=12, scale=0.10)
    # Drop half the vertices -> still must produce a valid per-vertex cell map.
    mesh.vertices = mesh.vertices[::2]
    with caplog.at_level(logging.INFO, logger="ext_dfh.writeback"):
        bl = snapshot_baseline_from_trimesh("/World/ground/flat", mesh, horizontal_scale_m=0.10)
    assert bl.num_vertices == 48  # 96 / 2
    assert bl.vertex_cell_i.shape == (48,)
    assert bl.vertex_cell_j.shape == (48,)
    assert (bl.vertex_cell_i >= 0).all() and (bl.vertex_cell_i < bl.grid_h).all()
    assert (bl.vertex_cell_j >= 0).all() and (bl.vertex_cell_j < bl.grid_w).all()


@pytest.mark.unit
def test_writeback_outside_kit_disables_after_first_call(caplog: pytest.LogCaptureFixture) -> None:
    mesh = _make_grid_mesh(grid_h=4, grid_w=4, scale=0.10)
    bl = snapshot_baseline_from_trimesh("/World/ground/flat", mesh, horizontal_scale_m=0.10)
    wb = PhysxHeightfieldWriteback(baseline=bl, dfh_grid_h=bl.grid_h, dfh_grid_w=bl.grid_w, device="cpu")

    h = torch.zeros(2, 4, 4)
    h[0, 1, 1] = -0.02
    with caplog.at_level(logging.WARNING, logger="ext_dfh.writeback"):
        wb(h)
    assert wb._mutation_failed is True
    assert any("USD points attribute unavailable" in r.message for r in caplog.records)

    # Subsequent calls are silent no-ops.
    caplog.clear()
    wb(h)
    assert len(caplog.records) == 0


@pytest.mark.unit
def test_writeback_aggregation_picks_min_across_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the cross-env reduction picks the most-deformed cell value."""
    mesh = _make_grid_mesh(grid_h=3, grid_w=3, scale=0.10)
    bl = snapshot_baseline_from_trimesh("/World/ground/flat", mesh, horizontal_scale_m=0.10)
    wb = PhysxHeightfieldWriteback(baseline=bl, dfh_grid_h=bl.grid_h, dfh_grid_w=bl.grid_w, device="cpu")

    captured: list[np.ndarray] = []

    class _FakeAttr:
        def Set(self, value: Any) -> None:
            captured.append(np.asarray(value))

    monkeypatch.setattr(wb, "_get_points_attr", lambda: _FakeAttr())
    monkeypatch.setattr(wb, "_mark_collision_dirty", lambda: None)
    # Stub Vt so we can run outside Kit.
    import sys
    import types
    pxr_mod = types.ModuleType("pxr")
    vt_mod = types.ModuleType("pxr.Vt")
    vt_mod.Vec3fArray = type("Vec3fArray", (), {"FromNumpy": staticmethod(lambda a: a)})
    gf_mod = types.ModuleType("pxr.Gf")
    pxr_mod.Vt = vt_mod
    pxr_mod.Gf = gf_mod
    sys.modules.setdefault("pxr", pxr_mod)
    sys.modules.setdefault("pxr.Vt", vt_mod)
    sys.modules.setdefault("pxr.Gf", gf_mod)

    h = torch.zeros(3, 3, 3)
    h[0, 1, 1] = -0.01
    h[1, 1, 1] = -0.05  # this env is the most-deformed
    h[2, 1, 1] = -0.02
    wb(h)

    assert len(captured) == 1
    new_xyz = captured[0]
    # Find the vertex whose precomputed cell map points at (1, 1).
    target = (bl.vertex_cell_i == 1) & (bl.vertex_cell_j == 1)
    assert target.any()
    z_at_target = new_xyz[target, 2]
    assert np.allclose(z_at_target, -0.05, atol=1e-6)


@pytest.mark.unit
def test_build_writeback_with_no_meshes_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    importer = _MockTerrainImporter(meshes={})
    with caplog.at_level(logging.WARNING, logger="ext_dfh.writeback"):
        wb = build_writeback(
            isaacsim_terrain=importer,
            terrain_prim_path="/World/ground",
            horizontal_scale_m=0.10,
            dfh_grid_h=4, dfh_grid_w=4,
            device="cpu",
        )
    assert wb is None
    assert any("no .meshes" in r.message for r in caplog.records)


@pytest.mark.unit
def test_build_writeback_returns_writer_when_mesh_present() -> None:
    mesh = _make_grid_mesh(grid_h=5, grid_w=5, scale=0.10)
    importer = _MockTerrainImporter(meshes={"flat": mesh})
    wb = build_writeback(
        isaacsim_terrain=importer,
        terrain_prim_path="/World/ground",
        horizontal_scale_m=0.10,
        dfh_grid_h=5, dfh_grid_w=5,
        device="cpu",
    )
    assert wb is not None
    assert wb.baseline.prim_path == "/World/ground/flat"
    assert wb.baseline.grid_h == 5 and wb.baseline.grid_w == 5
