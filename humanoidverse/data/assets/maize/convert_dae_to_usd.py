#!/usr/bin/env python3
"""Convert maize_01.dae and maize_02.dae to .usd.

`omni.kit.asset_converter` supports OBJ/FBX/STL/GLTF but NOT COLLADA (DAE), so
we bypass it and:

1. Load the DAE mesh with trimesh (available in the IsaacSim Python env).
2. Author a fresh USD stage with the mesh as a `UsdGeom.Mesh` prim.
3. Apply the maize_*.png texture as a `UsdShade.Material` if PIL can open it.

USD's stage reference loader handles the resulting .usd files natively.

Usage:
    $ISAAC_PATH/python.sh humanoidverse/data/assets/maize/convert_dae_to_usd.py
or with isaaclab conda env (whichever has trimesh + pxr installed):
    /home/anhar/miniconda3/envs/isaaclab/bin/python convert_dae_to_usd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

from pxr import Sdf, Usd, UsdGeom, UsdShade

ASSETS_DIR = Path(__file__).resolve().parent
MESHES_DIR = ASSETS_DIR / "meshes"
TEXTURES_DIR = ASSETS_DIR / "materials" / "textures"

JOBS = [
    ("maize_01", MESHES_DIR / "maize_01.dae", ASSETS_DIR / "maize_01.usd",
     TEXTURES_DIR / "maize_01.png"),
    ("maize_02", MESHES_DIR / "maize_02.dae", ASSETS_DIR / "maize_02.usd",
     TEXTURES_DIR / "maize_02.png"),
]


def _write_usd(name: str, mesh: trimesh.Trimesh, out: Path, texture: Path | None) -> None:
    """Author a textured USD stage with the given mesh."""
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    # World units = meters; matches the rest of the HumanoidVerse pipeline.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    mesh_prim = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    mesh_prim.CreatePointsAttr().Set([tuple(v) for v in mesh.vertices.astype(float)])
    mesh_prim.CreateFaceVertexIndicesAttr().Set(mesh.faces.astype(int).flatten().tolist())
    mesh_prim.CreateFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    # Disable subdivision so vertices stay where we put them.
    mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # UVs (st), if the trimesh visuals expose them.
    visual = getattr(mesh, "visual", None)
    if visual is not None and hasattr(visual, "uv") and visual.uv is not None:
        uvs = np.asarray(visual.uv, dtype=float)
        if uvs.shape[0] == len(mesh.vertices):
            primvars = UsdGeom.PrimvarsAPI(mesh_prim.GetPrim())
            st = primvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            # Map vertex-UVs to face-varying.
            face_uvs = uvs[mesh.faces.flatten()]
            st.Set([tuple(uv) for uv in face_uvs])

    if texture is not None and texture.exists():
        mat = UsdShade.Material.Define(stage, f"/{name}/material")
        pbr = UsdShade.Shader.Define(stage, f"/{name}/material/pbr")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        tex_reader = UsdShade.Shader.Define(stage, f"/{name}/material/diffuse_tex")
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_reader.ConnectableAPI(), "rgb"
        )
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)

    stage.GetRootLayer().Save()


def main() -> int:
    if not MESHES_DIR.exists():
        print(f"ERROR: {MESHES_DIR} missing", file=sys.stderr)
        return 1

    for name, src, dst, tex in JOBS:
        if not src.exists():
            print(f"SKIP {name}: {src} missing")
            continue
        print(f"converting {name}: {src.name} -> {dst.name}", flush=True)
        m = trimesh.load(src, force="mesh")
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            print(f"  WARN: loaded mesh has no faces", file=sys.stderr)
            continue
        print(f"  vertices={len(m.vertices)} faces={len(m.faces)} texture={'ok' if tex.exists() else 'missing'}",
              flush=True)
        _write_usd(name, m, dst, tex if tex.exists() else None)
        print(f"  wrote {dst} ({dst.stat().st_size} bytes)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
