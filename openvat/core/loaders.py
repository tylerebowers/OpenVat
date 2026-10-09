"""Load 3D files into MeshObjects.

Mesh formats (STL, OBJ, PLY, 3MF, OFF, GLTF/GLB...) go through trimesh.
STEP / IGES need a CAD kernel; we use CadQuery / OCP when it is installed
(``pip install openvat[step]``) and raise a friendly error otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .mesh import MeshObject

MESH_EXTENSIONS = {".stl", ".obj", ".ply", ".3mf", ".off", ".glb", ".gltf", ".dae"}
CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs"}
ALL_EXTENSIONS = sorted(MESH_EXTENSIONS | CAD_EXTENSIONS)

FILE_FILTER = ("3D models (" + " ".join(f"*{e}" for e in ALL_EXTENSIONS) + ");;"
               "Mesh files (" + " ".join(f"*{e}" for e in sorted(MESH_EXTENSIONS)) + ");;"
               "CAD files (" + " ".join(f"*{e}" for e in sorted(CAD_EXTENSIONS)) + ");;"
               "All files (*)")


def load_file(path: str | Path) -> list[MeshObject]:
    """Load a file and return one MeshObject per solid body found."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in CAD_EXTENSIONS:
        meshes = _load_cad(path)
    else:
        meshes = _load_mesh(path)

    objects = []
    for i, m in enumerate(meshes):
        if len(m.faces) == 0:
            continue
        name = path.stem if len(meshes) == 1 else f"{path.stem}_{i + 1}"
        obj = MeshObject(name=name, mesh=m)
        obj.center_on_plate()
        obj.drop_to_plate()
        objects.append(obj)
    if not objects:
        raise ValueError(f"No geometry found in {path.name}")
    return objects


def _load_mesh(path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(str(path), force="scene")
    meshes = []
    for geom in loaded.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            meshes.append(geom)
    # Scenes may carry per-node transforms; bake them in.
    if isinstance(loaded, trimesh.Scene):
        baked = []
        for node in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node]
            g = loaded.geometry[geom_name]
            if isinstance(g, trimesh.Trimesh):
                g = g.copy()
                g.apply_transform(transform)
                baked.append(g)
        if baked:
            meshes = baked
    return meshes


def _load_cad(path: Path) -> list[trimesh.Trimesh]:
    try:
        import cadquery as cq  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "STEP/IGES import needs CadQuery.  Install it with\n"
            "    pip install cadquery\n"
            "(or `pip install openvat[step]`)."
        ) from exc

    ext = path.suffix.lower()
    if ext in (".step", ".stp"):
        shape = cq.importers.importStep(str(path))
    else:
        shape = cq.importers.importShape(cq.importers.ImportTypes.STEP, str(path))

    meshes = []
    for solid in shape.solids().vals():
        verts, tris = solid.tessellate(tolerance=0.02, angularTolerance=0.2)
        v = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
        f = np.array(tris, dtype=np.int64)
        meshes.append(trimesh.Trimesh(vertices=v, faces=f, process=True))
    return meshes
