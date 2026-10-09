"""MeshObject: one printable part in the scene.

A MeshObject keeps the *source* mesh untouched and a 4x4 transform that
places it on the build plate.  Rotation is stored as Euler angles (degrees)
and scale as three factors so the UI can show and edit them directly.

Coordinate system: X/Y across the build plate (origin at the plate center),
Z up.  Units are millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np
import trimesh

from .hollow import Hole, HollowState

_ids = count(1)


@dataclass
class MeshObject:
    name: str
    mesh: trimesh.Trimesh                     # source geometry, never mutated by transforms
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # degrees, X Y Z
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    visible: bool = True
    elevation: float = 0.0                    # Z lift: height of the lowest point when dropped
    holes: tuple = ()                         # drain holes (Hole), in source mesh coordinates
    hollow: HollowState | None = None         # set by Hollow (see core/hollow.py); None = solid
    id: int = field(default_factory=lambda: next(_ids))

    # ---------------------------------------------------------------- transforms
    def matrix(self) -> np.ndarray:
        """4x4 model matrix = T * Rz * Ry * Rx * S, applied about the mesh's
        bounding-box center so rotation/scale feel natural in the UI."""
        center = self.mesh.bounds.mean(axis=0)
        t_back = trimesh.transformations.translation_matrix(-center)
        s = np.diag([*self.scale, 1.0])
        rx, ry, rz = np.radians(self.rotation)
        r = (trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
             @ trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
             @ trimesh.transformations.rotation_matrix(rx, [1, 0, 0]))
        t = trimesh.transformations.translation_matrix(self.position)
        return t @ r @ s @ t_back

    def transformed(self) -> trimesh.Trimesh:
        """A copy of the mesh with the model matrix applied (world space)."""
        m = self.mesh.copy()
        m.apply_transform(self.matrix())
        return m

    def world_bounds(self) -> np.ndarray:
        """[[xmin, ymin, zmin], [xmax, ymax, zmax]] in world space.

        The 3D view asks for this every frame, so the expensive part (every
        vertex rotated and scaled) is cached per mesh, rotation and scale;
        moving the object only shifts the cached box."""
        key = (self.rotation.tobytes(), self.scale.tobytes())
        cache = self.__dict__.get("_bounds_cache")
        if cache is None or cache[0] is not self.mesh or cache[1] != key:
            m = self.matrix()
            m[:3, 3] -= self.position                       # rotate and scale only
            verts = trimesh.transform_points(self.mesh.vertices, m)
            local = np.vstack([verts.min(axis=0), verts.max(axis=0)])
            cache = self._bounds_cache = (self.mesh, key, local)   # keeps the mesh alive: ids stay unique
        return cache[2] + self.position

    def drop_to_plate(self) -> None:
        """Move the object so its lowest point sits on the plate, or at
        ``elevation`` mm above it when Z lift is enabled."""
        zmin = self.world_bounds()[0][2]
        self.position[2] += self.elevation - zmin

    def center_on_plate(self) -> None:
        b = self.world_bounds()
        self.position[:2] -= (b[0][:2] + b[1][:2]) / 2.0

    def rotate(self, axis: int, degrees: float) -> None:
        self.rotation[axis] = (self.rotation[axis] + degrees) % 360.0
        self.drop_to_plate()

    def set_uniform_scale(self, factor: float) -> None:
        self.scale[:] = factor
        self.drop_to_plate()

    # ---------------------------------------------------------------- editing
    def mirror(self, axis: int) -> None:
        """Mirror the *source* mesh across one of its own axes (0=X,1=Y,2=Z).
        A new mesh is created so undo snapshots keep the old one."""
        center = self.mesh.bounds.mean(axis=0)
        flip = np.eye(4)
        flip[axis, axis] = -1.0
        m = (trimesh.transformations.translation_matrix(center)
             @ flip @ trimesh.transformations.translation_matrix(-center))
        mesh = self.mesh.copy()
        mesh.apply_transform(m)
        mesh.fix_normals()               # mirroring flips winding
        self.mesh = mesh
        self.holes = tuple(Hole(tuple((m @ np.r_[h.point, 1.0])[:3]),
                                tuple(np.asarray(h.normal) * np.diag(flip)[:3]), h.diameter)
                           for h in self.holes)
        self.drop_to_plate()

    def clone(self) -> "MeshObject":
        c = MeshObject(name=self.name, mesh=self.mesh.copy(),
                       position=self.position.copy(), rotation=self.rotation.copy(),
                       scale=self.scale.copy(), elevation=self.elevation,
                       holes=self.holes, hollow=self.hollow)
        return c

    def repair(self) -> dict:
        """Best-effort mesh repair on a copy. Returns a small report dict."""
        m = self.mesh.copy()
        before = dict(faces=len(m.faces), watertight=m.is_watertight)
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        trimesh.repair.fix_inversion(m)
        trimesh.repair.fix_normals(m)
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
        trimesh.repair.fix_winding(m)
        after = dict(faces=len(m.faces), watertight=m.is_watertight)
        self.mesh = m
        return {"before": before, "after": after}

    # ---------------------------------------------------------------- info
    def volume_mm3(self) -> float:
        m = self.transformed()
        return abs(m.volume) if m.is_volume else float(m.convex_hull.volume)

    def size(self) -> np.ndarray:
        b = self.world_bounds()
        return b[1] - b[0]

    def __repr__(self) -> str:  # pragma: no cover
        return f"MeshObject({self.name!r}, faces={len(self.mesh.faces)})"
