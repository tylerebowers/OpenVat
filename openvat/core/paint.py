"""Support paint: where supports must go on a model, and where they must not.

A MeshObject's ``paint`` holds one byte per face of its source mesh:
``NONE``, ``ENFORCE`` (supports go here) or ``BLOCK`` (no supports here).
Painting always makes a new array, so undo snapshots keep the old one.
Automatic support placement follows it (see placement.py).

``brushed`` paints the faces within a ball around a point (a brush that
does not reach through to the far side of the model); ``overhang_faces``
picks the faces that face down steeper than an angle, to paint all at once.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

NONE, ENFORCE, BLOCK = 0, 1, 2
PLATE_TOUCH = 0.05      # mm: faces this close to the plate lie on it (they need no support)

_TREE: list = []        # [(mesh, matrix bytes), tree] of the last model painted on


def paint_of(obj) -> np.ndarray | None:
    """The object's paint if it has any (and it still fits its mesh)."""
    p = getattr(obj, "paint", None)
    if p is None or len(p) != len(obj.mesh.faces) or not p.any():
        return None
    return p


def _centers(obj) -> cKDTree:
    """A search tree over the face centers as the model stands now."""
    key = (obj.mesh, obj.matrix().tobytes())
    if _TREE and _TREE[0][0] is key[0] and _TREE[0][1] == key[1]:
        return _TREE[1]
    m = obj.matrix()
    centers = np.asarray(obj.mesh.triangles_center) @ m[:3, :3].T + m[:3, 3]
    tree = cKDTree(centers)
    _TREE[:] = [key, tree]
    return tree


def brushed(obj, center, radius: float, value: int, face: int | None = None) -> np.ndarray | None:
    """The object's paint with every face whose center lies within
    ``radius`` mm of ``center`` (world) set to ``value`` - and ``face`` (the
    face under the cursor, which a big triangle's far-off center would
    miss); None when that changes nothing."""
    idx = list(_centers(obj).query_ball_point(np.asarray(center, float), float(radius)))
    if face is not None and 0 <= face < len(obj.mesh.faces):
        idx.append(int(face))
    if not idx:
        return None
    n = len(obj.mesh.faces)
    base = obj.paint if obj.paint is not None and len(obj.paint) == n else np.zeros(n, np.int8)
    idx = np.asarray(idx)
    if np.all(base[idx] == value):
        return None
    out = base.copy()
    out[idx] = value
    return out


def world_normals(obj) -> np.ndarray:
    """Face normals of the source mesh as the model stands now."""
    n = np.asarray(obj.mesh.face_normals) @ np.linalg.inv(obj.matrix()[:3, :3])   # = M^-T n, row-wise
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def overhang_faces(obj, angle: float) -> np.ndarray:
    """Faces that face down steeper than ``angle`` degrees from vertical
    (a vertical wall: 0, a flat ceiling: 90) as the model stands now - not
    the ones lying on the plate."""
    nz = world_normals(obj)[:, 2]
    m = obj.matrix()
    z = (np.asarray(obj.mesh.triangles) @ m[:3, :3].T + m[:3, 3])[:, :, 2]
    on_plate = z.max(axis=1) < PLATE_TOUCH
    return (nz < -math.sin(math.radians(min(max(angle, 0.0), 89.9)))) & ~on_plate


def painted_with(obj, faces: np.ndarray, value: int) -> np.ndarray:
    """The object's paint with ``faces`` (a mask or indices) set to ``value``."""
    n = len(obj.mesh.faces)
    out = obj.paint.copy() if obj.paint is not None and len(obj.paint) == n else np.zeros(n, np.int8)
    out[faces] = value
    return out
