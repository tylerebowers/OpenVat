"""Ray / mesh intersection (vectorized Möller–Trumbore, numpy only).

Used for mouse picking and for snapping supports to the model surface.
It tests every triangle, which is fast enough for picking on meshes with
up to a few million faces.
"""

from __future__ import annotations

import numpy as np
import trimesh

EPS = 1e-9


def ray_mesh_hits(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """All intersection distances t (>0) along the ray, sorted ascending."""
    tri = mesh.triangles                 # (N, 3, 3)
    if len(tri) == 0:
        return np.empty(0)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    d = np.asarray(direction, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)

    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > EPS
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]

    s = o - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("j,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv

    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > EPS)
    return np.sort(t[hit])


def ray_mesh_closest(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray):
    """(distance, point) of the closest hit, or None."""
    t = ray_mesh_hits(mesh, origin, direction)
    if len(t) == 0:
        return None
    return float(t[0]), np.asarray(origin) + np.asarray(direction) * t[0]


def ray_mesh_first(mesh: trimesh.Trimesh, origin, direction):
    """(distance, point, face index) of the closest hit, or None."""
    tri = mesh.triangles
    if len(tri) == 0:
        return None
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    d = np.asarray(direction, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > EPS
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = o - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("j,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = np.flatnonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > EPS))
    if len(hit) == 0:
        return None
    k = hit[np.argmin(t[hit])]
    return float(t[k]), o + d * t[k], int(k)


class VerticalRays:
    """Fast vertical ray casting against one mesh (many rays, same mesh).

    ``hits(x, y)`` returns every surface crossing at (x, y) as sorted
    (z values, face indices); ``first_above`` / ``first_below`` pick one.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.tri = np.asarray(mesh.triangles, dtype=np.float64)
        self.normals = np.asarray(mesh.face_normals)
        xy = self.tri[:, :, :2]
        self.lo, self.hi = xy.min(axis=1), xy.max(axis=1)

    def hits(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        cand = np.flatnonzero((self.lo[:, 0] <= x) & (self.hi[:, 0] >= x) &
                              (self.lo[:, 1] <= y) & (self.hi[:, 1] >= y))
        if len(cand) == 0:
            return np.empty(0), np.empty(0, dtype=int)
        a, b, c = self.tri[cand, 0], self.tri[cand, 1], self.tri[cand, 2]
        v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        px, py = x - a[:, 0], y - a[:, 1]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-14
        den = np.where(ok, den, 1.0)
        u = (px * v1[:, 1] - v1[:, 0] * py) / den
        v = (v0[:, 0] * py - px * v0[:, 1]) / den
        inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        z, f = z[inside], cand[inside]
        order = np.argsort(z)
        return z[order], f[order]

    def first_above(self, x: float, y: float, z: float):
        """Lowest surface strictly above z: (z_hit, face) or None."""
        zs, fs = self.hits(x, y)
        k = np.searchsorted(zs, z, side="right")
        return (float(zs[k]), int(fs[k])) if k < len(zs) else None

    def first_below(self, x: float, y: float, z: float):
        """Highest surface strictly below z: (z_hit, face) or None."""
        zs, fs = self.hits(x, y)
        k = np.searchsorted(zs, z, side="left") - 1
        return (float(zs[k]), int(fs[k])) if k >= 0 else None


class RayCaster:
    """First hits of rays in any direction against one mesh.  Only the
    triangles whose bounding boxes touch the ray's segment are tested, so
    short rays are cheap even on big meshes."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.tri = np.asarray(mesh.triangles, dtype=np.float64)
        self.normals = np.asarray(mesh.face_normals)
        self.lo, self.hi = self.tri.min(axis=1), self.tri.max(axis=1)

    def first(self, origin, direction, max_t: float):
        """(distance, point, face) of the closest hit within max_t, or None."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        end = o + d * max_t
        lo, hi = np.minimum(o, end) - 1e-9, np.maximum(o, end) + 1e-9
        cand = np.flatnonzero(np.all(self.hi >= lo, axis=1) & np.all(self.lo <= hi, axis=1))
        if len(cand) == 0:
            return None
        tri = self.tri[cand]
        v0 = tri[:, 0]
        e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > EPS
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        s = o - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("j,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = np.flatnonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > EPS) & (t <= max_t))
        if len(hit) == 0:
            return None
        k = hit[np.argmin(t[hit])]
        return float(t[k]), o + d * t[k], int(cand[k])
