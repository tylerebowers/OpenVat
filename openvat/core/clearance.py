"""Clearance field: how far points near a model are from its surface.

Support placement keeps asking "how much room is there around this rod?".
A distance field sampled on a voxel grid around the model answers that with
one array lookup per point, so a whole rod is checked by sampling points
along its axis and comparing each distance with the rod radius plus a margin.

Built once per object shape:

1. **occupancy** - each vertical column of cells is filled by counting
   surface crossings below every cell center (a winding number: faces that
   point down are entered going up, faces that point up are left);
2. **distance** - an exact Euclidean distance transform, truncated at
   ``cap`` (the largest clearance anyone asks for), done separably along X,
   Y and Z with numpy.

Distances are in mm, roughly to the surface (cell centers are half a cell
off): negative inside the model, ``cap`` far from it and anywhere outside
the grid.  Moving an object reuses its field; only a new mesh, rotation or
scale builds a new one (``clearance_field`` keeps a small cache).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import trimesh

MAX_CELLS = 6_000_000            # grid size limit: big models get coarser cells
MIN_CELL, MAX_CELL = 0.2, 1.5    # mm
_CHUNK = 400_000                 # (triangle, column) pairs rasterized at once
_JITTER = np.array([1.13e-4, 0.71e-4, 0.0])   # keeps cell centers off exact mesh edges


class ClearanceField:
    def __init__(self, mesh: trimesh.Trimesh, cap: float, cell: float | None = None) -> None:
        lo, hi = np.asarray(mesh.bounds, dtype=float)
        if cell is None:
            room = np.prod(hi - lo + 2 * (cap + 2 * MIN_CELL))
            cell = min(max((room / MAX_CELLS) ** (1 / 3), MIN_CELL), MAX_CELL)
        self.cell = float(cell)
        self.cap = float(max(cap, 2 * self.cell))
        pad = self.cap + 2 * self.cell
        self.origin = lo - pad + _JITTER
        self.shape = tuple(int(n) for n in np.ceil((hi - lo + 2 * pad) / self.cell))
        self.offset = np.zeros(3)                       # translation since it was built
        self.dist = _distance(_occupancy(mesh, self.origin, self.cell, self.shape), self.cell, self.cap)

    def moved(self, delta) -> "ClearanceField":
        """This field for the model translated by ``delta`` (shares the data)."""
        f = copy.copy(self)
        f.offset = np.asarray(delta, dtype=float).copy()
        return f

    # ------------------------------------------------------------ lookups
    def index(self, point) -> tuple[int, int, int]:
        i = np.floor((np.asarray(point, float) - self.offset - self.origin) / self.cell)
        return int(i[0]), int(i[1]), int(i[2])

    def layer_of(self, z: float) -> int:
        return int(math.floor((z - self.offset[2] - self.origin[2]) / self.cell))

    def center(self, i: int, j: int, k: int) -> np.ndarray:
        return self.origin + self.offset + (np.array([i, j, k], float) + 0.5) * self.cell

    def distance(self, points) -> np.ndarray:
        """Distance (mm) from each point to the model (trilinear between
        cell centers)."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        q = (p - self.offset - self.origin) / self.cell - 0.5
        base = np.floor(q).astype(np.int64)
        frac = q - base
        shape = np.array(self.shape)
        out = np.zeros(len(p))
        for corner in np.ndindex(2, 2, 2):
            idx = base + corner
            ok = np.all((idx >= 0) & (idx < shape), axis=1)
            val = np.full(len(p), self.cap)
            i = idx[ok]
            val[ok] = self.dist[i[:, 0], i[:, 1], i[:, 2]]
            w = np.prod(np.where(corner, frac, 1.0 - frac), axis=1)
            out += w * val
        return out

    def segment_clear(self, p0, p1, clearance: float) -> bool:
        """True if a rod of radius ``clearance`` from p0 to p1 misses the model."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = max(2, int(math.ceil(np.linalg.norm(p1 - p0) / (0.5 * self.cell))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        return bool((self.distance(p0 + (p1 - p0) * t) >= clearance).all())

    def layer(self, k: int, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
        """Distances of cells [i0:i1, j0:j1] of grid layer k (cap outside the grid)."""
        out = np.full((i1 - i0, j1 - j0), self.cap, dtype=np.float32)
        nx, ny, nz = self.shape
        a0, a1, b0, b1 = max(i0, 0), min(i1, nx), max(j0, 0), min(j1, ny)
        if 0 <= k < nz and a0 < a1 and b0 < b1:
            out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = self.dist[a0:a1, b0:b1, k]
        return out


# --------------------------------------------------------------------------
# building

def _occupancy(mesh: trimesh.Trimesh, origin: np.ndarray, cell: float, shape) -> np.ndarray:
    """Boolean grid: is each cell center inside the mesh?"""
    nx, ny, nz = shape
    tri = np.asarray(mesh.triangles, dtype=float)
    nzs = np.asarray(mesh.face_normals)[:, 2]
    keep = np.abs(nzs) > 1e-9                                  # vertical faces cross no vertical ray
    tri = tri[keep]
    sign = np.where(nzs[keep] < 0, 1, -1).astype(np.int16)     # +1 entering, -1 leaving (going up)
    # grid columns whose centers lie in each triangle's XY bounding box
    lo = np.ceil((tri[:, :, :2].min(axis=1) - origin[:2]) / cell - 0.5).astype(np.int64)
    hi = np.floor((tri[:, :, :2].max(axis=1) - origin[:2]) / cell - 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [nx - 1, ny - 1])
    cnt = np.maximum(hi - lo + 1, 0)
    total = cnt[:, 0] * cnt[:, 1]
    wind = np.zeros((nx, ny, nz + 1), dtype=np.int16)

    order = np.flatnonzero(total)
    ends = np.cumsum(total[order])
    start = 0
    while start < len(order):
        done = ends[start - 1] if start else 0
        stop = max(int(np.searchsorted(ends, done + _CHUNK, side="right")), start + 1)
        t = order[start:stop]
        n = total[t]
        t = np.repeat(t, n)
        local = np.arange(len(t)) - np.repeat(np.cumsum(n) - n, n)
        ci = lo[t, 0] + local // cnt[t, 1]
        cj = lo[t, 1] + local % cnt[t, 1]
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        qx = origin[0] + (ci + 0.5) * cell - a[:, 0]
        qy = origin[1] + (cj + 0.5) * cell - a[:, 1]
        v0x, v0y = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        v1x, v1y = c[:, 0] - a[:, 0], c[:, 1] - a[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            den = v0x * v1y - v1x * v0y
            u = (qx * v1y - v1x * qy) / den
            v = (v0x * qy - qx * v0y) / den
        hit = (u >= 0) & (v >= 0) & (u + v <= 1)
        z = a[hit, 2] + u[hit] * (b[hit, 2] - a[hit, 2]) + v[hit] * (c[hit, 2] - a[hit, 2])
        k = np.clip(np.floor((z - origin[2]) / cell - 0.5).astype(np.int64) + 1, 0, nz)   # first cell above
        np.add.at(wind, (ci[hit], cj[hit], k), sign[t[hit]])
        start = stop
    return np.cumsum(wind, axis=2, dtype=np.int16)[:, :, :nz] > 0


def _distance(inside: np.ndarray, cell: float, cap: float) -> np.ndarray:
    """Truncated Euclidean distance transform (mm): exact up to ``cap``."""
    reach = int(math.ceil(cap / cell)) + 1
    g = np.where(inside, np.float32(0), np.float32(3 * (reach + 1) ** 2))
    for axis in range(3):                     # squared distance, one axis at a time
        g = _min_plus(g, axis, reach)
    d = np.sqrt(g) * np.float32(cell) - np.float32(0.5 * cell)
    return np.minimum(d, np.float32(cap))


def _min_plus(g: np.ndarray, axis: int, reach: int) -> np.ndarray:
    """out[i] = min over |a| <= reach of g[i + a] + a^2 along one axis."""
    out = g.copy()
    gm, om = np.moveaxis(g, axis, 0), np.moveaxis(out, axis, 0)
    for a in range(1, min(reach, gm.shape[0] - 1) + 1):
        w = np.float32(a * a)
        np.minimum(om[:-a], gm[a:] + w, out=om[:-a])
        np.minimum(om[a:], gm[:-a] + w, out=om[a:])
    return out


# --------------------------------------------------------------------------
# cache: one field per (mesh, rotation, scale); translations reuse it

_CACHE: list[tuple] = []          # (source mesh, linear part, translation, field), newest last
_CACHE_SIZE = 4


def clearance_field(obj, cap: float) -> ClearanceField:
    """The clearance field of a MeshObject where it is now."""
    m = obj.matrix()
    linear, shift = m[:3, :3], m[:3, 3]
    for i, (src, lin, t0, field) in enumerate(_CACHE):
        if src is obj.mesh and np.allclose(lin, linear) and field.cap >= cap:
            _CACHE.append(_CACHE.pop(i))
            return field.moved(shift - t0)
    field = ClearanceField(obj.transformed(), cap)
    _CACHE.append((obj.mesh, linear.copy(), shift.copy(), field))
    del _CACHE[:-_CACHE_SIZE]
    return field
