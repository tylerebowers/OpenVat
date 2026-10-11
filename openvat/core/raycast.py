"""Ray / mesh intersection (vectorized Möller–Trumbore, numpy only).

Used for mouse picking and for snapping supports to the model surface.
It tests every triangle, which is fast enough for picking on meshes with
up to a few million faces.
"""

from __future__ import annotations

import math

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


class BoxGrid:
    """A uniform grid over boxes (here: the triangles' bounding boxes), to
    find the boxes that overlap a query box without testing every one.

    Each box is listed in every grid cell it touches; a query collects the
    boxes of the cells it touches and then tests just those exactly, so
    ``overlapping`` returns precisely the boxes a test of all of them would
    (in increasing order).  Boxes that would touch very many cells (long
    slivers across the model) are kept apart and always tested."""

    MAX_CELLS = 1 << 21          # grid cells at most
    MAX_BOX_CELLS = 4096         # a box touching more cells than this is kept apart
    BUILD_CHUNK = 1 << 20        # (box, cell) entries made at a time while building

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo, self.hi = lo, hi
        n, dim = lo.shape
        if n == 0:
            self.origin, self.cell, self.shape = np.zeros(dim), 1.0, np.ones(dim, np.int64)
            self.offsets, self.ids, self.big = np.zeros(2, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
            self._o, self._n = [0.0] * dim, [1] * dim
            return
        self.origin = lo.min(axis=0)
        extent = np.maximum(hi.max(axis=0) - self.origin, 1e-9)
        cell = max(2.0 * float(np.median((hi - lo).max(axis=1))), float(extent.max()) / 512, 1e-6)
        while np.prod(np.ceil(extent / cell)) > self.MAX_CELLS:
            cell *= 1.25
        self.cell = cell
        self.shape = np.maximum(np.ceil(extent / cell).astype(np.int64), 1)
        self._o = [float(v) for v in self.origin]          # plain floats: cheaper for one query box
        self._n = [int(v) for v in self.shape]
        i0, i1 = self._index(lo), self._index(hi)
        w = i1 - i0 + 1                                   # cells each box spans, per axis
        counts = np.prod(w, axis=1)
        small = counts <= self.MAX_BOX_CELLS
        self.big = np.flatnonzero(~small)
        boxes = np.flatnonzero(small)
        ends = np.cumsum(counts[boxes])
        flats, owners, done, first = [], [], 0, 0
        while first < len(boxes):                         # a few million entries at a time (memory)
            last = max(int(np.searchsorted(ends, done + self.BUILD_CHUNK, side="right")), first + 1)
            part = boxes[first:last]
            box = np.repeat(part, counts[part])
            k = np.arange(len(box)) - np.repeat(np.cumsum(counts[part]) - counts[part], counts[part])
            flat = np.zeros(len(box), np.int64)
            scale = 1
            for a in range(dim - 1, -1, -1):              # each box's own cells, in turn
                flat += (i0[box, a] + k % w[box, a]) * scale
                k = k // w[box, a]
                scale *= int(self.shape[a])
            flats.append(flat.astype(np.int32))           # = ravel_multi_index; cells < 2**31
            owners.append(box.astype(np.int32 if n < 2 ** 31 else np.int64))
            done, first = int(ends[last - 1]), last
        flat = np.concatenate(flats) if flats else np.empty(0, np.int32)
        owner = np.concatenate(owners) if owners else np.empty(0, np.int32)
        order = np.argsort(flat, kind="stable")           # ids stay increasing within a cell
        self.ids = owner[order]
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=int(np.prod(self.shape))))])

    def _index(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((x - self.origin) / self.cell).astype(np.int64), 0, self.shape - 1)

    def overlapping(self, qlo, qhi) -> np.ndarray:
        """Indices (increasing) of the boxes that overlap the box qlo..qhi."""
        qlo = [float(v) for v in qlo]
        qhi = [float(v) for v in qhi]
        c0, c1 = [], []
        for a, (o, n) in enumerate(zip(self._o, self._n)):         # = _index, for one box
            c0.append(min(max(math.floor((qlo[a] - o) / self.cell), 0), n - 1))
            c1.append(min(max(math.floor((qhi[a] - o) / self.cell), 0), n - 1))
        count = 1
        for a, b in zip(c0, c1):
            count *= b - a + 1
        if count * 8 > len(self.offsets):                          # a large part of the grid: test them all
            return self._exact(np.arange(len(self.lo)), qlo, qhi)
        if count == 1:                                             # one cell: its list is sorted already
            f = 0
            for a, n in zip(c0, self._n):
                f = f * n + a
            cand = self.ids[self.offsets[f]:self.offsets[f + 1]]
            if len(self.big):
                cand = np.union1d(cand, self.big)
            return self._exact(cand, qlo, qhi)
        flat = np.arange(c0[0], c1[0] + 1)
        for a in range(1, len(c0)):                                # = ravel_multi_index over the block
            flat = (flat[:, None] * self._n[a] + np.arange(c0[a], c1[a] + 1)[None, :]).ravel()
        starts = self.offsets[flat]
        lens = self.offsets[flat + 1] - starts
        total = int(lens.sum())
        cand = self.ids[np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(total)] if total \
            else np.empty(0, np.int64)
        if len(self.big):
            cand = np.concatenate([cand, self.big])
        return self._exact(np.unique(cand), qlo, qhi)

    def _exact(self, cand: np.ndarray, qlo: list, qhi: list) -> np.ndarray:
        """The boxes of ``cand`` that overlap the query box."""
        hi, lo = self.hi[cand], self.lo[cand]
        keep = (hi[:, 0] >= qlo[0]) & (lo[:, 0] <= qhi[0])
        for a in range(1, len(qlo)):
            keep &= (hi[:, a] >= qlo[a]) & (lo[:, a] <= qhi[a])
        return cand[keep]


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
        self.grid = BoxGrid(self.lo, self.hi)

    def hits(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        cand = self.grid.overlapping((x, y), (x, y))
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


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross(a, b) for (3,) or (n, 3) by (n, 3) - the same arithmetic,
    without its general-purpose overhead (it is called a lot here)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    out = np.empty((len(b), 3))
    out[:, 0] = a1 * b2 - a2 * b1
    out[:, 1] = a2 * b0 - a0 * b2
    out[:, 2] = a0 * b1 - a1 * b0
    return out


class RayCaster:
    """First hits of rays in any direction against one mesh.  Only the
    triangles whose bounding boxes touch the ray's segment are tested
    (found through a ``BoxGrid``), so short rays are cheap even on big
    meshes; a long ray is followed piece by piece through the mesh's
    bounding box and stops at the first piece holding a hit."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.tri = np.asarray(mesh.triangles, dtype=np.float64)
        self.normals = np.asarray(mesh.face_normals)
        self.lo, self.hi = self.tri.min(axis=1), self.tri.max(axis=1)
        self.grid = BoxGrid(self.lo, self.hi)
        if len(self.tri):
            box_lo, box_hi = self.lo.min(axis=0), self.hi.max(axis=0)
            pad = 1e-6 * (1.0 + float((box_hi - box_lo).max()))
            self.box_lo, self.box_hi = [float(v) for v in box_lo - pad], [float(v) for v in box_hi + pad]
        self.piece = 4.0 * self.grid.cell                 # length of each piece of a long ray

    def overlapping(self, lo, hi) -> np.ndarray:
        """Faces (increasing) whose bounding boxes overlap the box lo..hi."""
        return self.grid.overlapping(lo, hi)

    def _span(self, o: np.ndarray, d: np.ndarray, max_t: float):
        """The part (t0, t1) of the ray up to max_t inside the mesh's bounding
        box - no hit can lie elsewhere - or None."""
        if not len(self.tri):
            return None
        t0, t1 = 0.0, float(max_t)
        for a in range(3):
            if abs(d[a]) < 1e-15:
                if o[a] < self.box_lo[a] or o[a] > self.box_hi[a]:
                    return None
                continue
            ta, tb = (self.box_lo[a] - o[a]) / d[a], (self.box_hi[a] - o[a]) / d[a]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return None
        return t0, t1

    def first(self, origin, direction, max_t: float):
        """(distance, point, face) of the closest hit within max_t, or None.
        Of hits at the same distance, the lowest face index."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        span = self._span(o, d, max_t)
        if span is None:
            return None
        a, end = span
        best_t, best_face = None, -1
        of, df = [float(v) for v in o], [float(v) for v in d]
        while True:
            b = min(a + self.piece, end)
            p0 = [oi + di * a for oi, di in zip(of, df)]           # = o + d * a, as plain floats
            p1 = [oi + di * b for oi, di in zip(of, df)]
            cand = self.grid.overlapping([min(x, y) - 1e-9 for x, y in zip(p0, p1)],
                                         [max(x, y) + 1e-9 for x, y in zip(p0, p1)])
            if len(cand):
                tri = self.tri[cand]
                v0 = tri[:, 0]
                e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
                p = _cross(d, e2)
                det = np.einsum("ij,ij->i", e1, p)
                ok = np.abs(det) > EPS
                inv = np.zeros_like(det)
                inv[ok] = 1.0 / det[ok]
                s = o - v0
                u = np.einsum("ij,ij->i", s, p) * inv
                q = _cross(s, e1)
                v = np.einsum("j,ij->i", d, q) * inv
                t = np.einsum("ij,ij->i", e2, q) * inv
                hit = np.flatnonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > EPS) & (t <= max_t))
                if len(hit):
                    k = hit[np.argmin(t[hit])]
                    if best_t is None or t[k] < best_t or (t[k] == best_t and cand[k] < best_face):
                        best_t, best_face = float(t[k]), int(cand[k])
            # every hit closer than best_t lies in the pieces done, so it was seen
            if (best_t is not None and best_t <= b) or b >= end:
                break
            a = b
        if best_t is None:
            return None
        return best_t, o + d * best_t, best_face
