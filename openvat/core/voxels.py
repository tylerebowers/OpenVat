"""Rebuild the model from its slices as a voxel volume.

A voxel is one printer pixel (or a block of n x n pixels) in X/Y and one
layer in Z.  A voxel is solid when at least half of its area lies inside the
layer polygon; that is decided by rasterizing each layer at 3x3 sub-pixels
and counting (an odd factor avoids ties on walls that sit exactly on a
pixel center).  Layers whose geometry is identical to the previous layer
reuse its bitmap, so identical layers always give identical voxels.

The volume is stored bit-packed (a 20 mm cube at M7 Pro resolution is
~400 M voxels).  ``VoxelVolume.faces()`` lists the exposed voxel faces as
merged rectangles in grid units (flat surfaces cost one rectangle per run
of pixels, walls are merged across layers too); ``VoxelFaces`` orders
them in chunks for the GPU (16 bytes per rectangle, see
``ui/voxel_view.py``) and decides per frame which chunks can be seen.
``mesh()`` builds an ordinary triangle mesh from the same faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from .profiles import PrinterProfile
from .slicer import SliceResult, Layer

SUPERSAMPLE = 3                      # sub-pixels per pixel per axis (odd!)
MIN_COVERAGE = (SUPERSAMPLE * SUPERSAMPLE + 1) // 2   # >= 50 %
CHUNK = (64, 64, 32)                 # culling chunk: pixels in X, Y and layers
GRID_LIMIT = 65535                   # grid coordinates are packed into 16 bits

# face directions (outward normal): +X, -X, +Y, -Y, +Z, -Z
PX, NX, PY, NY, PZ, NZ = range(6)

ProgressFn = Callable[[int, int], None]


class VoxelVolume:
    def __init__(self, layers: list[Layer], origin: np.ndarray, voxel_xy: tuple[float, float],
                 nx: int, ny: int):
        self.layers = layers
        self.origin = origin                 # mm, lower-left corner of voxel (0, 0)
        self.vx, self.vy = voxel_xy          # voxel size in mm
        self.nx, self.ny = nx, ny
        self.packed: list[np.ndarray] = []   # one packed (ny, ceil(nx/8)) array per layer
        self.solid_count: list[int] = []
        self.z_edges = np.array([l.z_bottom for l in layers] + [layers[-1].z_top])

    @property
    def nz(self) -> int:
        return len(self.packed)

    def layer(self, i: int) -> np.ndarray:
        return np.unpackbits(self.packed[i], axis=1, count=self.nx).astype(bool)

    def solid_up_to(self, i: int) -> int:
        return int(sum(self.solid_count[: i + 1]))

    # ------------------------------------------------------------ faces
    def groups(self, max_layer: int | None = None) -> list[tuple[int, int]]:
        """Runs [first, last] of consecutive identical layers."""
        nz = self.nz if max_layer is None else min(max_layer + 1, self.nz)
        out: list[tuple[int, int]] = []
        start = 0
        for i in range(1, nz + 1):
            if i == nz or not np.array_equal(self.packed[i], self.packed[start]):
                out.append((start, i - 1))
                start = i
        return out

    def faces(self, max_layer: int | None = None, progress: ProgressFn | None = None) -> np.ndarray:
        """The exposed voxel faces up to ``max_layer`` (all layers when None)
        as an (N, 7) int32 array of rectangles in grid units:
        ``dir, i0, j0, k0, i1, j1, k1`` - direction (PX..NZ, the outward
        normal), then the low and high corner (i = pixel column 0..nx,
        j = pixel row 0..ny, k = layer edge 0..nz; the corners are equal
        along the normal's axis).

        Horizontal faces are runs of pixels merged into rectangles across
        rows; walls are runs merged across every layer they continue
        through, so straight walls cost one rectangle however tall."""
        if max(self.nx, self.ny, self.nz) > GRID_LIMIT:
            raise ValueError("The model is too large for the voxel preview.")
        horizontal: list[np.ndarray] = []
        walls: list[np.ndarray] = []                 # dir, plane, start, end, k0, k1
        empty = np.zeros_like(self.packed[0]) if self.packed else None
        below = empty                                # packed bits of the layer below
        groups = self.groups(max_layer)
        for n, (first, last) in enumerate(groups):
            if progress and n % 16 == 0:
                progress(n, len(groups))
            cur = self.packed[first]
            # work only where this layer or the one below has material (plus an
            # empty row and byte on every side), straight on the packed bits
            box = self._window(cur | below)
            if box is None:
                below = cur
                continue
            r0, r1, b0, b1 = box
            c0 = b0 * 8
            c, b = cur[r0:r1, b0:b1], below[r0:r1, b0:b1]
            # horizontal faces at the bottom of this group (every mask below is
            # ANDed with a layer, so the padding bits past nx stay zero)
            horizontal.append(_rect_faces(b & ~c, PZ, first, r0, c0))
            horizontal.append(_rect_faces(~b & c, NZ, first, r0, c0))
            k0, k1 = first, last + 1
            # walls between rows j-1 and j (normal +-Y): runs along X
            prev, this = c[:-1], c[1:]
            walls.append(_wall_runs(prev & ~this, PY, r0 + 1, c0, k0, k1))
            walls.append(_wall_runs(~prev & this, NY, r0 + 1, c0, k0, k1))
            # walls between columns i-1 and i (normal +-X): where each row's runs
            # end or start, joined across consecutive rows
            row, start, end = _packed_runs(c)
            walls.append(_column_walls(end + c0, row + r0, PX, k0, k1))
            walls.append(_column_walls(start + c0, row + r0, NX, k0, k1))
            below = cur
        if groups:                                   # the top of the last group
            box = self._window(below)
            if box is not None:
                r0, r1, b0, b1 = box
                horizontal.append(_rect_faces(below[r0:r1, b0:b1], PZ, groups[-1][1] + 1, r0, b0 * 8))
        if progress:
            progress(len(groups), len(groups))
        out = [h for h in horizontal if len(h)]
        w = _merge_walls(walls)
        if len(w):
            d, plane, s, e, k0, k1 = w.T
            x = d <= NX                              # +-X walls: plane is a column, the run goes along Y
            out.append(np.stack([d, np.where(x, plane, s), np.where(x, s, plane), k0,
                                 np.where(x, plane, e), np.where(x, e, plane), k1], 1))
        if not out:
            return np.zeros((0, 7), dtype=np.int32)
        return np.concatenate(out).astype(np.int32)

    def cap(self, max_layer: int) -> np.ndarray:
        """Faces closing the cut when only layers 0..max_layer are shown:
        the top of ``max_layer`` wherever the layer above is solid too (the
        rest of its top is among ``faces()`` already).  Works on the packed
        bits, so it stays fast on plate-sized layers (the slider calls it)."""
        if not 0 <= max_layer < self.nz - 1:
            return np.zeros((0, 7), dtype=np.int32)
        both = self.packed[max_layer] & self.packed[max_layer + 1]
        box = self._window(both)
        if box is None:
            return np.zeros((0, 7), dtype=np.int32)
        r0, r1, b0, b1 = box
        return _rect_faces(both[r0:r1, b0:b1], PZ, max_layer + 1, r0, b0 * 8).astype(np.int32)

    def _window(self, packed: np.ndarray) -> tuple[int, int, int, int] | None:
        """Rows [r0, r1) and packed byte columns [b0, b1) holding every set bit,
        grown by one row and one byte (8 voxels) on each side, or None."""
        rows = np.flatnonzero(packed.any(axis=1))
        if not len(rows):
            return None
        r0, r1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, packed.shape[0])
        cols = np.flatnonzero(packed[r0:r1].any(axis=0))
        return r0, r1, max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, packed.shape[1])

    def mesh(self, max_layer: int | None = None) -> trimesh.Trimesh:
        """A closed triangle mesh of the voxels up to ``max_layer``."""
        return faces_to_mesh(self.faces(max_layer), self)

    def world(self, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Low and high world corners (mm) of grid-unit faces."""
        f = np.asarray(faces)
        lo = np.stack([self.origin[0] + f[:, 1] * self.vx, self.origin[1] + f[:, 2] * self.vy,
                       self.z_edges[f[:, 3]]], 1)
        hi = np.stack([self.origin[0] + f[:, 4] * self.vx, self.origin[1] + f[:, 5] * self.vy,
                       self.z_edges[f[:, 6]]], 1)
        return lo, hi


def clip_faces(faces: np.ndarray, k_cut: int) -> np.ndarray:
    """What the GPU keeps of ``faces`` when showing layers below edge
    ``k_cut`` (mirrors the clip plane in the voxel shader): walls are cut at
    the edge, horizontal faces above it go, and so do downward faces on it
    (they are the bottoms of the next layer)."""
    f = np.array(faces, copy=True)
    d = f[:, 0]
    flat = d >= PZ
    keep = np.where(flat, (f[:, 3] < k_cut) | ((f[:, 3] == k_cut) & (d == PZ)), f[:, 3] < k_cut)
    f = f[keep]
    f[:, 6] = np.where(f[:, 0] >= PZ, f[:, 6], np.minimum(f[:, 6], k_cut))
    return f


def faces_to_mesh(faces: np.ndarray, vol: VoxelVolume) -> trimesh.Trimesh:
    """Triangles for grid-unit faces, wound counter-clockwise seen from
    outside (the corner order the voxel shader uses too)."""
    if not len(faces):
        return trimesh.Trimesh()
    lo, hi = vol.world(faces)
    axis, neg = faces[:, 0] // 2, faces[:, 0] % 2 == 1
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    eye = np.eye(3)
    eb, ec = eye[(axis + 1) % 3], eye[(axis + 2) % 3]
    u = np.where(neg[:, None], uv[None, :, 1], uv[None, :, 0])        # (N, 4): negative faces swap u/v
    v = np.where(neg[:, None], uv[None, :, 0], uv[None, :, 1])
    span = hi - lo
    corners = lo[:, None, :] + span[:, None, :] * (eb[:, None, :] * u[..., None] + ec[:, None, :] * v[..., None])
    verts = corners.reshape(-1, 3)
    base = np.arange(len(faces)) * 4
    tris = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)])
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)


# --------------------------------------------------------------------------
# faces in chunks, for drawing

@dataclass
class VoxelFaces:
    """Faces sorted into groups - one per culling chunk and direction -
    with each group's world bounds, so the viewer can skip whole groups
    that are off screen, face away from the camera or lie above the cut."""
    faces: np.ndarray            # (N, 7) int32, see VoxelVolume.faces(); sorted by group
    start: np.ndarray            # (G,) first face of each group
    end: np.ndarray              # (G,) one past its last face
    dir: np.ndarray              # (G,) direction of its faces
    lo: np.ndarray               # (G, 3) world bounds (mm)
    hi: np.ndarray

    @classmethod
    def build(cls, faces: np.ndarray, vol: VoxelVolume, chunk=CHUNK) -> "VoxelFaces":
        f = np.asarray(faces, dtype=np.int32)
        if not len(f):
            z = np.zeros(0, dtype=np.int64)
            return cls(f.reshape(0, 7), z, z, z, np.zeros((0, 3)), np.zeros((0, 3)))
        ci = f[:, 1].astype(np.int64) // chunk[0]
        cj = f[:, 2].astype(np.int64) // chunk[1]
        ck = f[:, 3].astype(np.int64) // chunk[2]
        nci, ncj = int(ci.max()) + 1, int(cj.max()) + 1
        key = ((ck * ncj + cj) * nci + ci) * 6 + f[:, 0]
        order = np.argsort(key, kind="stable")
        f, key = f[order], key[order]
        start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        end = np.r_[start[1:], len(f)]
        lo, hi = vol.world(f)
        return cls(f, start, end, f[start, 0].astype(np.int64),
                   np.minimum.reduceat(lo, start), np.maximum.reduceat(hi, start))

    def __len__(self) -> int:
        return len(self.faces)

    def packed(self) -> np.ndarray:
        return pack_faces(self.faces)

    def visible(self, eye, view_proj: np.ndarray, z_cut: float = np.inf) -> np.ndarray:
        """Indices of the groups worth drawing, nearest first: inside the
        view frustum, below the cut, and with at least one face turned
        towards the camera (an outward +X face is seen only from x greater
        than its plane, and so on)."""
        if not len(self.start):
            return np.zeros(0, dtype=np.int64)
        eye = np.asarray(eye, dtype=float)
        lo, hi, d = self.lo, self.hi, self.dir
        axis = d // 2
        rows = np.arange(len(d))
        positive = d % 2 == 0
        ok = np.where(positive, eye[axis] > lo[rows, axis], eye[axis] < hi[rows, axis])
        eps = 1e-4
        ok &= (lo[:, 2] <= z_cut + eps) & ~((d == NZ) & (lo[:, 2] >= z_cut - eps))
        m = np.asarray(view_proj, dtype=float)
        for plane in (m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]):
            far_corner = np.where(plane[:3] >= 0, hi, lo)          # the corner furthest along the plane normal
            ok &= far_corner @ plane[:3] + plane[3] >= 0
        idx = np.flatnonzero(ok)
        center = (lo[idx] + hi[idx]) / 2
        return idx[np.argsort(((center - eye) ** 2).sum(axis=1))]


def pack_faces(faces: np.ndarray) -> np.ndarray:
    """(N, 4) uint32 for the GPU: i0 | j0 << 16, i1 | j1 << 16,
    k0 | k1 << 16, dir - 16 bytes per face."""
    f = np.asarray(faces).astype(np.uint32).reshape(-1, 7)
    return np.ascontiguousarray(np.stack([f[:, 1] | (f[:, 2] << 16), f[:, 4] | (f[:, 5] << 16),
                                          f[:, 3] | (f[:, 6] << 16), f[:, 0]], 1))


# --------------------------------------------------------------------------
# helpers

def _transition_tables() -> tuple[np.ndarray, np.ndarray]:
    """For every byte together with the last bit of the byte before it
    (9 bits, index = prev << 8 | byte): at which of its 8 pixels a run of
    set bits starts, and at which one ends (np.packbits order: the first
    pixel is the high bit)."""
    v = np.arange(512)
    bits = (v[:, None] >> np.arange(8, -1, -1)) & 1          # previous pixel, then pixels 0..7
    return (bits[:, 1:] == 1) & (bits[:, :-1] == 0), (bits[:, 1:] == 0) & (bits[:, :-1] == 1)


_STARTS, _ENDS = _transition_tables()


def _packed_runs(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, start, end) of every run of set bits in a packed 2D mask, rows
    in order and runs left to right.  Only bytes where a run starts or ends
    are looked at - all-empty and all-full bytes are skipped - so this is
    fast even on plate-sized layers.  Bits past the last pixel must be 0."""
    p = np.asarray(packed, dtype=np.uint8)
    if not p.size:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    prev = np.zeros_like(p)
    prev[:, 1:] = p[:, :-1] & 1                               # last pixel of the byte before
    look = ((p != 0) & (p != 255)) | (prev != (p >> 7))
    r, k = np.nonzero(look)
    v = (prev[r, k].astype(np.int64) << 8) | p[r, k]
    i, j = np.nonzero(_STARTS[v])
    srow, start = r[i], k[i] * 8 + j
    i, j = np.nonzero(_ENDS[v])
    erow, end = r[i], k[i] * 8 + j
    full = np.flatnonzero(p[:, -1] & 1)                       # runs reaching the last bit
    if len(full):
        erow = np.concatenate([erow, full])
        end = np.concatenate([end, np.full(len(full), p.shape[1] * 8)])
        order = np.lexsort((end, erow))
        erow, end = erow[order], end[order]
    return srow.astype(np.int64), start.astype(np.int64), end.astype(np.int64)


def _wall_runs(mask: np.ndarray, direction: int, plane0: int, offset: int, k0: int, k1: int) -> np.ndarray:
    """Runs of a packed wall mask (one row per plane) as rows of
    (dir, plane, start, end, k0, k1)."""
    r, s, e = _packed_runs(mask)
    n = len(r)
    if not n:
        return np.zeros((0, 6), dtype=np.int32)
    return np.stack([np.full(n, direction), r + plane0, s + offset, e + offset,
                     np.full(n, k0), np.full(n, k1)], 1).astype(np.int32)


def _column_walls(plane: np.ndarray, row: np.ndarray, direction: int, k0: int, k1: int) -> np.ndarray:
    """X walls from the columns where row runs start or end: the same column
    in consecutive rows becomes one wall (dir, plane, row0, row1, k0, k1)."""
    if not len(plane):
        return np.zeros((0, 6), dtype=np.int32)
    order = np.lexsort((row, plane))
    plane, row = plane[order], row[order]
    new = np.r_[True, (plane[1:] != plane[:-1]) | (row[1:] != row[:-1] + 1)]
    first = np.flatnonzero(new)
    last = np.r_[first[1:], len(row)] - 1
    n = len(first)
    return np.stack([np.full(n, direction), plane[first], row[first], row[last] + 1,
                     np.full(n, k0), np.full(n, k1)], 1).astype(np.int32)


def _merge_walls(walls: list[np.ndarray]) -> np.ndarray:
    """Join wall runs that continue in the next layer group (same direction,
    plane and run) into one taller rectangle."""
    walls = [w for w in walls if len(w)]
    if not walls:
        return np.zeros((0, 6), dtype=np.int32)
    w = np.concatenate(walls)
    w = w[np.lexsort((w[:, 4], w[:, 3], w[:, 2], w[:, 1], w[:, 0]))]
    same = np.all(w[1:, :4] == w[:-1, :4], axis=1) & (w[1:, 4] == w[:-1, 5])
    first = np.flatnonzero(np.r_[True, ~same])
    last = np.r_[first[1:], len(w)] - 1
    out = w[first].copy()
    out[:, 5] = w[last, 5]
    return out


def _rect_faces(mask: np.ndarray, direction: int, k: int, row0: int, col0: int) -> np.ndarray:
    """Horizontal faces covering a packed 2D mask: row runs with the same
    start and end in consecutive rows become one rectangle."""
    r, x0, x1 = _packed_runs(mask)
    if not len(r):
        return np.zeros((0, 7), dtype=np.int32)
    order = np.lexsort((r, x1, x0))            # group identical spans, rows ascending
    r, x0, x1 = r[order], x0[order], x1[order]
    new = np.ones(len(r), dtype=bool)
    new[1:] = (x0[1:] != x0[:-1]) | (x1[1:] != x1[:-1]) | (r[1:] != r[:-1] + 1)
    first = np.flatnonzero(new)
    last = np.append(first[1:], len(r)) - 1
    n = len(first)
    return np.stack([np.full(n, direction), x0[first] + col0, r[first] + row0, np.full(n, k),
                     x1[first] + col0, r[last] + 1 + row0, np.full(n, k)], 1).astype(np.int32)


# --------------------------------------------------------------------------
# rasterization

def _raster(layer: Layer, origin: np.ndarray, vx: float, vy: float, nx: int, ny: int) -> np.ndarray:
    """Boolean (ny, nx) bitmap: True where >= 50 % of the voxel is inside."""
    if layer.geometry.is_empty:
        return np.zeros((ny, nx), dtype=bool)
    s = SUPERSAMPLE
    sx, sy = vx / s, vy / s
    img = Image.new("1", (nx * s, ny * s), 0)
    draw = ImageDraw.Draw(img)
    # PIL fills every sub-pixel its outline touches; shrinking the polygon by
    # half a sub-pixel turns that into sampling at sub-pixel centers.
    shrunk = layer.geometry.buffer(-min(sx, sy) / 2, join_style="mitre")
    polys = [shrunk] if isinstance(shrunk, Polygon) else [g for g in getattr(shrunk, "geoms", []) if isinstance(g, Polygon)]
    for poly in sorted(polys, key=lambda p: -p.area):      # islands inside holes last
        if poly.is_empty:
            continue
        draw.polygon(_to_px(poly.exterior.coords, origin, sx, sy), fill=1)
        for ring in poly.interiors:
            draw.polygon(_to_px(ring.coords, origin, sx, sy), fill=0)
    sub = np.asarray(img, dtype=np.uint8)
    cover = sub.reshape(ny, s, nx, s).sum(axis=(1, 3))
    return cover >= MIN_COVERAGE


def _to_px(coords, origin, sx, sy):
    c = np.asarray(coords)
    return [(float((x - origin[0]) / sx), float((y - origin[1]) / sy)) for x, y in c]


def _same_geometry(a, b) -> bool:
    if a.is_empty and b.is_empty:
        return True
    if a.is_empty != b.is_empty or abs(a.area - b.area) > 1e-9:
        return False
    try:
        return a.symmetric_difference(b).area < 1e-9
    except Exception:
        return False


def build_volume(result: SliceResult, pixels_per_voxel: int = 1,
                 progress: ProgressFn | None = None) -> VoxelVolume:
    """Rasterize every layer into a VoxelVolume whose voxels are
    ``pixels_per_voxel`` printer pixels wide (X/Y) and one layer tall."""
    printer: PrinterProfile = result.printer
    layers = result.layers
    vx = printer.pixel_x_mm * pixels_per_voxel
    vy = printer.pixel_y_mm * pixels_per_voxel
    bounds = [l.geometry.bounds for l in layers if not l.geometry.is_empty]
    if not bounds:
        raise ValueError("Nothing sliced.")
    b = np.array(bounds)
    # align the voxel grid with the printer's pixel grid (plate center = pixel edge)
    lo = b[:, :2].min(axis=0)
    hi = b[:, 2:].max(axis=0)
    ix0, iy0 = int(np.floor(lo[0] / vx)) - 1, int(np.floor(lo[1] / vy)) - 1
    ix1, iy1 = int(np.ceil(hi[0] / vx)) + 1, int(np.ceil(hi[1] / vy)) + 1
    origin = np.array([ix0 * vx, iy0 * vy])
    nx, ny = ix1 - ix0, iy1 - iy0

    vol = VoxelVolume(layers, origin, (vx, vy), nx, ny)
    prev_geom, prev_bits, prev_count = None, None, 0
    for i, layer in enumerate(layers):
        if prev_geom is not None and _same_geometry(layer.geometry, prev_geom):
            bits, count = prev_bits, prev_count
        else:
            mask = _raster(layer, origin, vx, vy, nx, ny)
            bits, count = np.packbits(mask, axis=1), int(mask.sum())
            prev_geom, prev_bits, prev_count = layer.geometry, bits, count
        vol.packed.append(bits)
        vol.solid_count.append(count)
        if progress and i % 10 == 0:
            progress(i + 1, len(layers))
    if progress:
        progress(len(layers), len(layers))
    return vol
