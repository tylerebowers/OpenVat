"""Hollowing: drain holes, a shell of a set wall thickness, and internal
support poles (a 3-axis lattice).

Workflow (enforced by the UI): a model first gets at least one drain hole,
then it can be hollowed with a wall thickness, and a hollow model can get
internal supports.  The per-model state is small and immutable -
``MeshObject.holes`` (a tuple of ``Hole``) and ``MeshObject.hollow`` (a
``HollowState`` or None) - so undo snapshots can share it.

Nothing changes the model's mesh.  The slicer applies everything to the
layer outlines of the model, layer by layer:

* **shell** - the cavity of a layer is the model's solid there eroded by a
  ball of radius *t* (the wall thickness): the layer's outline shrunk by t,
  intersected with the outlines a little above and below shrunk by less
  (sqrt(t² - dz²)).  Sampling the ball at 9 heights evenly spread in angle
  leaves walls at least 98 % of t, so t is raised by that 2 % - walls,
  floors and ceilings are never thinner than asked.
* **internal supports** - three families of straight poles at right
  angles to each other (an X, Y, Z grid with ``spacing`` mm between nodes),
  the whole grid turned about the X, Y and Z axes in turn (like a model's
  rotation: X first, then Y, then Z).  At 0/0/0 it stands upright (X and Y
  poles horizontal, Z poles vertical); the default 0/45/0 turns it 45°
  about the front-to-back (Y) axis, so the X and Z poles run diagonally,
  both 45° from horizontal, crossing like an X seen from the front, and the
  Y poles run front to back between them.  A pole's cross-section at a layer is
  computed exactly (an ellipse cut by the pole's ends), and only what falls
  in the cavity is added.
* **drain holes** - cylinders from 0.5 mm outside the surface inward along
  the surface normal, deep enough to open the cavity; they are cut last, so
  they stay clear of poles.

For the 3D view: ``tubes`` turns segments into a mesh, and ``cavity_clip``
keeps the parts of pole segments inside the cavity (a voxel depth field).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields
from typing import Callable

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ERODE_STEPS = 4                          # ball samples per quarter turn (9 heights in all)
HOLE_OUTSIDE = 0.5                       # mm a drain hole starts outside the surface
SECTION_POINTS = 10                      # points along each side of a pole's cross-section

ProgressFn = Callable[[int, int], None]


# --------------------------------------------------------------------------
# data

@dataclass(frozen=True)
class Hole:
    """A drain hole: where it is on the model's surface and the outward
    surface normal there, both in the model's own (source mesh) coordinates,
    so the hole moves, turns and scales with the model."""
    point: tuple
    normal: tuple
    diameter: float


@dataclass(frozen=True)
class LatticeParams:
    angle_x: float = 0.0                # the X/Y/Z grid turned about X, then Y, then Z (deg, 0..90)
    angle_y: float = 45.0
    angle_z: float = 0.0
    spacing: float = 8.0                # mm between nodes along every pole
    diameter: float = 1.2               # mm

    @property
    def angles(self) -> tuple[float, float, float]:
        return self.angle_x, self.angle_y, self.angle_z


@dataclass(frozen=True)
class HollowState:
    thickness: float                    # wall thickness (mm)
    lattice: LatticeParams | None = None


@dataclass
class HollowSettings:
    """The settings window: values the next Hollow / Add internal supports
    (and new holes) use."""
    hole_diameter: float = 3.0
    wall_thickness: float = 2.0
    lattice_angle_x: float = 0.0
    lattice_angle_y: float = 45.0
    lattice_angle_z: float = 0.0
    lattice_spacing: float = 8.0
    pole_diameter: float = 1.2

    def lattice(self) -> LatticeParams:
        return LatticeParams(self.lattice_angle_x, self.lattice_angle_y, self.lattice_angle_z,
                             self.lattice_spacing, self.pole_diameter)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HollowSettings":
        d = dict(d or {})
        if "lattice_angle" in d and "lattice_angle_y" not in d:      # one angle (about Y) before
            d["lattice_angle_y"] = d["lattice_angle"]
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def describe(state: HollowState | None, holes: int) -> str:
    """'hollow 2.0 mm · 2 holes · 3-axis poles turned 0/45/0°, 8 mm'"""
    s = "s" if holes != 1 else ""
    if state is None:
        return f"solid · {holes} hole{s}" if holes else "solid"
    text = f"hollow {state.thickness:.1f} mm · {holes} hole{s}"
    lat = state.lattice
    if lat is not None:
        text += f" · 3-axis poles turned {lat.angle_x:g}/{lat.angle_y:g}/{lat.angle_z:g}°, {lat.spacing:g} mm"
    return text


# --------------------------------------------------------------------------
# model space -> world

def hole_world(matrix: np.ndarray, hole: Hole) -> tuple[np.ndarray, np.ndarray]:
    """(centre on the surface, outward unit normal) in world space."""
    m = np.asarray(matrix, float)
    center = (m @ np.r_[np.asarray(hole.point, float), 1.0])[:3]
    n = np.linalg.inv(m[:3, :3]).T @ np.asarray(hole.normal, float)
    return center, n / max(np.linalg.norm(n), 1e-12)


def hole_from_world(matrix: np.ndarray, point, normal, diameter: float) -> Hole:
    """A hole clicked at ``point`` (world) on a face with outward ``normal``."""
    m = np.asarray(matrix, float)
    p = np.linalg.solve(m, np.r_[np.asarray(point, float), 1.0])[:3]
    n = m[:3, :3].T @ np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    return Hole(tuple(float(v) for v in p), tuple(float(v) for v in n), float(diameter))


def hole_depth(thickness: float) -> float:
    """How far a drain hole reaches below the surface: through the wall,
    into the cavity, also where the surface curves."""
    return 1.5 * thickness + 0.5


def hole_segments(matrix: np.ndarray, holes, depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(start, end, radius) per hole, world space: from just outside the
    surface inward along the normal."""
    if not holes:
        z = np.zeros((0, 3))
        return z, z, np.zeros(0)
    starts, ends, radii = [], [], []
    for h in holes:
        c, n = hole_world(matrix, h)
        starts.append(c + n * HOLE_OUTSIDE)
        ends.append(c - n * depth)
        radii.append(h.diameter / 2)
    return np.array(starts), np.array(ends), np.array(radii)


# --------------------------------------------------------------------------
# the lattice

def lattice_rotation(angle_x: float, angle_y: float = 0.0, angle_z: float = 0.0) -> np.ndarray:
    """The grid's X, Y and Z pole directions (rows), turned about X, then Y,
    then Z (R = Rz·Ry·Rx, the order a model's rotation uses).  About Y
    alone: X tilts down to the right, Z leans right."""
    cx, sx = math.cos(math.radians(angle_x)), math.sin(math.radians(angle_x))
    cy, sy = math.cos(math.radians(angle_y)), math.sin(math.radians(angle_y))
    cz, sz = math.cos(math.radians(angle_z)), math.sin(math.radians(angle_z))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx).T                                               # rows: the turned X, Y, Z


def lattice_segments(lo, hi, lat: LatticeParams, origin=None) -> tuple[np.ndarray, np.ndarray]:
    """Pole axes clipped to the box [lo, hi], as (start, end) arrays.

    Nodes: origin + s·(i·X + j·Y + k·Z) with X, Y, Z the turned grid
    directions (``lattice_rotation``) and s the spacing.  X poles join
    nodes along i, Y poles along j, Z poles along k - three sets of
    poles at right angles to each other, meeting at every node."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    origin = (lo + hi) / 2 if origin is None else np.asarray(origin, float)
    s = float(lat.spacing)
    axes = lattice_rotation(*lat.angles)                                   # rows: X, Y, Z
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    coords = (corners - origin) @ axes.T / s                               # grid coordinates (orthonormal axes)
    cmin, cmax = np.floor(coords.min(axis=0)) - 1, np.ceil(coords.max(axis=0)) + 1
    starts, ends = [], []
    for m in range(3):
        a, b = [k for k in range(3) if k != m]
        ga, gb = np.meshgrid(np.arange(cmin[a], cmax[a] + 1), np.arange(cmin[b], cmax[b] + 1), indexing="ij")
        base = origin + s * (ga.reshape(-1, 1) * axes[a] + gb.reshape(-1, 1) * axes[b])
        p0, p1 = _clip_lines(base, axes[m], lo, hi)
        starts.append(p0)
        ends.append(p1)
    return np.concatenate(starts), np.concatenate(ends)


def _clip_lines(points: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Clip the lines points + t·d to the box (slab test)."""
    t0 = np.full(len(points), -np.inf)
    t1 = np.full(len(points), np.inf)
    ok = np.ones(len(points), bool)
    for ax in range(3):
        if abs(d[ax]) < 1e-12:
            ok &= (points[:, ax] >= lo[ax]) & (points[:, ax] <= hi[ax])
            continue
        a = (lo[ax] - points[:, ax]) / d[ax]
        b = (hi[ax] - points[:, ax]) / d[ax]
        t0 = np.maximum(t0, np.minimum(a, b))
        t1 = np.minimum(t1, np.maximum(a, b))
    ok &= t1 > t0 + 1e-9
    p = points[ok]
    return p + t0[ok, None] * d, p + t1[ok, None] * d


# --------------------------------------------------------------------------
# cross-sections of cylinders (poles, holes)

_T = (1 - np.cos(np.pi * np.arange(SECTION_POINTS) / (SECTION_POINTS - 1))) / 2    # denser near the ends


def cylinder_sections(p0: np.ndarray, p1: np.ndarray, radius, z: float) -> np.ndarray:
    """The cross-sections at height z of cylinders (flat ends) from p0 to p1
    - an array of shapely Polygons, only those that are not empty.

    In the layer, measure ``a`` along the axis's horizontal direction from
    p0 and ``v`` across it.  A point is within the radius of the axis when
    (a·e - w·c)² + v² <= r² (c, e: horizontal and vertical part of the axis
    direction, w: height above p0), and between the two ends when
    0 <= a·c + w·e <= length.  For every ``a`` in the allowed range, ``v``
    runs over ±sqrt(r² - (a·e - w·c)²) - exact for any tilt, horizontal
    and vertical included."""
    p0 = np.asarray(p0, float).reshape(-1, 3)
    p1 = np.asarray(p1, float).reshape(-1, 3)
    r = np.broadcast_to(np.asarray(radius, float), (len(p0),))
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    good = length > 1e-9
    p0, d, length, r = p0[good], d[good], length[good], r[good]
    if not len(p0):
        return np.array([], dtype=object)
    dn = d / length[:, None]
    c = np.hypot(dn[:, 0], dn[:, 1])
    e = dn[:, 2]
    flat = c > 1e-9
    eu = np.where(flat[:, None], dn[:, :2] / np.where(flat, c, 1.0)[:, None], np.array([1.0, 0.0]))
    ev = np.stack([-eu[:, 1], eu[:, 0]], axis=1)
    w = z - p0[:, 2]
    wc = w * c
    inf = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        steep = np.abs(e) > 1e-9
        ca = np.where(steep, (wc - r) / e, np.where(np.abs(wc) <= r, -inf, inf))
        cb = np.where(steep, (wc + r) / e, np.where(np.abs(wc) <= r, inf, -inf))
        cyl_lo, cyl_hi = np.minimum(ca, cb), np.maximum(ca, cb)
        we = w * e
        inside_ends = (we >= 0) & (we <= length)
        cap_lo = np.where(flat, -we / np.where(flat, c, 1.0), np.where(inside_ends, -inf, inf))
        cap_hi = np.where(flat, (length - we) / np.where(flat, c, 1.0), np.where(inside_ends, inf, -inf))
    a_lo, a_hi = np.maximum(cyl_lo, cap_lo), np.minimum(cyl_hi, cap_hi)
    ok = np.isfinite(a_lo) & np.isfinite(a_hi) & (a_hi > a_lo + 1e-9)
    if not ok.any():
        return np.array([], dtype=object)
    a_lo, a_hi, e, wc, r = a_lo[ok], a_hi[ok], e[ok], wc[ok], r[ok]
    A = a_lo[:, None] + (a_hi - a_lo)[:, None] * _T[None, :]
    g = np.sqrt(np.maximum(r[:, None] ** 2 - (A * e[:, None] - wc[:, None]) ** 2, 0.0))
    a_all = np.concatenate([A, A[:, ::-1]], axis=1)
    v_all = np.concatenate([g, -g[:, ::-1]], axis=1)
    xy = (p0[ok, None, :2] + a_all[..., None] * eu[ok, None, :] + v_all[..., None] * ev[ok, None, :])
    polys = shapely.polygons(xy)
    return polys[~shapely.is_empty(polys) & (shapely.area(polys) > 1e-9)]


# --------------------------------------------------------------------------
# per layer

def _safe(op, a, b):
    """A shapely set operation that survives slightly broken input."""
    try:
        return op(a, b)
    except shapely.errors.GEOSException:
        try:
            return op(shapely.make_valid(a), shapely.make_valid(b))
        except shapely.errors.GEOSException:
            return op(a, b, grid_size=1e-6)


def _polys(geom) -> MultiPolygon:
    if geom is None or geom.is_empty:
        return MultiPolygon()
    parts = [p for p in shapely.get_parts(geom) if isinstance(p, Polygon) and not p.is_empty and p.area > 1e-10]
    return MultiPolygon(parts) if parts else MultiPolygon()


def cavities(solids: list, zc: np.ndarray, thickness: float,
             progress: ProgressFn | None = None) -> list:
    """The cavity of every layer: ``solids[i]`` (a layer of one model, at
    centre height ``zc[i]``) eroded by a ball of radius ``thickness``, i.e.
    everything at least that far from the model's surface in 3D."""
    n = len(solids)
    t = thickness / math.cos(math.pi / (4 * ERODE_STEPS))        # walls >= thickness between samples
    angles = [k * math.pi / (2 * ERODE_STEPS) for k in range(ERODE_STEPS + 1)]
    rings = [(t * math.cos(a), t * math.sin(a)) for a in angles]   # (dz, radius); k = 0: dz = t, radius 0
    shrunk: dict[tuple[int, int], object] = {}

    def eroded(j: int, k: int):
        if (j, k) not in shrunk:
            r = rings[k][1]
            g = solids[j]
            shrunk[(j, k)] = g if r < 1e-9 else shapely.buffer(g, -r, quad_segs=8)
        return shrunk[(j, k)]

    out = []
    for i in range(n):
        acc = None
        if solids[i] is not None and not solids[i].is_empty:
            acc = eroded(i, ERODE_STEPS)                            # this layer, shrunk by t
            for k in range(ERODE_STEPS - 1, -1, -1):                # then above and below, nearest first
                dz = rings[k][0]
                for sign in (1, -1):
                    if acc is None or acc.is_empty:
                        break
                    zt = zc[i] + sign * dz
                    if zt < zc[0] - 1e-9 or zt > zc[-1] + 1e-9:
                        acc = None                                  # above the top / below the bottom
                        break
                    j = int(np.clip(np.searchsorted(zc, zt), 1, n - 1))
                    j = j - 1 if abs(zc[j - 1] - zt) <= abs(zc[j] - zt) else j
                    g = solids[j]
                    if g is None or g.is_empty:
                        acc = None
                        break
                    acc = _safe(shapely.intersection, acc, eroded(j, k))
        out.append(None if acc is None or acc.is_empty else acc)
        below = zc[i] - t - 1e-6                                    # never needed again: drop
        for key in [key for key in shrunk if zc[key[0]] < below]:
            del shrunk[key]
        if progress:
            progress(i + 1, n)
    return out


HELD_OVERHANG = 60.0                     # deg: like the slicer's island check, a pole piece is held when it ...
HELD_TOLERANCE = 0.05                    # ... comes this close to held material one layer down (mm, at least)


def hollow_layers(solids: list, zc: np.ndarray, state: HollowState,
                  holes: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                  poles: tuple[np.ndarray, np.ndarray] | None,
                  progress: ProgressFn | None = None) -> list[MultiPolygon]:
    """One model's layers after hollowing: the solid minus its cavity, plus
    the poles that pass through the cavity, minus the drain holes.

    Poles never add islands: going up layer by layer, a piece of pole is
    only kept when it touches the wall in the same layer or what was kept
    one layer down.  A pole cut by a drain hole just under the ceiling, for
    example, would otherwise start in mid-air - that piece is left out.  The
    model's own walls are never removed."""
    caves = cavities(solids, zc, state.thickness, progress)
    pole_r = state.lattice.diameter / 2 if (state.lattice is not None and poles is not None) else 0.0
    if pole_r > 0 and len(poles[0]):
        pz0 = np.minimum(poles[0][:, 2], poles[1][:, 2]) - pole_r
        pz1 = np.maximum(poles[0][:, 2], poles[1][:, 2]) + pole_r
    else:
        pole_r = 0.0
    if holes is not None and len(holes[0]):
        hz0 = np.minimum(holes[0][:, 2], holes[1][:, 2]) - holes[2]
        hz1 = np.maximum(holes[0][:, 2], holes[1][:, 2]) + holes[2]
    reach = math.tan(math.radians(HELD_OVERHANG))
    out = []
    below = None                                         # this model's material one layer down
    for i, (solid, cave) in enumerate(zip(solids, caves)):
        if solid is None or solid.is_empty:
            out.append(MultiPolygon())
            below = None
            continue
        z = float(zc[i])
        shell = solid if cave is None else _safe(shapely.difference, solid, cave)
        bars = None                                      # pole cross-sections inside the cavity
        if cave is not None and pole_r > 0:
            near = (pz0 <= z) & (pz1 >= z)
            sec = cylinder_sections(poles[0][near], poles[1][near], pole_r, z)
            if len(sec):
                shapely.prepare(cave)
                sec = sec[shapely.intersects(cave, sec)]
                if len(sec):
                    bars = _safe(shapely.intersection, cave, unary_union(sec))
        if holes is not None and len(holes[0]):
            near = (hz0 <= z) & (hz1 >= z)
            if near.any():
                sec = cylinder_sections(holes[0][near], holes[1][near], holes[2][near], z)
                if len(sec):
                    cut = unary_union(sec)
                    shell = _safe(shapely.difference, shell, cut)
                    if bars is not None:
                        bars = _safe(shapely.difference, bars, cut)
        geom = shell
        if bars is not None and not bars.is_empty:
            pieces = np.array([p for p in shapely.get_parts(bars) if isinstance(p, Polygon) and not p.is_empty],
                              dtype=object)
            if len(pieces):
                thickness = float(zc[i] - zc[i - 1]) if i > 0 else 0.05
                tol = max(HELD_TOLERANCE, thickness * reach)
                keep = np.zeros(len(pieces), bool)
                if not shell.is_empty:
                    keep |= shapely.dwithin(pieces, shell, 1e-6)
                if below is not None and not below.is_empty:
                    keep |= shapely.dwithin(pieces, below, tol)
                if keep.any():
                    geom = _safe(shapely.union, shell, unary_union(pieces[keep]))
        geom = _polys(geom)
        out.append(geom)
        below = geom
        if below.geoms:
            shapely.prepare(below)
    return out


# --------------------------------------------------------------------------
# for the 3D view

def tubes(p0: np.ndarray, p1: np.ndarray, radius, sides: int = 10) -> trimesh.Trimesh:
    """Closed tubes (cylinders with flat ends) along segments, one mesh."""
    p0 = np.asarray(p0, float).reshape(-1, 3)
    p1 = np.asarray(p1, float).reshape(-1, 3)
    r = np.broadcast_to(np.asarray(radius, float), (len(p0),))
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    keep = length > 1e-9
    p0, p1, d, length, r = p0[keep], p1[keep], d[keep], length[keep], r[keep]
    n = len(p0)
    if not n:
        return trimesh.Trimesh()
    axis = d / length[:, None]
    helper = np.where(np.abs(axis[:, 2:3]) < 0.9, np.array([0.0, 0, 1]), np.array([1.0, 0, 0]))
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(axis, u)
    ang = 2 * np.pi * np.arange(sides) / sides
    ring = (np.cos(ang)[None, :, None] * u[:, None, :] + np.sin(ang)[None, :, None] * v[:, None, :]) * r[:, None, None]
    bottom, top = p0[:, None, :] + ring, p1[:, None, :] + ring
    verts = np.concatenate([bottom, top, p0[:, None, :], p1[:, None, :]], axis=1)    # (n, 2s + 2, 3)
    k = np.arange(sides)
    k1 = (k + 1) % sides
    b0, t0 = k, sides + k
    b1, t1 = k1, sides + k1
    cb, ct = 2 * sides, 2 * sides + 1
    local = np.concatenate([np.stack([b0, b1, t1], 1), np.stack([b0, t1, t0], 1),
                            np.stack([np.full(sides, cb), b1, b0], 1), np.stack([np.full(sides, ct), t0, t1], 1)])
    offset = (np.arange(n) * (2 * sides + 2))[:, None, None]
    faces = (local[None, :, :] + offset).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts.reshape(-1, 3), faces=faces, process=False)


def cavity_clip(mesh: trimesh.Trimesh, thickness: float, p0: np.ndarray, p1: np.ndarray,
                into_wall: float | None = None, max_cells: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """The parts of segments that run through the cavity of ``mesh`` (world
    space) hollowed to ``thickness``, each lengthened a little into the wall
    - for showing the internal supports.  Uses a voxel depth field (how far
    inside the model each cell is); an approximation the slicer does not use."""
    from .clearance import _occupancy, _distance, MIN_CELL, MAX_CELL
    lo, hi = np.asarray(mesh.bounds, float)
    if not len(p0):
        return p0, p1
    cell = min(max((np.prod(hi - lo + 1.0) / max_cells) ** (1 / 3), MIN_CELL, thickness / 4), MAX_CELL)
    origin = lo - 1.5 * cell
    shape = tuple(int(v) for v in np.ceil((hi - lo + 3 * cell) / cell))
    inside = _occupancy(mesh, origin, cell, shape)
    depth = _distance(~inside, cell, thickness + 2 * cell)                     # inside: distance to outside
    cave = inside & (depth >= thickness)
    into = thickness * 0.6 if into_wall is None else into_wall
    out0, out1 = [], []
    for a, b in zip(np.asarray(p0, float), np.asarray(p1, float)):
        length = float(np.linalg.norm(b - a))
        steps = max(int(length / (cell / 2)), 2)
        t = np.linspace(0.0, 1.0, steps)
        pts = a + (b - a) * t[:, None]
        idx = np.floor((pts - origin) / cell).astype(int)
        valid = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        inc = np.zeros(steps, bool)
        inc[valid] = cave[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
        if not inc.any():
            continue
        edges = np.diff(np.r_[0, inc.astype(np.int8), 0])
        for s0, s1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            u0 = max(t[s0] - into / length, 0.0)
            u1 = min(t[s1] + into / length, 1.0)
            out0.append(a + (b - a) * u0)
            out1.append(a + (b - a) * u1)
    if not out0:
        z = np.zeros((0, 3))
        return z, z
    return np.array(out0), np.array(out1)
