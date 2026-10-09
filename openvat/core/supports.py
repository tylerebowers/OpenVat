"""Support structures: what a support is and how it is built.

Two kinds of support:

**plate** - runs from the build plate up to a *contact point* on the part.
From bottom to top:

    foot      a short wide cylinder on the plate (or the raft)
    pillar    a vertical rod
    detour    tilted rods around parts of the model that are in the way
              (``Support.route``: the joints below the pivot)
    bulb      a sphere as wide as the rod at every joint, so tilted parts
              meet without gaps; rods and arms pivot about its center
    arm       a cone from the pivot (the "socket") into the contact point.
              Straight up when the face is (nearly) flat; otherwise it
              meets the face perpendicularly, tilted at most
              ``max_arm_angle``.  Its tip reaches ``tip_depth`` into the part.

Pillars can **split**: nearby contacts share one trunk that rises to a split
bulb and branches out to each contact's pivot (``Support.trunk``).

**nook** - a small pillar for caves and other pockets with no way down to
the plate: it stands on the part itself (``Support.foot``).  A thin rod
(``nook_diameter``) between two small contact cones
(``nook_contact_diameter``), straight from the contact to the floor below,
so it leaves small marks on both surfaces.

Where supports go and which kind they are is decided in placement.py.  Also
here: ``generate_braces`` (X braces between neighboring pillars) and
``raft_mesh`` (one perforated plate under each object's supports).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, fields
from itertools import count

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

from .clearance import ClearanceField

_ids = count(1)

DENSITY_SPACING = {"low": 7.0, "medium": 4.5, "high": 3.0}   # mm between contact points


@dataclass
class SupportSettings:
    # -- base: the foot of each pillar and the raft under everything
    z_lift: bool = False                # Automatic raises the model this high first
    z_lift_height: float = 5.0          # height of the model's lowest point (mm)
    base_height: float = 0.6            # foot / raft thickness (mm)
    base_diameter: float = 3.0          # foot diameter (mm)
    raft: bool = True                   # one plate under each object's supports
    raft_margin: float = 2.0            # how far the raft extends past the outer feet (mm)
    perforations: bool = True           # honeycomb holes in the raft
    perforation_size: float = 2.0       # across-flats size of each hex hole (mm)
    perforation_spacing: float = 3.5    # hex lattice pitch (mm)

    # -- pillars, branches and X braces
    pillar_diameter: float = 1.6        # also the diameter of branches, detours and bulbs
    branches: bool = True               # let pillars split to reach several contacts
    branch_max_count: int = 3           # contacts per split pillar
    branch_max_distance: float = 6.0    # contacts closer than this may share a pillar (mm)
    branch_angle: float = 30.0          # max tilt of branches and detours from vertical (deg)
    braces: bool = True                 # X-shaped braces between neighboring pillars
    brace_diameter: float = 0.8
    brace_max_distance: float = 10.0    # only brace pillars closer than this (mm)
    brace_angle: float = 45.0           # angle of each diagonal above horizontal (deg)

    # -- paths and nooks
    routing: bool = True                # route pillars around the part to reach the plate
    model_clearance: float = 0.4        # gap kept between rods and the part (mm)
    nook_diameter: float = 1.0          # nook supports (standing on the part): rod diameter...
    nook_contact_diameter: float = 0.4  # ...and the diameter of both contacts (mm)
    nook_max_gap: float = 3.0           # narrower gaps (grooves, slots) get a nook straight across (mm)
    max_bridge: float = 3.0             # ceilings spanning less than this between two walls need no support (mm)

    # -- interface: the part that touches the model
    tip_depth: float = 0.3              # distance the tip reaches into the part (mm)
    tip_diameter: float = 0.5           # contact diameter (mm); 0 = automatic
    tip_length: float = 2.5             # vertical length of the arm (mm)
    straight_below: float = 10.0        # faces tilted less than this get a straight arm (deg)
    max_arm_angle: float = 45.0         # arms meet tilted faces perpendicularly, up to this (deg)

    # -- other
    density: str = "medium"             # low / medium / high  -> contact point spacing
    overhang_angle: float = 45.0        # surfaces tilted more than this from vertical get supports
    min_support_height: float = 0.3     # ignore contact points lower than this (mm)
    island_anchors: int = 4             # supports at the lowest point of each island
    contact_min: float = 0.4            # automatic tip diameter for short supports (mm)
    contact_max: float = 0.8            # automatic tip diameter for tall supports (mm)
    contact_scale_height: float = 30.0  # support height at which contact_max is reached (mm)

    @property
    def spacing(self) -> float:
        return DENSITY_SPACING.get(self.density, 4.5)

    def contact_diameter(self, height: float) -> float:
        if self.tip_diameter > 0:
            return self.tip_diameter
        t = min(max(height / max(self.contact_scale_height, 1e-6), 0.0), 1.0)
        return self.contact_min + (self.contact_max - self.contact_min) * t

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SupportSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


Trunk = tuple  # (x, y, split_z, base_z): a shared pillar that splits at split_z


@dataclass
class Support:
    tip: np.ndarray                      # contact point on the part (world)
    owner_id: int | None = None          # id of the MeshObject it supports
    kind: str = "plate"                  # "plate" (to the build plate) or "nook" (stands on the part)
    lean: np.ndarray = field(default_factory=lambda: np.zeros(2))  # unit XY direction of the arm
    angle: float = 0.0                   # arm tilt from vertical (deg)
    trunk: Trunk | None = None           # set when this contact branches off a shared pillar
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))  # face at the contact
    stretch: float = 1.0                 # arm length factor (longer when that clears the part)
    route: tuple = ()                    # detour joints below the pivot, top to bottom
    foot: tuple | None = None            # nook: where it stands on the part (x, y, z)
    width: float = 0.0                   # nook: rod diameter when thinned to fit (0 = the setting)
    status: str = ""                     # how placement went (shown in the UI)
    id: int = field(default_factory=lambda: next(_ids))

    # everything placement decides (undo and failed drags restore these)
    STATE = ("tip", "kind", "lean", "angle", "trunk", "normal", "stretch", "route", "foot", "width", "status")

    def state(self) -> dict:
        return {k: _copy(getattr(self, k)) for k in self.STATE}

    def restore(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, _copy(v))

    def move_xy(self, dx: float, dy: float) -> None:
        self.tip[0] += dx
        self.tip[1] += dy

    # ------------------------------------------------------------ geometry
    @property
    def is_nook(self) -> bool:
        return self.kind == "nook" and self.foot is not None

    @property
    def base_z(self) -> float:
        """Height the support stands on: 0 = the plate, else the part (nook)."""
        return float(self.foot[2]) if self.is_nook else 0.0

    def radius(self, s: SupportSettings) -> float:
        return (self.width or s.nook_diameter if self.is_nook else s.pillar_diameter) / 2

    def bottom_z(self, s: SupportSettings) -> float:
        """Where the vertical pillar starts (the top of the foot)."""
        return s.base_height

    def arm_axis(self) -> np.ndarray:
        """Unit vector from the contact point to the pivot."""
        t = math.radians(self.angle)
        return np.array([self.lean[0] * math.sin(t), self.lean[1] * math.sin(t), -math.cos(t)])

    def pivot(self, s: SupportSettings) -> np.ndarray:
        """Where the arm meets its pillar, detour or branch (a bulb center)."""
        drop = s.tip_length * self.stretch
        xy = self.tip[:2] + self.lean * math.tan(math.radians(min(self.angle, 89.0))) * drop
        z = float(self.tip[2]) - drop
        if not self.route and self.trunk is None:
            z = max(z, self.bottom_z(s))
        return np.array([xy[0], xy[1], z])

    def pillar_xy(self, s: SupportSettings) -> np.ndarray:
        """XY of the vertical pillar (below the detour, if there is one)."""
        return np.array(self.route[-1][:2], float) if self.route else self.pivot(s)[:2]

    def pillar_top_z(self, s: SupportSettings) -> float:
        return float(self.route[-1][2]) if self.route else float(self.pivot(s)[2])

    def to_mesh(self, s: SupportSettings | None = None) -> trimesh.Trimesh:
        """This contact's own geometry.  Plate: its pillar and detour (or its
        branch off a shared trunk), the pivot bulb and the arm - trunks are
        built by ``trunk_mesh``.  Nook: the whole small pillar."""
        s = s or SupportSettings()
        if self.is_nook:
            return _nook_mesh(np.asarray(self.tip, float), np.asarray(self.foot, float), 2 * self.radius(s), s)
        r = self.radius(s)
        p = self.pivot(s)
        parts = []
        if self.trunk is not None:
            parts.append(_tube(self.trunk[:3], p, r, r))                       # branch
        else:
            joints = [p, *(np.asarray(j, float) for j in self.route)]
            for a, b in zip(joints, joints[1:]):                               # detour
                parts += [_tube(a, b, r, r), _bulb(b, r)]
            parts += _column_parts(self.pillar_xy(s), self.pillar_top_z(s), s)
        parts.append(_bulb(p, r))
        tip_d = min(s.contact_diameter(float(self.tip[2])), 2 * r)
        into = np.asarray(self.tip, float) - self.arm_axis() * s.tip_depth   # arm, into the contact
        parts.append(_tube(p, into, r, tip_d / 2))
        return trimesh.util.concatenate([m for m in parts if len(m.faces)])


def _copy(v):
    return v.copy() if isinstance(v, np.ndarray) else v


def nook_cone_length(length: float, s: SupportSettings) -> float:
    """Length of each contact cone of a nook support ``length`` mm long."""
    return min(s.tip_length * 0.5, 0.35 * length)


def _nook_mesh(tip: np.ndarray, foot: np.ndarray, width: float, s: SupportSettings) -> trimesh.Trimesh:
    """Contact cone, thin rod, contact cone - straight from the contact down to the foot."""
    axis = foot - tip
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return trimesh.Trimesh()
    axis /= length
    r, rc = width / 2, s.nook_contact_diameter / 2
    cone = nook_cone_length(length, s)
    top, bottom = tip + axis * cone, foot - axis * cone
    parts = [_tube(tip - axis * s.tip_depth, top, rc, r)]
    if length > 2 * cone + 1e-6:
        parts.append(_tube(top, bottom, r, r))
    parts.append(_tube(bottom, foot + axis * s.tip_depth, r, rc))
    return trimesh.util.concatenate([m for m in parts if len(m.faces)])


def _column_parts(xy, top_z: float, s: SupportSettings) -> list:
    """Foot on the plate and the vertical rod up to top_z."""
    x, y = float(xy[0]), float(xy[1])
    r = s.pillar_diameter / 2
    parts = []
    if s.base_height > 0:
        parts.append(_tube([x, y, 0.0], [x, y, s.base_height], s.base_diameter / 2, s.base_diameter / 2))
    if top_z > s.base_height:
        parts.append(_tube([x, y, s.base_height], [x, y, top_z], r, r))
    return parts


def trunk_mesh(trunk: Trunk, s: SupportSettings) -> trimesh.Trimesh:
    """A shared pillar from the plate up to the split bulb."""
    x, y, z, _base = trunk
    parts = _column_parts((x, y), z, s) + [_bulb([x, y, z], s.pillar_diameter / 2)]
    return trimesh.util.concatenate([m for m in parts if len(m.faces)])


def columns(supports: list["Support"], s: SupportSettings) -> list[tuple]:
    """Every vertical plate pillar once: (xy, bottom_z, top_z, base_z, owner_id).
    Shared trunks count once (their top is the split point); nook supports
    are not columns."""
    out, seen = [], set()
    for sup in supports:
        if sup.is_nook:
            continue
        if sup.trunk is not None:
            if sup.trunk in seen:
                continue
            seen.add(sup.trunk)
            x, y, z, _b = sup.trunk
            out.append((np.array([x, y]), s.base_height, z, 0.0, sup.owner_id))
        else:
            out.append((sup.pillar_xy(s), s.base_height, sup.pillar_top_z(s), 0.0, sup.owner_id))
    return out


def arm_for_normal(normal: np.ndarray, max_angle: float, straight_below: float = 10.0
                   ) -> tuple[np.ndarray, float]:
    """Arm for a contact on a face with the given outward normal.

    Straight up when the face is (nearly) flat - tilted less than
    ``straight_below`` from horizontal.  Otherwise the arm leaves the face
    perpendicularly (along the normal), tilted at most ``max_angle``."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    horiz = np.hypot(n[0], n[1])
    tilt = math.degrees(math.atan2(horiz, max(-n[2], 1e-9)))     # face tilt = normal's angle from down
    if horiz < 1e-6 or max_angle <= 0 or tilt < straight_below:
        return np.zeros(2), 0.0
    return n[:2] / horiz, float(min(tilt, max_angle))


def face_tilt(normal) -> float:
    """Degrees a downward face is tilted from horizontal (0 = flat ceiling)."""
    n = np.asarray(normal, float)
    return math.degrees(math.atan2(np.hypot(n[0], n[1]), max(-n[2], 1e-9)))


# --------------------------------------------------------------------------
# geometry helpers

def _tube(p0, p1, r0: float, r1: float, sections: int = 20) -> trimesh.Trimesh:
    """A closed frustum from p0 (radius r0) to p1 (radius r1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return trimesh.Trimesh()
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    circle = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(sections)])
    verts = np.vstack([circle * r0, circle * r1 + [0, 0, length], [[0, 0, 0]], [[0, 0, length]]])
    cb, ct = 2 * sections, 2 * sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[i, j, sections + j], [i, sections + j, sections + i], [cb, j, i], [ct, sections + i, sections + j]]
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    m.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis / length))
    m.apply_translation(p0)
    return m


def hex_lattice(xmin: float, xmax: float, ymin: float, ymax: float, pitch: float) -> np.ndarray:
    """Points of a hexagonal lattice covering the rectangle (N, 2).  The
    lattice is anchored at the origin so every layer uses the same points."""
    dy = pitch * math.sqrt(3) / 2
    rows = np.arange(math.floor(ymin / dy), math.ceil(ymax / dy) + 1)
    pts = []
    for i in rows:
        offset = pitch / 2 if i % 2 else 0.0
        xs = np.arange(math.floor((xmin - offset) / pitch), math.ceil((xmax - offset) / pitch) + 1) * pitch + offset
        pts.append(np.column_stack([xs, np.full_like(xs, i * dy)]))
    return np.vstack(pts) if pts else np.empty((0, 2))


def _hexagon(cx: float, cy: float, across_flats: float) -> Polygon:
    r = across_flats / math.sqrt(3)         # circumradius
    ang = np.radians(np.arange(0, 360, 60) + 30)
    return Polygon(np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)]))


def _polys(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    return [g for g in shapely.get_parts(geom) if isinstance(g, Polygon) and not g.is_empty]


def _valid(geom):
    """Only valid polygons.  Growing and shrinking thin slivers (the overhang
    bookkeeping below) can leave rings that touch themselves, which later set
    operations refuse (GEOS 'non-noded intersection')."""
    if geom.is_empty or geom.is_valid:
        return geom
    parts = shapely.get_parts(shapely.get_parts(shapely.make_valid(geom)))     # flatten collections
    polys = [p for p in parts if isinstance(p, Polygon) and p.area > 1e-9]
    return unary_union(polys) if polys else Polygon()


def _safe(op, a, b):
    """A shapely set operation (``shapely.union`` / ``shapely.difference``)
    that survives slightly broken input: retried on repaired geometry, then
    with coordinates snapped to a fine grid."""
    try:
        return op(a, b)
    except shapely.errors.GEOSException:
        a, b = _valid(a), _valid(b)
        try:
            return op(a, b)
        except shapely.errors.GEOSException:
            return _valid(op(a, b, grid_size=1e-6))


def _bulb(center, r: float) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=1, radius=r)
    m.apply_translation(np.asarray(center, float))
    return m


# --------------------------------------------------------------------------
# braces and raft

def _brace_pairs(xy: np.ndarray, max_dist: float, neighbors: int = 2) -> set[tuple[int, int]]:
    """Each column is braced to its nearest neighbors (up to ``neighbors``)."""
    pairs = set()
    for i in range(len(xy)):
        d = np.linalg.norm(xy - xy[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:neighbors]:
            if d[j] <= max_dist:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def generate_braces(supports: list[Support], settings: SupportSettings,
                    fields: dict[int, ClearanceField] | None = None) -> trimesh.Trimesh:
    """X braces between neighboring columns: two diagonals crossing between
    each pair, stacked in panels whose diagonals rise at ``brace_angle``.
    With ``fields`` (owner id -> clearance field) a panel whose diagonals
    would pass through the model is left out."""
    cols = columns(supports, settings)
    if not settings.braces or len(cols) < 2:
        return trimesh.Trimesh()
    xy = np.array([c[0] for c in cols])
    bottoms = np.array([c[1] for c in cols])
    tops = np.array([c[2] for c in cols])
    owners = [c[4] for c in cols]
    r = settings.brace_diameter / 2
    rise = math.tan(math.radians(min(max(settings.brace_angle, 10.0), 80.0)))
    parts = []
    for i, j in _brace_pairs(xy, settings.brace_max_distance):
        dist = float(np.linalg.norm(xy[i] - xy[j]))
        z0 = max(bottoms[i], bottoms[j]) + 0.5
        z1 = min(tops[i], tops[j]) - 0.5
        if dist < 1e-6 or z1 - z0 < dist * rise * 0.5:      # too short for a sensible X
            continue
        checks = [fields[o] for o in {owners[i], owners[j]} if fields and o in fields]
        panels = max(1, round((z1 - z0) / (dist * rise)))
        h = (z1 - z0) / panels
        for k in range(panels):
            za, zb = z0 + k * h, z0 + (k + 1) * h
            diagonals = (([*xy[i], za], [*xy[j], zb]), ([*xy[j], za], [*xy[i], zb]))
            if any(not f.segment_clear(a, b, r + settings.model_clearance) for f in checks for a, b in diagonals):
                continue
            parts += [_tube(a, b, r, r, sections=10) for a, b in diagonals]
    return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()


def raft_polygon(supports: list[Support], settings: SupportSettings,
                 keep_out=None) -> Polygon | MultiPolygon | None:
    """One plate under each object's plate-standing columns: the convex
    hull of their feet plus ``raft_margin``, optionally perforated.
    ``keep_out`` (the footprint of models standing on the plate) is cut
    away with ``model_clearance`` around it."""
    if not settings.raft:
        return None
    groups: dict = {}
    for xy, _b, _t, base_z, owner in columns(supports, settings):
        if base_z <= 0:
            groups.setdefault(owner, []).append(xy)
    plates, feet = [], []
    for pts in groups.values():
        hull = MultiPoint([tuple(p) for p in pts]).convex_hull
        plates.append(hull.buffer(settings.base_diameter / 2 + settings.raft_margin))
        feet.extend(pts)
    if not plates:
        return None
    plate = _valid(unary_union(plates))
    if settings.perforations and settings.perforation_size > 0:
        x0, y0, x1, y1 = plate.bounds
        wall = max(settings.perforation_spacing - settings.perforation_size, 0.5)
        keep_clear = (settings.base_diameter + settings.perforation_size) / 2 + 0.3
        feet_arr = np.array(feet)
        holes = [_hexagon(cx, cy, settings.perforation_size)
                 for cx, cy in hex_lattice(x0, x1, y0, y1, settings.perforation_spacing)
                 if np.min(np.hypot(feet_arr[:, 0] - cx, feet_arr[:, 1] - cy)) >= keep_clear]
        if holes:
            rim = _safe(shapely.difference, plate, plate.buffer(-wall))           # solid outer rim
            plate = _valid(_safe(shapely.union, _safe(shapely.difference, plate, unary_union(holes)), rim).buffer(0))
    if keep_out is not None and not keep_out.is_empty:
        plate = _safe(shapely.difference, plate, _valid(keep_out).buffer(settings.model_clearance + 0.5))
    return plate if not plate.is_empty else None


def raft_mesh(supports: list[Support], settings: SupportSettings, keep_out=None) -> trimesh.Trimesh:
    poly = raft_polygon(supports, settings, keep_out)
    if poly is None:
        return trimesh.Trimesh()
    # buffer/union leave near-duplicate vertices that break the extrusion
    # (non-watertight, falls apart); a light simplify removes them
    parts = []
    for g in _polys(poly.simplify(0.01)):
        if g.area > 1e-6:
            m = trimesh.creation.extrude_polygon(g, settings.base_height)
            m.merge_vertices()
            parts.append(m)
    return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()
