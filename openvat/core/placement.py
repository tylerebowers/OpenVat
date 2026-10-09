"""Where supports go, and what kind they are.

Automatic placement runs in two phases (``generate_supports``):

**1. Plan: where supports are needed, and of which kind.**

* Contact points: islands first (parts that start in mid-air - supported
  exactly where they start; one that starts as a corner or tip also gets
  anchors spread around its lowest point), then overhangs steeper than the
  overhang angle (``find_contact_points``).  Each sits on a downward face
  with open space under it (faces buried where shells overlap are skipped);
  overhangs take the flattest face nearby.
* The plate-reach map (``PlateReach``): every cell of the model's clearance
  grid from which a pillar can get down to the build plate without touching
  the part - rods tilted at most ``branch_angle``, ``model_clearance`` away
  from the part, with room for the foot.  Built once, from the plate up.
* Bridges: an overhang contact on a ceiling spanning at most ``max_bridge``
  between two walls facing each other needs no support (``on_bridge``).
* Each contact's kind, in order of preference (``_plan``): a short **nook**
  straight across a narrow gap - a rung square to the walls of a groove or
  slot (``_find_nook``); a **plate** support, with the best arm that clears
  the part (``_arm_candidates``) whose pivot can drop straight to the plate
  or lies on the map; a **nook** standing on the part below.  Island
  contacts stay exactly where the island starts.  A spot with room for
  none is reported as skipped (unless a support close by holds it) - a
  support is never pushed through the part.

**2. Route: plate supports down to the build plate.**

Straight down when the rod and the foot clear the part.  Otherwise a
downward sweep that only steps on map cells (so it always arrives) finds the
plate spot closest to the pivot; the path heads there as directly as the map
allows (tilted rods under the part, then a vertical pillar) and is pulled
tight to a few joints whose segments are checked exactly against the part.  Finally nearby straight pillars merge
into split pillars.

Manual and dragged supports go through the same steps (``place_support``).
"""

from __future__ import annotations

import copy
import math
from functools import lru_cache

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import polylabel, unary_union

from .clearance import ClearanceField, clearance_field
from .raycast import RayCaster, VerticalRays
from .supports import (Support, SupportSettings, arm_for_normal, face_tilt, hex_lattice, _polys, _safe,
                       _valid, nook_cone_length)

DETECTION_STEP = 0.1           # mm between analysis slices
ARM_TILT_STEP = 7.5            # deg between the arm tilts tried when the nominal arm is blocked
ARM_STRETCH = (1.0, 1.5, 2.0)  # arm lengths tried (x tip_length): at most twice as long
ARM_NORMAL_LIMIT = 70.0        # an arm meeting its face more obliquely than this can't hold it
STRAIGHT_TRIES = 3             # best arms tried for a straight pillar
CRACK_MIN_SPAN = 2.0           # a detour with 2+ joints must drop at least this much (mm)
ANGLE_SLACK = 3.0              # deg of grid slack allowed on detour segments
GOAL_TRIES = 4                 # plate cells tried per route (closest first)
NOOK_MIN_GAP = 0.5             # mm: a nook lower than this closes up by itself
NOOK_SLIDE = 2.0               # mm a nook's contact may slide along its face to find room
NOOK_SLIDE_STEP = 0.25         # mm between the spots tried while sliding
ANCHOR_HEIGHT = 3.0            # mm above an island's lowest point where its anchors go
ANCHOR_INSET = 0.4             # mm anchors keep inside the island's outline
NOOK_FACE_MATCH = 0.985        # the slid contact must stay on a face this parallel (cos ~10 deg)
NOOK_END_LIMIT = 35.0          # deg: a nook meets both faces at most this far from square


# --------------------------------------------------------------------------
# the model as placement sees it

def field_cap(s: SupportSettings) -> float:
    """The largest clearance placement asks the field about (mm)."""
    return max(s.base_diameter, s.pillar_diameter) / 2 + s.model_clearance + 1.0


def _rod_clearance(s: SupportSettings) -> float:
    return s.pillar_diameter / 2 + s.model_clearance


def _route_tilt(s: SupportSettings) -> float:
    return min(max(s.branch_angle, 5.0), 60.0)


class ModelSpace:
    """A MeshObject in world space: its mesh, vertical rays, rays in any
    direction, clearance field and (built on first use) its plate-reach map."""

    def __init__(self, obj, s: SupportSettings) -> None:
        self.mesh = obj.transformed()
        self.rays = VerticalRays(self.mesh)
        self.caster = RayCaster(self.mesh)
        self.field = clearance_field(obj, field_cap(s))
        self.settings = s
        self._reach = None

    @property
    def reach(self) -> "PlateReach":
        if self._reach is None:
            self._reach = plate_reach(self.field, self.settings)
        return self._reach


def model_space(obj, s: SupportSettings) -> ModelSpace:
    return ModelSpace(obj, s)


class PlateReach:
    """Which cells of a clearance field a pillar can drop from to the plate.

    Built from the plate up, ``step`` layers at a time: a cell is on the map
    when a rod fits there and in the layers down to the previous step, and
    a cell one step lower that is on the map lies within reach of a rod
    tilted at most ``branch_angle``.  At the bottom: wherever the foot fits
    (everywhere, when the plate is below the grid).  Outside the grid there
    is no part at all, so everything there counts as on the map."""

    def __init__(self, field: ClearanceField, s: SupportSettings) -> None:
        self.field = field
        self.tan = math.tan(math.radians(_route_tilt(s)))
        self.step = max(1, math.ceil(1.5 / self.tan))            # a step can move >= 1.5 cells
        nx, ny, nz = field.shape
        margin = 0.5 * field.cell                                # stricter than the exact checks
        blocked = np.cumsum(field.dist < _rod_clearance(s) + margin, axis=2, dtype=np.int16)
        self.base_layer = kb = max(field.layer_of(s.base_height), -1)
        if kb >= 0:                                              # the plate is in the grid: room for the foot
            k0 = min(max(field.layer_of(0.0), 0), kb)
            foot = s.base_diameter / 2 + s.model_clearance + margin
            self.base = (field.dist[:, :, k0:kb + 1] >= foot).all(axis=2)
        else:
            self.base = np.ones((nx, ny), bool)
        self.cells = np.zeros(field.shape, bool)
        self.cells[:, :, :kb + 1] = self.base[:, :, None]
        for k in range(kb + 1, nz):
            prev = max(k - self.step, kb)
            below = self.base if prev == kb else self.cells[:, :, prev]
            open_column = blocked[:, :, k] == (blocked[:, :, prev] if prev >= 0 else 0)
            self.cells[:, :, k] = open_column & _dilate(below, (k - prev) * self.tan, outside=True)

    def on(self, field: ClearanceField) -> "PlateReach":
        """The same map for the field moved sideways (shares the data)."""
        r = copy.copy(self)
        r.field = field
        return r

    def contains(self, points) -> np.ndarray:
        """Can a pillar get from each point down to the plate?"""
        f = self.field
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        idx = np.floor((p - f.offset - f.origin) / f.cell).astype(np.int64)
        nx, ny, nz = f.shape
        out = np.ones(len(p), bool)                    # beside or below the grid: nothing in the way
        in_xy = (idx[:, 0] >= 0) & (idx[:, 0] < nx) & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
        out[in_xy & (idx[:, 2] >= nz)] = False         # above the part: unknown, say no
        grid = in_xy & (idx[:, 2] >= 0) & (idx[:, 2] < nz)
        i = idx[grid]
        out[grid] = self.cells[i[:, 0], i[:, 1], i[:, 2]]
        return out

    def layer(self, k: int, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
        """Map cells [i0:i1, j0:j1] of layer k (True outside the grid)."""
        nx, ny, _nz = self.field.shape
        out = np.ones((i1 - i0, j1 - j0), bool)
        src = self.base if k <= self.base_layer else self.cells[:, :, k]
        a0, a1, b0, b1 = max(i0, 0), min(i1, nx), max(j0, 0), min(j1, ny)
        if a0 < a1 and b0 < b1:
            out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = src[a0:a1, b0:b1]
        return out


_REACH_CACHE: list[tuple] = []        # (field data, key, map), newest last


def plate_reach(field: ClearanceField, s: SupportSettings) -> PlateReach:
    """The plate-reach map for a field (cached: the map only changes when the
    model moves up or down or the pillar settings change)."""
    key = (round(float(field.offset[2]), 6), _route_tilt(s), s.pillar_diameter, s.model_clearance,
           s.base_diameter, s.base_height)
    for i, (data, k, reach) in enumerate(_REACH_CACHE):
        if data is field.dist and k == key:
            _REACH_CACHE.append(_REACH_CACHE.pop(i))
            return reach.on(field)
    reach = PlateReach(field, s)
    _REACH_CACHE.append((field.dist, key, reach))
    del _REACH_CACHE[:-4]
    return reach


# --------------------------------------------------------------------------
# automatic placement: plan, then route

def generate_supports(obj, settings: SupportSettings, report: dict | None = None,
                      layer_height: float = 0.05) -> list[Support]:
    """Supports for every island and unsupported overhang of a MeshObject.
    ``report``, if given, receives how many are straight, routed, split and
    nook, how many spots were skipped (no room for any support) and how
    many need none after all (narrow ceilings bridged by walls on both sides)."""
    space = ModelSpace(obj, settings)
    # phase 1: where supports are needed, and of which kind
    planned, failed, anchors, unheld = [], [], set(), []
    bridged = 0
    for tip, face, kind in _contacts(space, settings, layer_height):
        sup = Support(tip=tip, owner_id=obj.id, normal=np.array(space.rays.normals[face], float))
        fixed = kind != "overhang"                        # islands: exactly where they start
        if kind == "anchor":
            anchors.add(sup.id)
        if not fixed and settings.max_bridge > 0 and on_bridge(space, tip, sup.normal, settings.max_bridge):
            bridged += 1                                  # a short bridge holds itself
            continue
        if not _plan(sup, space, settings, fixed):
            (unheld if fixed else failed).append(sup)
        elif fixed or not (sup.is_nook and any(o.is_nook and np.linalg.norm(o.tip - sup.tip) < settings.nook_diameter
                                               for o in planned)):  # nooks that slid onto the same spot
            planned.append(sup)
    # overhang spots with no room for a support: fine when a support close by
    # on the same face holds them; islands with no room are always reported
    skipped = len(unheld)
    for sup in failed:
        if not any(np.linalg.norm(o.tip - sup.tip) < 0.75 * settings.spacing
                   and o.normal @ sup.normal >= NOOK_FACE_MATCH * np.linalg.norm(o.normal) * np.linalg.norm(sup.normal)
                   for o in planned):
            skipped += 1
    # phase 2: route the plate supports down to the build plate
    supports = []
    for sup in planned:
        if sup.kind == "plate" and not _route(sup, space, settings) and not _make_nook(sup, space, settings):
            skipped += 1
            continue
        supports.append(sup)
    if settings.branches:
        split_pillars(supports, space.field, settings, keep=anchors)
    if report is not None:
        for key, n in support_counts(supports).items():
            report[key] = report.get(key, 0) + n
        report["skipped"] = report.get("skipped", 0) + skipped
        report["bridged"] = report.get("bridged", 0) + bridged
    return supports


def support_counts(supports: list[Support]) -> dict:
    counts = dict(straight=0, routed=0, split=0, nook=0)
    for s in supports:
        kind = "nook" if s.is_nook else "split" if s.trunk else "routed" if s.route else "straight"
        counts[kind] += 1
    return counts


def place_support(sup: Support, space: ModelSpace, s: SupportSettings) -> bool:
    """Plan and route one support (manual or dragged).  False when there is
    no room for a support at its contact."""
    sup.kind, sup.route, sup.trunk, sup.foot = "plate", (), None, None
    if not _plan(sup, space, s, fixed=True):
        return False
    if sup.kind == "plate" and not _route(sup, space, s):
        return _make_nook(sup, space, s)
    return True


def manual_support(obj, point: np.ndarray, settings: SupportSettings) -> Support | None:
    """A support at a clicked point, its arm following the surface there
    (None when there is no room for one)."""
    space = ModelSpace(obj, settings)
    tri = space.mesh.triangles
    closest = trimesh.triangles.closest_point(tri, np.repeat([point], len(tri), axis=0))
    f = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    sup = Support(tip=np.array(point, float), owner_id=obj.id, normal=np.array(space.mesh.face_normals[f], float))
    return sup if place_support(sup, space, settings) else None


def snap_support(sup: Support, obj, settings: SupportSettings) -> bool:
    """Re-seat a dragged support: its contact goes to the open downward face
    at its XY nearest its old height (following the face there), and it is
    placed again.  False when there is no room for a support there."""
    space = ModelSpace(obj, settings)
    faces = _downward_faces(space, sup.tip[0], sup.tip[1], settings)
    if not faces:
        return False
    z, f = min(faces, key=lambda zf: abs(zf[0] - sup.tip[2]))
    sup.tip[2] = z
    sup.normal = np.array(space.rays.normals[f], float)
    return place_support(sup, space, settings)


# --------------------------------------------------------------------------
# phase 1: contacts and kinds

def _contacts(space: ModelSpace, s: SupportSettings, layer_height: float) -> list[tuple[np.ndarray, int, str]]:
    """(contact point, face, kind) of every spot that needs support, islands
    and anchors first."""
    out = []
    search = 0.35 * s.spacing
    points = find_contact_points(space.mesh, s, layer_height)
    points.sort(key=lambda p: p[3] == "overhang")        # islands and anchors first: highest priority
    for x, y, z_found, kind in points:
        if kind != "overhang":                           # supported exactly where it starts
            hit = _downward_face(space, x, y, z_found + layer_height, s)
            if hit is None:
                continue
            (z, f), cx, cy = hit, x, y
        else:                                            # overhang: slide onto a flat face
            hit = _contact_on_flat_face(space, x, y, z_found, s, search)
            if hit is None:
                continue
            cx, cy, z, f = hit
        out.append((np.array([cx, cy, z]), f, kind))
    return out


def _plan(sup: Support, space: ModelSpace, s: SupportSettings, fixed: bool = False) -> bool:
    """Choose the contact's arm and kind.  In order: a short nook straight
    across a narrow gap (a groove, a slot - up to ``nook_max_gap``); a plate
    support if any clear arm leads down to the plate (straight first, else
    via the map); any nook.  ``fixed`` keeps the contact exactly where it is
    (islands: a nook may not slide away from the island's first layer).
    False when nothing fits."""
    if s.nook_max_gap > 0 and _make_nook(sup, space, s, max_gap=s.nook_max_gap, slide=not fixed):
        return True
    field = space.field
    arms = _arm_candidates(sup, field, s)
    for arm in arms[:STRAIGHT_TRIES]:
        _set_arm(sup, arm)
        if _straight_clear(field, sup.pivot(s), s):
            sup.kind = "plate"
            return True
    if s.routing and arms:
        on_map = space.reach.contains(np.array([_pivot_of(sup, arm, s) for arm in arms]))
        for arm, ok in zip(arms, on_map):
            if ok:
                _set_arm(sup, arm)
                sup.kind = "plate"
                return True
    return _make_nook(sup, space, s, slide=not fixed)


def _nominal_arm(sup: Support, s: SupportSettings) -> tuple:
    lean, angle = arm_for_normal(sup.normal, s.max_arm_angle, s.straight_below)
    return lean, angle, 1.0


def _set_arm(sup: Support, arm: tuple) -> None:
    sup.lean, sup.angle, sup.stretch = np.array(arm[0], float), float(arm[1]), float(arm[2])


def _pivot_of(sup: Support, arm: tuple, s: SupportSettings) -> np.ndarray:
    keep = (sup.lean, sup.angle, sup.stretch)
    _set_arm(sup, arm)
    p = sup.pivot(s)
    sup.lean, sup.angle, sup.stretch = keep
    return p


def _arm_note(sup: Support, s: SupportSettings) -> str:
    lean, angle, _k = _nominal_arm(sup, s)
    same = sup.stretch == 1.0 and abs(sup.angle - angle) < 1e-6 and (angle == 0 or np.allclose(sup.lean, lean))
    return "" if same else "; arm turned or lengthened to clear the part"


def _arm_candidates(sup: Support, field: ClearanceField, s: SupportSettings) -> list[tuple]:
    """Arms that clear the part, best first: (lean, angle, stretch).  The
    nominal arm (straight, or perpendicular to the face) comes first
    whenever it is clear; the others are scored: turned away from the
    nominal arm, more oblique to the face, longer, tilted."""
    n = np.asarray(sup.normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    cands = [_nominal_arm(sup, s)]
    azimuths = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    for k in ARM_STRETCH:
        for tilt in np.arange(0.0, s.max_arm_angle + 1e-6, ARM_TILT_STEP):
            if tilt == 0:
                cands.append((np.zeros(2), 0.0, k))
            else:
                cands += [(np.array([math.cos(a), math.sin(a)]), float(tilt), k) for a in azimuths]
    lean = np.array([c[0] for c in cands])
    tilt = np.radians([c[1] for c in cands])
    k = np.array([c[2] for c in cands])
    axis = np.column_stack([lean * np.sin(tilt)[:, None], -np.cos(tilt)])
    length = s.tip_length * k / np.cos(tilt)

    # sample every arm from just off the contact up to its pivot
    frac = np.linspace(0, 1, 9)[1:]
    t = length[:, None] * frac                                   # mm from the contact
    pts = np.asarray(sup.tip, float) + axis[:, None, :] * t[..., None]
    r_rod = s.pillar_diameter / 2
    r_tip = min(s.contact_diameter(float(sup.tip[2])), 2 * r_rod) / 2
    cos_n = np.clip(axis @ n, -1.0, 1.0)
    # near the contact an arm is naturally close to its own face (~ t * cos_n
    # away), so the full clearance is only demanded where that face allows it
    need = np.minimum(r_tip + (r_rod - r_tip) * frac + s.model_clearance, t * cos_n[:, None] - field.cell)
    d = field.distance(pts.reshape(-1, 3)).reshape(t.shape)
    oblique = np.degrees(np.arccos(cos_n))
    ok = (d >= need).all(axis=1) & (oblique <= ARM_NORMAL_LIMIT)
    ok &= (float(sup.tip[2]) - s.tip_length * k >= s.base_height) | (k == 1.0)

    dev = np.degrees(np.arccos(np.clip(axis @ axis[0], -1.0, 1.0)))
    worse = np.maximum(0.0, oblique - oblique[0])
    extra = (k - 1.0) * s.tip_length
    score = dev + 2 * worse + 20 * extra + 10 * extra ** 2 + 5 * extra ** 3 + 0.1 * np.degrees(tilt)
    out, pivots = [], []
    for i in np.argsort(score, kind="stable"):
        if not ok[i]:
            continue
        pivot = pts[i, -1]
        if any(np.linalg.norm(pivot - q) < 0.2 for q in pivots[-8:]):    # (nearly) the same arm again
            continue
        pivots.append(pivot)
        out.append(cands[i])
    return out


def _make_nook(sup: Support, space: ModelSpace, s: SupportSettings, max_gap: float | None = None,
               slide: bool = True) -> bool:
    """Turn the support into a nook support if one fits (see ``_find_nook``)."""
    found = _find_nook(sup, space, s, max_gap, slide)
    if found is None:
        return False
    tip, normal, foot, width = found
    sup.tip, sup.normal = tip, normal
    sup.kind, sup.foot, sup.route, sup.trunk = "nook", tuple(float(v) for v in foot), (), None
    sup.lean, sup.angle, sup.stretch = np.zeros(2), 0.0, 1.0
    sup.width = 0.0 if width == s.nook_diameter else width
    gap = float(np.linalg.norm(foot - tip))
    sup.status = (f"nook support across a {gap:.1f} mm gap" if max_gap is not None
                  else "nook support: stands on the part (no path to the plate)")
    if sup.width:
        sup.status += f", thinned to {width:.2f} mm to fit"
    return True


def _nook_widths(s: SupportSettings) -> list[float]:
    """Rod diameters tried for a nook: the setting, then thinner ones for tight spots."""
    floor = s.nook_contact_diameter + 0.1
    return [s.nook_diameter] + [w for w in (0.75 * s.nook_diameter, 0.5 * s.nook_diameter) if w >= floor]


def _find_nook(sup: Support, space: ModelSpace, s: SupportSettings, max_gap: float | None = None,
               slide: bool = True):
    """Where a nook support for this contact fits: (contact, its face normal,
    foot, rod diameter) or None.

    Best is a rod perpendicular to the contact's face (straight down under a
    flat ceiling, across the gap in a slanted groove), reaching the opposite
    face; the contact may slide up to ``NOOK_SLIDE`` mm along its face to
    where the rod has room.  With ``max_gap`` only such short perpendicular
    nooks count.  Otherwise any direction within ``max_arm_angle`` of
    straight down is tried from the contact itself, those closest to the
    perpendicular first.  Where the rod doesn't fit, thinner ones are tried.
    ``slide=False`` keeps the contact where it is."""
    n = np.asarray(sup.normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    lean, angle = arm_for_normal(n, s.max_arm_angle, s.straight_below)
    a = math.radians(angle)
    across = np.array([lean[0] * math.sin(a), lean[1] * math.sin(a), -math.cos(a)])   # perpendicular-ish
    reach = max_gap if max_gap is not None else 1e4
    tip = np.asarray(sup.tip, float)
    # sliding only helps when there is something across from the contact
    across_hit = space.caster.first(tip + across * 1e-3, across, reach + NOOK_SLIDE)
    spots = None
    for width in _nook_widths(s):
        foot = _nook_fit(space, tip, n, across, reach, width, s)
        if foot is not None:
            return tip, n, foot, width
        if across_hit is not None and slide:
            if spots is None:
                spots = list(_face_spots(space, tip, n))[1:]
            for spot, normal in spots:
                foot = _nook_fit(space, spot, normal, across, reach, width, s)
                if foot is not None:
                    return spot, normal, foot, width
        if max_gap is None:
            for d in _nook_directions(across, s):
                foot = _nook_fit(space, tip, n, d, reach, width, s)
                if foot is not None:
                    return tip, n, foot, width
    return None


def _face_spots(space: ModelSpace, tip: np.ndarray, n: np.ndarray):
    """The contact, then spots around it on the same face, nearest first:
    (point, face normal)."""
    yield tip, n
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    for a, b in _slide_offsets():
        p = tip + a * e1 + b * e2
        hit = space.caster.first(p + n * 0.5, -n, 1.0)
        if hit is not None and space.caster.normals[hit[2]] @ n >= NOOK_FACE_MATCH:
            yield hit[1], np.asarray(space.caster.normals[hit[2]], float)


def on_bridge(space: ModelSpace, tip, normal, max_span: float) -> bool:
    """Is the contact on a bridge - a ceiling spanning at most ``max_span``
    mm between two walls that face each other (the bottom of a groove, the
    roof of a slot)?  Rays along the ceiling, just under it, in a fan of
    directions: some direction must hit a wall on both sides within the
    span, the two walls facing each other.  (Two walls meeting at a corner
    don't count: the ceiling there still overhangs past the corner.)"""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    start = np.asarray(tip, float) + n * 0.05
    for a in np.radians(np.arange(0, 180, 15)):
        d = math.cos(a) * e1 + math.sin(a) * e2
        one = space.caster.first(start, d, max_span)
        if one is None:
            continue
        other = space.caster.first(start, -d, max_span - one[0])
        if other is not None and space.caster.normals[one[2]] @ space.caster.normals[other[2]] < -0.5:
            return True
    return False


@lru_cache(maxsize=1)
def _slide_offsets() -> tuple:
    r = int(NOOK_SLIDE / NOOK_SLIDE_STEP)
    pts = [(i * NOOK_SLIDE_STEP, j * NOOK_SLIDE_STEP) for i in range(-r, r + 1) for j in range(-r, r + 1)
           if (i or j) and math.hypot(i, j) <= r]
    return tuple(sorted(pts, key=lambda p: math.hypot(*p)))


def _nook_directions(first: np.ndarray, s: SupportSettings) -> list[np.ndarray]:
    """Directions within ``max_arm_angle`` of straight down, closest to ``first`` first."""
    dirs = [first]
    for tilt in np.arange(0.0, s.max_arm_angle + 1e-6, ARM_TILT_STEP):
        t = math.radians(tilt)
        for az in (np.linspace(0, 2 * math.pi, 12, endpoint=False) if tilt else [0.0]):
            dirs.append(np.array([math.sin(t) * math.cos(az), math.sin(t) * math.sin(az), -math.cos(t)]))
    return sorted(dirs, key=lambda d: -float(d @ first))


def _nook_fit(space: ModelSpace, tip, n_top, direction, reach: float, width: float, s: SupportSettings):
    """The foot of a nook from ``tip`` along ``direction`` (the first surface
    hit), if a rod ``width`` mm thick fits there; else None."""
    hit = space.caster.first(tip + direction * 1e-3, direction, reach)
    if hit is None or hit[0] < NOOK_MIN_GAP:
        return None
    _t, foot, face = hit
    n_foot = np.asarray(space.caster.normals[face], float)
    if -(direction @ n_foot) < math.cos(math.radians(NOOK_END_LIMIT)):     # meets the far face askew
        return None
    return foot if _nook_fits(space, tip, n_top, foot, n_foot, width, s) else None


def _nook_fits(space: ModelSpace, tip, n_top, foot, n_foot, width: float, s: SupportSettings) -> bool:
    """Does a nook from tip to foot meet both faces squarely (within
    ``NOOK_END_LIMIT`` of their normals) and stay out of the part?

    The surface of its rod and contact cones is sampled.  A quick look at
    the clearance field first: every point must keep ``model_clearance``
    from the part, except where only the face at that end is that close.
    Then exactly, against the mesh: every point must keep
    ``model_clearance`` from every face except the two it stands on - so it
    can't slip into a corner or the bottom of a groove."""
    field = space.field
    axis = np.asarray(foot, float) - tip
    length = float(np.linalg.norm(axis))
    axis /= length
    cos_top, cos_foot = float(axis @ n_top), float(-(axis @ n_foot))     # 1 = square to the face
    limit = math.cos(math.radians(NOOK_END_LIMIT))
    if cos_top < limit or cos_foot < limit:
        return False
    r, rc = width / 2, s.nook_contact_diameter / 2
    cone = nook_cone_length(length, s)
    t = np.linspace(0.0, length, max(7, int(math.ceil(length / (0.5 * field.cell))) + 1))
    radius = rc + (r - rc) * np.clip(np.minimum(t, length - t) / max(cone, 1e-6), 0.0, 1.0)
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v                   # (12, 3)
    pts = (tip + axis * t[:, None])[:, None, :] + radius[:, None, None] * ring[None]
    pts = pts.reshape(-1, 3)
    faces = np.minimum((pts - tip) @ n_top, (pts - foot) @ n_foot)               # distance to the end faces
    need = np.minimum(s.model_clearance, faces - field.cell)
    if not (field.distance(pts) >= need).all():
        return False
    return _clear_of_other_faces(space.caster, pts, (tip, n_top), (foot, n_foot), clearance=s.model_clearance)


def _clear_of_other_faces(caster: RayCaster, pts: np.ndarray, *ends, clearance: float) -> bool:
    """Exact check: are all points at least ``clearance`` from every mesh
    face, except faces in the planes of ``ends`` ((point, normal) pairs)?"""
    lo, hi = pts.min(axis=0) - clearance, pts.max(axis=0) + clearance
    cand = np.flatnonzero(np.all(caster.hi >= lo, axis=1) & np.all(caster.lo <= hi, axis=1))
    for point, normal in ends:
        same = (caster.normals[cand] @ normal >= NOOK_FACE_MATCH) & \
               (np.abs((caster.tri[cand, 0] - point) @ normal) < 0.05)
        cand = cand[~same]
    if len(cand) == 0:
        return True
    tri = np.tile(caster.tri[cand], (len(pts), 1, 1))
    p = np.repeat(pts, len(cand), axis=0)
    d = np.linalg.norm(trimesh.triangles.closest_point(tri, p) - p, axis=1)
    return bool(d.min() >= clearance - 1e-6)


# --------------------------------------------------------------------------
# phase 2: routing plate supports

def _route(sup: Support, space: ModelSpace, s: SupportSettings) -> bool:
    """Straight down if that clears the part, else around it via the map."""
    sup.route, sup.trunk = (), None
    p = sup.pivot(s)
    if _straight_clear(space.field, p, s):
        sup.status = "straight to the plate" + _arm_note(sup, s)
        return True
    if s.routing:
        route = route_to_plate(space.reach, p, s)
        if route is not None:
            sup.route = route
            n = len(route)
            sup.status = f"routed around the part to the plate ({n} joint{'s' if n != 1 else ''})" + _arm_note(sup, s)
            return True
    return False


def _foot_clear(field: ClearanceField, xy, s: SupportSettings) -> bool:
    pts = [[xy[0], xy[1], z] for z in (0.0, 0.5 * s.base_height, s.base_height)]
    return bool((field.distance(pts) >= s.base_diameter / 2 + s.model_clearance).all())


def _straight_clear(field: ClearanceField, top, s: SupportSettings) -> bool:
    """Can a pillar drop straight from ``top`` to a foot on the plate?"""
    top = np.asarray(top, float)
    bottom = np.array([top[0], top[1], s.base_height])
    return field.segment_clear(top, bottom, _rod_clearance(s)) and _foot_clear(field, top[:2], s)


def route_to_plate(reach: PlateReach, socket, s: SupportSettings) -> tuple | None:
    """Joints of a path from ``socket`` (a pivot) down to the plate that keeps
    every rod ``model_clearance`` away from the part, or None.

    Going down the map a step at a time, the cells the pillar can reach grow
    by what a rod tilted at most ``branch_angle`` covers, and only cells on
    the map are kept - so from a socket on the map the sweep always arrives.
    The plate cell closest to the socket wins.  The path heads for it as
    directly as the map allows (tilted rods up top, then a vertical pillar)
    and is pulled tight to the few joints whose straight segments still
    clear the part; every segment, the foot and the crack-span rule are
    checked once more."""
    field = reach.field
    m, tan = reach.step, reach.tan
    socket = np.asarray(socket, float)
    ci, cj, ck = field.index(socket)
    kb = reach.base_layer
    if ck <= kb or not reach.contains(socket)[0]:
        return None
    w = int(math.ceil((ck - kb) * tan)) + 2                 # the farthest a path can drift (cells)
    i0, j0, size = ci - w, cj - w, 2 * w + 1
    ii, jj = np.mgrid[-w:w + 1, -w:w + 1]
    drift = ii * ii + jj * jj
    reachable = np.zeros((size, size), bool)
    reachable[w, w] = True
    steps = [(ck, reachable, 0.0)]                          # (layer, reachable cells, step radius)
    k = ck
    while k > kb:
        prev = max(k - m, kb)
        radius = (k - prev) * tan
        reachable = _dilate(reachable, radius) & reach.layer(prev, i0, i0 + size, j0, j0 + size)
        if not reachable.any():
            return None
        steps.append((prev, reachable, radius))
        k = prev
    # plate cells closest to the socket first (a few, in case the exact checks reject one)
    goals = np.argwhere(reachable)
    goals = goals[np.argsort(drift[reachable], kind="stable")][:GOAL_TRIES]
    for gi, gj in goals:
        route = _path_to(field, steps, (int(gi), int(gj)), (w, i0, j0), socket, s)
        if route is not None:
            return route
    return None


def _path_to(field: ClearanceField, steps: list, goal: tuple, window: tuple, socket, s: SupportSettings):
    """The joints of the sweep's path from the socket to one plate cell, or None."""
    w, i0, j0 = window
    gi, gj = goal
    size = 2 * w + 1
    h = field.cell
    tilt = _route_tilt(s)
    clearance = _rod_clearance(s)
    # the cells from which that plate cell can still be reached, step by step
    back = [None] * len(steps)
    back[-1] = np.zeros((size, size), bool)
    back[-1][gi, gj] = True
    for n in range(len(steps) - 1, 0, -1):
        back[n - 1] = _dilate(back[n], steps[n][2]) & steps[n - 1][1]
    # head for it as directly as the map allows: tilted rods up top, then a
    # vertical pillar down to the foot
    cells = [(w, w)]
    for n in range(1, len(steps)):
        i, j = cells[-1]
        options = [(i + a, j + b) for a, b in _disk(steps[n][2])
                   if 0 <= i + a < size and 0 <= j + b < size and back[n][i + a, j + b]]
        cells.append(min(options, key=lambda c: ((c[0] - gi) ** 2 + (c[1] - gj) ** 2,
                                                  (c[0] - i) ** 2 + (c[1] - j) ** 2)))
    z0 = field.origin[2] + field.offset[2]
    pts = [socket] + [np.array([*field.center(i0 + i, j0 + j, 0)[:2], z0 + steps[n][0] * h])
                      for n, (i, j) in enumerate(cells) if 0 < n < len(cells) - 1]
    gx, gy = field.center(i0 + gi, j0 + gj, 0)[:2]
    pts.append(np.array([gx, gy, s.base_height]))

    # the vertical run at the bottom becomes the pillar; pull the rest tight
    q = len(pts) - 1
    while q > 1 and np.allclose(pts[q - 1][:2], pts[-1][:2]):
        q -= 1
    joints, i = [], 0
    while i < q:
        j = next((j for j in range(q, i, -1) if _detour_ok(field, pts[i], pts[j], clearance, tilt)), None)
        if j is None:
            return None
        joints.append(pts[j])
        i = j
    if not field.segment_clear(joints[-1], pts[-1], clearance) or not _foot_clear(field, pts[-1][:2], s):
        return None
    if len(joints) >= 2 and socket[2] - joints[-1][2] < CRACK_MIN_SPAN:
        return None                                                      # squeezed through a crack
    return tuple(tuple(float(v) for v in p) for p in joints)


def _detour_ok(field: ClearanceField, a, b, clearance: float, tilt: float) -> bool:
    d = np.asarray(b, float) - np.asarray(a, float)
    if d[2] >= 0:
        return False
    if math.degrees(math.atan2(math.hypot(d[0], d[1]), -d[2])) > tilt + ANGLE_SLACK:
        return False
    return field.segment_clear(a, b, clearance)


@lru_cache(maxsize=64)
def _disk(radius: float) -> tuple:
    """Integer offsets (a, b) with a^2 + b^2 <= radius^2."""
    r = int(math.floor(radius))
    return tuple((a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if a * a + b * b <= radius * radius)


def _dilate(mask: np.ndarray, radius: float, outside: bool = False) -> np.ndarray:
    """Grow a 2D mask by a disk; ``outside`` is what lies beyond its edges."""
    r = int(math.floor(radius))
    if r and outside:
        mask = np.pad(mask, r, constant_values=True)
    out = mask.copy()
    n0, n1 = mask.shape
    for a, b in _disk(radius):
        if a or b:
            out[max(a, 0):n0 + min(a, 0), max(b, 0):n1 + min(b, 0)] |= \
                mask[max(-a, 0):n0 - max(a, 0), max(-b, 0):n1 - max(b, 0)]
    return out[r:-r, r:-r] if r and outside else out


def split_pillars(supports: list[Support], field: ClearanceField, s: SupportSettings,
                  keep: set = frozenset()) -> None:
    """Group nearby straight plate pillars onto shared trunks that split into
    branches.  A group is kept only if every branch stays within
    ``branch_angle`` of vertical, the trunk below the split is at least a
    few millimeters tall, and trunk, foot and branches all clear the part.
    Supports whose id is in ``keep`` (island anchors) keep their own pillars."""
    min_trunk = 3.0                                       # mm of trunk below the split
    rise = math.tan(math.radians(_route_tilt(s)))
    clearance = _rod_clearance(s)
    free = [sup for sup in supports if not sup.is_nook and sup.trunk is None and not sup.route
            and sup.id not in keep]
    free.sort(key=lambda sup: sup.pillar_top_z(s))
    used: set[int] = set()
    for sup in free:
        if sup.id in used:
            continue
        p0 = sup.pivot(s)
        others = [o for o in free if o.id not in used and o.id != sup.id]
        others.sort(key=lambda o: np.linalg.norm(o.pillar_xy(s) - p0[:2]))
        group = [sup] + [o for o in others if np.linalg.norm(o.pillar_xy(s) - p0[:2]) <= s.branch_max_distance]
        group = group[:max(int(round(s.branch_max_count)), 1)]
        while len(group) >= 2:
            piv = np.array([g.pivot(s) for g in group])
            center = piv[:, :2].mean(axis=0)
            spread = np.linalg.norm(piv[:, :2] - center, axis=1).max()
            split = np.array([center[0], center[1], piv[:, 2].min() - spread / rise])
            if (split[2] - s.base_height >= min_trunk and _straight_clear(field, split, s)
                    and all(field.segment_clear(split, p, clearance) for p in piv)):
                trunk = (float(split[0]), float(split[1]), float(split[2]), 0.0)
                for g in group:
                    g.trunk = trunk
                    g.status = f"on a split pillar ({len(group)} contacts)" + _arm_note(g, s)
                    used.add(g.id)
                break
            group = group[:-1]                            # drop the farthest and retry


# --------------------------------------------------------------------------
# finding contact points

def find_contact_points(mesh: trimesh.Trimesh, settings: SupportSettings, layer_height: float = 0.05,
                        step: float = DETECTION_STEP) -> list[tuple[float, float, float, str]]:
    """(x, y, z, kind) of every spot that needs support, z = height of the
    slice where it was found, kind = 'island', 'anchor' or 'overhang'.

    Simulates the print: slices the mesh at every printed layer
    (``layer_height``), and analyses overhangs every ``step`` mm:

    * a polygon touching nothing in the layer below is an **island**: it
      gets a support at once, in the very layer where it starts (on the
      lattice, or at its most interior point);
    * an island that starts small (a corner or a tip pointing down) and
      stays apart from the rest for ``ANCHOR_HEIGHT`` mm gets **anchors**:
      more supports spread around its lowest point, up to ``island_anchors``
      in all - so the part's first layers don't hang on a single support
      while the vat film peels off.  The island's own points become anchors
      too (anchors keep their own pillars);
    * the part of a slice reaching more than ``step * tan(overhang_angle)``
      beyond the slice ``step`` below is an **overhang**.  Overhang area that no
      support covers yet (each support holds a disc of radius ~0.6 *
      spacing) is collected slice after slice, and once a piece of it grows
      large enough a support goes in its middle.  Flat ceilings appear in
      one slice and get supports on a hexagonal lattice.
    """
    from .slicer import slice_mesh_at          # local import: slicer -> scene -> supports

    lo, hi = mesh.bounds
    layer = min(max(layer_height, 0.005), step)
    every = max(1, int(round(step / layer)))    # overhangs are analysed every `every` layers
    step = layer * every
    # a face exactly at the overhang angle counts as overhang (whatever the rounding)
    reach = step * math.tan(math.radians(min(settings.overhang_angle, 89.0))) * 0.995
    pitch = settings.spacing
    radius = 0.6 * pitch                       # area held by one support
    min_area = 0.15 * pitch * pitch            # collect at least this much before adding one
    lattice = hex_lattice(lo[0], hi[0], lo[1], hi[1], pitch)
    lattice_pts = shapely.points(lattice) if len(lattice) else None
    on_plate = lo[2] < settings.min_support_height          # the bottom rests on the plate
    anchor_count = max(int(round(settings.island_anchors)), 1)
    anchor_gap = max(1.5, 0.35 * pitch)        # anchors at least this far apart

    out: list[tuple[float, float, float, str]] = []
    pending = Polygon()                        # overhang area still without support
    anchoring: list[dict] = []                 # small islands being followed up for anchors

    def covered(z: float, window: float):
        discs = [Point(x, y).buffer(radius, 16) for x, y, zz, _k in out if z - zz <= window]
        return unary_union(discs) if discs else Polygon()

    def add(x: float, y: float, z: float, kind: str) -> None:
        out.append((x, y, z, kind))

    def anchor_spots(island: dict, z: float) -> None:
        """More spots spread around a small island's lowest point, as far as
        possible from the supports it has (kept aside until it proves to be
        a real island - see below)."""
        inner = island["outline"].buffer(-ANCHOR_INSET)
        if inner.is_empty:
            return
        cands = shapely.get_coordinates(shapely.segmentize(inner.boundary, 0.25))
        near = island["outline"].buffer(0.5)                   # supports it has: its own and any nearby
        held = [(x, y) for x, y, _z, _k in out if near.contains(Point(x, y))]
        held += [(x, y) for x, y, _z in island["spots"]]
        while len(held) < anchor_count and len(cands):
            gap = np.min(np.linalg.norm(cands[:, None, :] - np.array(held)[None, :, :], axis=2), axis=1)
            k = int(np.argmax(gap))
            if gap[k] < anchor_gap:
                return
            island["spots"].append((float(cands[k, 0]), float(cands[k, 1]), z))
            held.append((float(cands[k, 0]), float(cands[k, 1])))

    def follow(island: dict, cur, below) -> bool:
        """Track a small island one slice up; False once it is done.  An
        island that joins other material within ``ANCHOR_HEIGHT`` mm was only
        cut off in the slices (a groove of engraved text, say) and keeps its
        one support; one that stays apart gets its anchors."""
        parts = [p for p in _polys(cur) if p.intersects(island["outline"])]
        others = [q for q in _polys(below) if not q.intersects(island["outline"])]
        if not parts or any(p.intersects(q) for p in parts for q in others):
            return False                                   # gone, or joined the rest: no anchors
        island["outline"] = unary_union(parts)
        if z - island["z0"] < ANCHOR_HEIGHT and len(island["spots"]) + len(island["points"]) < anchor_count:
            anchor_spots(island, z)
            return True
        anchor_spots(island, z)
        for i in island["points"]:                         # a real island: anchor it
            x, y, zz, _k = out[i]
            out[i] = (x, y, zz, "anchor")
        for x, y, zz in island["spots"]:
            add(x, y, zz, "anchor")
        return False

    prev = MultiPolygon()                      # the slice one printed layer down (islands)
    prev_step = MultiPolygon()                 # the slice one analysis step down (overhangs)
    z = lo[2] + layer / 2
    first, k = True, 0
    while z < hi[2]:
        cur = slice_mesh_at(mesh, z)
        if not cur.is_empty and not (first and on_plate):
            islands = []
            for poly in _polys(cur):
                if prev.is_empty or not poly.intersects(prev):            # island
                    islands.append(poly)
                    pts = []
                    if lattice_pts is not None:
                        pts = [tuple(p) for p in lattice[shapely.contains(poly, lattice_pts)]]
                    if not pts:
                        c = polylabel(poly, tolerance=0.01)
                        pts = [(c.x, c.y)]
                    if len(pts) < anchor_count:                            # small: follow it up
                        anchoring.append(dict(outline=poly, z0=z, spots=[],
                                              points=list(range(len(out), len(out) + len(pts)))))
                    for x, y in pts:
                        add(x, y, z, "island")
            if k % every == 0:
                grown = prev_step.buffer(reach) if not prev_step.is_empty else None
                for poly in _polys(cur):
                    if any(poly.equals(i) for i in islands):
                        continue
                    region = _safe(shapely.difference, poly, grown) if grown is not None else poly   # overhang
                    if region.is_empty or region.area < 1e-6:
                        continue
                    if region.area > min_area and lattice_pts is not None:     # flat ceiling
                        for x, y in lattice[shapely.contains(region, lattice_pts)]:
                            add(float(x), float(y), z, "overhang")
                    pending = _safe(shapely.union, pending, region)
                if not pending.is_empty:
                    # consecutive slivers are separated by the reach gap; close it
                    # so a sloped overhang accumulates into one growing piece
                    with np.errstate(divide="ignore", invalid="ignore"):    # degenerate slivers are fine
                        pending = pending.buffer(reach, join_style="mitre").buffer(-reach, join_style="mitre")
                    pending = _safe(shapely.difference, _valid(pending), covered(z, pitch))
                    for part in sorted(_polys(pending), key=lambda p: -p.area):
                        guard = 0
                        while part.area > min_area and guard < 50:
                            c = polylabel(part, tolerance=0.05)
                            add(c.x, c.y, z, "overhang")
                            part = _safe(shapely.difference, part, Point(c.x, c.y).buffer(radius, 16))
                            guard += 1
                    pending = _safe(shapely.difference, pending, covered(z, pitch))
                anchoring = [island for island in anchoring if follow(island, cur, prev_step)]
        if k % every == 0:
            prev_step = cur
        prev, first = cur, False
        z += layer
        k += 1
    return out


def _contact_on_flat_face(space: ModelSpace, x: float, y: float, z_found: float,
                          s: SupportSettings, search: float):
    """Best contact near (x, y) for an overhang found at height z_found:
    prefer the flattest downward face, then the closest one.
    Returns (x, y, z, face) or None."""
    best, best_score = None, np.inf
    for dx, dy in _SEARCH_PATTERN * search:
        px, py = x + dx, y + dy
        hit = _downward_face(space, px, py, z_found + search + DETECTION_STEP, s)
        if hit is None:
            continue
        z, f = hit
        score = face_tilt(space.rays.normals[f]) + 4.0 * math.hypot(dx, dy)
        if score < best_score:
            best, best_score = (px, py, z, f), score
    return best


def _downward_faces(space: ModelSpace, x: float, y: float, s: SupportSettings) -> list[tuple[float, int]]:
    """Downward-facing surfaces at (x, y) with open space under them, as
    (z, face) from the bottom up.  Faces buried inside the model (where
    shells overlap) have none and are skipped."""
    zs, fs = space.rays.hits(x, y)
    down = [(float(z), int(f)) for z, f in zip(zs, fs)
            if space.rays.normals[f][2] < -1e-3 and z > s.min_support_height]
    if not down:
        return []
    gap = max(0.3, space.field.cell)
    open_below = space.field.distance([[x, y, z - gap] for z, _f in down]) > 0
    return [zf for zf, ok in zip(down, open_below) if ok]


def _downward_face(space: ModelSpace, x: float, y: float, z_max: float, s: SupportSettings):
    """The highest open downward face at (x, y) no higher than z_max: (z, face) or None."""
    below = [zf for zf in _downward_faces(space, x, y, s) if zf[0] <= z_max]
    return below[-1] if below else None


# center + two hexagonal rings (unit radius)
_SEARCH_PATTERN = np.vstack([[0.0, 0.0]] + [
    [ring * math.cos(a), ring * math.sin(a)]
    for ring, n in ((0.5, 6), (1.0, 12)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])
