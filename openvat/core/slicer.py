"""Slicing: turn the scene into a stack of 2D layers.

Each layer is a shapely (Multi)Polygon in build-plate millimeters (origin at
the plate center).  Keeping layers as vectors rather than bitmaps is what the
Anycubic .pwsz format wants, and it also keeps memory low; the layer viewer
rasterizes on demand.

Pipeline for one layer:
    1. intersect every world-space mesh with the plane at the model height
       the layer stands for: its center, unless the resin scales layers in
       Z or stretches the lowest ones (see ``layer_schedule`` and
       ``unstretched_z``)
    2. chain the resulting segments into closed polygons (with holes)
    3. scale each model's polygons by the layer's X/Y factors about the
       model's own center (bottom, transition and normal layers each have
       their own); supports go with the model they stand under
    4. union the polygons of all objects/supports

Support meshes are sets of overlapping closed parts; each part is outlined
on its own and the outlines are united (``shell_labels``), so crossing parts
never cancel out.  Hollowed models are sliced on their own first: their
cavity, internal poles and drain holes are worked out from their layers
(see ``core/hollow.py``) before they join the rest.  After slicing,
``find_islands`` walks the stack from the plate up and lists regions the
print would not hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import shapely
import shapely.affinity
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .contours import segments_to_multipolygon
from .hollow import HollowState, hollow_layers, hole_segments, hole_depth, lattice_segments
from .profiles import PrinterProfile, ResinProfile
from .scene import Scene

ProgressFn = Callable[[int, int], None]   # (done, total)


@dataclass
class Layer:
    index: int
    z_bottom: float          # mm, bottom of the layer
    thickness: float         # mm
    exposure: float          # seconds
    off_time: float
    wait_before_lift: float
    wait_after_lift: float
    lift_distance: float
    lift_speed: float
    retract_speed: float
    geometry: MultiPolygon   # solid area of this layer

    @property
    def z_center(self) -> float:
        return self.z_bottom + self.thickness / 2

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.thickness

    @property
    def area(self) -> float:
        return float(self.geometry.area)

    def polygons(self) -> list[Polygon]:
        return list(self.geometry.geoms)


@dataclass
class SliceResult:
    layers: list[Layer]
    printer: PrinterProfile
    resin: ResinProfile
    model_bounds: np.ndarray             # unscaled scene bounds [[x0,y0,z0],[x1,y1,z1]]
    object_names: list[str] = field(default_factory=list)
    islands: list["Island"] = field(default_factory=list)   # unsupported regions (see find_islands)

    @property
    def height(self) -> float:
        return self.layers[-1].z_top if self.layers else 0.0

    def volume_mm3(self) -> float:
        return sum(l.area * l.thickness for l in self.layers)

    def print_time_s(self) -> float:
        """Rough estimate: exposure + rest + lift/retract travel per layer."""
        total = 0.0
        for l in self.layers:
            travel = l.lift_distance / max(l.lift_speed, 0.1) + l.lift_distance / max(l.retract_speed, 0.1)
            total += l.exposure + l.off_time + l.wait_before_lift + l.wait_after_lift + travel
        return total


@dataclass
class Island:
    """A region of the print that is not connected to the build plate
    through the layers below it - it would be cured onto the vat film
    instead of the part."""
    first: int                           # first layer index
    last: int                            # last layer it is still unconnected
    regions: dict[int, list[Polygon]]    # layer index -> its outline(s) there
    xy: tuple[float, float]              # where it starts


ISLAND_OVERHANG = 60.0                   # deg: a region this close to the layer below still sticks to it...
ISLAND_TOLERANCE = 0.05                  # ...and at least this close (mm)


def find_islands(layers: list[Layer]) -> list[Island]:
    """Simulate the print layer by layer: a region is held when it touches a
    held region of the layer below, starting from the plate - or comes
    within what a ``ISLAND_OVERHANG`` overhang reaches in one layer (at
    least ``ISLAND_TOLERANCE``).  Whatever is never held is an island,
    grouped across layers for as long as it stays unconnected."""
    reach = math.tan(math.radians(ISLAND_OVERHANG))
    islands: list[Island] = []
    held = None                                   # held material of the layer below
    open_islands: list[Island] = []
    for layer in layers:
        tolerance = max(ISLAND_TOLERANCE, layer.thickness * reach)
        polys = [p for p in shapely.get_parts(layer.geometry) if isinstance(p, Polygon) and not p.is_empty]
        if held is None:                          # the first layer lies on the plate
            loose = []
            keep = polys
        else:
            shapely.prepare(held)
            near = shapely.dwithin(np.array(polys, dtype=object), held, tolerance) if polys else np.array([], bool)
            keep = [p for p, ok in zip(polys, near) if ok]
            loose = [p for p, ok in zip(polys, near) if not ok]
        still_open = []
        for p in loose:                           # continue an island from the layer below, or start one
            isl = next((i for i in open_islands
                        if any(p.distance(q) <= tolerance for q in i.regions[layer.index - 1])), None)
            if isl is None:
                c = p.representative_point()
                isl = Island(layer.index, layer.index, {}, (float(c.x), float(c.y)))
                islands.append(isl)
            isl.last = layer.index
            isl.regions.setdefault(layer.index, []).append(p)
            if isl not in still_open:
                still_open.append(isl)
        open_islands = still_open
        held = unary_union(keep) if keep else Polygon()
    return islands


# --------------------------------------------------------------------------
# layer schedule

LAYER_ONLY = ("scale", "cut_z")            # layer_schedule keys that are not Layer fields


def layer_schedule(height: float, resin: ResinProfile) -> list[dict]:
    """Return the per-layer parameters (without geometry) for a model of the
    given height.  Bottom layers, optional transition layers, normal layers.

    Besides the Layer fields every entry has ``scale``, the layer's (X, Y,
    Z) factors, and ``cut_z``, the model height it is cut at.  A layer
    stands for thickness / Z of the model, so with Z > 1 more layers cover
    the same height and that part prints taller.  Transition layers ramp
    exposure and scale linearly from the bottom values to the normal ones."""
    out: list[dict] = []
    z = 0.0                                      # print height
    u = 0.0                                      # model height covered so far
    n_bottom = resin.bottom_layers
    n_trans = max(resin.transition_layers, 0)
    b, n = resin.bottom, resin.normal
    b_scale = (b.scale_x, b.scale_y, b.scale_z)
    n_scale = (n.scale_x, n.scale_y, n.scale_z)
    i = 0
    while u < height - 1e-6:
        src = b if i < n_bottom else n
        p = dict(thickness=src.thickness, exposure=src.exposure, off_time=src.off_time,
                 wait_before_lift=src.wait_before_lift, wait_after_lift=src.wait_after_lift,
                 lift_distance=src.lift_distance, lift_speed=src.lift_speed,
                 retract_speed=src.retract_speed)
        scale = b_scale if i < n_bottom else n_scale
        k = i - n_bottom
        if 0 <= k < n_trans:
            # linear ramp from the bottom values to the normal ones
            t = (k + 1) / (n_trans + 1)
            p["exposure"] = b.exposure + (n.exposure - b.exposure) * t
            scale = tuple(bv + (nv - bv) * t for bv, nv in zip(b_scale, n_scale))
        step = p["thickness"] / max(scale[2], 1e-3)
        p["index"] = i
        p["z_bottom"] = z
        p["scale"] = tuple(float(v) for v in scale)
        p["cut_z"] = u + step / 2
        out.append(p)
        z += p["thickness"]
        u += step
        i += 1
    return out


# --------------------------------------------------------------------------
# geometry helpers

@dataclass
class _HollowJob:
    """A hollowed model: its world mesh and what hollowing adds (world space)."""
    mesh: trimesh.Trimesh
    center: np.ndarray                            # XY: its layers scale about this
    state: HollowState
    holes: tuple                                  # (starts, ends, radii)
    poles: tuple | None                           # (starts, ends) of the internal support poles


def _world_meshes(scene: Scene) -> tuple[list, tuple | None, list[_HollowJob]]:
    """All printable geometry in world space: the models as (mesh, XY
    center) pairs, the supports as one (mesh, shell labels) pair (None
    without supports), and the hollowed models apart.  Models are sliced as
    they are (even-odd, so models that come with cavities keep them);
    support meshes are made of overlapping closed shells and carry a shell
    label per face."""
    models, jobs = [], []
    for obj in scene.objects:
        if not obj.visible:
            continue
        m = obj.transformed()
        center = m.bounds.mean(axis=0)[:2]
        if obj.hollow is None:
            models.append((m, center))
            continue
        state = obj.hollow
        starts, ends, radii = hole_segments(obj.matrix(), obj.holes, hole_depth(state.thickness))
        poles = None
        if state.lattice is not None:
            lo, hi = m.bounds
            poles = lattice_segments(lo, hi, state.lattice)
        jobs.append(_HollowJob(m, center, state, (starts, ends, radii), poles))
    support = None
    meshes = scene.support_meshes()
    if meshes:                                    # one mesh, so each layer is cut once
        m = trimesh.util.concatenate(meshes)
        support = (m, shell_labels(m))
    return models, support, jobs


def shell_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    """For every face, the id of the closed shell (connected piece) it belongs to."""
    faces = np.asarray(mesh.faces)
    label = np.arange(len(mesh.vertices))
    while True:                                   # spread the smallest id over each piece
        new = label.copy()
        low = label[faces].min(axis=1)
        for k in range(3):
            np.minimum.at(new, faces[:, k], low)
        new = new[new]                            # pointer jumping: converges in a few rounds
        if np.array_equal(new, label):
            break
        label = new
    return label[faces[:, 0]]


def _scaled(geom: MultiPolygon, sx: float, sy: float, center) -> MultiPolygon:
    """A layer outline scaled in X/Y about ``center`` (scale compensation,
    as PhotonWorkshop does it: about each model's own center)."""
    if (sx == 1.0 and sy == 1.0) or geom.is_empty:
        return geom
    return _as_multi(shapely.affinity.scale(geom, sx, sy, origin=(float(center[0]), float(center[1]))))


def _scaled_supports(geom: MultiPolygon, sx: float, sy: float,
                     boxes: np.ndarray, centers: np.ndarray) -> list[MultiPolygon]:
    """The support outlines of one layer, each scaled about the center of
    the model it stands under (the nearest model footprint, ``boxes`` being
    their XY bounds [x0, y0, x1, y1]), so supports stay under their contact
    points."""
    if (sx == 1.0 and sy == 1.0) or geom.is_empty or not len(centers):
        return [geom]
    polys = np.array([g for g in shapely.get_parts(geom) if isinstance(g, Polygon) and not g.is_empty],
                     dtype=object)
    if not len(polys):
        return [geom]
    pts = shapely.get_coordinates(shapely.point_on_surface(polys))
    dx = np.maximum(np.maximum(boxes[None, :, 0] - pts[:, None, 0], pts[:, None, 0] - boxes[None, :, 2]), 0)
    dy = np.maximum(np.maximum(boxes[None, :, 1] - pts[:, None, 1], pts[:, None, 1] - boxes[None, :, 3]), 0)
    to_center = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    owner = np.argmin(np.hypot(dx, dy) * 1e6 + to_center, axis=1)   # inside a footprint first
    return [_scaled(MultiPolygon(list(polys[owner == k])), sx, sy, centers[k]) for k in np.unique(owner)]


def _z_stretch(resin: ResinProfile) -> tuple[float, float] | None:
    """(height, factor) of the resin's Z stretch, None when it is off."""
    h, f = float(resin.stretch_height), float(resin.stretch_factor)
    if h <= 0 or f <= 0 or f == 1.0:
        return None
    return h, f


def stretched_z(z, resin: ResinProfile) -> np.ndarray:
    """Model height -> print height (before the layers' own Z scale): up to
    ``stretch_height`` everything is ``stretch_factor`` times taller, above
    that shifted up by the difference."""
    z = np.asarray(z, dtype=float)
    s = _z_stretch(resin)
    if s is None:
        return z
    h, f = s
    return np.where(z <= h, z * f, z + h * (f - 1))


def unstretched_z(z, resin: ResinProfile) -> np.ndarray:
    """Print height -> model height (the inverse of ``stretched_z``): where a
    layer printed at height z cuts the model (again before the layers' own
    Z scale, which ``layer_schedule`` works out)."""
    z = np.asarray(z, dtype=float)
    s = _z_stretch(resin)
    if s is None:
        return z
    h, f = s
    return np.where(z <= h * f, z / f, z - h * (f - 1))


def slice_mesh_at(mesh: trimesh.Trimesh, z: float, shells: np.ndarray | None = None) -> MultiPolygon:
    """Cross-section of one mesh at height z as a MultiPolygon.  With
    ``shells`` (see ``shell_labels``) every closed shell is outlined on its
    own and the outlines are united, so overlapping parts add up."""
    if shells is None:
        segments = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z])
        return segments_to_multipolygon(segments[:, :, :2]) if len(segments) else MultiPolygon()
    segments, faces = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z],
                                                       return_faces=True)
    if len(segments) == 0:
        return MultiPolygon()
    return segments_to_multipolygon(segments[:, :, :2], shells[faces])


class MeshSlicer:
    """Cross-sections of one mesh at many heights - the same as
    ``slice_mesh_at`` for each, without looking at every face every time.

    Each slice hands trimesh only the faces whose z range reaches that
    height, in their original order, so the segments (and so the outline)
    come out exactly as from the whole mesh.  The faces are kept sorted by
    their lowest point; the few much taller than the rest are kept apart and
    always checked, so the sorted ones can be found from a short window."""

    def __init__(self, mesh: trimesh.Trimesh, shells: np.ndarray | None = None):
        self.mesh, self.shells = mesh, shells
        z = np.asarray(mesh.vertices)[np.asarray(mesh.faces)][:, :, 2] if len(mesh.faces) else np.zeros((0, 3))
        self.zmin, self.zmax = z.min(axis=1), z.max(axis=1)
        h = self.zmax - self.zmin
        tall = h > max(8.0 * float(np.median(h)), 1e-6) if len(h) else np.zeros(0, bool)
        self.tall = np.flatnonzero(tall)
        short = np.flatnonzero(~tall)
        self.short = short[np.argsort(self.zmin[short], kind="stable")]
        self.short_zmin = self.zmin[self.short]
        self.reach = float(h[short].max()) if len(short) else 0.0
        self.pad = 1e-6 * (1.0 + float(np.abs(z).max())) if len(z) else 1e-6   # well above trimesh's tolerance
        self.vz = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64)[:, 2])

    def faces_at(self, z: float) -> np.ndarray:
        """Faces (in increasing order) that may cross the plane at height z."""
        a = np.searchsorted(self.short_zmin, z - self.pad - self.reach, side="left")
        b = np.searchsorted(self.short_zmin, z + self.pad, side="right")
        near = self.short[a:b]
        near = near[self.zmax[near] >= z - self.pad]
        tall = self.tall[(self.zmin[self.tall] <= z + self.pad) & (self.zmax[self.tall] >= z - self.pad)]
        return np.sort(np.concatenate([near, tall]))

    def at(self, z: float) -> MultiPolygon:
        faces = self.faces_at(z)
        if len(faces) == 0:
            return MultiPolygon()
        dots = self.vz - z                                # what trimesh would work out: the height above z
        if self.shells is None:
            segments = trimesh.intersections.mesh_plane(self.mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z],
                                                        local_faces=faces, cached_dots=dots)
            return segments_to_multipolygon(segments[:, :, :2]) if len(segments) else MultiPolygon()
        segments, index = trimesh.intersections.mesh_plane(self.mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z],
                                                           return_faces=True, local_faces=faces, cached_dots=dots)
        if len(segments) == 0:
            return MultiPolygon()
        return segments_to_multipolygon(segments[:, :, :2], self.shells[index])


def _union(parts: list) -> MultiPolygon:
    """Union that survives slightly invalid input (slivers from meshes)."""
    if not parts:
        return MultiPolygon()
    try:
        return _as_multi(unary_union(parts))
    except Exception:
        fixed = [shapely.make_valid(p) for p in parts]
        return _as_multi(unary_union(fixed))


def _as_multi(geom) -> MultiPolygon:
    if geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    polys = [g for g in shapely.get_parts(geom) if isinstance(g, Polygon) and not g.is_empty]
    return MultiPolygon(polys)


# --------------------------------------------------------------------------
# the slicer

def slice_scene(scene: Scene, printer: PrinterProfile, resin: ResinProfile,
                progress: ProgressFn | None = None) -> SliceResult:
    models, support, jobs = _world_meshes(scene)
    if not models and not jobs:
        raise ValueError("Nothing to slice - add a model first.")

    bounds = scene.bounds()
    every = [m for m, _c in models] + [j.mesh for j in jobs] + ([support[0]] if support else [])
    height = max(m.bounds[1][2] for m in every)
    schedule = layer_schedule(float(stretched_z(height, resin)), resin)
    n = len(schedule)
    total = n * (1 + len(jobs))
    # where each layer cuts the model: its center, unless the resin scales
    # layers in Z or stretches the lowest ones
    zc = unstretched_z([p["cut_z"] for p in schedule], resin)
    # model footprints and centers: X/Y scale compensation works about these
    owners = [(m.bounds, c) for m, c in models] + [(j.mesh.bounds, j.center) for j in jobs]
    boxes = np.array([[lo[0], lo[1], hi[0], hi[1]] for (lo, hi), _c in owners])
    centers = np.array([c for _b, c in owners])

    # 1. cut every layer: the plain geometry united, hollowed models kept apart
    plain: list[MultiPolygon] = []
    solids: list[list] = [[] for _ in jobs]
    slicers = [(MeshSlicer(m), m.bounds, c) for m, c in models]
    job_slicers = [MeshSlicer(job.mesh) for job in jobs]
    support_slicer = (MeshSlicer(*support), support[0].bounds) if support else None
    for p, z in zip(schedule, zc):
        sx, sy, _sz = p["scale"]
        parts = [_scaled(sl.at(z), sx, sy, c) for sl, b, c in slicers if b[0][2] <= z <= b[1][2]]
        if support_slicer is not None and support_slicer[1][0][2] <= z <= support_slicer[1][1][2]:
            parts += _scaled_supports(support_slicer[0].at(z), sx, sy, boxes, centers)
        plain.append(_union([g for g in parts if not g.is_empty]))
        for job, sl, out in zip(jobs, job_slicers, solids):
            lo, hi = job.mesh.bounds
            out.append(sl.at(z) if lo[2] <= z <= hi[2] else None)
        if progress:
            progress(p["index"] + 1, total)

    # 2. hollow those models (each layer needs its neighbours) and add them
    extra: list[list] = [[] for _ in range(n)]
    for q, (job, out) in enumerate(zip(jobs, solids)):
        step = (lambda d, _t, q=q: progress(n * (1 + q) + d, total)) if progress else None
        for i, g in enumerate(hollow_layers(out, zc, job.state, job.holes, job.poles, step)):
            if not g.is_empty:
                sx, sy, _sz = schedule[i]["scale"]
                extra[i].append(_scaled(g, sx, sy, job.center))

    layers: list[Layer] = []
    for p, geom, more in zip(schedule, plain, extra):
        if more:
            geom = _union([g for g in [geom, *more] if not g.is_empty])
        layers.append(Layer(geometry=geom, **{k: v for k, v in p.items() if k not in LAYER_ONLY}))

    return SliceResult(layers=layers, printer=printer, resin=resin,
                       model_bounds=bounds,
                       object_names=[o.name for o in scene.objects if o.visible],
                       islands=find_islands(layers))
