"""Slicing: turn the scene into a stack of 2D layers.

Each layer is a shapely (Multi)Polygon in build-plate millimeters (origin at
the plate center).  Keeping layers as vectors rather than bitmaps is what the
Anycubic .pwsz format wants, and it also keeps memory low; the layer viewer
rasterizes on demand.

Pipeline for one layer:
    1. intersect every world-space mesh with the plane z = layer center
    2. chain the resulting segments into closed polygons (with holes)
    3. union the polygons of all objects/supports
    4. bottom layers: elephant-foot compensation - every edge is moved
       inward by ``elephant_foot_mm`` (negative polygon offset; holes grow)

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

def layer_schedule(height: float, resin: ResinProfile) -> list[dict]:
    """Return the per-layer parameters (without geometry) for a model of the
    given height.  Bottom layers, optional transition layers, normal layers."""
    out: list[dict] = []
    z = 0.0
    n_bottom = resin.bottom_layers
    n_trans = max(resin.transition_layers, 0)
    b, n = resin.bottom, resin.normal
    i = 0
    while z < height - 1e-6:
        if i < n_bottom:
            p = dict(thickness=b.thickness, exposure=b.exposure, off_time=b.off_time,
                     wait_before_lift=b.wait_before_lift, wait_after_lift=b.wait_after_lift,
                     lift_distance=b.lift_distance, lift_speed=b.lift_speed,
                     retract_speed=b.retract_speed)
        else:
            p = dict(thickness=n.thickness, exposure=n.exposure, off_time=n.off_time,
                     wait_before_lift=n.wait_before_lift, wait_after_lift=n.wait_after_lift,
                     lift_distance=n.lift_distance, lift_speed=n.lift_speed,
                     retract_speed=n.retract_speed)
            k = i - n_bottom
            if k < n_trans:
                # linear ramp from bottom exposure down to normal exposure
                t = (k + 1) / (n_trans + 1)
                p["exposure"] = b.exposure + (n.exposure - b.exposure) * t
        p["index"] = i
        p["z_bottom"] = z
        out.append(p)
        z += p["thickness"]
        i += 1
    return out


# --------------------------------------------------------------------------
# geometry helpers

@dataclass
class _HollowJob:
    """A hollowed model: its world mesh and what hollowing adds (world space,
    shrinkage compensated like the mesh)."""
    mesh: trimesh.Trimesh
    state: HollowState
    holes: tuple                                  # (starts, ends, radii)
    poles: tuple | None                           # (starts, ends) of the internal support poles


def _world_meshes(scene: Scene, resin: ResinProfile) -> tuple[list, list[_HollowJob]]:
    """All printable geometry in world space with shrinkage compensation:
    (mesh, shell labels) pairs, and the hollowed models apart.  Models are
    sliced as they are (even-odd, so models that come with cavities keep
    them); support meshes are made of overlapping closed shells and carry a
    shell label per face."""
    meshes, jobs = [], []
    for obj in scene.objects:
        if not obj.visible:
            continue
        m = obj.transformed()
        shrink = _shrink_matrix(m, resin)
        if shrink is not None:
            m.apply_transform(shrink)
        if obj.hollow is None:
            meshes.append((m, None))
            continue
        state = obj.hollow
        starts, ends, radii = hole_segments(obj.matrix(), obj.holes, hole_depth(state.thickness))
        poles = None
        if state.lattice is not None:
            lo, hi = m.bounds
            poles = lattice_segments(lo, hi, state.lattice)
        if shrink is not None and len(starts):
            starts, ends = (trimesh.transform_points(p, shrink) for p in (starts, ends))
        jobs.append(_HollowJob(m, state, (starts, ends, radii), poles))
    supports = scene.support_meshes()
    if supports:                                  # one mesh, so each layer is cut once
        m = trimesh.util.concatenate(supports)
        meshes.append((_apply_shrink(m, resin), shell_labels(m)))
    return meshes, jobs


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


def _shrink_matrix(m: trimesh.Trimesh, resin: ResinProfile) -> np.ndarray | None:
    """Shrinkage compensation as PhotonWorkshop does it: X/Y scaled about the
    mesh's own center, Z scaled up from the build plate (z = 0).  None when
    the resin does not shrink."""
    if resin.shrink_x == 1.0 and resin.shrink_y == 1.0 and resin.shrink_z == 1.0:
        return None
    c = m.bounds.mean(axis=0)
    s = np.diag([resin.shrink_x, resin.shrink_y, resin.shrink_z, 1.0])
    return (trimesh.transformations.translation_matrix([c[0], c[1], 0])
            @ s @ trimesh.transformations.translation_matrix([-c[0], -c[1], 0]))


def _apply_shrink(m: trimesh.Trimesh, resin: ResinProfile) -> trimesh.Trimesh:
    shrink = _shrink_matrix(m, resin)
    if shrink is not None:
        m.apply_transform(shrink)
    return m


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
    meshes, jobs = _world_meshes(scene, resin)
    if not meshes and not jobs:
        raise ValueError("Nothing to slice - add a model first.")

    bounds = scene.bounds()
    height = max(m.bounds[1][2] for m in [m for m, _shells in meshes] + [j.mesh for j in jobs])
    schedule = layer_schedule(height, resin)
    n = len(schedule)
    total = n * (1 + len(jobs))
    zc = np.array([p["z_bottom"] + p["thickness"] / 2 for p in schedule])

    # 1. cut every layer: the plain geometry united, hollowed models kept apart
    plain: list[MultiPolygon] = []
    solids: list[list] = [[] for _ in jobs]
    for p, z in zip(schedule, zc):
        parts = [slice_mesh_at(m, z, shells) for m, shells in meshes if m.bounds[0][2] <= z <= m.bounds[1][2]]
        plain.append(_union([g for g in parts if not g.is_empty]))
        for job, out in zip(jobs, solids):
            lo, hi = job.mesh.bounds
            out.append(slice_mesh_at(job.mesh, z) if lo[2] <= z <= hi[2] else None)
        if progress:
            progress(p["index"] + 1, total)

    # 2. hollow those models (each layer needs its neighbours) and add them
    extra: list[list] = [[] for _ in range(n)]
    for q, (job, out) in enumerate(zip(jobs, solids)):
        step = (lambda d, _t, q=q: progress(n * (1 + q) + d, total)) if progress else None
        for i, g in enumerate(hollow_layers(out, zc, job.state, job.holes, job.poles, step)):
            if not g.is_empty:
                extra[i].append(g)

    layers: list[Layer] = []
    for p, geom, more in zip(schedule, plain, extra):
        if more:
            geom = _union([g for g in [geom, *more] if not g.is_empty])
        if p["index"] < resin.bottom_layers and resin.elephant_foot_mm > 0 and not geom.is_empty:
            geom = _as_multi(geom.buffer(-resin.elephant_foot_mm, join_style="mitre"))
        layers.append(Layer(geometry=geom, **p))

    return SliceResult(layers=layers, printer=printer, resin=resin,
                       model_bounds=bounds,
                       object_names=[o.name for o in scene.objects if o.visible],
                       islands=find_islands(layers))
