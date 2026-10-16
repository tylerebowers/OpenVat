"""Move arrows and rotate rings for the selected model (pure numpy, no GL).

The gizmos sit at the model's center (its pivot, ``MeshObject.position``)
and keep the same size on screen however far the camera is: ``world_size``
turns pixels into millimeters at that depth.  The meshes here are built
once at unit size along / around +Z; ``axis_frame`` turns them onto each
world axis.

Picking works in screen space (distance from the mouse to the projected
arrow or ring), which is cheap and stays forgiving when the arrows are thin.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

AXIS_COLORS = ((0.90, 0.29, 0.29, 1.0), (0.35, 0.76, 0.32, 1.0), (0.29, 0.52, 0.95, 1.0))   # X, Y, Z
HOVER_COLOR = (1.0, 0.80, 0.22, 1.0)
FACE_COLOR = (0.15, 0.78, 0.86, 0.55)      # "lay on face" candidates (stands out from the model colors)
FACE_HOVER_COLOR = (0.62, 0.96, 1.0, 0.90)

ARROW_PX = 95          # arrow length on screen
ARROW_START = 0.16     # the shaft starts this far out (share of the length): the center stays clear
RING_PX = 80           # ring radius on screen
PICK_PX = 9            # how close (px) the mouse has to be to grab a handle
RING_SAMPLES = 72


# --------------------------------------------------------------------------
# geometry (unit size, along / around +Z)

def arrow_mesh() -> trimesh.Trimesh:
    """An arrow from ARROW_START to 1 along +Z: a shaft and a cone."""
    head = 0.24
    shaft = trimesh.creation.cylinder(radius=0.022, segment=[[0, 0, ARROW_START], [0, 0, 1 - head]], sections=16)
    cone = trimesh.creation.cone(radius=0.075, height=head, sections=24)
    cone.apply_translation([0, 0, 1 - head])
    return trimesh.util.concatenate([shaft, cone])


def ring_mesh(tube: float = 0.026, segments: int = 96, sides: int = 10) -> trimesh.Trimesh:
    """A thin torus of radius 1 in the XY plane."""
    u = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    v = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    r = 1 + tube * np.cos(vv)
    verts = np.stack([r * np.cos(uu), r * np.sin(uu), tube * np.sin(vv)], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(segments), np.arange(sides), indexing="ij")
    a = i * sides + j
    b = ((i + 1) % segments) * sides + j
    c = ((i + 1) % segments) * sides + (j + 1) % sides
    d = i * sides + (j + 1) % sides
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return trimesh.Trimesh(verts, faces, process=False)


def axis_frame(axis: int) -> np.ndarray:
    """3x3 rotation taking +Z onto world axis ``axis`` (0 = X, 1 = Y, 2 = Z)."""
    if axis == 0:
        return np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])      # z -> x
    if axis == 1:
        return np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])      # z -> y
    return np.eye(3)


def model_matrix(center, axis: int, size: float) -> np.ndarray:
    """4x4 placing a unit gizmo part on ``axis`` at ``center``, ``size`` mm big."""
    m = np.eye(4)
    m[:3, :3] = axis_frame(axis) * size
    m[:3, 3] = center
    return m


# --------------------------------------------------------------------------
# screen space

def world_size(camera, center, pixels: float, height_px: int) -> float:
    """How many mm ``pixels`` screen pixels span at ``center``'s depth."""
    depth = max(float((np.asarray(center, float) - camera.position()) @ camera.forward()), 1e-3)
    return pixels * 2.0 * depth * math.tan(math.radians(camera.fov) / 2) / max(height_px, 1)


def project(vp: np.ndarray, points, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Screen positions (px, y down) of world points, and which are in front
    of the camera."""
    p = np.atleast_2d(np.asarray(points, float))
    c = np.hstack([p, np.ones((len(p), 1))]) @ vp.T
    w = c[:, 3]
    front = w > 1e-9
    w = np.where(front, w, 1e-9)
    x = (c[:, 0] / w + 1) / 2 * width
    y = (1 - c[:, 1] / w) / 2 * height
    return np.stack([x, y], axis=1), front


def segment_distance(p, a, b) -> float:
    """Distance from 2D point ``p`` to the segment a-b."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def polyline_distance(p, pts: np.ndarray, closed: bool = True) -> float:
    """Distance from 2D point ``p`` to a polyline (vectorised)."""
    a = np.asarray(pts, float)
    b = np.roll(a, -1, axis=0) if closed else a[1:]
    a = a if closed else a[:-1]
    ab = b - a
    ap = np.asarray(p, float) - a
    den = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / den, 0.0, 1.0)
    d = ap - ab * t[:, None]
    return float(np.sqrt(np.einsum("ij,ij->i", d, d)).min())


def ring_points(center, axis: int, radius: float, n: int = RING_SAMPLES) -> np.ndarray:
    """World points around the ring of ``axis``."""
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local = np.stack([np.cos(u), np.sin(u), np.zeros_like(u)], axis=1) * radius
    return local @ axis_frame(axis).T + np.asarray(center, float)


def pick(vp: np.ndarray, camera, center, width: int, height: int, mouse,
         arrows: bool, rings: bool) -> tuple[str, int] | None:
    """The handle under the mouse: ("move", axis), ("rotate", axis) or None."""
    best, best_d = None, PICK_PX
    c_px, front = project(vp, [center], width, height)
    if not front[0]:
        return None
    if arrows:
        size = world_size(camera, center, ARROW_PX, height)
        for axis in range(3):
            e = np.eye(3)[axis]
            ends, ok = project(vp, [center + e * size * ARROW_START, center + e * size], width, height)
            if ok.all():
                d = segment_distance(mouse, ends[0], ends[1])
                if d < best_d:
                    best, best_d = ("move", axis), d
    if rings:
        size = world_size(camera, center, RING_PX, height)
        for axis in range(3):
            pts, ok = project(vp, ring_points(center, axis, size), width, height)
            if ok.all():
                d = polyline_distance(mouse, pts)
                if d < best_d:
                    best, best_d = ("rotate", axis), d
    return best


# --------------------------------------------------------------------------
# dragging

def axis_parameter(origin, direction, center, axis_dir) -> float | None:
    """Where along the line center + t * axis_dir the mouse ray passes
    closest (None when the ray runs along the axis)."""
    d1 = np.asarray(axis_dir, float)
    d2 = np.asarray(direction, float)
    w = np.asarray(center, float) - np.asarray(origin, float)
    b = d1 @ d2
    den = 1.0 - b * b                                       # both unit vectors
    if den < 1e-6:
        return None
    return float((b * (d2 @ w) - (d1 @ w)) / den)


def plane_angle(origin, direction, center, axis: int) -> float | None:
    """Angle (radians, right-handed about ``axis``) of the point where the
    mouse ray meets the ring's plane, or None when it does not."""
    n = np.eye(3)[axis]
    d = np.asarray(direction, float)
    den = float(d @ n)
    if abs(den) < 1e-6:
        return None
    t = float((np.asarray(center, float) - np.asarray(origin, float)) @ n) / den
    if t <= 0:
        return None
    p = np.asarray(origin, float) + t * d - np.asarray(center, float)
    frame = axis_frame(axis)
    return math.atan2(float(p @ frame[:, 1]), float(p @ frame[:, 0]))


EDGE_ON = 0.2          # |ray . axis| below this: the ring is seen nearly edge-on (turn by screen angle)


def screen_angle(center_px, mouse) -> float:
    """Angle (radians, counter-clockwise on screen) of the mouse around a point."""
    return math.atan2(-(mouse[1] - center_px[1]), mouse[0] - center_px[0])


def unwrap(delta: float) -> float:
    """An angle step folded into -pi..pi (crossing the +-pi seam)."""
    return (delta + math.pi) % (2 * math.pi) - math.pi


def ray_triangles(origin, direction, tri: np.ndarray) -> float | None:
    """Distance along the ray to the nearest of the triangles (n, 3, 3), or None."""
    tri = np.asarray(tri, float)
    if not len(tri):
        return None
    d = np.asarray(direction, float)
    v0 = tri[:, 0]
    e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = np.asarray(origin, float) - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return float(t[hit].min()) if hit.any() else None
