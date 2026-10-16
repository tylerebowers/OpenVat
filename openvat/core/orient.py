"""Orienting a model: rotations about world axes and "lay on face".

A model's rotation is stored as X/Y/Z angles in degrees, applied as
Rz @ Ry @ Rx about the model's center (see ``MeshObject.matrix``).  The
rotate rings of the 3D view turn the model about a *world* axis, and "lay on
face" turns it so a chosen face points straight down; both work on the
rotation matrix and convert back to angles.

``flat_faces`` lists the faces a model can be laid on: flat regions of its
surface whose plane has the whole model on one side - the planes of its
convex hull that the model actually touches.  Laid on one of them, nothing
ends up below the build plate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

DOWN = np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------
# rotations

def rotation_matrix(angles_deg) -> np.ndarray:
    """3x3 matrix of X/Y/Z angles in degrees, as the model uses them (Rz @ Ry @ Rx)."""
    ax, ay, az = np.radians(np.asarray(angles_deg, dtype=float))
    return trimesh.transformations.euler_matrix(ax, ay, az, "sxyz")[:3, :3]


def angles_of(matrix: np.ndarray) -> np.ndarray:
    """X/Y/Z angles in degrees (-180..180] of a rotation matrix; the inverse
    of ``rotation_matrix``.  Rounded to 1e-6 degrees, so a quarter turn reads
    90, not 89.99999999999999."""
    m = np.eye(4)
    m[:3, :3] = matrix
    angles = np.degrees(trimesh.transformations.euler_from_matrix(m, "sxyz"))
    angles = np.round(angles, 6) + 0.0                     # + 0.0: no -0.0
    angles[angles <= -180.0] += 360.0
    return angles


def axis_rotation(axis, degrees: float) -> np.ndarray:
    """3x3 rotation by ``degrees`` about the (world) direction ``axis``."""
    return trimesh.transformations.rotation_matrix(np.radians(degrees), np.asarray(axis, float))[:3, :3]


def rotated_about(angles_deg, axis, degrees: float) -> np.ndarray:
    """The model's angles after turning it ``degrees`` about the world
    direction ``axis`` (through its center)."""
    return angles_of(axis_rotation(axis, degrees) @ rotation_matrix(angles_deg))


def turn_onto(direction, target=DOWN) -> np.ndarray:
    """The smallest rotation that turns ``direction`` onto ``target``."""
    a = np.asarray(direction, float)
    a = a / np.linalg.norm(a)
    b = np.asarray(target, float)
    b = b / np.linalg.norm(b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:                                   # opposite: half a turn about any perpendicular
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return axis_rotation(perp, 180.0)
    return axis_rotation(np.cross(a, b), np.degrees(np.arccos(c)))


def laid_on(angles_deg, normal) -> np.ndarray:
    """The model's angles after turning it so the face with world normal
    ``normal`` points straight down (onto the build plate)."""
    return angles_of(turn_onto(normal) @ rotation_matrix(angles_deg))


# --------------------------------------------------------------------------
# faces to lay a model on

@dataclass(eq=False)
class FlatFace:
    """A flat region of the model's surface that it can lie on."""
    normal: np.ndarray          # outward unit normal (world)
    area: float                 # mm^2 of the model in that plane (what touches the plate)
    faces: np.ndarray           # indices of the model's triangles in that plane


def flat_faces(mesh: trimesh.Trimesh, limit: int = 24, min_share: float = 0.01,
               min_area: float = 1.0, min_of_largest: float = 0.1) -> list[FlatFace]:
    """The model's largest flat faces it can be laid on, biggest first.

    Candidates are the planes of the convex hull: the whole model lies on
    one side of each, so laid on one nothing goes below the plate.  The
    model's own triangles in such a plane are the face (a hull plane that
    only spans a gap, under an arch say, touches nothing and is skipped).
    Faces smaller than ``min_share`` of the hull's surface, ``min_area``
    mm^2 or ``min_of_largest`` of the largest face are left out (so the
    strips of a curved surface do not count).

    The faces stay the same when the model is moved, turned or scaled, so
    this can be worked out once on the source mesh."""
    if len(mesh.faces) < 4:
        return []
    try:
        hull = mesh.convex_hull
    except Exception:                                       # flat or degenerate: no volume
        return []
    if len(hull.faces) < 4:
        return []
    groups = [np.asarray(f) for f in hull.facets]
    grouped = np.zeros(len(hull.faces), bool)
    for f in groups:
        grouped[f] = True
    groups += [np.array([i]) for i in np.flatnonzero(~grouped)]
    areas = np.array([hull.area_faces[g].sum() for g in groups])
    least = max(min_area, min_share * float(hull.area))
    order = [k for k in np.argsort(-areas) if areas[k] >= least]   # a hull plane bounds what touches it

    tol = max(1e-4 * float(mesh.scale), 1e-6)               # mm off the plane still counts as on it
    normals = np.asarray(mesh.face_normals)
    tri = np.asarray(mesh.triangles)
    face_area = np.asarray(mesh.area_faces)
    found: list[FlatFace] = []
    planes: list[tuple[np.ndarray, float]] = []
    for k in order:
        g = groups[k]
        n = hull.area_faces[g] @ hull.face_normals[g]       # area-weighted: one normal for the plane
        n = n / np.linalg.norm(n)
        d = float(hull.vertices[hull.faces[g[0], 0]] @ n)
        if any(n @ m > 1 - 1e-6 and abs(d - e) < tol for m, e in planes):
            continue                                        # the same plane, split in two by the hull
        planes.append((n, d))
        cand = np.flatnonzero(normals @ n > 1 - 1e-4)       # triangles facing the same way...
        if not len(cand):
            continue
        dist = np.abs(tri[cand] @ n - d).max(axis=1)        # ...lying in the plane
        on = cand[dist <= tol]
        area = float(face_area[on].sum())
        if area >= least:
            found.append(FlatFace(n, area, on))
            if len(found) >= limit:
                break
    found.sort(key=lambda f: -f.area)
    return [f for f in found if f.area >= min_of_largest * found[0].area]
