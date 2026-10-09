"""Tiny software renderer for preview thumbnails.

Painter's algorithm with flat shading via Pillow: plenty for a thumbnail
and it works headless (no OpenGL needed), so the CLI slicer can make the
same previews as the GUI.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image, ImageDraw

MAX_FACES = 200_000


def render_preview(meshes: list[trimesh.Trimesh], size: tuple[int, int],
                   background=(0.93, 0.94, 0.96), model_color=(0.17, 0.60, 0.62),
                   azimuth_deg: float = 35.0, elevation_deg: float = 30.0) -> Image.Image:
    w, h = size
    bg = tuple(int(c * 255) for c in background)
    img = Image.new("RGBA", (w, h), bg + (255,))
    valid = [m for m in meshes if len(m.faces)]
    if not valid:
        return img
    mesh = trimesh.util.concatenate(valid) if len(valid) > 1 else valid[0]
    if len(mesh.faces) > MAX_FACES:
        # keep previews fast on huge meshes: draw a random subset of faces
        idx = np.random.default_rng(0).choice(len(mesh.faces), MAX_FACES, replace=False)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[idx], process=False)

    # camera: look at the model center from an elevated, rotated viewpoint
    center = mesh.bounds.mean(axis=0)
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    rz = trimesh.transformations.rotation_matrix(-az, [0, 0, 1])
    rx = trimesh.transformations.rotation_matrix(-(np.pi / 2 - el), [1, 0, 0])
    view = rx @ rz @ trimesh.transformations.translation_matrix(-center)
    verts = trimesh.transform_points(mesh.vertices, view)      # camera space, -Z forward

    # orthographic fit with a margin
    xy = verts[:, :2]
    span = (xy.max(axis=0) - xy.min(axis=0)).max()
    scale = 0.8 * min(w, h) / max(span, 1e-6)
    mid = (xy.max(axis=0) + xy.min(axis=0)) / 2
    px = (xy[:, 0] - mid[0]) * scale + w / 2
    py = h / 2 - (xy[:, 1] - mid[1]) * scale

    # shading from the face normal in camera space
    normals = trimesh.transform_points(mesh.face_normals, view, translate=False)
    light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    lit = np.clip(normals @ light, 0, 1) * 0.65 + 0.35

    depth = verts[mesh.faces][:, :, 2].mean(axis=1)
    order = np.argsort(depth)                                  # far to near
    draw = ImageDraw.Draw(img)
    base = np.array(model_color)
    faces = mesh.faces
    for fi in order:
        if normals[fi, 2] <= 0:          # back face
            continue
        c = tuple(int(v) for v in np.clip(base * lit[fi] * 255, 0, 255))
        tri = [(px[i], py[i]) for i in faces[fi]]
        draw.polygon(tri, fill=c + (255,))
    return img
