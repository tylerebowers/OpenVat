"""Turn the loose line segments of a mesh cross-section into polygons.

mesh/plane intersection yields one segment per triangle, in no particular
order.  We snap endpoints to a fine grid, chain segments into closed loops,
then nest loops by containment: a loop at even depth is an outer boundary,
a loop at odd depth is a hole in the loop that contains it.

A mesh made of several closed shells that overlap (a support: rods, bulbs
and cones simply placed together) must not be nested as one - a bulb's
outline inside a rod's would become a hole.  Give ``segments_to_multipolygon``
a shell label per segment and each shell is nested on its own, then all are
united.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

SNAP = 1e-4          # mm, grid used to merge coincident endpoints


def chain_segments(segments: np.ndarray, labels: np.ndarray | None = None):
    """segments: (N, 2, 2) array of 2D segments -> list of closed loops,
    each an (M, 2) array of points (first point not repeated at the end).
    With ``labels`` (one per segment) returns (loops, label of each loop)."""
    if len(segments) == 0:
        return ([], []) if labels is not None else []
    pts = segments.reshape(-1, 2)
    keys = np.round(pts / SNAP).astype(np.int64)
    # map each distinct snapped point to an index
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1, 2)          # (N, 2) node ids per segment

    # drop zero-length segments
    keep = inverse[:, 0] != inverse[:, 1]
    inverse, segments = inverse[keep], segments[keep]
    if labels is not None:
        labels = np.asarray(labels)[keep]
    n = len(segments)
    if n == 0:
        return ([], []) if labels is not None else []

    adjacency: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(inverse):
        adjacency.setdefault(int(a), []).append(i)
        adjacency.setdefault(int(b), []).append(i)

    used = np.zeros(n, dtype=bool)
    loops: list[np.ndarray] = []
    loop_labels: list = []
    for start in range(n):
        if used[start]:
            continue
        used[start] = True
        a, b = int(inverse[start, 0]), int(inverse[start, 1])
        loop_pts = [segments[start, 0]]
        first_node, node = a, b
        while node != first_node:
            nxt = next((s for s in adjacency[node] if not used[s]), None)
            if nxt is None:
                break
            used[nxt] = True
            s0, s1 = int(inverse[nxt, 0]), int(inverse[nxt, 1])
            if s0 == node:
                loop_pts.append(segments[nxt, 0]); node = s1
            else:
                loop_pts.append(segments[nxt, 1]); node = s0
        if len(loop_pts) >= 3:
            # Open chains (from non-watertight meshes) are closed with a
            # straight line - better a slightly wrong layer than a missing one.
            loops.append(np.asarray(loop_pts, dtype=np.float64))
            if labels is not None:
                loop_labels.append(labels[start])
    return (loops, loop_labels) if labels is not None else loops


def loops_to_multipolygon(loops: list[np.ndarray]) -> MultiPolygon:
    """Nest closed loops into polygons with holes using even-odd depth."""
    rings = []
    for pts in loops:
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-9:
                continue
            # buffer(0) can split a bow-tie into pieces; keep each piece
            parts = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
            rings.extend(p for p in parts if isinstance(p, Polygon) and p.area > 1e-9)
        except Exception:
            continue
    if not rings:
        return MultiPolygon()

    rings.sort(key=lambda p: p.area, reverse=True)
    depth = [0] * len(rings)
    parent = [-1] * len(rings)
    for i, r in enumerate(rings):
        probe = r.representative_point()
        # the smallest ring that contains this one is its parent
        for j in range(i - 1, -1, -1):
            if rings[j].contains(probe):
                parent[i] = j
                depth[i] = depth[j] + 1
                break

    polys = []
    for i, r in enumerate(rings):
        if depth[i] % 2 == 0:
            holes = [rings[k].exterior.coords for k in range(len(rings)) if parent[k] == i]
            polys.append(Polygon(r.exterior.coords, holes))
    try:
        merged = unary_union(polys)
    except Exception:
        merged = unary_union([shapely.make_valid(p) for p in polys])
    if merged.is_empty:
        return MultiPolygon()
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    if isinstance(merged, MultiPolygon):
        return merged
    return MultiPolygon([g for g in merged.geoms if isinstance(g, Polygon)])


def segments_to_multipolygon(segments: np.ndarray, labels: np.ndarray | None = None) -> MultiPolygon:
    """Polygons from cross-section segments.  ``labels`` (the closed shell
    each segment comes from) nests every shell on its own and unites them."""
    segments = np.asarray(segments, dtype=np.float64)
    if labels is None:
        return loops_to_multipolygon(chain_segments(segments))
    loops, loop_labels = chain_segments(segments, labels)
    if not loops:
        return MultiPolygon()
    loop_labels = np.asarray(loop_labels)
    shells = [loops_to_multipolygon([loops[i] for i in np.flatnonzero(loop_labels == lab)])
              for lab in np.unique(loop_labels)]
    shells = [g for g in shells if not g.is_empty]
    if len(shells) == 1:
        return shells[0]
    try:
        merged = unary_union(shells)
    except Exception:
        merged = unary_union([shapely.make_valid(g) for g in shells])
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    if isinstance(merged, MultiPolygon):
        return merged
    return MultiPolygon([g for g in shapely.get_parts(merged) if isinstance(g, Polygon)])
