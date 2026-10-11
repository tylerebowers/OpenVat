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


def _node_ids(pts: np.ndarray) -> np.ndarray:
    """One id per point, the same for points that snap to the same grid
    point (``SNAP``)."""
    keys = np.round(pts / SNAP).astype(np.int64)
    k0, k1 = keys[:, 0] - keys[:, 0].min(), keys[:, 1] - keys[:, 1].min()
    span = int(k1.max()) + 1
    if int(k0.max()) < (1 << 62) // span:                 # both in one integer: a plain 1-D sort
        _, inverse = np.unique(k0 * span + k1, return_inverse=True)
    else:
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def chain_segments(segments: np.ndarray, labels: np.ndarray | None = None):
    """segments: (N, 2, 2) array of 2D segments -> list of closed loops,
    each an (M, 2) array of points (first point not repeated at the end).
    With ``labels`` (one per segment) returns (loops, label of each loop).

    Each loop starts at the lowest-numbered segment not used yet, runs from
    its first end to its second, and at every point takes the
    lowest-numbered unused segment there."""
    if len(segments) == 0:
        return ([], []) if labels is not None else []
    inverse = _node_ids(segments.reshape(-1, 2)).reshape(-1, 2)     # (N, 2) node ids per segment

    # drop zero-length segments
    keep = inverse[:, 0] != inverse[:, 1]
    inverse, segments = inverse[keep], segments[keep]
    if labels is not None:
        labels = np.asarray(labels)[keep]
    n = len(segments)
    if n == 0:
        return ([], []) if labels is not None else []

    # the segments at each point, lowest-numbered first (plain lists: the walk is a Python loop)
    ends = inverse.ravel()
    order = np.argsort(ends, kind="stable")
    first = np.searchsorted(ends[order], np.arange(int(ends.max()) + 2)).tolist()
    at = (order // 2).tolist()
    nodes = inverse.tolist()
    flat = segments.reshape(-1, 2)

    used = bytearray(n)
    loops: list[np.ndarray] = []
    loop_labels: list = []
    for start in range(n):
        if used[start]:
            continue
        used[start] = 1
        first_node, node = nodes[start]
        ends_taken = [2 * start]                          # (segment, end) as 2 * segment + end
        while node != first_node:
            nxt = -1
            for k in range(first[node], first[node + 1]):
                if not used[at[k]]:
                    nxt = at[k]
                    break
            if nxt < 0:
                break
            used[nxt] = 1
            s0, s1 = nodes[nxt]
            if s0 == node:
                ends_taken.append(2 * nxt)
                node = s1
            else:
                ends_taken.append(2 * nxt + 1)
                node = s0
        if len(ends_taken) >= 3:
            # Open chains (from non-watertight meshes) are closed with a
            # straight line - better a slightly wrong layer than a missing one.
            loops.append(np.asarray(flat[ends_taken], dtype=np.float64))
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
