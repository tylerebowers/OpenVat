"""Problems with the plate as it is now, for the Problems list.

``find_problems`` runs a set of quick checks (it is called after every
edit) and returns them worst first:

* errors: a model outside the build volume, a model floating above the
  plate without supports, islands in the sliced layers (with the layer, so
  the list can show it), no resin profile for the printer;
* warnings: models overlapping each other, a hollow model without a drain
  hole (resin stays trapped inside), a mesh with holes (not watertight),
  spots the last Automatic run could not support, too little support for
  the release force;
* notes: a printer OpenVat can slice for but not write files for.

Qt-free; the overlap test compares the models' footprints (convex hulls
seen from above) where their boxes meet, kept per mesh, turn and scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon

ERROR, WARNING, NOTE = "error", "warning", "note"
_ORDER = {ERROR: 0, WARNING: 1, NOTE: 2}
FLOAT_GAP = 0.05             # mm: a model whose lowest point is higher than this floats
OVERLAP_AREA = 0.5           # mm^2 of shared footprint before two models count as overlapping

_HULLS: dict = {}            # (mesh, rotation, scale) -> footprint around the model's own center


@dataclass
class Problem:
    level: str               # ERROR / WARNING / NOTE
    text: str
    object_id: int | None = None     # the model to select when it is clicked
    layer: int | None = None         # the layer to show (islands)
    kind: str = ""                   # what was checked ("outside", "floating", ...)


def find_problems(scene, printer=None, resin=None, result=None, can_export: str = "") -> list[Problem]:
    """Problems of ``scene`` on ``printer`` with ``resin``.  ``result``: the
    sliced layers if they match the plate as it is (for islands);
    ``can_export``: why the printer's files cannot be written ("" = they can)."""
    out: list[Problem] = []
    objects = [o for o in scene.objects if o.visible]
    supported = {s.owner_id for s in scene.supports}

    if printer is not None:
        fits = scene.fits(printer)
        for o in objects:
            if not fits.get(o.id, True):
                out.append(Problem(ERROR, f"{o.name} is outside the build volume", o.id, kind="outside"))
    for o in objects:
        lo = o.world_bounds()[0][2]
        if lo > FLOAT_GAP and o.id not in supported:
            out.append(Problem(ERROR, f"{o.name} floats {lo:.1f} mm above the plate with no supports",
                               o.id, kind="floating"))
        if o.hollow is not None and not o.holes:
            out.append(Problem(WARNING, f"{o.name} is hollow with no drain hole: resin stays trapped inside",
                               o.id, kind="trapped"))
        if not o.mesh.is_watertight:
            out.append(Problem(WARNING, f"{o.name} has holes in its mesh (Edit → Repair mesh)", o.id,
                               kind="open_mesh"))
    out += _overlaps(objects)

    report = scene.support_report if scene.supports else {}
    skipped = int(report.get("skipped", 0) or 0)
    if skipped:
        s = "s" if skipped != 1 else ""
        out.append(Problem(WARNING, f"{skipped} spot{s} that need support got none (no room for one)",
                           kind="skipped"))
    for check in report.get("release", []) or []:
        if not check.ok:
            text = check.text()
            out.append(Problem(WARNING, text[:1].upper() + text[1:], kind="release"))

    if result is not None:
        for isl in result.islands[:20]:
            span = f"layers {isl.first}–{isl.last}" if isl.last > isl.first else f"layer {isl.first}"
            out.append(Problem(ERROR, f"Island in the sliced layers ({span}): add supports there",
                               layer=isl.first, kind="island"))
        if len(result.islands) > 20:
            out.append(Problem(ERROR, f"… and {len(result.islands) - 20} more islands", kind="island"))
    if printer is not None and resin is None:
        out.append(Problem(ERROR, f"No resin profile for {printer.name} yet: add one before slicing",
                           kind="no_resin"))
    if can_export:
        out.append(Problem(NOTE, can_export, kind="no_export"))
    out.sort(key=lambda p: _ORDER[p.level])           # stable: the order above within each level
    return out


def _overlaps(objects) -> list[Problem]:
    """Pairs of models whose footprints overlap where their heights do too."""
    if len(objects) < 2:
        return []
    boxes = np.array([o.world_bounds() for o in objects])
    lo, hi = boxes[:, 0], boxes[:, 1]
    out = []
    for i in range(len(objects)):
        meet = np.flatnonzero(np.all(lo[i + 1:] < hi[i], axis=1) & np.all(hi[i + 1:] > lo[i], axis=1)) + i + 1
        for j in meet:
            a, b = objects[i], objects[j]
            shared = shapely.intersection(_footprint(a), _footprint(b)).area
            if shared > OVERLAP_AREA:
                out.append(Problem(WARNING, f"{a.name} and {b.name} overlap", a.id, kind="overlap"))
    return out


def _footprint(obj) -> Polygon:
    """The model's outline seen from above (convex hull), where it stands."""
    key = (id(obj.mesh), obj.rotation.tobytes(), obj.scale.tobytes())
    entry = _HULLS.get(key)
    if entry is None or entry[0] is not obj.mesh:
        m = obj.matrix()
        m[:3, 3] -= obj.position                        # turn and scale only
        xy = np.asarray(obj.mesh.vertices) @ m[:2, :3].T + m[:2, 3]
        hull = shapely.convex_hull(shapely.multipoints(xy))
        if len(_HULLS) > 256:
            _HULLS.clear()
        entry = _HULLS[key] = (obj.mesh, hull)
    return affinity.translate(entry[1], float(obj.position[0]), float(obj.position[1]))
