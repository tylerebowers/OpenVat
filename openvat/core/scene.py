"""The Scene: everything sitting on the build plate.

It owns the list of MeshObjects and Supports, the selection, and simple
operations that involve more than one object (auto-arrange, clone).  It is
deliberately Qt-free so it can be used from the CLI and from tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from shapely.ops import unary_union

from .mesh import MeshObject
from .profiles import PrinterProfile
from .clearance import clearance_field
from .placement import field_cap, generate_supports
from .supports import Support, SupportSettings, generate_braces, raft_mesh, trunk_mesh
from .hollow import HollowSettings
from .undo import UndoStack


class Scene:
    def __init__(self) -> None:
        self.objects: list[MeshObject] = []
        self.supports: list[Support] = []
        self.selected: MeshObject | None = None
        self.selected_support: Support | None = None
        # Callbacks the UI can hook to refresh itself.
        self.listeners: list[Callable[[], None]] = []
        self.undo_stack = UndoStack(self)
        self.support_settings = SupportSettings()
        self.hollow_settings = HollowSettings()   # the Hollowing settings window
        self._extras_cache: tuple | None = None
        self.extras_version = 0           # bumps whenever support_meshes() is rebuilt
        self.support_report: dict = {}    # what the last Automatic run made (and skipped)

    def push_undo(self) -> None:
        """Call before changing anything the user might want to undo."""
        self.undo_stack.push()

    # ------------------------------------------------------------- notify
    def changed(self) -> None:
        self._extras_cache = None
        for cb in self.listeners:
            cb()

    # ------------------------------------------------------------- objects
    def add(self, obj: MeshObject) -> None:
        self.push_undo()
        self.objects.append(obj)
        self.selected = obj
        self.changed()

    def remove(self, obj: MeshObject) -> None:
        self.push_undo()
        self.objects.remove(obj)
        self.supports = [s for s in self.supports if s.owner_id != obj.id]
        if self.selected is obj:
            self.selected = None
        self.changed()

    def clear(self) -> None:
        self.push_undo()
        self.objects.clear()
        self.supports.clear()
        self.selected = None
        self.selected_support = None
        self.changed()

    def clone(self, obj: MeshObject) -> MeshObject:
        c = obj.clone()
        c.name = _unique_name(obj.name, [o.name for o in self.objects])
        c.position[0] += obj.size()[0] + 5.0
        self.add(c)
        return c

    def move_object(self, obj: MeshObject, new_position: np.ndarray) -> None:
        """Move an object and carry its supports along with it."""
        delta = np.asarray(new_position, dtype=float) - obj.position
        obj.position[:] = new_position
        for s in self.supports:
            if s.owner_id == obj.id:
                s.tip += delta
                if s.foot is not None:            # a nook stands on the part: moves with it
                    s.foot = tuple(float(v) for v in np.add(s.foot, delta))
                if s.route:                       # so does a detour around it
                    s.route = tuple(tuple(float(v) for v in np.add(j, delta)) for j in s.route)
        trunks = {s.trunk for s in self.supports if s.owner_id == obj.id and s.trunk is not None}
        for t in trunks:                          # shared pillars move with their contacts
            moved = (t[0] + delta[0], t[1] + delta[1], t[2] + delta[2], t[3] + (delta[2] if t[3] > 0 else 0.0))
            for s in self.supports:
                if s.trunk == t:
                    s.trunk = moved

    def find(self, obj_id: int) -> MeshObject | None:
        return next((o for o in self.objects if o.id == obj_id), None)

    # ------------------------------------------------------------- supports
    def add_support(self, support: Support) -> None:
        self.push_undo()
        self.supports.append(support)
        self.selected_support = support
        self.changed()

    def remove_support(self, support: Support) -> None:
        self.push_undo()
        self.supports.remove(support)
        if self.selected_support is support:
            self.selected_support = None
        self.changed()

    def lift_height(self) -> float:
        s = self.support_settings
        return float(s.z_lift_height) if s.z_lift else 0.0

    def set_elevation(self, obj: MeshObject, height: float) -> None:
        """Seat an object with its lowest point ``height`` above the plate."""
        obj.elevation = height
        target = obj.position.copy()
        target[2] += height - obj.world_bounds()[0][2]
        self.move_object(obj, target)

    def auto_supports(self, obj: MeshObject | None = None, layer_height: float = 0.05) -> int:
        """Generate supports for one object (or all visible ones), replacing
        their existing supports.  With Z lift enabled the objects are first
        raised to the lift height (otherwise they stay where they are).
        ``layer_height`` is the printed layer thickness islands are looked
        for at.  Returns the number of supports created; ``support_report``
        says how many are straight, routed, split, nook, how many spots are
        bridges and how many were skipped."""
        targets = [obj] if obj is not None else [o for o in self.objects if o.visible]
        self.push_undo()
        ids = {o.id for o in targets}
        self.supports = [s for s in self.supports if s.owner_id not in ids]
        self.support_report = {}
        created = 0
        for o in targets:
            if self.support_settings.z_lift:
                self.set_elevation(o, self.lift_height())
            new = generate_supports(o, self.support_settings, self.support_report, layer_height)
            self.supports.extend(new)
            created += len(new)
        self.selected_support = None
        self.changed()
        return created

    def clear_supports(self) -> None:
        """Remove all supports and put lifted objects back on the plate."""
        self.push_undo()
        self.supports.clear()
        self.selected_support = None
        for o in self.objects:
            if o.elevation:
                self.set_elevation(o, 0.0)
        self.changed()

    def support_meshes(self) -> list:
        """World-space meshes of all supports, shared trunks, braces and raft (cached
        until the scene changes).  Used by the viewport, the slicer and the
        exporter so they always agree."""
        if self._extras_cache is None:
            s = self.support_settings
            meshes = [sup.to_mesh(s) for sup in self.supports]
            meshes += [trunk_mesh(t, s) for t in dict.fromkeys(sup.trunk for sup in self.supports if sup.trunk)]
            owners = {sup.owner_id for sup in self.supports}
            fields = {o.id: clearance_field(o, field_cap(s)) for o in self.objects if o.id in owners}
            meshes.append(generate_braces(self.supports, s, fields))     # braces keep off the model
            meshes.append(raft_mesh(self.supports, s, self._plate_footprint(owners)))
            self._extras_cache = tuple(m for m in meshes if len(m.faces))
            self.extras_version += 1
        return list(self._extras_cache)

    def _plate_footprint(self, ids: set):
        """Where the given objects touch the plate (the raft keeps off it)."""
        from .slicer import slice_mesh_at                 # local import: slicer imports scene
        s = self.support_settings
        parts = []
        for o in self.objects:
            if o.id in ids and o.world_bounds()[0][2] < s.base_height:
                mesh = o.transformed()
                parts += [slice_mesh_at(mesh, z) for z in (0.02, max(s.base_height - 0.02, 0.03))]
        return unary_union(parts) if parts else None

    # ------------------------------------------------------------- layout
    def arrange(self, printer: PrinterProfile, spacing: float = 4.0) -> None:
        """Shelf-pack all objects by footprint, biggest first, centered on
        the plate.  Simple, predictable and good enough for most jobs."""
        if not self.objects:
            return
        self.push_undo()
        half_x, half_y = printer.print_x / 2, printer.print_y / 2
        items = sorted(self.objects, key=lambda o: -(o.size()[0] * o.size()[1]))

        rows: list[list[MeshObject]] = []
        row_w = 0.0
        for o in items:
            w = o.size()[0] + spacing
            if rows and row_w + w <= printer.print_x:
                rows[-1].append(o)
                row_w += w
            else:
                rows.append([o])
                row_w = w

        row_heights = [max(o.size()[1] for o in r) + spacing for r in rows]
        total_h = sum(row_heights)
        y = min(total_h / 2, half_y)
        for r, h in zip(rows, row_heights):
            widths = [o.size()[0] + spacing for o in r]
            x = -min(sum(widths) / 2, half_x)
            for o, w in zip(r, widths):
                b = o.world_bounds()
                center = (b[0] + b[1]) / 2
                target = np.array([x + w / 2, y - h / 2])
                shift = target - center[:2]
                self.move_object(o, o.position + [shift[0], shift[1], 0.0])
                x += w
            y -= h
        self.changed()

    def fits(self, printer: PrinterProfile) -> dict[int, bool]:
        """Which objects are inside the build volume (by id)."""
        hx, hy = printer.print_x / 2, printer.print_y / 2
        result = {}
        for o in self.objects:
            lo, hi = o.world_bounds()
            result[o.id] = bool(lo[0] >= -hx and lo[1] >= -hy and lo[2] >= -1e-6
                                and hi[0] <= hx and hi[1] <= hy and hi[2] <= printer.print_z)
        return result

    def bounds(self) -> np.ndarray | None:
        if not self.objects:
            return None
        b = np.array([o.world_bounds() for o in self.objects])
        return np.vstack([b[:, 0].min(axis=0), b[:, 1].max(axis=0)])


def _unique_name(base: str, existing: list[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"
