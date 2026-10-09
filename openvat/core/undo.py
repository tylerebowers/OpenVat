"""Snapshot-based undo / redo for the Scene.

Before every edit the UI calls ``scene.push_undo()``; that stores a cheap
snapshot (transforms are copied, meshes are shared by reference).  Operations
that change the mesh itself (mirror, repair) replace ``obj.mesh`` with a new
mesh instead of editing in place, so older snapshots stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .mesh import MeshObject
from .supports import Support

MAX_STEPS = 100


@dataclass
class _ObjectState:
    obj: MeshObject
    mesh: object
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    visible: bool
    name: str
    elevation: float
    holes: tuple
    hollow: object


@dataclass
class _Snapshot:
    objects: list[_ObjectState]
    supports: list[tuple[Support, dict]]
    selected_id: int | None
    selected_support_id: int | None
    support_settings: object
    hollow_settings: object


class UndoStack:
    def __init__(self, scene) -> None:
        self.scene = scene
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []

    # ------------------------------------------------------------ snapshots
    def _capture(self) -> _Snapshot:
        s = self.scene
        return _Snapshot(
            objects=[_ObjectState(o, o.mesh, o.position.copy(), o.rotation.copy(),
                                  o.scale.copy(), o.visible, o.name, o.elevation, o.holes, o.hollow)
                     for o in s.objects],
            supports=[(sup, sup.state()) for sup in s.supports],
            selected_id=s.selected.id if s.selected else None,
            selected_support_id=s.selected_support.id if s.selected_support else None,
            support_settings=replace(s.support_settings),
            hollow_settings=replace(s.hollow_settings),
        )

    def _restore(self, snap: _Snapshot) -> None:
        s = self.scene
        s.objects = []
        for st in snap.objects:
            o = st.obj
            o.mesh, o.name, o.visible, o.elevation = st.mesh, st.name, st.visible, st.elevation
            o.holes, o.hollow = st.holes, st.hollow            # immutable: shared safely
            o.position[:], o.rotation[:], o.scale[:] = st.position, st.rotation, st.scale
            s.objects.append(o)
        s.supports = []
        for sup, state in snap.supports:
            sup.restore(state)
            s.supports.append(sup)
        s.support_settings = replace(snap.support_settings)
        s.hollow_settings = replace(snap.hollow_settings)
        s.selected = s.find(snap.selected_id) if snap.selected_id else None
        s.selected_support = next((x for x in s.supports if x.id == snap.selected_support_id), None)
        s.changed()

    # ------------------------------------------------------------ API
    def push(self) -> None:
        """Record the current state as the point to return to on undo."""
        self._undo.append(self._capture())
        del self._undo[:-MAX_STEPS]
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._capture())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._capture())
        self._restore(self._redo.pop())
        return True

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

