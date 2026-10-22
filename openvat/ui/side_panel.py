"""Right-hand panel: printer list, resin list, object list and the view
options (build plate, see-through models, section view), each in a
collapsible (the lists also resizable) card with a summary in its header."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QLabel, QMessageBox, QGridLayout, QCheckBox, QSlider,
                               QDoubleSpinBox, QButtonGroup, QFrame)

from ..core.profiles import ProfileStore, PrinterProfile, ResinProfile
from ..core.scene import Scene
from .collapsible import SectionColumn
from .profile_dialogs import PrinterDialog, ResinDialog


class ProfileList(QWidget):
    """A list with Add / Edit / Copy / Delete buttons, shared by the
    printer and resin sections.

    ``make_dialog(profile, adding)`` returns the editor dialog (with presets
    when adding); ``new_profile()`` gives the starting values for Add.
    """
    changed = Signal()
    renamed = Signal(str, str)       # old name, new name
    removed = Signal(str)            # name

    def __init__(self, items: list, make_dialog, new_profile, min_items: int = 0,
                 kind: str = "profile", parent=None):
        super().__init__(parent)
        self.items = items
        self.make_dialog = make_dialog
        self.new_profile = new_profile
        self.min_items = min_items
        self.kind = kind
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setMinimumHeight(60)
        self.list.setMinimumWidth(50)
        layout.addWidget(self.list)
        row = QHBoxLayout()
        for label, slot in (("Add", self.add_profile), ("Edit", self._edit),
                            ("Copy", self._duplicate), ("Delete", self._delete)):
            b = QPushButton(label); b.clicked.connect(slot); b.setMinimumWidth(30); row.addWidget(b)
        layout.addLayout(row)
        self.list.itemDoubleClicked.connect(lambda _: self._edit())
        self.list.currentRowChanged.connect(lambda _: self.changed.emit())
        self.refresh()

    def set_items(self, items: list, select: int = 0) -> None:
        self.items = items
        self.refresh(select)

    def refresh(self, select: int | None = None) -> None:
        row = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for it in self.items:
            self.list.addItem(it.name)
        if self.items:
            self.list.setCurrentRow(min(max(row, 0), len(self.items) - 1))
        self.list.blockSignals(False)

    def current(self):
        i = self.list.currentRow()
        return self.items[i] if 0 <= i < len(self.items) else None

    def _unique(self, name: str, skip: int = -1) -> str:
        """Profile names are file names, so keep them unique."""
        taken = {it.name for i, it in enumerate(self.items) if i != skip}
        if name not in taken:
            return name
        n = 2
        while f"{name} ({n})" in taken:
            n += 1
        return f"{name} ({n})"

    def add_profile(self) -> bool:
        dlg = self.make_dialog(self.new_profile(), True)
        if not dlg.exec():
            return False
        prof = dlg.result_profile()
        self.items.append(replace(prof, name=self._unique(prof.name)))
        self.refresh(len(self.items) - 1)
        self.changed.emit()
        return True

    def _edit(self) -> None:
        cur = self.current()
        if cur is None:
            return
        dlg = self.make_dialog(cur, False)
        if dlg.exec():
            row = self.list.currentRow()
            prof = dlg.result_profile()
            prof = replace(prof, name=self._unique(prof.name, skip=row))
            self.items[row] = prof
            if prof.name != cur.name:
                self.renamed.emit(cur.name, prof.name)
            self.refresh()
            self.changed.emit()

    def _duplicate(self) -> None:
        cur = self.current()
        if cur is None:
            return
        self.items.append(replace(cur, name=self._unique(f"{cur.name} (copy)")))
        self.refresh(len(self.items) - 1)
        self.changed.emit()

    def _delete(self) -> None:
        cur = self.current()
        if cur is None:
            return
        if len(self.items) <= self.min_items:
            QMessageBox.information(self, "Delete", f"At least one {self.kind} must remain.")
            return
        if QMessageBox.question(self, "Delete", f"Delete {self.kind} \"{cur.name}\"?") != QMessageBox.Yes:
            return
        del self.items[self.list.currentRow()]
        self.removed.emit(cur.name)
        self.refresh()
        self.changed.emit()


def printer_dialog(store: ProfileStore, profile: PrinterProfile, adding: bool, parent=None,
                   intro: str = "") -> PrinterDialog:
    return PrinterDialog(profile, parent, presets=store.printer_presets if adding else None, intro=intro)


def add_first_printer(store: ProfileStore, parent=None) -> bool:
    """Startup check: no printers yet -> ask for one.  False if cancelled."""
    start = store.printer_presets[0] if store.printer_presets else PrinterProfile()
    dlg = printer_dialog(store, start, True, parent,
                         intro="<b>Welcome to OpenVat.</b> No printer is set up yet - pick your "
                               "printer from the presets (or enter its specs) and save.")
    if not dlg.exec():
        return False
    store.printers.append(dlg.result_profile())
    store.save()
    return True


class SectionControls(QWidget):
    """The section view: cut the 3D view at a plane across X, Y or Z, at a
    distance along that axis, showing one side of it or (Flip) the other."""
    changed = Signal(object)                      # (axis, position, flip), or None when off

    STEPS = 1000                                  # slider resolution over the scene's extent

    def __init__(self, scene: Scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.lo, self.hi = 0.0, 100.0             # the scene's extent on the chosen axis
        self._sync = False
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        self.enable = QCheckBox("Cut the view")
        self.enable.setToolTip("Show the models and supports cut open at a plane")
        g.addWidget(self.enable, 0, 0, 1, 4)
        self.axes = QButtonGroup(self)
        for i, name in enumerate("XYZ"):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setToolTip(f"Cut across the {name} axis")
            self.axes.addButton(b, i)
            g.addWidget(b, 1, i)
        self.axes.button(2).setChecked(True)
        self.flip = QPushButton("Flip")
        self.flip.setCheckable(True)
        self.flip.setToolTip("Show the other side of the cut")
        g.addWidget(self.flip, 1, 3)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setValue(self.STEPS // 2)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.1)
        self.spin.setSuffix(" mm")
        self.spin.setKeyboardTracking(False)
        g.addWidget(self.slider, 2, 0, 1, 2)
        g.addWidget(self.spin, 2, 2, 1, 2)
        self.enable.toggled.connect(self._enabled)
        self.axes.idToggled.connect(lambda _i, on: on and self._axis_changed())
        self.flip.toggled.connect(lambda _on: self._emit())
        self.slider.valueChanged.connect(self._slid)
        self.spin.valueChanged.connect(self._typed)
        self._enabled(False)
        scene.listeners.append(self.refresh_range)
        self.refresh_range()

    # ------------------------------------------------------------ state
    def axis(self) -> int:
        return max(self.axes.checkedId(), 0)

    def state(self):
        if not self.enable.isChecked():
            return None
        return self.axis(), self.spin.value(), self.flip.isChecked()

    def summary(self) -> str:
        st = self.state()
        if st is None:
            return "off"
        axis, pos, flip = st
        return f"{'XYZ'[axis]} {'≥' if flip else '≤'} {pos:.2f} mm"

    # ------------------------------------------------------------ range
    def _extent(self, axis: int) -> tuple[float, float]:
        """The scene's extent on an axis: its models and their supports."""
        boxes = [o.world_bounds() for o in self.scene.objects if o.visible]
        if self.scene.supports:
            boxes += [m.bounds for m in self.scene.support_meshes() if len(m.faces)]
        if not boxes:
            return 0.0, 100.0
        b = np.array(boxes)
        lo, hi = float(b[:, 0, axis].min()), float(b[:, 1, axis].max())
        return (lo, hi) if hi - lo > 1e-6 else (lo - 1.0, hi + 1.0)

    def refresh_range(self) -> None:
        """The models moved or changed: the slider spans their new extent and
        the cut stays where it was (inside that extent)."""
        if not self.enable.isChecked():
            return
        pos = self.spin.value()
        self._set_range(*self._extent(self.axis()), pos)

    def _set_range(self, lo: float, hi: float, pos: float) -> None:
        self.lo, self.hi = lo, hi
        self._sync = True
        self.spin.setRange(lo, hi)
        pos = min(max(pos, lo), hi)
        self.spin.setValue(pos)
        self.slider.setValue(round((pos - lo) / (hi - lo) * self.STEPS))
        self._sync = False

    # ------------------------------------------------------------ input
    def _enabled(self, on: bool) -> None:
        for w in (*self.axes.buttons(), self.flip, self.slider, self.spin):
            w.setEnabled(on)
        if on:
            lo, hi = self._extent(self.axis())
            self._set_range(lo, hi, (lo + hi) / 2)
        self._emit()

    def _axis_changed(self) -> None:
        if self.enable.isChecked():
            lo, hi = self._extent(self.axis())
            self._set_range(lo, hi, (lo + hi) / 2)       # a new axis starts in the middle
        self._emit()

    def _slid(self, value: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.spin.setValue(self.lo + (self.hi - self.lo) * value / self.STEPS)
        self._sync = False
        self._emit()

    def _typed(self, pos: float) -> None:
        if self._sync:
            return
        self._sync = True
        self.slider.setValue(round((pos - self.lo) / max(self.hi - self.lo, 1e-9) * self.STEPS))
        self._sync = False
        self._emit()

    def _emit(self) -> None:
        self.changed.emit(self.state())


class ViewControls(QWidget):
    """The View card: show the build plate, see-through models, and the
    section view (SectionControls) under them."""
    plate_toggled = Signal(bool)
    see_through_toggled = Signal(bool)

    def __init__(self, scene: Scene, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plate = QCheckBox("Build plate")
        self.plate.setChecked(True)
        self.plate.setToolTip("Show the build plate and the build volume (also in View menu)")
        self.see_through = QCheckBox("Semi-transparent models")
        self.see_through.setToolTip("See supports and holes through the models (also in View menu)")
        self.plate.toggled.connect(self.plate_toggled)
        self.see_through.toggled.connect(self.see_through_toggled)
        lay.addWidget(self.plate)
        lay.addWidget(self.see_through)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        lay.addWidget(line)
        self.section = SectionControls(scene)
        lay.addWidget(self.section)

    def summary(self) -> str:
        """'plate · see-through · cut Z ≤ 12.00 mm' (what is on)."""
        parts = ["plate" if self.plate.isChecked() else "no plate"]
        if self.see_through.isChecked():
            parts.append("see-through")
        cut = self.section.state()
        if cut is not None:
            parts.append("cut " + self.section.summary())
        return " · ".join(parts)


class SidePanel(SectionColumn):
    printer_changed = Signal(object)
    resin_changed = Signal(object)
    section_changed = Signal(object)             # see SectionControls

    def __init__(self, store: ProfileStore, scene: Scene, parent=None):
        super().__init__(width=260, parent=parent)
        self.store = store
        self.scene = scene
        self._syncing = False

        self.printers = ProfileList(
            store.printers, lambda prof, adding: printer_dialog(store, prof, adding, self),
            lambda: store.printer_presets[0] if store.printer_presets else PrinterProfile(),
            min_items=1, kind="printer")
        self.resins = ProfileList(
            [], self._resin_dialog, self._new_resin, min_items=0, kind="resin profile")
        self.add("printers", "Printers", self.printers, stretch=1)
        self.add("resins", "Resin profiles", self.resins, stretch=1)
        self.printers.refresh(self._index_of(store.printers, store.default_printer()))
        self._bind_resins()
        self.printers.changed.connect(self._printer_list_changed)
        self.printers.renamed.connect(store.rename_printer)
        self.printers.removed.connect(store.forget_printer)
        self.resins.changed.connect(self._profiles_changed)

        objects = QWidget()
        ol = QVBoxLayout(objects); ol.setContentsMargins(0, 0, 0, 0)
        self.objects = QListWidget()
        self.objects.currentRowChanged.connect(self._object_selected)
        self.objects.itemChanged.connect(self._visibility_changed)
        ol.addWidget(self.objects)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        ol.addWidget(self.info)
        self.add("objects", "Objects", objects, stretch=1)
        self.view = ViewControls(scene)
        self.section = self.view.section
        self.add("view", "View", self.view, expanded=False)
        self.section.changed.connect(self._section_changed)
        self.view.plate_toggled.connect(lambda _on: self._view_summary())
        self.view.see_through_toggled.connect(lambda _on: self._view_summary())
        self._view_summary()
        self.finish()

        scene.listeners.append(self.refresh_objects)
        self._profiles_changed(save=False)
        self.refresh_objects()

    # ------------------------------------------------------------ profiles
    def current_printer(self) -> PrinterProfile | None:
        return self.printers.current() or (self.store.printers[0] if self.store.printers else None)

    def current_resin(self) -> ResinProfile | None:
        return self.resins.current()

    @staticmethod
    def _index_of(items, item) -> int:
        return items.index(item) if item in items else 0

    def _resin_dialog(self, profile: ResinProfile, adding: bool) -> ResinDialog:
        printer = self.current_printer()
        return ResinDialog(profile, self, presets=self.store.resin_presets if adding else None,
                           printer_name=printer.name if printer else "")

    def _new_resin(self) -> ResinProfile:
        printer = self.current_printer()
        presets = self.store.resin_presets_for(printer) if printer else []
        return presets[0] if presets else ResinProfile()

    def _bind_resins(self) -> None:
        """Show the resin profiles of the selected printer."""
        printer = self.current_printer()
        items = self.store.resins_for(printer) if printer else []
        self.resins.set_items(items, self._index_of(items, self.store.default_resin(printer)) if printer else 0)

    def _printer_list_changed(self) -> None:
        self._bind_resins()
        self._profiles_changed()

    def add_resin(self) -> bool:
        """Open the "new resin profile" dialog for the selected printer."""
        return self.resins.add_profile()

    def _profiles_changed(self, save: bool = True) -> None:
        if save:
            self.store.save()
        printer, resin = self.current_printer(), self.current_resin()
        self.store.remember_selection(printer, resin)
        self.sections["printers"].set_summary(printer.name if printer else "none")
        self.sections["resins"].set_summary(resin.name if resin else "none - click Add")
        self.printer_changed.emit(printer)
        self.resin_changed.emit(resin)

    def reload_profiles(self) -> None:
        self.printers.set_items(self.store.printers, self._index_of(self.store.printers, self.store.default_printer()))
        self._bind_resins()
        self._profiles_changed(save=False)

    def _section_changed(self, state) -> None:
        self._view_summary()
        self.section_changed.emit(state)

    def _view_summary(self) -> None:
        self.sections["view"].set_summary(self.view.summary())

    # ------------------------------------------------------------ objects
    def refresh_objects(self) -> None:
        self._syncing = True
        self.objects.clear()
        for obj in self.scene.objects:
            item = QListWidgetItem(obj.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if obj.visible else Qt.Unchecked)
            item.setData(Qt.UserRole, obj.id)
            self.objects.addItem(item)
            if obj is self.scene.selected:
                self.objects.setCurrentItem(item)
        self._syncing = False
        sel = self.scene.selected
        if sel is not None:
            s = sel.size()
            self.info.setText(f"{len(sel.mesh.faces)} faces, "
                              f"{s[0]:.1f} x {s[1]:.1f} x {s[2]:.1f} mm, "
                              f"{'watertight' if sel.mesh.is_watertight else 'open mesh'}")
        else:
            self.info.setText(f"{len(self.scene.objects)} object(s), {len(self.scene.supports)} support(s)")
        n = len(self.scene.objects)
        self.sections["objects"].set_summary(f"{n} object{'s' if n != 1 else ''}")

    def _object_selected(self, row: int) -> None:
        if self._syncing:
            return
        item = self.objects.item(row)
        obj = self.scene.find(item.data(Qt.UserRole)) if item else None
        if obj is not self.scene.selected:
            self.scene.selected = obj
            self.scene.selected_support = None
            self.scene.changed()

    def _visibility_changed(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        obj = self.scene.find(item.data(Qt.UserRole))
        if obj:
            self.scene.push_undo()
            obj.visible = item.checkState() == Qt.Checked
            self.scene.changed()
