"""Right-hand panel: printer list, resin list and object list, each in a
collapsible / resizable section with a summary in its header."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QLabel, QMessageBox)

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


class SidePanel(SectionColumn):
    printer_changed = Signal(object)
    resin_changed = Signal(object)

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
