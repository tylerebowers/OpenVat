"""Left-hand model editing panel: position / rotation / scale / edit /
supports / hollowing.  Each group is a collapsible, resizable section whose
header shows a summary while folded.  Every control acts on
``scene.selected``.

Hollowing goes in order: drain holes first (click the model), then Hollow
(wall thickness from the settings window), then internal supports."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QGridLayout, QDoubleSpinBox, QPushButton,
                               QLabel, QCheckBox, QComboBox, QMessageBox, QApplication)

from ..core.scene import Scene
from ..core.hollow import HollowState, hole_from_world, hole_world, describe
from ..core.placement import manual_support
from ..core.supports import DENSITY_SPACING
from .collapsible import SectionColumn
from .support_dialog import SupportSettingsDialog
from .hollow_dialog import HollowSettingsDialog


class ToolsPanel(SectionColumn):
    mesh_edited = Signal(object)        # object whose source mesh changed (GPU refresh)
    arrange_requested = Signal()
    support_mode_toggled = Signal(bool)
    settings_changed = Signal()         # support settings edited (persist them)
    hole_mode_toggled = Signal(bool)    # clicking the model places / removes drain holes
    show_inside_toggled = Signal(bool)  # hollow models see-through, internal supports shown
    hollow_settings_changed = Signal()  # hollowing settings edited (persist them)

    def __init__(self, scene: Scene, parent=None):
        super().__init__(width=240, parent=parent)
        self.scene = scene
        self._updating = False

        self.add("position", "Position", self._build_position())
        self.add("rotation", "Rotation", self._build_rotation())
        self.add("scale", "Scale", self._build_scale())
        self.add("edit", "Edit", self._build_edit())
        self.add("supports", "Supports", self._build_supports())
        self.add("hollow", "Hollowing", self._build_hollowing())
        self.finish()

        scene.listeners.append(self.refresh)
        self.refresh()

    # ------------------------------------------------------------ builders
    @staticmethod
    def _spin(lo, hi, step) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(2)
        sp.setKeyboardTracking(False)
        return sp

    def _build_position(self) -> QWidget:
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        self.pos_spins = [self._spin(-500, 500, 0.1) for _ in range(3)]
        for sp in self.pos_spins:
            sp.setSuffix(" mm")
        for i, (axis, sp) in enumerate(zip("XYZ", self.pos_spins)):
            g.addWidget(QLabel(axis), i, 0); g.addWidget(sp, i, 1)
            sp.valueChanged.connect(self._apply_position)
        row = QHBoxLayout()
        drop = QPushButton("Drop to plate"); drop.clicked.connect(self._drop)
        center = QPushButton("Center"); center.clicked.connect(self._center)
        row.addWidget(drop); row.addWidget(center)
        g.addLayout(row, 3, 0, 1, 2)
        arrange = QPushButton("Auto-arrange all"); arrange.clicked.connect(self.arrange_requested)
        g.addWidget(arrange, 4, 0, 1, 2)
        return w

    def _build_rotation(self) -> QWidget:
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        self.rot_spins = [self._spin(-360, 360, 1.0) for _ in range(3)]
        for i, (axis, sp) in enumerate(zip("XYZ", self.rot_spins)):
            g.addWidget(QLabel(axis), i, 0)
            minus = QPushButton("-45"); minus.setFixedWidth(38)
            plus = QPushButton("+45"); plus.setFixedWidth(38)
            minus.clicked.connect(lambda _=False, a=i: self._rotate(a, -45))
            plus.clicked.connect(lambda _=False, a=i: self._rotate(a, 45))
            g.addWidget(minus, i, 1); g.addWidget(sp, i, 2); g.addWidget(plus, i, 3)
            sp.valueChanged.connect(self._apply_rotation)
        reset = QPushButton("Reset rotation"); reset.clicked.connect(self._reset_rotation)
        g.addWidget(reset, 3, 0, 1, 4)
        return w

    def _build_scale(self) -> QWidget:
        """Per axis of the model: scale in % and the size that gives in mm -
        either can be typed in."""
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        for col, text in ((1, "Scale"), (2, "Size")):
            head = QLabel(text)
            head.setStyleSheet("color: #888;")
            head.setAlignment(Qt.AlignCenter)
            g.addWidget(head, 0, col)
        self.scale_spins = [self._spin(1, 10000, 1.0) for _ in range(3)]
        self.mm_spins = [self._spin(0.01, 100000, 0.5) for _ in range(3)]
        for i, axis in enumerate("XYZ"):
            pct, mm = self.scale_spins[i], self.mm_spins[i]
            pct.setSuffix(" %"); mm.setSuffix(" mm")
            pct.setToolTip(f"Scale along the model's {axis} axis")
            mm.setToolTip(f"The model's size along its {axis} axis")
            g.addWidget(QLabel(axis), i + 1, 0); g.addWidget(pct, i + 1, 1); g.addWidget(mm, i + 1, 2)
            pct.valueChanged.connect(lambda v, a=i: self._apply_scale(a, v))
            mm.valueChanged.connect(lambda v, a=i: self._apply_size(a, v))
        self.uniform = QCheckBox("Uniform"); self.uniform.setChecked(True)
        self.uniform.setToolTip("Keep the proportions: changing one axis scales all three")
        g.addWidget(self.uniform, 4, 0, 1, 3)
        self.size_label = QLabel("On the plate: -")
        self.size_label.setToolTip("The space the model takes on the build plate (after rotation)")
        self.size_label.setStyleSheet("color: #888;")
        g.addWidget(self.size_label, 5, 0, 1, 3)
        return w

    def _build_edit(self) -> QWidget:
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(QLabel("Mirror"), 0, 0)
        for i, axis in enumerate("XYZ"):
            b = QPushButton(axis); b.setFixedWidth(36)
            b.clicked.connect(lambda _=False, a=i: self._mirror(a))
            g.addWidget(b, 0, 1 + i)
        clone = QPushButton("Clone"); clone.clicked.connect(self.clone_selected)
        repair = QPushButton("Repair mesh"); repair.clicked.connect(self.repair_selected)
        delete = QPushButton("Delete"); delete.clicked.connect(self.delete_selected)
        g.addWidget(clone, 1, 0, 1, 2); g.addWidget(repair, 1, 2, 1, 2)
        g.addWidget(delete, 2, 0, 1, 4)
        return w

    def _build_hollowing(self) -> QWidget:
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        self.hole_btn = QPushButton("Add holes")
        self.hole_btn.setCheckable(True)
        self.hole_btn.setToolTip("Click the model to place a drain hole; click a hole again to remove it")
        self.hole_btn.toggled.connect(self.hole_mode_toggled)
        self.clear_holes_btn = QPushButton("Clear holes")
        self.clear_holes_btn.clicked.connect(self._clear_holes)
        g.addWidget(self.hole_btn, 0, 0); g.addWidget(self.clear_holes_btn, 0, 1)
        self.hole_info = QLabel("")
        self.hole_info.setWordWrap(True)
        self.hole_info.setStyleSheet("color: #888;")
        g.addWidget(self.hole_info, 1, 0, 1, 2)
        self.hollow_btn = QPushButton("Hollow")
        self.hollow_btn.clicked.connect(self._toggle_hollow)
        g.addWidget(self.hollow_btn, 2, 0, 1, 2)
        self.lattice_btn = QPushButton("Add internal supports")
        self.lattice_btn.setToolTip("Poles along three axes through the cavity (angles and spacing in Settings…)")
        self.lattice_btn.clicked.connect(self._toggle_lattice)
        g.addWidget(self.lattice_btn, 3, 0, 1, 2)
        settings = QPushButton("Settings…"); settings.clicked.connect(self.edit_hollow_settings)
        self.show_inside = QCheckBox("Show inside")
        self.show_inside.setToolTip("Draw hollow models see-through, with their internal supports and holes")
        self.show_inside.toggled.connect(self.show_inside_toggled)
        g.addWidget(settings, 4, 0); g.addWidget(self.show_inside, 4, 1)
        self.hollow_info = QLabel("")
        self.hollow_info.setTextFormat(Qt.RichText)
        self.hollow_info.setWordWrap(True)
        g.addWidget(self.hollow_info, 5, 0, 1, 2)
        self.hole_message = ""                     # about the hole just placed
        return w

    def _build_supports(self) -> QWidget:
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        self.add_support_btn = QPushButton("Manual (click model)")
        self.add_support_btn.setCheckable(True)
        self.add_support_btn.toggled.connect(self.support_mode_toggled)
        auto = QPushButton("Automatic"); auto.clicked.connect(self.auto_supports)
        g.addWidget(self.add_support_btn, 0, 0); g.addWidget(auto, 0, 1)
        g.addWidget(QLabel("Density"), 1, 0)
        self.density = QComboBox()
        for key in DENSITY_SPACING:
            self.density.addItem(key.capitalize(), key)
        self.density.currentIndexChanged.connect(self._density_changed)
        g.addWidget(self.density, 1, 1)
        self.z_lift = QCheckBox("Z lift")
        self.z_lift.setToolTip("When Automatic runs, first raise the model this high off the "
                               "plate so supports go underneath (height in Settings…)")
        self.z_lift.toggled.connect(self._z_lift_toggled)
        self.z_lift_label = QLabel("")
        self.z_lift_label.setStyleSheet("color: #888;")
        g.addWidget(self.z_lift, 3, 0); g.addWidget(self.z_lift_label, 3, 1)
        settings = QPushButton("Settings…"); settings.clicked.connect(self.edit_support_settings)
        clear = QPushButton("Clear all"); clear.clicked.connect(self._clear_supports)
        g.addWidget(settings, 2, 0); g.addWidget(clear, 2, 1)
        hint = QLabel("<i>Click a support to select it, Del removes it. Drag to move.</i>")
        hint.setWordWrap(True)
        g.addWidget(hint, 4, 0, 1, 2)
        self.support_info = QLabel("")             # how the selected support was placed / last run
        self.support_message = ""                  # result of the last Automatic or manual placement
        self.layer_height = lambda: 0.05            # printed layer thickness (set by the main window)
        self.support_info.setWordWrap(True)
        self.support_info.setStyleSheet("color: #888;")
        g.addWidget(self.support_info, 5, 0, 1, 2)
        return w

    # ------------------------------------------------------------ refresh
    def refresh(self) -> None:
        """Pull values from the selected object into the controls and the
        section summaries."""
        obj = self.scene.selected
        self._updating = True
        enabled = obj is not None
        for w in (*self.pos_spins, *self.rot_spins, *self.scale_spins, *self.mm_spins):
            w.setEnabled(enabled)
        sec = self.sections
        if obj is not None:
            for sp, v in zip(self.pos_spins, obj.position):
                sp.setValue(float(v))
            for sp, v in zip(self.rot_spins, obj.rotation):
                sp.setValue(float(v))
            for sp, v in zip(self.scale_spins, obj.scale):
                sp.setValue(float(v) * 100)
            dims = self._model_size(obj)
            for sp, v in zip(self.mm_spins, dims):
                sp.setValue(float(v))
            s = obj.size()
            self.size_label.setText(f"On the plate: {s[0]:.2f} × {s[1]:.2f} × {s[2]:.2f} mm")
            p, r, sc = obj.position, obj.rotation, obj.scale
            sec["position"].set_summary(f"{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} mm")
            sec["rotation"].set_summary(f"{r[0]:.0f}°, {r[1]:.0f}°, {r[2]:.0f}°")
            pct = (f"{sc[0] * 100:.0f}%" if np.allclose(sc, sc[0]) else
                   f"{sc[0] * 100:.0f}%, {sc[1] * 100:.0f}%, {sc[2] * 100:.0f}%")
            sec["scale"].set_summary(f"{dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} mm · {pct}")
            sec["edit"].set_summary(obj.name)
        else:
            self.size_label.setText("On the plate: -")
            for k in ("position", "rotation", "scale", "edit"):
                sec[k].set_summary("no selection")
        self._refresh_hollowing(obj)
        n = len(self.scene.supports)
        sec["supports"].set_summary(f"{n} support{'s' if n != 1 else ''}")
        ss = self.scene.support_settings
        self.density.setCurrentIndex(max(self.density.findData(ss.density), 0))
        self.z_lift.setChecked(ss.z_lift)
        self.z_lift_label.setText(f"{ss.z_lift_height:.1f} mm")
        if ss.z_lift:
            sec["supports"].set_summary(f"{n} support{'s' if n != 1 else ''} · Z lift {ss.z_lift_height:.1f} mm")
        sel = self.scene.selected_support
        lines = [self.support_message, f"Selected support: {sel.status}" if sel is not None and sel.status else ""]
        self.support_info.setText("\n".join(line for line in lines if line))
        self.support_info.setVisible(bool(self.support_info.text()))
        self._updating = False

    def _refresh_hollowing(self, obj) -> None:
        """Buttons follow the order: holes, then Hollow, then internal supports."""
        has = obj is not None
        holes = len(obj.holes) if has else 0
        hollow = obj.hollow if has else None
        self.hole_btn.setEnabled(has)
        self.clear_holes_btn.setEnabled(holes > 0)
        if not has:
            self.hole_info.setText("Select a model.")
        elif holes == 0:
            self.hole_info.setText("Add at least one drain hole first: click Add holes, then the model.")
        else:
            s = "s" if holes != 1 else ""
            self.hole_info.setText(f"{holes} drain hole{s}." + (f" {self.hole_message}" if self.hole_message else ""))
        self.hollow_btn.setEnabled(has and (hollow is not None or holes > 0))
        self.hollow_btn.setText("Make solid" if hollow is not None
                                else f"Hollow ({self.scene.hollow_settings.wall_thickness:g} mm walls)")
        self.hollow_btn.setToolTip("" if holes or hollow is not None else "Add a drain hole first")
        self.lattice_btn.setEnabled(hollow is not None)
        self.lattice_btn.setText("Remove internal supports" if hollow is not None and hollow.lattice is not None
                                 else "Add internal supports")
        lines = []
        if hollow is not None:
            lines.append(f"<span style='color:#888'>{describe(hollow, holes)}</span>")
            if holes == 0:
                lines.append("<span style='color:#e05a5a'>⚠ No drain hole: uncured resin would stay "
                             "trapped inside.</span>")
        self.hollow_info.setText("<br>".join(lines))
        self.hollow_info.setVisible(bool(lines))
        self.sections["hollow"].set_summary(describe(hollow, holes) if has else "no selection")

    # ------------------------------------------------------------ hollowing
    def add_hole_at(self, obj, point, normal) -> None:
        """A click on the model in hole mode: remove the hole there, or add one."""
        self.scene.selected, self.scene.selected_support = obj, None
        m = obj.matrix()
        point = np.asarray(point, float)
        for h in obj.holes:
            c, _n = hole_world(m, h)
            if np.linalg.norm(c - point) <= max(h.diameter / 2, 1.0) + 0.5:
                self.scene.push_undo()
                obj.holes = tuple(x for x in obj.holes if x is not h)
                self.hole_message = ""
                self.scene.changed()
                return
        hole = hole_from_world(m, point, normal, self.scene.hollow_settings.hole_diameter)
        self.scene.push_undo()
        obj.holes = obj.holes + (hole,)
        lowest = obj.world_bounds()[0][2]
        touches = point[2] - hole.diameter / 2 - 0.3 <= lowest and lowest < 0.05
        self.hole_message = ("This one touches the build plate: the ring around it would start in mid-air "
                             "(the layer view will show an island) - a little higher up the side is better."
                             if touches else "")
        self.scene.changed()

    def _clear_holes(self) -> None:
        obj = self.scene.selected
        if obj is None or not obj.holes:
            return
        self.scene.push_undo()
        obj.holes = ()
        self.hole_message = ""
        self.scene.changed()

    def _toggle_hollow(self) -> None:
        obj = self.scene.selected
        if obj is None:
            return
        if obj.hollow is None and not obj.holes:
            QMessageBox.information(self, "Hollowing", "Add at least one drain hole first: click Add holes, "
                                    "then the model.  Without one, uncured resin stays trapped inside.")
            return
        self.scene.push_undo()
        obj.hollow = None if obj.hollow is not None else HollowState(self.scene.hollow_settings.wall_thickness)
        self.scene.changed()

    def _toggle_lattice(self) -> None:
        obj = self.scene.selected
        if obj is None or obj.hollow is None:
            return
        self.scene.push_undo()
        lattice = None if obj.hollow.lattice is not None else self.scene.hollow_settings.lattice()
        obj.hollow = replace(obj.hollow, lattice=lattice)
        self.scene.changed()

    def edit_hollow_settings(self) -> None:
        obj = self.scene.selected
        target = obj if obj is not None and obj.hollow is not None else None
        dlg = HollowSettingsDialog(self.scene.hollow_settings, self,
                                   applies_to=f"'{target.name}', which is hollow" if target else "")
        if not dlg.exec():
            return
        self.scene.push_undo()
        settings = dlg.result_settings()
        self.scene.hollow_settings = settings
        if target is not None:                      # the selected hollow model follows the new values
            target.hollow = HollowState(settings.wall_thickness,
                                        settings.lattice() if target.hollow.lattice is not None else None)
        self.scene.changed()
        self.hollow_settings_changed.emit()

    # ------------------------------------------------------------ slots
    def _apply_position(self) -> None:
        obj = self.scene.selected
        if self._updating or obj is None:
            return
        self.scene.push_undo()
        self.scene.move_object(obj, [sp.value() for sp in self.pos_spins])
        self.scene.changed()

    def _apply_rotation(self) -> None:
        obj = self.scene.selected
        if self._updating or obj is None:
            return
        self.scene.push_undo()
        obj.rotation[:] = [sp.value() for sp in self.rot_spins]
        obj.drop_to_plate()
        self.scene.changed()

    def _rotate(self, axis: int, deg: float) -> None:
        obj = self.scene.selected
        if obj is None:
            return
        self.scene.push_undo()
        obj.rotate(axis, deg)
        self.scene.changed()

    def _reset_rotation(self) -> None:
        obj = self.scene.selected
        if obj is None:
            return
        self.scene.push_undo()
        obj.rotation[:] = 0
        obj.drop_to_plate()
        self.scene.changed()

    @staticmethod
    def _model_size(obj) -> np.ndarray:
        """The model's size along its own X, Y, Z axes (scaled, before rotation)."""
        return np.asarray(obj.mesh.extents, float) * np.asarray(obj.scale, float)

    def _apply_size(self, axis: int, mm: float) -> None:
        """A size typed in mm: the scale that gives it (all three axes when Uniform)."""
        obj = self.scene.selected
        if self._updating or obj is None:
            return
        extent = float(obj.mesh.extents[axis])
        if extent < 1e-9:
            return
        self._apply_scale(axis, mm / extent * 100.0)

    def _apply_scale(self, axis: int, percent: float) -> None:
        obj = self.scene.selected
        if self._updating or obj is None:
            return
        self.scene.push_undo()
        if self.uniform.isChecked():
            obj.set_uniform_scale(percent / 100.0)
        else:
            obj.scale[axis] = percent / 100.0
            obj.drop_to_plate()
        self.scene.changed()

    def _drop(self) -> None:
        if self.scene.selected:
            self.scene.push_undo()
            self.scene.selected.drop_to_plate()
            self.scene.changed()

    def _center(self) -> None:
        if self.scene.selected:
            self.scene.push_undo()
            self.scene.selected.center_on_plate()
            self.scene.changed()

    def _mirror(self, axis: int) -> None:
        obj = self.scene.selected
        if obj is None:
            return
        self.scene.push_undo()
        obj.mirror(axis)
        self.mesh_edited.emit(obj)
        self.scene.changed()

    def clone_selected(self) -> None:
        if self.scene.selected:
            self.scene.clone(self.scene.selected)

    def delete_selected(self) -> None:
        if self.scene.selected:
            self.scene.remove(self.scene.selected)

    def repair_selected(self) -> None:
        obj = self.scene.selected
        if obj is None:
            return
        self.scene.push_undo()
        report = obj.repair()
        self.mesh_edited.emit(obj)
        self.scene.changed()
        b, a = report["before"], report["after"]
        QMessageBox.information(
            self, "Mesh repair",
            f"{obj.name}\n\nFaces: {b['faces']} -> {a['faces']}\n"
            f"Watertight: {b['watertight']} -> {a['watertight']}")

    def delete_selected_support(self) -> None:
        if self.scene.selected_support:
            self.scene.remove_support(self.scene.selected_support)

    def delete_key(self) -> None:
        """Del: remove the selected support if there is one, else the object."""
        if self.scene.selected_support:
            self.delete_selected_support()
        else:
            self.delete_selected()

    def auto_supports(self) -> None:
        if not self.scene.objects:
            return
        target = self.scene.selected if self.scene.selected else None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            n = self.scene.auto_supports(target, self.layer_height())
        finally:
            QApplication.restoreOverrideCursor()
        report = self.scene.support_report
        if n == 0 and not report.get("skipped"):
            QMessageBox.information(self, "Automatic supports",
                                    "Nothing needs support with the current settings: no islands and "
                                    "no overhangs steeper than the overhang angle.")
        self.support_message = _report_text(report)
        self.refresh()

    def _density_changed(self) -> None:
        if self._updating:
            return
        self.scene.support_settings.density = self.density.currentData()
        self.settings_changed.emit()

    def _z_lift_toggled(self, on: bool) -> None:
        """Only records the choice - the model is raised when Automatic runs."""
        if self._updating or on == self.scene.support_settings.z_lift:
            return
        self.scene.support_settings.z_lift = on
        self.settings_changed.emit()
        self.refresh()

    def edit_support_settings(self) -> None:
        dlg = SupportSettingsDialog(self.scene.support_settings, self)
        if dlg.exec():
            self.scene.push_undo()
            self.scene.support_settings = dlg.result_settings()
            self.scene.changed()
            self.settings_changed.emit()

    def _clear_supports(self) -> None:
        self.support_message = ""
        self.scene.clear_supports()

    def add_support_at(self, obj, point: np.ndarray) -> None:
        sup = manual_support(obj, np.asarray(point, float), self.scene.support_settings)
        if sup is None:
            self.support_message = ("No room for a support there: no path to the plate and no "
                                    "floor below to stand on.")
            self.refresh()
            return
        self.support_message = ""
        self.scene.add_support(sup)


def _report_text(report: dict) -> str:
    """'Automatic: 12 straight · 5 routed · 3 nook · 1 spot skipped …'"""
    names = (("straight", "straight"), ("split", "on split pillars"), ("routed", "routed around the part"),
             ("nook", "nook"))
    parts = [f"{report[k]} {label}" for k, label in names if report.get(k)]
    text = "Automatic: " + (" · ".join(parts) if parts else "no supports")
    bridged = report.get("bridged", 0)
    if bridged:
        text += f" · {bridged} spot{'s' if bridged != 1 else ''} on short bridges (no support needed)"
    skipped = report.get("skipped", 0)
    if skipped:
        text += (f" · {skipped} spot{'s' if skipped != 1 else ''} skipped (no path to the plate "
                 f"and no room for a nook support)")
    return text
