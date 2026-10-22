"""Left-hand model editing panel, in two tabs: Manipulation (position /
rotation / scale / edit) and Preparation (supports / hollowing).  Each
group is a collapsible, resizable section whose header shows a summary
while folded.  Every control acts on ``scene.selected``.

Hollowing goes in order: drain holes first (click the model), then Hollow
(wall thickness from the settings window), then internal supports."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QSize, QEvent
from PySide6.QtGui import QIcon, QPainter, QPainterPath, QPen, QPixmap, QColor
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QGridLayout, QDoubleSpinBox, QPushButton,
                               QLabel, QCheckBox, QComboBox, QMessageBox, QApplication, QToolButton,
                               QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QVBoxLayout)

from ..core.scene import Scene
from ..core.hollow import HollowState, hole_from_world, hole_world, describe
from ..core.orient import DOWN, best_face, laid_on
from ..core.paint import NONE, ENFORCE, BLOCK, overhang_faces, painted_with, paint_of
from ..core.profiles import PrinterProfile
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
    move_tool_toggled = Signal(bool)    # move arrows on the selected model
    rotate_tool_toggled = Signal(bool)  # rotate rings on the selected model
    lay_face_toggled = Signal(bool)     # highlight the faces the model can lie on
    paint_mode_toggled = Signal(bool)   # dragging on the model paints where supports go / don't
    paint_options_changed = Signal()    # paint value or brush size
    supports_made = Signal(int)         # supports placed (Automatic, or one by a click)

    def __init__(self, scene: Scene, parent=None):
        super().__init__(width=240, parent=parent)
        self.scene = scene
        self._updating = False

        self.add("position", "Position", self._build_position())
        self.add("rotation", "Rotation", self._build_rotation())
        self.add("scale", "Scale", self._build_scale())
        self.add("edit", "Edit", self._build_edit())
        self.add("supports", "Supports", self._build_supports())
        self.add("paint", "Support paint", self._build_paint(), expanded=False)
        self.add("hollow", "Hollowing", self._build_hollowing())
        self.move_eye = _eye_button("Show move arrows on the selected model:\n"
                                    "drag one to move the model along that axis (Ctrl: whole mm)")
        self.rotate_eye = _eye_button("Show rotate rings on the selected model:\n"
                                      "drag one to turn the model about that axis (Ctrl: 15° steps)")
        self.sections["position"].add_header_widget(self.move_eye)
        self.sections["rotation"].add_header_widget(self.rotate_eye)
        self.move_eye.toggled.connect(self.move_tool_toggled)
        self.rotate_eye.toggled.connect(self.rotate_tool_toggled)
        self.set_pages([("Manipulation", ["position", "rotation", "scale", "edit"]),
                        ("Preparation", ["supports", "paint", "hollow"])])
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
        reset = QPushButton("Reset"); reset.clicked.connect(self._reset_rotation)
        reset.setToolTip("Reset the rotation to 0°, 0°, 0°")
        self.lay_btn = QPushButton("Lay on face")
        self.lay_btn.setCheckable(True)
        self.lay_btn.setToolTip("Highlight the model's large flat faces it can lie on (nothing goes below\n"
                                "the build plate); click one to put the model down on it")
        self.lay_btn.toggled.connect(self.lay_face_toggled)
        g.addWidget(reset, 3, 0, 1, 2); g.addWidget(self.lay_btn, 3, 2, 1, 2)
        self.best_btn = QPushButton("Lay flat automatically")
        self.best_btn.setToolTip("Put the model down on its best flat face: the one that leaves the least\n"
                                 "surface needing supports (facing down steeper than 45°) for the most on\n"
                                 "the plate; when that is a close call, the one that prints lowest")
        self.best_btn.clicked.connect(self._lay_best)
        g.addWidget(self.best_btn, 4, 0, 1, 4)
        self.lay_info = QLabel("")
        self.lay_info.setWordWrap(True)
        self.lay_info.setStyleSheet("color: #888;")
        self.lay_info.hide()
        g.addWidget(self.lay_info, 5, 0, 1, 4)
        self.lay_message: tuple | None = None       # (object, its rotation, text) after Lay flat automatically
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
        clone.setToolTip("One copy beside the model, with its supports (Ctrl+D)")
        self.clones_btn = _ThemedButton(_copies_icon, "Clone several times…")
        self.clones_btn.clicked.connect(self.clone_many_selected)
        repair = QPushButton("Repair mesh"); repair.clicked.connect(self.repair_selected)
        delete = QPushButton("Delete"); delete.clicked.connect(self.delete_selected)
        row = QHBoxLayout(); row.setSpacing(4)
        row.addWidget(clone, stretch=1); row.addWidget(self.clones_btn); row.addWidget(repair, stretch=1)
        g.addLayout(row, 1, 0, 1, 4)
        g.addWidget(delete, 2, 0, 1, 4)
        self.clone_count, self.clone_arrange = 4, True      # the window starts with the last choice
        self.printer = lambda: PrinterProfile()             # the printer Auto-arrange packs for (main window)
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

    def _build_paint(self) -> QWidget:
        """Paint where supports must go (blue) and where they must not (red);
        Automatic follows it."""
        w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
        self.paint_btn = QPushButton("Paint")
        self.paint_btn.setCheckable(True)
        self.paint_btn.setToolTip("Drag on the model to paint (its faces within the brush)")
        self.paint_btn.toggled.connect(self.paint_mode_toggled)
        self.paint_value = QComboBox()
        self.paint_value.addItem("Supports here", ENFORCE)
        self.paint_value.addItem("No supports", BLOCK)
        self.paint_value.addItem("Erase", NONE)
        self.paint_value.currentIndexChanged.connect(lambda _i: self.paint_options_changed.emit())
        g.addWidget(self.paint_btn, 0, 0); g.addWidget(self.paint_value, 0, 1)
        g.addWidget(QLabel("Brush"), 1, 0)
        self.brush = self._spin(0.2, 50, 0.5)
        self.brush.setValue(2.0)
        self.brush.setSuffix(" mm")
        self.brush.setToolTip("Brush radius: faces whose center lies this close to the cursor are painted")
        self.brush.valueChanged.connect(lambda _v: self.paint_options_changed.emit())
        g.addWidget(self.brush, 1, 1)
        self.overhang_btn = QPushButton("Paint overhangs")
        self.overhang_btn.setToolTip("Paint every face that faces down steeper than this angle from vertical\n"
                                     "(a wall: 0°, a flat ceiling: 90°) with the paint chosen above")
        self.overhang_btn.clicked.connect(self._paint_overhangs)
        self.overhang_angle = self._spin(0, 89, 5)
        self.overhang_angle.setDecimals(0)
        self.overhang_angle.setValue(45)
        self.overhang_angle.setSuffix(" °")
        g.addWidget(self.overhang_btn, 2, 0); g.addWidget(self.overhang_angle, 2, 1)
        self.clear_paint_btn = QPushButton("Clear paint")
        self.clear_paint_btn.clicked.connect(self._clear_paint)
        g.addWidget(self.clear_paint_btn, 3, 0, 1, 2)
        hint = QLabel("<i>Blue: supports go here · red: no supports.  Automatic follows the paint.</i>")
        hint.setWordWrap(True)
        g.addWidget(hint, 4, 0, 1, 2)
        return w

    def paint_settings(self) -> dict:
        """The paint value and brush radius (mm), as keywords of Viewport3D.set_paint."""
        return {"value": int(self.paint_value.currentData()), "radius": float(self.brush.value())}

    def _paint_overhangs(self) -> None:
        obj = self.scene.selected
        if obj is None:
            return
        mask = overhang_faces(obj, self.overhang_angle.value())
        if not mask.any():
            QMessageBox.information(self, "Support paint", "No face of this model faces down that steeply.")
            return
        self.scene.push_undo()
        obj.paint = painted_with(obj, mask, self.paint_settings()["value"])
        self.scene.changed()

    def _clear_paint(self) -> None:
        obj = self.scene.selected
        if obj is None or obj.paint is None:
            return
        self.scene.push_undo()
        obj.paint = None
        self.scene.changed()

    # ------------------------------------------------------------ refresh
    def refresh(self) -> None:
        """Pull values from the selected object into the controls and the
        section summaries."""
        obj = self.scene.selected
        self._updating = True
        enabled = obj is not None
        for w in (*self.pos_spins, *self.rot_spins, *self.scale_spins, *self.mm_spins, self.lay_btn,
                  self.best_btn, self.clones_btn, self.overhang_btn):
            w.setEnabled(enabled)
        painted = paint_of(obj) if obj is not None else None
        self.clear_paint_btn.setEnabled(painted is not None)
        if painted is None:
            self.sections["paint"].set_summary("no paint" if obj is not None else "no selection")
        else:
            n_on, n_off = int((painted == ENFORCE).sum()), int((painted == BLOCK).sum())
            self.sections["paint"].set_summary(f"{n_on} faces here · {n_off} blocked")
        msg = self.lay_message
        shown = msg is not None and msg[0] is obj and np.allclose(msg[1], obj.rotation)
        self.lay_info.setText(msg[2] if shown else "")
        self.lay_info.setVisible(shown)
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

    def _lay_best(self) -> None:
        """Lay flat automatically: the best of the model's flat faces down."""
        obj = self.scene.selected
        if obj is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            best = best_face(obj)
        finally:
            QApplication.restoreOverrideCursor()
        if best is None:
            text = "This model has no flat face it can lie on."
        else:
            if best.normal @ DOWN < 1 - 1e-6:
                self.scene.push_undo()
                obj.rotation[:] = laid_on(obj.rotation, best.normal)
                obj.drop_to_plate()
                head = "Laid on its best face"
            else:
                head = "Already on its best face"
            support = (f"{best.overhang:.0f} mm² facing down to support" if best.overhang >= 1
                       else "nothing facing down to support")
            text = f"{head}: {best.contact:.0f} mm² on the plate, {support}, {best.height:.1f} mm tall."
        self.lay_message = (obj, obj.rotation.copy(), text)
        self.scene.changed()

    def lay_on_face(self, obj, normal) -> None:
        """A click on a highlighted face: turn the model so it lies on it."""
        if obj is None:
            return
        self.scene.push_undo()
        obj.rotation[:] = laid_on(obj.rotation, normal)
        obj.drop_to_plate()
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

    def clone_many_selected(self) -> None:
        """The copies button: how many clones, and arrange the plate after."""
        obj = self.scene.selected
        if obj is None:
            return
        dlg = CloneDialog(obj.name, self.clone_count, self.clone_arrange, self)
        if not dlg.exec():
            return
        self.clone_count, self.clone_arrange = dlg.count.value(), dlg.arrange.isChecked()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.scene.clone_many(obj, self.clone_count, self.printer() if self.clone_arrange else None)
        finally:
            QApplication.restoreOverrideCursor()

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
        if n:
            self.supports_made.emit(n)
        report = self.scene.support_report
        if n == 0 and report.get("blocked"):
            QMessageBox.information(self, "Automatic supports",
                                    "No supports: every spot that needs one is painted \"No supports\".")
        elif n == 0 and not report.get("skipped"):
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
        self.supports_made.emit(1)


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
    blocked = report.get("blocked", 0)
    if blocked:
        text += f" · {blocked} spot{'s' if blocked != 1 else ''} left out (painted: no supports)"
    added = report.get("release_added", 0)
    if added:
        text += f" · {added} added to hold against the release force"
    checks = [c for c in report.get("release", []) if not c.on_plate]
    if checks:
        worst = min(checks, key=lambda c: c.ratio)
        text += "\n" + ("Release force: " if worst.ok else "⚠ Release force: ") + worst.text()
    return text


# --------------------------------------------------------------------------
def _eye_pixmap(color: QColor, open_: bool, size: int = 18, ratio: float = 2.0) -> QPixmap:
    """An eye (open) or a struck-through one (closed), drawn in ``color``."""
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    if not open_:
        c.setAlphaF(0.45)
    p.setPen(QPen(c, 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    mid, w = size / 2, size * 0.42
    lid = QPainterPath(QPointF(mid - w, mid))
    lid.quadTo(QPointF(mid, mid - w * 1.15), QPointF(mid + w, mid))
    lid.quadTo(QPointF(mid, mid + w * 1.15), QPointF(mid - w, mid))
    p.drawPath(lid)
    p.setBrush(c)
    r = size * 0.13
    p.drawEllipse(QRectF(mid - r, mid - r, 2 * r, 2 * r))
    if not open_:
        p.drawLine(QPointF(mid - w * 0.85, mid + w * 0.85), QPointF(mid + w * 0.85, mid - w * 0.85))
    p.end()
    return pm


class _ThemedButton(QToolButton):
    """A small icon button whose icon is drawn in the theme's text color
    (and redrawn when the theme changes)."""

    def __init__(self, make_icon, tip: str, checkable: bool = False, flat: bool = False):
        super().__init__()
        self._make_icon = make_icon
        self.setCheckable(checkable)
        self.setAutoRaise(flat)
        self.setToolTip(tip)
        self.setIconSize(QSize(18, 18))
        self.setIcon(make_icon(self.palette().buttonText().color()))

    def changeEvent(self, e) -> None:
        if e.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self.setIcon(self._make_icon(self.palette().buttonText().color()))
        super().changeEvent(e)


def _eye_icon(color: QColor) -> QIcon:
    icon = QIcon()
    icon.addPixmap(_eye_pixmap(color, True), QIcon.Normal, QIcon.On)
    icon.addPixmap(_eye_pixmap(color, False), QIcon.Normal, QIcon.Off)
    return icon


def _eye_button(tip: str) -> QToolButton:
    """A small checkable eye for a card header (on: the tool shows)."""
    return _ThemedButton(_eye_icon, tip, checkable=True, flat=True)



class CloneDialog(QDialog):
    """How many copies of the selected model, and whether to arrange the
    plate afterwards."""

    def __init__(self, name: str, count: int, arrange: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clone")
        root = QVBoxLayout(self)
        label = QLabel(f"Copies of <b>{name}</b>, each with its supports:")
        label.setTextFormat(Qt.RichText)
        root.addWidget(label)
        form = QFormLayout()
        self.count = QSpinBox()
        self.count.setRange(1, 200)
        self.count.setValue(max(1, min(int(count), 200)))
        form.addRow("Number of copies", self.count)
        root.addLayout(form)
        self.arrange = QCheckBox("Auto-arrange the plate afterwards")
        self.arrange.setChecked(arrange)
        self.arrange.setToolTip("Pack every model on the plate once the copies are made\n"
                                "(otherwise the copies line up beside the model)")
        root.addWidget(self.arrange)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Clone")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.count.selectAll()


def _copies_icon(color: QColor, size: int = 18, ratio: float = 2.0) -> QIcon:
    """Two stacked squares with a plus: "several copies"."""
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    back = QColor(c)
    back.setAlphaF(0.55)
    p.setPen(QPen(back, 1.3))
    p.drawRoundedRect(QRectF(2.5, 2.5, 9.5, 9.5), 1.8, 1.8)
    front = QRectF(6.0, 6.0, 9.5, 9.5)
    p.setCompositionMode(QPainter.CompositionMode_Clear)      # the front square hides the back one
    p.fillRect(front.adjusted(-1.0, -1.0, 1.0, 1.0), Qt.transparent)
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    p.setPen(QPen(c, 1.4))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(front, 1.8, 1.8)
    mid = front.center()
    p.drawLine(QPointF(mid.x() - 2.6, mid.y()), QPointF(mid.x() + 2.6, mid.y()))
    p.drawLine(QPointF(mid.x(), mid.y() - 2.6), QPointF(mid.x(), mid.y() + 2.6))
    p.end()
    return QIcon(pm)
