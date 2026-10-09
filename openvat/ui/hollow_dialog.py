"""The Hollowing settings window: drain holes, wall thickness, and the
internal supports (a 3-axis lattice of poles, turned about X, Y and Z -
with a small front and top view of how it is turned)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QVBoxLayout,
                               QDoubleSpinBox, QCheckBox, QLabel, QWidget)

from ..core.hollow import HollowSettings, lattice_rotation


def _double(lo, hi, step, decimals=2, suffix="") -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setDecimals(decimals)
    if suffix:
        sp.setSuffix(suffix)
    return sp


ANGLES = ("lattice_angle_x", "lattice_angle_y", "lattice_angle_z")

GROUPS = {
    "Drain holes": [
        ("hole_diameter", "Diameter of new holes", lambda: _double(0.5, 15, 0.5, 2, " mm")),
    ],
    "Shell": [
        ("wall_thickness", "Wall thickness", lambda: _double(0.5, 15, 0.1, 2, " mm")),
    ],
    "Internal supports (poles along 3 axes)": [
        ("lattice_angle_x", "Rotate about X", lambda: _double(0, 90, 5, 1, " °")),
        ("lattice_angle_y", "Rotate about Y", lambda: _double(0, 90, 5, 1, " °")),
        ("lattice_angle_z", "Rotate about Z", lambda: _double(0, 90, 5, 1, " °")),
        ("lattice_spacing", "Pole spacing", lambda: _double(2, 50, 0.5, 1, " mm")),
        ("pole_diameter", "Pole diameter", lambda: _double(0.3, 6, 0.1, 2, " mm")),
    ],
}

AXIS_COLORS = {"X": QColor(200, 45, 60), "Y": QColor(40, 100, 200), "Z": QColor(40, 165, 105)}


class LatticePreview(QWidget):
    """The turned pole directions seen from the front (X right, Z up, like
    looking at the printer) and from above (X right, Y away).  A pole
    pointing straight at the viewer shows as a ringed dot."""

    VIEWS = (("front", 0, 2), ("top", 0, 1))            # (title, screen-right axis, screen-up axis)

    def __init__(self, angles=(0.0, 45.0, 0.0), parent=None):
        super().__init__(parent)
        self.angles = tuple(float(a) for a in angles)
        self.setFixedSize(QSize(250, 140))
        self.setToolTip("How the pole grid is turned: seen from the front and from above")

    def set_angles(self, ax: float, ay: float, az: float) -> None:
        self.angles = (float(ax), float(ay), float(az))
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        axes = lattice_rotation(*self.angles)            # rows: the turned X, Y, Z
        text = self.palette().windowText().color()
        dim = QColor(text); dim.setAlpha(110)
        small = QFont(self.font()); small.setPointSizeF(max(small.pointSizeF() - 1.5, 6.5))
        half = self.width() / 2
        for k, (title, right, up) in enumerate(self.VIEWS):
            c = QPointF(half * k + half / 2, self.height() / 2 - 2)
            r = min(half, self.height()) * 0.34
            p.setFont(small)
            p.setPen(dim)
            p.drawText(QRectF(half * k, self.height() - 16, half, 14), Qt.AlignCenter, title)
            if title == "front":                          # the build plate under the front view
                p.setPen(QPen(dim, 1.2))
                p.drawLine(QPointF(half * k + 14, self.height() - 20), QPointF(half * (k + 1) - 14, self.height() - 20))
            order = sorted(range(3), key=lambda i: abs(axes[i][3 - right - up]))   # flattest-to-viewer last
            for i in reversed(order):
                name = "XYZ"[i]
                v = QPointF(axes[i][right] * r, -axes[i][up] * r)
                if (v.x() ** 2 + v.y() ** 2) ** 0.5 < 0.2 * r:            # pointing at us
                    p.setPen(QPen(AXIS_COLORS[name], 2.2))
                    p.setBrush(self.palette().window())
                    p.drawEllipse(c, 6.5, 6.5)
                    p.setBrush(AXIS_COLORS[name])
                    p.drawEllipse(c, 2.0, 2.0)
                    p.setBrush(Qt.NoBrush)
                    p.setPen(AXIS_COLORS[name])
                    p.drawText(QRectF(c.x() + 5, c.y() + 3, 16, 16), Qt.AlignCenter, name)
                    continue
                p.setPen(QPen(AXIS_COLORS[name], 5, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(c - v, c + v)
                p.setPen(AXIS_COLORS[name])
                tip = c + v * 1.22
                p.drawText(QRectF(tip.x() - 8, tip.y() - 8, 16, 16), Qt.AlignCenter, name)


class HollowSettingsDialog(QDialog):
    def __init__(self, settings: HollowSettings, parent=None, applies_to: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Hollowing settings")
        self.widgets: dict[str, object] = {}
        root = QVBoxLayout(self)
        for title, rows in GROUPS.items():
            box = QGroupBox(title)
            form = QFormLayout()
            for attr, label, make in rows:
                value = getattr(settings, attr)
                if make is None:
                    w = QCheckBox()
                    w.setChecked(bool(value))
                else:
                    w = make()
                    w.setValue(float(value))
                form.addRow(label, w)
                self.widgets[attr] = w
            if rows[0][0] in ANGLES:
                self.preview = LatticePreview([getattr(settings, a) for a in ANGLES])
                for a in ANGLES:
                    self.widgets[a].valueChanged.connect(self._angles_changed)
                lay = QVBoxLayout(box)
                lay.addLayout(form)
                lay.addWidget(self.preview, 0, Qt.AlignHCenter)
            else:
                box.setLayout(form)
            root.addWidget(box)

        note = QLabel(
            "<i>A model needs at least one drain hole before it can be hollowed - place holes where "
            "resin can run out, usually low on the side (not on a face resting on the plate). "
            "Hollowing keeps a wall of this thickness everywhere, floors and ceilings included.<br><br>"
            "Internal supports are straight poles through the cavity in three directions at right "
            "angles to each other (X, Y and Z), meeting at nodes.  The grid is turned about X, then Y, "
            "then Z: 0/0/0 = upright (X and Y poles horizontal); 0/45/0 (the default) = X and Z "
            "poles diagonal, crossing like an X seen from the front, Y poles running front to back; "
            "45/35/0 stands the grid on a corner, so no pole is flatter than 35°.  The spacing is "
            "the distance between nodes.</i>"
            + (f"<br><br><b>OK also applies these to {applies_to}.</b>" if applies_to else ""))
        note.setWordWrap(True)
        note.setMinimumWidth(380)
        root.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(lambda: self._load(HollowSettings()))
        root.addWidget(buttons)

    def _angles_changed(self, *_args) -> None:
        self.preview.set_angles(*(self.widgets[a].value() for a in ANGLES))

    def _load(self, s: HollowSettings) -> None:
        for attr, w in self.widgets.items():
            v = getattr(s, attr)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(float(v))

    def result_settings(self) -> HollowSettings:
        values = {}
        for attr, w in self.widgets.items():
            values[attr] = w.isChecked() if isinstance(w, QCheckBox) else w.value()
        return HollowSettings(**values)
