"""Dialog for editing SupportSettings: Base, Pillars, Paths and nooks, Interface, Other."""

from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QGridLayout,
                               QVBoxLayout, QDoubleSpinBox, QCheckBox, QComboBox, QLabel)

from ..core.supports import SupportSettings, DENSITY_SPACING


def _double(lo, hi, step, decimals=2, suffix="") -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(decimals)
    if suffix:
        sp.setSuffix(suffix)
    return sp


# (attribute, label, factory) per group
GROUPS = {
    "Base": [
        ("z_lift", "Z lift when generating automatically", None),
        ("z_lift_height", "Z lift height (lowest point of model)", lambda: _double(0, 100, 0.5, 2, " mm")),
        ("base_height", "Plate / foot height", lambda: _double(0.1, 10, 0.1, 2, " mm")),
        ("base_diameter", "Foot diameter", lambda: _double(0.5, 20, 0.5, 2, " mm")),
        ("raft", "Raft: one plate under each object's supports", None),
        ("raft_margin", "Raft margin past the outer feet", lambda: _double(0, 20, 0.5, 2, " mm")),
        ("perforations", "Perforations (hex pattern)", None),
        ("perforation_size", "Hole size (across flats)", lambda: _double(0.5, 20, 0.5, 2, " mm")),
        ("perforation_spacing", "Hole pitch", lambda: _double(1, 30, 0.5, 2, " mm")),
    ],
    "Pillars, branches and braces": [
        ("pillar_diameter", "Pillar / branch / bulb diameter", lambda: _double(0.3, 10, 0.1, 2, " mm")),
        ("branches", "Split pillars (several contacts per pillar)", None),
        ("branch_max_count", "Max contacts per pillar", lambda: _double(2, 6, 1, 0)),
        ("branch_max_distance", "Max distance between contacts", lambda: _double(1, 30, 0.5, 1, " mm")),
        ("branch_angle", "Max branch / detour tilt from vertical", lambda: _double(5, 60, 5, 1, " °")),
        ("braces", "X braces between neighboring pillars", None),
        ("brace_diameter", "Brace diameter", lambda: _double(0.2, 5, 0.1, 2, " mm")),
        ("brace_max_distance", "Max pillar distance for a brace", lambda: _double(1, 50, 0.5, 1, " mm")),
        ("brace_angle", "Brace angle above horizontal", lambda: _double(10, 80, 5, 1, " °")),
    ],
    "Paths and nook supports": [
        ("routing", "Route pillars around the part to the plate", None),
        ("model_clearance", "Clearance from the part", lambda: _double(0, 5, 0.1, 2, " mm")),
        ("nook_diameter", "Nook support diameter", lambda: _double(0.2, 10, 0.1, 2, " mm")),
        ("nook_contact_diameter", "Nook contact diameter (both ends)", lambda: _double(0.1, 5, 0.05, 2, " mm")),
        ("nook_max_gap", "Nook straight across gaps up to", lambda: _double(0, 20, 0.5, 1, " mm")),
    ],
    "Interface": [
        ("tip_depth", "Distance into model", lambda: _double(0, 3, 0.05, 2, " mm")),
        ("tip_diameter", "Contact diameter (0 = automatic)", lambda: _double(0, 5, 0.05, 2, " mm")),
        ("tip_length", "Arm length (vertical)", lambda: _double(0.2, 20, 0.5, 2, " mm")),
        ("straight_below", "Straight arm if face tilt below", lambda: _double(0, 45, 1, 1, " °")),
        ("max_arm_angle", "Max arm tilt (perpendicular to face)", lambda: _double(0, 80, 5, 1, " °")),
    ],
    "Other": [
        ("density", "Density", None),
        ("overhang_angle", "Overhang angle (needs support beyond)", lambda: _double(0, 90, 5, 1, " °")),
        ("max_bridge", "Bridges up to this need no support", lambda: _double(0, 20, 0.5, 1, " mm")),
        ("min_support_height", "Minimum support height", lambda: _double(0, 50, 0.1, 2, " mm")),
        ("island_anchors", "Supports at each island's lowest point", lambda: _double(1, 12, 1, 0)),
        ("contact_min", "Automatic contact diameter: min", lambda: _double(0.1, 5, 0.05, 2, " mm")),
        ("contact_max", "Automatic contact diameter: max", lambda: _double(0.1, 5, 0.05, 2, " mm")),
        ("contact_scale_height", "…reached at support height", lambda: _double(1, 300, 5, 1, " mm")),
    ],
}


class SupportSettingsDialog(QDialog):
    def __init__(self, settings: SupportSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Support settings")
        self.widgets: dict[str, object] = {}
        root = QVBoxLayout(self)
        grid = QGridLayout()
        root.addLayout(grid)

        for index, (title, rows) in enumerate(GROUPS.items()):
            box = QGroupBox(title)
            form = QFormLayout(box)
            for attr, label, make in rows:
                value = getattr(settings, attr)
                if attr == "density":
                    w = QComboBox()
                    for key in DENSITY_SPACING:
                        w.addItem(f"{key.capitalize()} ({DENSITY_SPACING[key]:.1f} mm spacing)", key)
                    w.setCurrentIndex(max(w.findData(value), 0))
                elif make is None:
                    w = QCheckBox()
                    w.setChecked(bool(value))
                else:
                    w = make()
                    w.setValue(float(value))
                form.addRow(label, w)
                self.widgets[attr] = w
            grid.addWidget(box, index // 3, index % 3)

        note = QLabel("<i>Automatic first simulates the print: it slices the model at the resin's layer "
                      "height and finds where supports are needed - every island (a region that starts in "
                      "mid-air, supported in the very layer it starts, and anchored by several supports "
                      "if it starts as a corner or tip), then overhangs steeper than the overhang angle, "
                      "spaced by the density, except short bridges between two walls. Then it decides each "
                      "support's kind - a small nook straight across a narrow gap, a plate support when a "
                      "pillar can reach the build plate without touching the part, otherwise a nook "
                      "standing on the part - and routes the plate supports around the part. After "
                      "slicing, the layer view marks any region the print would not hold in red. Re-run "
                      "Automatic after changing these.</i>")
        note.setWordWrap(True)
        root.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore)
        root.addWidget(buttons)

    def _restore(self) -> None:
        self._load(SupportSettings())

    def _load(self, s: SupportSettings) -> None:
        for attr, w in self.widgets.items():
            v = getattr(s, attr)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(max(w.findData(v), 0))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(float(v))

    def result_settings(self) -> SupportSettings:
        values = {}
        for attr, w in self.widgets.items():
            if isinstance(w, QComboBox):
                values[attr] = w.currentData()
            elif isinstance(w, QCheckBox):
                values[attr] = w.isChecked()
            else:
                values[attr] = w.value()
        return SupportSettings(**values)
