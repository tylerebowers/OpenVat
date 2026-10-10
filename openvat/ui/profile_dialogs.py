"""Dialogs for editing printer and resin profiles.

Both dialogs are built from a small declarative field list so adding a
setting later is a one-line change.  When opened from "Add" they get a list
of presets on the left: clicking one loads its values into the form, which
can then be adjusted and saved as a new profile.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout, QLineEdit,
                               QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QHBoxLayout, QWidget,
                               QListWidget, QLabel)

from ..core.profiles import PrinterProfile, ResinProfile, LayerSettings, OUTPUT_TYPES


def _double(lo, hi, step, decimals=3) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(decimals)
    return sp


def _int(lo, hi) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    return sp


class _ProfileDialog(QDialog):
    """Shared layout: [preset panel] [editor], buttons underneath.

    ``presets`` is either a list (printers) or a dict {group: list} (resins
    grouped by printer); a dict adds a combo box to pick the group first.
    Subclasses build the editor widget and implement load() / result_profile().
    """

    def __init__(self, title: str, editor: QWidget, presets, parent=None,
                 preset_group: str | None = None, intro: str = ""):
        super().__init__(parent)
        self.groups: dict[str, list] = presets if isinstance(presets, dict) else ({"": presets} if presets else {})
        self.presets: list = []
        root = QVBoxLayout(self)
        if intro:
            label = QLabel(intro)
            label.setWordWrap(True)
            root.addWidget(label)
        body = QHBoxLayout()
        root.addLayout(body)

        adding = presets is not None
        self.setWindowTitle(f"New {title.lower()}" if adding else title)
        if adding:
            box = QGroupBox("Presets")
            bl = QVBoxLayout(box)
            self.group_combo = None
            if isinstance(presets, dict):
                bl.addWidget(QLabel("1. Printer"))
                self.group_combo = QComboBox()
                self.group_combo.addItems(list(self.groups))
                bl.addWidget(self.group_combo)
                bl.addWidget(QLabel("2. Resin"))
            hint = QLabel("Click a preset to load its settings,\nthen adjust and save.")
            hint.setStyleSheet("color: #888;")
            bl.addWidget(hint)
            self.preset_list = QListWidget()
            self.preset_list.setMinimumWidth(380 if isinstance(presets, dict) else 260)
            self.preset_list.currentRowChanged.connect(self._preset_selected)
            bl.addWidget(self.preset_list, stretch=1)
            if not self.groups:
                none = QLabel("No presets available - fill in\nthe settings by hand.")
                none.setStyleSheet("color: #888;")
                bl.addWidget(none)
            body.addWidget(box)
            if self.group_combo is not None:
                self.group_combo.currentTextChanged.connect(self._show_group)
                if preset_group in self.groups:
                    self.group_combo.setCurrentText(preset_group)
                self._show_group(self.group_combo.currentText())
            else:
                self._show_group("")
        body.addWidget(editor, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save if adding else QDialogButtonBox.Ok)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _show_group(self, group: str) -> None:
        self.presets = self.groups.get(group, [])
        self.preset_list.blockSignals(True)
        self.preset_list.clear()
        for p in self.presets:
            self.preset_list.addItem(p.name)
            item = self.preset_list.item(self.preset_list.count() - 1)
            item.setToolTip(p.name)
            if getattr(p, "can_export", True) is False:
                # printers whose layer format OpenVat can't write yet
                item.setForeground(QColor(140, 140, 140))
                item.setToolTip(f"{p.name}\n.{p.file_extension} files ({p.layer_format}, Photon Workshop "
                                f"version {p.file_version}): slicing and previews work, export is not "
                                "supported yet")
        self.preset_list.blockSignals(False)

    def _preset_selected(self, row: int) -> None:
        if 0 <= row < len(self.presets):
            self.load(self.presets[row])

    def load(self, profile) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------
class PrinterDialog(_ProfileDialog):
    FIELDS = [
        # attribute, label, widget factory
        ("res_x", "X resolution (px)", lambda: _int(1, 100000)),
        ("res_y", "Y resolution (px)", lambda: _int(1, 100000)),
        ("print_x", "Print size X (mm)", lambda: _double(1, 2000, 0.1)),
        ("print_y", "Print size Y (mm)", lambda: _double(1, 2000, 0.1)),
        ("print_z", "Print size Z (mm)", lambda: _double(1, 2000, 0.1)),
        ("pixel_x_um", "X pixel size (µm)", lambda: _double(1, 500, 0.1)),
        ("pixel_y_um", "Y pixel size (µm)", lambda: _double(1, 500, 0.1)),
    ]

    def __init__(self, printer: PrinterProfile, parent=None, presets: list[PrinterProfile] | None = None,
                 intro: str = ""):
        self.printer = printer
        editor = QWidget()
        form = QFormLayout(editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.output = QComboBox()
        for key, label in OUTPUT_TYPES.items():
            self.output.addItem(label, key)
        form.addRow("Output type", self.output)
        self.extension = QLineEdit()
        self.extension.setPlaceholderText("pwsz")
        form.addRow("File extension", self.extension)
        self.widgets = {}
        for attr, label, make in self.FIELDS:
            w = make()
            form.addRow(label, w)
            self.widgets[attr] = w
        super().__init__("Printer profile", editor, presets, parent, intro=intro)
        self.load(printer)

    def load(self, printer: PrinterProfile) -> None:
        self.printer = printer
        self.name.setText(printer.name)
        self.output.setCurrentIndex(max(self.output.findData(printer.output_type), 0))
        self.extension.setText(printer.file_extension)
        for attr, w in self.widgets.items():
            w.setValue(getattr(printer, attr))

    def result_profile(self) -> PrinterProfile:
        values = {attr: w.value() for attr, w in self.widgets.items()}
        output = self.output.currentData()
        if output == "anycubic_pwsz":
            layer_format = "pwszImg"
        else:                                    # bitmap: keep the preset's format, pw0Img for new ones
            layer_format = self.printer.layer_format if self.printer.layer_format != "pwszImg" else "pw0Img"
        return replace(self.printer, name=self.name.text().strip() or "Printer",
                       output_type=output, layer_format=layer_format,
                       file_extension=self.extension.text().strip().lstrip(".") or "pwsz", **values)


# --------------------------------------------------------------------------
class _LayerSettingsBox(QGroupBox):
    FIELDS = [
        ("thickness", "Layer thickness (mm)", lambda: _double(0.005, 1.0, 0.005)),
        ("exposure", "Exposure time (s)", lambda: _double(0, 600, 0.1, 2)),
        ("off_time", "Off time (s)", lambda: _double(0, 600, 0.1, 2)),
        ("wait_before_lift", "Wait before lift (s)", lambda: _double(0, 600, 0.1, 2)),
        ("wait_after_lift", "Wait after lift (s)", lambda: _double(0, 600, 0.1, 2)),
        ("lift_distance", "Z lift distance (mm)", lambda: _double(0, 100, 0.5, 2)),
        ("lift_speed", "Z lift speed (mm/s)", lambda: _double(0.1, 100, 0.5, 2)),
        ("retract_speed", "Z retract speed (mm/s)", lambda: _double(0.1, 100, 0.5, 2)),
    ]

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        form = QFormLayout(self)
        self.widgets = {}
        for attr, label, make in self.FIELDS:
            w = make()
            form.addRow(label, w)
            self.widgets[attr] = w
        self.form = form

    def load(self, settings: LayerSettings) -> None:
        for attr, w in self.widgets.items():
            w.setValue(getattr(settings, attr))

    def value(self) -> LayerSettings:
        return LayerSettings(**{attr: w.value() for attr, w in self.widgets.items()})


class ResinDialog(_ProfileDialog):
    def __init__(self, resin: ResinProfile, parent=None,
                 presets: dict[str, list[ResinProfile]] | None = None, printer_name: str = ""):
        self.resin = resin
        editor = QWidget()
        root = QVBoxLayout(editor)
        root.setContentsMargins(0, 0, 0, 0)

        top = QFormLayout()
        self.name = QLineEdit(); top.addRow("Name", self.name)
        self.brand = QLineEdit(); top.addRow("Brand", self.brand)
        self.density = _double(0.5, 3.0, 0.01, 3); top.addRow("Density (g/cm³)", self.density)
        self.price = _double(0, 10000, 1, 2); top.addRow("Price per liter", self.price)
        root.addLayout(top)

        columns = QHBoxLayout()
        self.normal_box = _LayerSettingsBox("Normal layers")
        self.bottom_box = _LayerSettingsBox("Bottom layers")
        self.bottom_layers = _int(0, 100)
        self.bottom_box.form.insertRow(0, "Bottom layer count", self.bottom_layers)
        self.elephant = _double(0, 2.0, 0.01, 3)
        self.elephant.setSuffix(" mm inward")
        tip = ("Elephant-foot compensation, applied to the bottom layers only.\n"
               "Every outline edge is moved inward by this distance (a polygon offset,\n"
               "not a radius or a scale): outer edges move in, holes grow.\n"
               "Example: with 0.2 mm a 10 x 10 mm square becomes 9.6 x 9.6 mm,\n"
               "and a 4 mm hole becomes 4.4 mm.")
        self.elephant.setToolTip(tip)
        self.bottom_box.form.addRow("Elephant foot (edge offset)", self.elephant)
        self.bottom_box.form.labelForField(self.elephant).setToolTip(tip)
        columns.addWidget(self.bottom_box)
        columns.addWidget(self.normal_box)
        root.addLayout(columns)

        extra = QGroupBox("Transition, anti-aliasing and shrinkage")
        form = QFormLayout(extra)
        self.transition = _int(0, 100)
        form.addRow("Transition layers (bottom → normal)", self.transition)
        self.aa = QComboBox()
        for n in (1, 2, 4, 8, 16):
            self.aa.addItem("Off" if n == 1 else f"{n}x", n)
        form.addRow("Anti-aliasing", self.aa)
        shrink = QWidget(); sl = QHBoxLayout(shrink); sl.setContentsMargins(0, 0, 0, 0)
        self.shrink = []
        for axis in "XYZ":
            sp = _double(0.5, 1.5, 0.001, 4); sp.setPrefix(f"{axis} ")
            sl.addWidget(sp); self.shrink.append(sp)
        form.addRow("Shrinkage compensation (scale)", shrink)
        root.addWidget(extra)

        intro = f"Resin profile for <b>{printer_name}</b>" if printer_name else ""
        super().__init__("Resin profile", editor, presets, parent, preset_group=printer_name, intro=intro)
        self.load(resin)

    def load(self, resin: ResinProfile) -> None:
        self.resin = resin
        self.name.setText(resin.name)
        self.brand.setText(resin.brand)
        self.density.setValue(resin.density)
        self.price.setValue(resin.price_per_liter)
        self.normal_box.load(resin.normal)
        self.bottom_box.load(resin.bottom)
        self.bottom_layers.setValue(resin.bottom_layers)
        self.elephant.setValue(resin.elephant_foot_mm)
        self.transition.setValue(resin.transition_layers)
        self.aa.setCurrentIndex(max(self.aa.findData(int(resin.anti_aliasing)), 0))
        for sp, v in zip(self.shrink, (resin.shrink_x, resin.shrink_y, resin.shrink_z)):
            sp.setValue(v)

    def result_profile(self) -> ResinProfile:
        return ResinProfile(
            name=self.name.text().strip() or "Resin",
            brand=self.brand.text().strip(),
            density=self.density.value(),
            price_per_liter=self.price.value(),
            currency=self.resin.currency,
            normal=self.normal_box.value(),
            bottom=self.bottom_box.value(),
            bottom_layers=self.bottom_layers.value(),
            elephant_foot_mm=self.elephant.value(),
            transition_layers=self.transition.value(),
            anti_aliasing=self.aa.currentData(),
            shrink_x=self.shrink[0].value(),
            shrink_y=self.shrink[1].value(),
            shrink_z=self.shrink[2].value(),
        )
