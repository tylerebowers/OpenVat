"""Layer viewer page: shows one sliced layer as the LCD would display it
(white = exposed), with a slider on the right to walk through the stack.
Unsupported islands (regions the print would not hold, see
``slicer.find_islands``) are drawn in red.

Everything else is in the bar at the bottom, in groups split by a vertical
line: the layer (number / last), this layer (z, thickness, exposure, area),
the whole print (height, resin, time), and islands when there are any."""

from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPainter, QPainterPath, QColor, QPen, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QLabel,
                               QPushButton, QFrame)

ISLAND_RED = "#e05a5a"


def separator() -> QFrame:
    """A short vertical line between groups in a bottom bar."""
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(20)
    return line


def duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes} min"

from ..core.slicer import SliceResult, Layer


class LayerCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layer: Layer | None = None
        self.islands: list = []            # outlines of unsupported islands on this layer
        self.plate = (223.642, 126.48)     # mm, set from the printer profile
        self.background = QColor(25, 26, 30)   # around the LCD; follows the 3D view's theme
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self._last = None
        self.setMinimumSize(400, 300)

    def set_layer(self, layer: Layer | None, islands: list | None = None) -> None:
        self.layer = layer
        self.islands = islands or []
        self.update()

    def set_background(self, color) -> None:
        """``color``: a QColor, '#rrggbb' or an (r, g, b, a) tuple of 0..1 floats."""
        self.background = QColor.fromRgbF(*color[:3]) if isinstance(color, tuple) else QColor(color)
        self.update()

    def reset_view(self) -> None:
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()

    def _scale(self) -> float:
        px, py = self.plate
        return min(self.width() / px, self.height() / py) * 0.92 * self.zoom

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self.background)
        s = self._scale()
        p.translate(self.width() / 2 + self.offset.x(), self.height() / 2 + self.offset.y())
        p.scale(s, -s)                       # mm -> px, Y up
        px, py = self.plate
        p.setPen(QPen(QColor(90, 95, 110), 1.0 / s))
        p.setBrush(QColor(0, 0, 0))
        p.drawRect(QRectF(-px / 2, -py / 2, px, py))
        if self.layer is None or self.layer.geometry.is_empty:
            return
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        for poly in self.layer.geometry.geoms:
            for ring in (poly.exterior, *poly.interiors):
                path.addPolygon([QPointF(x, y) for x, y in ring.coords])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(245, 245, 245))
        p.drawPath(path)
        if self.islands:                     # unsupported: red, with a ring so tiny ones show
            red = QPainterPath()
            red.setFillRule(Qt.OddEvenFill)
            for poly in self.islands:
                for ring in (poly.exterior, *poly.interiors):
                    red.addPolygon([QPointF(x, y) for x, y in ring.coords])
            p.setBrush(QColor(235, 60, 60))
            p.drawPath(red)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(235, 60, 60), 2.0 / s))
            for poly in self.islands:
                c = poly.centroid
                p.drawEllipse(QPointF(c.x, c.y), 12.0 / s, 12.0 / s)

    def wheelEvent(self, e: QWheelEvent) -> None:
        self.zoom = max(0.2, min(50.0, self.zoom * (1.15 ** (e.angleDelta().y() / 120))))
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self._last = e.position()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._last is not None and e.buttons():
            self.offset += e.position() - self._last
            self._last = e.position()
            self.update()

    def mouseDoubleClickEvent(self, _e) -> None:
        self.reset_view()


class PageButtons(QHBoxLayout):
    """Bottom-right row of navigation buttons shared by the result pages."""

    def __init__(self, buttons: list[tuple[str, object]]):
        super().__init__()
        self.setContentsMargins(6, 4, 6, 4)
        self.addStretch(1)
        for label, slot in buttons:
            b = QPushButton(label)
            b.setMinimumHeight(32)
            b.setStyleSheet("padding: 4px 14px;")
            b.clicked.connect(slot)
            self.addWidget(b)


class LayerViewPage(QWidget):
    back_requested = Signal()
    export_requested = Signal()
    voxels_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: SliceResult | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 0)

        body = QHBoxLayout()
        self.canvas = LayerCanvas()
        frame = QFrame(); frame.setFrameShape(QFrame.StyledPanel)
        fl = QVBoxLayout(frame); fl.setContentsMargins(0, 0, 0, 0); fl.addWidget(self.canvas)
        body.addWidget(frame, stretch=1)
        self.slider = QSlider(Qt.Vertical)
        self.slider.valueChanged.connect(self._show_layer)
        body.addWidget(self.slider)
        root.addLayout(body, stretch=1)

        # the bottom bar: layer | this layer | the whole print | islands, then the page buttons
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("Layer"))
        self.spin = QSpinBox()
        self.spin.setToolTip("Layers are numbered from 0 (the first layer on the plate)")
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.spin.setValue)
        bottom.addWidget(self.spin)
        self.layer_max = QLabel("/ 0")
        bottom.addWidget(self.layer_max)
        bottom.addWidget(separator())
        self.layer_info = QLabel("")                # z (thickness) · exposure · area
        bottom.addWidget(self.layer_info)
        bottom.addWidget(separator())
        self.summary = QLabel("")                   # height · resin · time
        bottom.addWidget(self.summary)
        self.island_sep = separator()
        bottom.addWidget(self.island_sep)
        self.island_info = QLabel("")
        self.island_info.setStyleSheet(f"color: {ISLAND_RED}; font-weight: bold;")
        bottom.addWidget(self.island_info)
        self.next_island = QPushButton("Next island ▸")
        self.next_island.clicked.connect(self._goto_next_island)
        bottom.addWidget(self.next_island)
        bottom.addLayout(PageButtons([("← Back to 3D view", self.back_requested),
                                      ("Export…", self.export_requested),
                                      ("Voxel preview →", self.voxels_requested)]), stretch=1)
        root.addLayout(bottom)
        self._show_islands([])

    def set_result(self, result: SliceResult) -> None:
        self.result = result
        n = len(result.layers)
        self.canvas.plate = (result.printer.print_x, result.printer.print_y)
        self.slider.setRange(0, max(n - 1, 0))
        self.spin.setRange(0, max(n - 1, 0))
        self.layer_max.setText(f"/ {max(n - 1, 0)}")
        self.slider.setValue(0)
        self.canvas.reset_view()
        self._show_layer(0)
        self.summary.setText(f"max z {result.height:.2f} mm · {result.volume_mm3() / 1000:.2f} mL · "
                             f"{duration(result.print_time_s())}")
        self.summary.setToolTip(f"{n} layers on {result.printer.name}")
        self._show_islands(result.islands)

    def _show_islands(self, islands) -> None:
        for w in (self.island_sep, self.island_info, self.next_island):
            w.setVisible(bool(islands))
        if not islands:
            return
        where = ", ".join(f"{i.first}–{i.last}" if i.last > i.first else f"{i.first}" for i in islands[:8])
        if len(islands) > 8:
            where += f" and {len(islands) - 8} more"
        s = "s" if len(islands) != 1 else ""
        self.island_info.setText(f"⚠ {len(islands)} island{s}")
        tip = (f"{len(islands)} unsupported island{s} (red), layers {where}: these regions would cure "
               "onto the vat film - add supports there.")
        self.island_info.setToolTip(tip)
        self.next_island.setToolTip(tip)

    def _show_layer(self, i: int) -> None:
        if self.result is None or not (0 <= i < len(self.result.layers)):
            return
        layer = self.result.layers[i]
        here = [p for isl in self.result.islands for p in isl.regions.get(i, [])]
        self.canvas.set_layer(layer, here)
        self.layer_info.setText(
            f"z {layer.z_top:.3f} mm ({layer.thickness:.3f} mm) · exposure {layer.exposure:.2f} s · "
            f"area {layer.area:.1f} mm²")
        self.layer_info.setToolTip(f"layer {i}: {layer.z_bottom:.3f} – {layer.z_top:.3f} mm, "
                                   f"{sum(1 + len(p.interiors) for p in layer.geometry.geoms)} contours")

    def _goto_next_island(self) -> None:
        """Jump to the first layer of the next island after the current layer."""
        if self.result is None or not self.result.islands:
            return
        cur = self.slider.value()
        starts = sorted({i.first for i in self.result.islands})
        nxt = next((k for k in starts if k > cur), starts[0])
        self.slider.setValue(nxt)
