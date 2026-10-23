"""Interface zoom: Ctrl + / Ctrl - / Ctrl + mouse wheel anywhere, View →
Zoom, or Edit → Preferences.

Qt scales a whole interface only at start-up (QT_SCALE_FACTOR, set from the
saved zoom - see app_settings_dialog.apply_ui_scale), so a zoom while
OpenVat runs scales the application font instead: text, buttons, spin
boxes, lists and the panels made from them grow and shrink at once.  The
zoom is saved, and the next start applies it through QT_SCALE_FACTOR (then
everything, pictures and icons too, is at that size).

Widgets that use a font larger or smaller than the normal one (headings,
the big numbers on the home page, the page buttons) register it with
``relative_font`` so they follow the zoom too; sizes given in pixels can
use ``scaled`` (the zoom since start-up)."""

from __future__ import annotations

import weakref

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

ZOOM_LEVELS = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
WHEEL_NOTCH = 120

_relative: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()   # widget -> (factor, bold)
_current: list = [None]                    # the UiZoom in use (for ``scaled``)


def relative_font(widget, factor: float = 1.0, bold: bool | None = None) -> None:
    """Give ``widget`` the application font times ``factor`` (and bold or
    not), now and whenever the zoom changes."""
    _relative[widget] = (factor, bold)
    _apply_relative(widget, factor, bold)


def _apply_relative(widget, factor: float, bold: bool | None) -> None:
    f = QFont(QApplication.font())
    if f.pointSizeF() > 0:
        f.setPointSizeF(f.pointSizeF() * factor)
    elif f.pixelSize() > 0:
        f.setPixelSize(max(1, round(f.pixelSize() * factor)))
    if bold is not None:
        f.setBold(bold)
    widget.setFont(f)


def scaled(px: float) -> int:
    """A size in pixels at the zoom since start-up."""
    zoom = _current[0]
    return round(px * (zoom.relative() if zoom is not None else 1.0))


class UiZoom(QObject):
    """Keeps the zoom (``settings["ui_scale"]``) and applies it."""
    changed = Signal(float)                # the new zoom

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        app = QApplication.instance()
        self.start = self._saved()             # what QT_SCALE_FACTOR was set from
        self.zoom = self.start
        self.base = QFont(app.font())          # the font at start-up
        self._wheel = 0
        app.installEventFilter(self)
        _current[0] = self

    def _saved(self) -> float:
        try:
            return min(max(float(self.store.settings.get("ui_scale", 1.0)), ZOOM_LEVELS[0]), ZOOM_LEVELS[-1])
        except (TypeError, ValueError):
            return 1.0

    def relative(self) -> float:
        """The zoom since start-up (1.0 = as Qt scaled it at the start)."""
        return self.zoom / self.start

    def set_zoom(self, zoom: float) -> None:
        zoom = min(max(float(zoom), ZOOM_LEVELS[0]), ZOOM_LEVELS[-1])
        if abs(zoom - self.zoom) < 1e-6:
            return
        self.zoom = zoom
        f = QFont(self.base)
        if f.pointSizeF() > 0:
            f.setPointSizeF(self.base.pointSizeF() * self.relative())
        elif f.pixelSize() > 0:
            f.setPixelSize(max(1, round(self.base.pixelSize() * self.relative())))
        QApplication.setFont(f)
        for widget, (factor, bold) in list(_relative.items()):
            try:
                _apply_relative(widget, factor, bold)
            except RuntimeError:                # its C++ side is gone
                _relative.pop(widget, None)
        self.store.settings["ui_scale"] = zoom
        self.store.save_settings()
        self.changed.emit(zoom)

    def step(self, steps: int) -> None:
        """``steps`` zoom levels in (+) or out (-)."""
        i = min(range(len(ZOOM_LEVELS)), key=lambda k: abs(ZOOM_LEVELS[k] - self.zoom))
        self.set_zoom(ZOOM_LEVELS[min(max(i + steps, 0), len(ZOOM_LEVELS) - 1)])

    def reset(self) -> None:
        self.set_zoom(1.0)

    def eventFilter(self, obj, event) -> bool:
        """Ctrl + wheel anywhere zooms the interface (not the 3D view, not a
        spin box)."""
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            self._wheel += delta
            while abs(self._wheel) >= WHEEL_NOTCH:
                notch = 1 if self._wheel > 0 else -1
                self._wheel -= notch * WHEEL_NOTCH
                self.step(notch)
            return True
        return False
