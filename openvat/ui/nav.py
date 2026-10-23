"""The buttons that move between the pages (Home, Slice ▶, ← Back to 3D
view, Export…, Voxel preview →, Prepare → ...): a little taller than
ordinary buttons with larger text filling more of them, floating ``NAV_GAP``
px off the window's bottom and side, and they follow the interface zoom."""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QPushButton

from .zoom import relative_font, scaled

NAV_SCALE = 1.3          # text size (times the normal text)
NAV_HEIGHT = 1.75        # button height (times the text's line height)
NAV_GAP = 14             # px between the buttons and the window's bottom / side (at 100 % zoom)


def nav_gap() -> int:
    """NAV_GAP at the zoom now."""
    return scaled(NAV_GAP)


class NavButton(QPushButton):
    def __init__(self, text: str = "", parent=None, bold: bool = False):
        super().__init__(text, parent)
        relative_font(self, NAV_SCALE, bold)
        self._fit()

    def _fit(self) -> None:
        """Height and padding from the font, so the button grows with it."""
        h = QFontMetrics(self.font()).height()
        self.setMinimumHeight(round(h * NAV_HEIGHT))
        self.setStyleSheet(f"padding: {round(h * 0.18)}px {round(h * 0.7)}px;")

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == QEvent.FontChange:
            self._fit()
