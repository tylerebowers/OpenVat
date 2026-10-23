"""The "Export to USB" button and the watcher that finds USB drives.

``UsbWatcher`` looks for removable drives every few seconds (see
core/usb.py) and says when the set changes; ``UsbButton`` shows up next to
"Export…" while a drive is plugged in - "Export to USB (KINGSTON)", or a
menu of drives when there are several.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QMenu

from ..core.usb import Drive, removable_drives
from .nav import NavButton


class UsbWatcher(QObject):
    changed = Signal(list)              # [Drive, ...] whenever drives come or go

    def __init__(self, parent=None, interval_ms: int = 2500, probe=removable_drives):
        super().__init__(parent)
        self._probe = probe
        self.drives: list[Drive] = []
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        QTimer.singleShot(0, self.refresh)

    def refresh(self) -> list[Drive]:
        drives = self._probe()
        if drives != self.drives:
            self.drives = drives
            self.changed.emit(list(drives))
        return self.drives


class UsbButton(NavButton):
    export_to = Signal(object)          # the Drive picked

    def __init__(self, parent=None):
        super().__init__("Export to USB", parent)
        self.drives: list[Drive] = []
        self.clicked.connect(self._clicked)
        self.set_drives([])

    def set_drives(self, drives: list[Drive]) -> None:
        self.drives = list(drives)
        self.setVisible(bool(self.drives))
        self.setMenu(None)
        if len(self.drives) == 1:
            d = self.drives[0]
            self.setText(f"Export to USB ({d.label})")
            self.setToolTip(f"Save the print file to {d.describe()}" + (f"\n{d.path}" if d.path else ""))
        elif self.drives:
            self.setText("Export to USB")
            self.setToolTip("Save the print file to a USB drive")
            menu = QMenu(self)
            for d in self.drives:
                menu.addAction(d.describe(), lambda _=False, d=d: self.export_to.emit(d))
            self.setMenu(menu)

    def _clicked(self) -> None:
        if len(self.drives) == 1:
            self.export_to.emit(self.drives[0])
