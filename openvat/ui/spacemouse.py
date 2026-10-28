"""The SpaceMouse in the running program: reads it without blocking (a
QSocketNotifier on spacenavd's socket or the device node), applies its
motion about 60 times a second while the cap is pushed, and looks for it
again every few seconds while it is not there (plugged in later,
spacenavd started later, unplugged and back).  What the motion and the
buttons do is up to the main window (``moved`` and ``button``)."""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QSocketNotifier, QTimer, Signal

from ..core import spacemouse as core
from ..core.spacemouse import Button, Motion, SpaceMouseSettings

TICK_MS = 16                 # apply the motion this often while the cap is pushed
RETRY_MS = 5000              # look for the SpaceMouse this often while it is not there
MAX_DT = 0.1                 # s: a stalled event loop does not turn into one big jump


class SpaceMouse(QObject):
    moved = Signal(tuple, float)          # normalized axes (core.spacemouse), seconds they apply for
    button = Signal(int, bool)            # button (0 = the first), pressed
    status_changed = Signal(str)

    def __init__(self, settings: SpaceMouseSettings | None = None, parent=None, connect=core.connect):
        super().__init__(parent)
        self.settings = settings or SpaceMouseSettings()
        self._connect = connect
        self.source = None
        self._notifier: QSocketNotifier | None = None
        self.axes = (0.0,) * 6
        self.status = ""
        self._last = 0.0
        self._tick = QTimer(self)
        self._tick.setInterval(TICK_MS)
        self._tick.timeout.connect(self._apply)
        self._retry = QTimer(self)
        self._retry.setInterval(RETRY_MS)
        self._retry.timeout.connect(self.try_connect)
        if self.settings.enabled:
            self.try_connect()
        else:
            self._set_status("Off")

    @property
    def connected(self) -> bool:
        return self.source is not None

    def set_settings(self, settings: SpaceMouseSettings) -> None:
        self.settings = settings
        if settings.enabled and not self.connected:
            self.try_connect()
        elif not settings.enabled:
            self.stop()
            self._set_status("Off")

    def try_connect(self) -> None:
        if self.connected or not self.settings.enabled:
            return
        source, status = self._connect()
        self._set_status(status)
        if source is None:
            if not self._retry.isActive():
                self._retry.start()
            return
        self._retry.stop()
        self.source = source
        self._notifier = QSocketNotifier(source.fileno(), QSocketNotifier.Read, self)
        self._notifier.activated.connect(self._readable)

    def stop(self) -> None:
        """Let go of the device (and stop looking for it)."""
        self._retry.stop()
        self._drop()

    def _drop(self) -> None:
        self._tick.stop()
        self.axes = (0.0,) * 6
        if self._notifier is not None:
            self._notifier.setEnabled(False)
            self._notifier.deleteLater()
            self._notifier = None
        if self.source is not None:
            self.source.close()
            self.source = None

    def _readable(self, *_args) -> None:
        if self.source is None:
            return
        try:
            events = self.source.read()
        except EOFError:
            self._drop()
            self._set_status("The SpaceMouse went away - looking for it again")
            self._retry.start()
            return
        for ev in events:
            if isinstance(ev, Motion):
                self._set_axes(ev.axes)
            elif isinstance(ev, Button):
                self.button.emit(ev.index, ev.pressed)

    def _set_axes(self, axes) -> None:
        self.axes = tuple(axes)
        if any(self.axes) and not self._tick.isActive():
            self._last = time.monotonic()
            self._tick.start()

    def _apply(self) -> None:
        if self.source is not None:
            settled = self.source.settle()
            if settled is not None:
                self.axes = settled.axes
        if not any(self.axes):
            self._tick.stop()
            return
        now = time.monotonic()
        dt, self._last = min(max(now - self._last, 0.0), MAX_DT), now
        if dt > 0:
            self.moved.emit(self.axes, dt)

    def _set_status(self, text: str) -> None:
        if text != self.status:
            self.status = text
            self.status_changed.emit(text)
