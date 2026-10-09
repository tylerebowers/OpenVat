"""Application entry point: ``openvat`` or ``python -m openvat``."""

from __future__ import annotations

import faulthandler
import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from . import APP_NAME
from .ui.glplatform import default_surface_format, match_pyopengl

# Nothing above may import PyOpenGL (OpenGL.GL): it has to wait until
# match_pyopengl() has told it whether Qt uses GLX or EGL.


def _report_crashes() -> None:
    """A crash in native code (a segmentation fault in a driver, say) prints
    the Python call stack to the terminal, so it can be reported."""
    if sys.platform == "win32" or sys.stderr is None:    # Windows: drivers trip it harmlessly
        return
    try:
        faulthandler.enable()
    except Exception:                                    # pragma: no cover - no usable stderr
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    _report_crashes()
    from .core.profiles import ProfileStore
    from .ui.app_settings_dialog import apply_ui_scale, apply_dark_mode, dark_mode

    store = ProfileStore()                    # no Qt needed; read before the app exists
    apply_ui_scale(store)                     # interface zoom (Edit -> Preferences)

    QSurfaceFormat.setDefaultFormat(default_surface_format())
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    match_pyopengl()                          # PyOpenGL must use Qt's GLX / EGL (see glplatform)
    apply_dark_mode(dark_mode(store))         # dark (or the user's light) theme, before any window

    from .ui.viewport3d import probe_gl       # imports PyOpenGL - only now
    probe_gl()                                # which GPU / renderer (see Help -> About)

    from .ui.main_window import MainWindow   # after QApplication exists
    from .ui.side_panel import add_first_printer

    if not store.printers and not add_first_printer(store):
        return 0                              # no printer set up: nothing to do
    window = MainWindow(store)
    window.show()
    for path in argv[1:]:                     # models given on the command line
        window.load_model(path)
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
