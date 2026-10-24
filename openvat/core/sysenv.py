"""The environment for programs OpenVat starts (udisksctl and gio to mount
and eject USB drives, xdg-open for "Show in folder" and links, ...).

A release build (PyInstaller: the AppImage, the Windows and macOS apps)
starts with its own folder first in the library search path
(LD_LIBRARY_PATH, keeping the old value as LD_LIBRARY_PATH_ORIG) and with
Qt's plugin paths pointing at its own Qt.  The bundle carries copies of
system libraries (GLib, GIO, D-Bus, OpenSSL ...) from the machine it was
built on.  Programs started from it inherit that environment and load the
bundle's older copies instead of their own: udisksctl and gio stop with a
symbol lookup error before doing anything - which is why USB drives could
not be mounted or ejected from the AppImage - and a Qt program such as the
file manager picks up the wrong Qt plugins.

``child_env`` is the environment as it was before the bundle changed it;
``restore_for_children`` makes that the process environment once OpenVat
no longer needs the bundle's values (the dynamic loader reads them when
the program starts, Qt when the QApplication is made).
"""

from __future__ import annotations

import os
import sys

QPA_MARK = "OPENVAT_SET_QPA"       # the AppImage's start script set QT_QPA_PLATFORM (it was not set)


def frozen() -> bool:
    """Running from a release build (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def child_env(env: dict | None = None) -> dict:
    """``env`` (default: this process's) without what the release build
    added; unchanged when OpenVat runs from source."""
    env = dict(os.environ if env is None else env)
    if not frozen():
        return env
    bundle = os.path.normpath(getattr(sys, "_MEIPASS", "") or "")
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBPATH"):
        original = env.pop(key + "_ORIG", None)
        if original is not None:
            env[key] = original
        elif key in env:
            kept = [p for p in env[key].split(os.pathsep) if p and not _inside(p, bundle)]
            if kept:
                env[key] = os.pathsep.join(kept)
            else:
                del env[key]
    for key in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH"):
        if key in env and bundle and all(_inside(p, bundle) for p in env[key].split(os.pathsep) if p):
            del env[key]
    if sys.platform == "win32" and "PATH" in env and bundle:     # the PySide6 hook put the bundle first
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if not _inside(p, bundle))
    if env.pop(QPA_MARK, None):
        env.pop("QT_QPA_PLATFORM", None)
    for key in [k for k in env if k.startswith("_PYI_")]:      # the bootloader's notes to itself (another
        del env[key]                                             # frozen program would take them as its own)
    return env


def restore_for_children() -> None:
    """Make ``child_env`` the process environment (call after the
    QApplication exists and Qt's plugin paths are pinned)."""
    if not frozen():
        return
    clean = child_env()
    for key in list(os.environ):
        if key not in clean:
            del os.environ[key]
    os.environ.update(clean)


def _inside(path: str, folder: str) -> bool:
    if not folder:
        return False
    path = os.path.normpath(path)
    return path == folder or path.startswith(folder + os.sep)
