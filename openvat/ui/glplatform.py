"""The OpenGL setup that has to happen before PyOpenGL is imported: the
surface format the views ask for, and making PyOpenGL use the same window
system interface (GLX or EGL) as Qt.

PyOpenGL picks GLX or EGL once, when ``OpenGL.GL`` is first imported, from
the desktop *session* (XDG_SESSION_TYPE, WAYLAND_DISPLAY) - not from what Qt
actually uses.  On a Wayland desktop where Qt runs through XWayland
(QT_QPA_PLATFORM=xcb, as the AppImage does) it would load EGL while Qt's
contexts are GLX, and the first real GL call crashes the program with a
segmentation fault.  So once QApplication exists, ``match_pyopengl`` makes a
throwaway context, asks which interface made it current and sets
PYOPENGL_PLATFORM accordingly.

Nothing in this module imports PyOpenGL; import it before ``viewport3d``.
"""

from __future__ import annotations

import ctypes
import os
import sys

from PySide6.QtGui import QSurfaceFormat

# (PYOPENGL_PLATFORM value, libraries that may hold it, "current context?" function)
_INTERFACES = (("egl", ("libEGL.so.1",), "eglGetCurrentContext"),
               ("glx", ("libGLX.so.0", "libGL.so.1"), "glXGetCurrentContext"))

# Qt platform plugin -> the interface it draws with (when probing fails)
_QT_PLATFORMS = {"xcb": "glx", "wayland": "egl", "wayland-egl": "egl", "eglfs": "egl",
                 "minimalegl": "egl"}


def default_surface_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    return fmt


def current_interface() -> str | None:
    """'egl' or 'glx': which interface holds the OpenGL context that is
    current on this thread (None when neither does or neither is installed)."""
    for name, libraries, function in _INTERFACES:
        for library in libraries:
            try:
                get = getattr(ctypes.CDLL(library), function)
            except (OSError, AttributeError):
                continue
            get.restype = ctypes.c_void_p
            get.argtypes = []
            if get():
                return name
            break
    return None


def _probe_interface() -> str | None:
    """Make a context current on an offscreen surface and see which
    interface Qt used for it."""
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
    ctx = QOpenGLContext()
    ctx.setFormat(QSurfaceFormat.defaultFormat())
    if not ctx.create():
        return None
    surface = QOffscreenSurface()
    surface.setFormat(ctx.format())
    surface.create()
    try:
        if not (surface.isValid() and ctx.makeCurrent(surface)):
            return None
        try:
            return current_interface()
        finally:
            ctx.doneCurrent()
    finally:
        surface.destroy()


def _guess_interface() -> str | None:
    """From Qt's platform plugin, for when probing finds nothing."""
    from PySide6.QtGui import QGuiApplication
    name = QGuiApplication.platformName().lower()
    if name == "xcb" and os.environ.get("QT_XCB_GL_INTEGRATION", "").lower() == "xcb_egl":
        return "egl"
    return _QT_PLATFORMS.get(name)


def match_pyopengl() -> str | None:
    """Make PyOpenGL use the interface Qt draws with.  Call it once the
    QApplication exists and before anything imports ``OpenGL.GL``.  Returns
    the interface (None where there is no choice: Windows, macOS).  A
    PYOPENGL_PLATFORM the user set is left alone.  Never fails."""
    if not sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd", "dragonfly")):
        return None                        # WGL and CGL: PyOpenGL has only one option there
    if os.environ.get("PYOPENGL_PLATFORM"):
        return os.environ["PYOPENGL_PLATFORM"]
    try:
        interface = _probe_interface() or _guess_interface()
    except Exception as exc:                                     # pragma: no cover
        print(f"OpenVat: could not tell whether Qt uses GLX or EGL ({exc})", file=sys.stderr)
        return None
    if not interface:
        return None
    loaded = sys.modules.get("OpenGL.platform")
    if loaded is not None:                 # too late to choose: say so if it is the wrong one
        if not type(loaded.PLATFORM).__name__.lower().startswith(interface):
            print(f"OpenVat: PyOpenGL was loaded with {type(loaded.PLATFORM).__name__} but Qt uses "
                  f"{interface.upper()} - the 3D views may crash; set PYOPENGL_PLATFORM={interface}",
                  file=sys.stderr)
        return interface
    os.environ["PYOPENGL_PLATFORM"] = interface
    return interface
