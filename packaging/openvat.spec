# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec shared by all three platforms.

    pyinstaller --distpath build/<platform>/dist --workpath build/<platform>/work packaging/openvat.spec

Produces <distpath>/OpenVat/ (one-folder build) and, on macOS,
<distpath>/OpenVat.app.  The build scripts (packaging/build_*) wrap it into
the AppImage, the setup.exe or the DMG in dist/.
"""

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent          # repository root (SPECPATH = packaging/)
VERSION = re.search(r'^__version__ = "(.+)"', (ROOT / "openvat" / "__init__.py").read_text(), re.M)[1]

# Ship the presets and the logo only (users' profiles live in ~/.openvat).
RES = ROOT / "openvat" / "resources"
datas = [(str(RES / "printers"), "openvat/resources/printers"),        # presets: a printer + its resins per file
         (str(RES / "logo.png"), "openvat/resources")]
datas += collect_data_files("trimesh")
datas += collect_data_files("shapely")

hiddenimports = (
    collect_submodules("trimesh")
    + collect_submodules("shapely")
    + collect_submodules("OpenGL")
    # SciPy: k-d trees (support paint, pillar grouping) and sparse graphs (painted
    # patches); imported inside functions, so listed here to be sure
    + collect_submodules("scipy.spatial")
    + collect_submodules("scipy.sparse")
    + ["OpenGL.platform.glx", "OpenGL.platform.egl", "OpenGL.platform.win32", "OpenGL.platform.darwin",
       "OpenGL.arrays.numpymodule", "OpenGL.arrays.ctypesarrays", "OpenGL.arrays.lists",
       "OpenGL.arrays.strings", "OpenGL.arrays.nones", "OpenGL.arrays.numbers",
       "PIL.PngImagePlugin", "networkx"]
)

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "PySide6.QtWebEngineCore",
              "PySide6.QtWebEngineWidgets", "PySide6.QtMultimedia", "PySide6.Qt3DCore"],
    noarchive=False,
)
if sys.platform.startswith("linux"):
    # Libraries that must be the desktop's own, never the build machine's (the
    # AppImage project's exclude list, github.com/AppImageCommunity/pkg2appimage,
    # file "excludelist").  Every Linux desktop has them, at least as new as on
    # the oldest distribution we build on:
    # - fonts: an old bundled fontconfig reads the desktop's newer font
    #   configuration and font files - text came out in the wrong font or not
    #   at all; FreeType and HarfBuzz must match that fontconfig;
    # - the C++ runtime: Mesa's GPU drivers (Intel iris, AMD radeonsi, ...) need
    #   the one they were built against, and an older bundled copy makes them
    #   fail to load - OpenGL then silently falls back to software rendering;
    # - X11 / OpenGL / Wayland and the rest of the base system.
    # Renamed private copies that wheels ship (libfreetype-1a2b3c4d.so.6, ...)
    # are only loaded by their own module and stay.
    HOST_LIBS = {
        "libstdc++.so.6", "libgcc_s.so.1",
        "libfontconfig.so.1", "libfreetype.so.6", "libharfbuzz.so.0", "libfribidi.so.0", "libexpat.so.1",
        "libGL.so.1", "libEGL.so.1", "libGLdispatch.so.0", "libGLX.so.0", "libOpenGL.so.0", "libglapi.so.0",
        "libdrm.so.2", "libgbm.so.1", "libwayland-client.so.0", "libxcb.so.1", "libxcb-dri2.so.0",
        "libxcb-dri3.so.0", "libX11.so.6", "libX11-xcb.so.1", "libICE.so.6", "libSM.so.6",
        "libz.so.1", "libuuid.so.1", "libcom_err.so.2", "libgpg-error.so.0", "libgmp.so.10",
        "libasound.so.2", "libjack.so.0", "libpipewire-0.3.so.0", "libusb-1.0.so.0",
    }
    a.binaries = [b for b in a.binaries if Path(b[0]).name not in HOST_LIBS]

pyz = PYZ(a.pure)

icon = None
if sys.platform == "win32" and (ROOT / "packaging" / "openvat.ico").exists():
    icon = str(ROOT / "packaging" / "openvat.ico")
elif sys.platform == "darwin" and (ROOT / "packaging" / "openvat.icns").exists():
    icon = str(ROOT / "packaging" / "openvat.icns")

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="OpenVat",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="OpenVat")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OpenVat.app",
        icon=icon,
        bundle_identifier="org.openvat.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "CFBundleDocumentTypes": [{
                "CFBundleTypeExtensions": ["stl", "obj", "3mf", "ply", "step", "stp"],
                "CFBundleTypeRole": "Viewer",
            }],
        },
    )
