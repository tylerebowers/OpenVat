# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec shared by all three platforms.

    pyinstaller packaging/openvat.spec

Produces dist/OpenVat/ (one-folder build) and, on macOS, dist/OpenVat.app.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent          # repository root (SPECPATH = packaging/)

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
    excludes=["tkinter", "matplotlib", "scipy", "pytest", "IPython", "PySide6.QtWebEngineCore",
              "PySide6.QtWebEngineWidgets", "PySide6.QtMultimedia", "PySide6.Qt3DCore"],
    noarchive=False,
)
if sys.platform.startswith("linux"):
    # Use the system's C++ runtime, never the build machine's: Mesa's GPU drivers
    # (Intel iris, AMD radeonsi, ...) need the one they were built against, and an
    # older bundled copy makes them fail to load - OpenGL then silently falls back
    # to software rendering.  Every desktop has these, at least as new as the
    # oldest distribution we build on.
    a.binaries = [b for b in a.binaries
                  if not Path(b[0]).name.startswith(("libstdc++.so", "libgcc_s.so"))]

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
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeExtensions": ["stl", "obj", "3mf", "ply", "step", "stp"],
                "CFBundleTypeRole": "Viewer",
            }],
        },
    )
