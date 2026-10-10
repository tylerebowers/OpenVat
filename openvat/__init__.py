"""OpenVat - an open resin (MSLA) slicer.

Package layout:
    core/     geometry, scene, slicing, profiles (no Qt imports)
    formats/  output file writers (Anycubic .pwsz family, Photon Workshop binary files)
    ui/       the Qt user interface
    cli.py    headless command-line slicer
"""

__version__ = "0.1.1"
APP_NAME = "OpenVat"
AUTHORS = [
    "Tyler Bowers",
    "Anthropic Fable 5.1",
]
GITHUB_URL = "https://github.com/tylerebowers/OpenVat"
