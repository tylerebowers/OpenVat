"""OpenVat - an open resin (MSLA) slicer.

Package layout:
    core/     geometry, scene, slicing, profiles (no Qt imports)
    formats/  output file writers (Anycubic .pwsz)
    ui/       the Qt user interface
    cli.py    headless command-line slicer
"""

__version__ = "0.1.0"
APP_NAME = "OpenVat"
AUTHORS = [
    "Tyler Bowers",
    "Claude (Fable 5.1 & Opus 5.5)",
]
GITHUB_URL = "https://github.com/tylerebowers/OpenVat"
