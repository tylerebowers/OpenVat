"""Pick the print file writer for a printer.

    .pwsz family (pwszImg printers)        -> pwsz.write_pwsz      (.pwsz, .pm7, .pm7m, .pm4u, .pp1, .pp1m)
        vector layers, or pw0Img bitmaps when the resin anti-aliases
    pw0Img, file version 1 / 516 / 517     -> pwbinary.write_pwbinary
        (.pwx, .pw0 / .pm3m, .pm3, .pwmx, .pwma / .dl2p, .pwmb, .pm3r, .pm3n, .pm4n, .pm5, ...)

Binary files of versions 515 and 518 (.pwmo, .pwms, .pmsq, .dlp, .pm5s,
.m5sp) and pwsImg (.pws: Photon, Photon S) cannot be written yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import trimesh

from ..core.profiles import PrinterProfile
from ..core.slicer import SliceResult
from . import pwbinary
from .pwsz import write_pwsz

ProgressFn = Callable[[int, int], None]


def why_not(printer: PrinterProfile) -> str:
    """Why OpenVat cannot write files for ``printer`` ("" if it can)."""
    if printer.can_export:
        return ""
    versions = ", ".join(map(str, pwbinary.VERSIONS))
    if printer.output_type == "anycubic_bitmap" and printer.layer_format == pwbinary.LAYER_FORMAT:
        return (f"{printer.name} reads Photon Workshop file version {printer.file_version} "
                f"(.{printer.file_extension}), which OpenVat cannot write yet - only versions {versions}.")
    return f"{printer.name} uses {printer.layer_format} layer files, which OpenVat cannot write yet."


def write_print_file(path: str | Path, result: SliceResult, preview_meshes: list[trimesh.Trimesh],
                     progress: ProgressFn | None = None) -> Path:
    printer = result.printer
    if printer.output_type == "anycubic_pwsz":
        return write_pwsz(path, result, preview_meshes, progress)
    if pwbinary.supports(printer):
        return pwbinary.write_pwbinary(path, result, preview_meshes, progress)
    raise ValueError(why_not(printer))
