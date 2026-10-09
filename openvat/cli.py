"""Headless command-line slicer.

    openvat-slice model.stl -o model.pwsz [--printer NAME] [--resin NAME]
    openvat-slice --list-profiles
    openvat-slice --inspect file.pwsz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.loaders import load_file
from .core.profiles import ProfileStore, RESOURCES, PRINTERS
from .core.scene import Scene
from .core.slicer import slice_scene


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openvat-slice", description="Slice 3D models to Anycubic print files")
    ap.add_argument("models", nargs="*", help="STL / OBJ / 3MF / STEP files")
    ap.add_argument("-o", "--output", help="output file (default: model name + the printer's extension)")
    ap.add_argument("--printer", help="printer profile name (default: last used in the GUI)")
    ap.add_argument("--resin", help="resin profile name (default: last used in the GUI)")
    ap.add_argument("--arrange", action="store_true", help="auto-arrange multiple models")
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--inspect", metavar="PWSZ", help="print a summary of an existing .pwsz")
    ap.add_argument("--import-presets", metavar="FILE", nargs="+",
                    help="add printer + resin presets from PhotonWorkshop .pwsp files (or exports "
                         "that contain one) to the presets folder of your profiles (~/.openvat/presets)")
    ap.add_argument("--shipped", action="store_true",
                    help="with --import-presets: write into the program's own presets "
                         "(openvat/resources/printers - for contributors) instead")
    args = ap.parse_args(argv)

    store = ProfileStore()
    if args.import_presets:
        return _import_presets(args.import_presets, RESOURCES / PRINTERS if args.shipped else None)
    if args.list_profiles:
        print("Printers (resin profiles indented):")
        for p in store.printers:
            print(f"  {p.name}  ({p.res_x}x{p.res_y}, {p.print_x}x{p.print_y}x{p.print_z} mm)")
            for r in store.resins_for(p):
                print(f"      {r.name}  ({r.normal.thickness} mm, {r.normal.exposure} s)")
        print("Resin presets:")
        for printer, presets in store.resin_presets.items():
            print(f"  {printer}: {len(presets)} presets")
        return 0

    if args.inspect:
        from .formats.pwsz import read_pwsz
        f = read_pwsz(args.inspect)
        print(f"{f.printer_name}: {f.res[0]}x{f.res[1]} px, {f.print_size} mm")
        print(f"layers: {len(f.layer_images)}, print time {f.print_info.get('print_time')} s, "
              f"volume {f.print_info.get('volume'):.3f} mL")
        for i in (0, len(f.layer_images) // 2, len(f.layer_images) - 1):
            img = f.layer_images[i]
            print(f"  layer {i}: area {img.area2 / 2:.3f} mm^2, {img.contours} contours, "
                  f"{len(img.segments)} segments, exposure {f.layer_paras[i]['exposure_time']} s")
        return 0

    if not args.models:
        ap.error("no model files given")

    printer = _pick(store.printers, args.printer, store.default_printer())
    if printer is None:
        sys.exit("No printer profiles yet - start the OpenVat GUI once to add your printer.")
    # a printer's own resins first, then the shipped presets for that printer
    resin = _pick(store.resins_for(printer) + store.resin_presets_for(printer), args.resin,
                  store.default_resin(printer))
    if resin is None:
        sys.exit(f"No resin profile for {printer.name}. Add one in the GUI or pass --resin.")
    if not printer.can_export:
        sys.exit(f"{printer.name} uses {printer.layer_format} layer files, which OpenVat cannot write yet.")
    out = Path(args.output or Path(args.models[0]).with_suffix(f".{printer.file_extension}"))

    scene = Scene()
    for m in args.models:
        for obj in load_file(m):
            scene.add(obj)
            print(f"loaded {obj.name}: {len(obj.mesh.faces)} faces, size {obj.size().round(2)} mm")
    if args.arrange or len(scene.objects) > 1:
        scene.arrange(printer)

    bad = [o.name for o, ok in zip(scene.objects, scene.fits(printer).values()) if not ok]
    if bad:
        print(f"warning: outside build volume: {', '.join(bad)}", file=sys.stderr)

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print(f"  slicing {done}/{total}", flush=True)

    result = slice_scene(scene, printer, resin, progress)
    print()
    from .formats.pwsz import write_pwsz
    meshes = [o.transformed() for o in scene.objects] + scene.support_meshes()
    write_pwsz(out, result, meshes)
    print(f"wrote {out}: {len(result.layers)} layers, {result.volume_mm3() / 1000:.2f} mL, "
          f"~{result.print_time_s() / 60:.0f} min")
    if result.islands:
        where = ", ".join(f"{i.first}-{i.last}" if i.last > i.first else str(i.first) for i in result.islands)
        print(f"warning: {len(result.islands)} unsupported island(s) at layers {where}", file=sys.stderr)
    return 0


def _import_presets(paths: list[str], folder=None) -> int:
    from .formats.photonworkshop import import_presets
    for path in paths:
        printer, resins = import_presets(path, folder)
        note = "" if printer.can_export else f"  (layer format {printer.layer_format}: export not supported)"
        print(f"{printer.name}: printer preset + {len(resins)} resin presets{note}")
    return 0


def _pick(items, name, default):
    if name is None:
        return default
    for it in items:
        if it.name.lower() == name.lower():
            return it
    sys.exit(f"unknown profile '{name}'. Use --list-profiles to see the options.")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
