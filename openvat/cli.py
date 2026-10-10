"""Headless command-line slicer.

    openvat-slice model.stl -o model.pwsz [--printer NAME] [--resin NAME]
    openvat-slice --list-profiles
    openvat-slice --inspect file.pwsz   (or .pp1, .pwx, .pm3m, .dl2p, ...)
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
    ap.add_argument("--inspect", metavar="FILE", help="print a summary of an existing print file "
                                                          "(.pwsz, .pp1, .pm7, ..., .pwx, .pm3m, .dl2p, ...)")
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
        from .formats.pwbinary import is_pwbinary
        if is_pwbinary(args.inspect):
            return _inspect_binary(args.inspect)
        from .formats.pwsz import read_pwsz
        f = read_pwsz(args.inspect)
        print(f"{f.printer_name}: {f.res[0]}x{f.res[1]} px, {f.print_size} mm")
        count = len(f.layer_paras)
        print(f"layers: {count} ({'pw0Img bitmaps' if f.bitmap else 'pwszImg vectors'}), "
              f"print time {f.print_info.get('print_time')} s, volume {f.print_info.get('volume'):.3f} mL")
        for i in (0, count // 2, count - 1) if count else ():
            if f.bitmap:
                img = f.layer_bitmap(i)
                lit = img > 0
                print(f"  layer {i}: {int(lit.sum())} lit pixels ({len(set(img[lit].tolist()))} grey levels), "
                      f"exposure {f.layer_paras[i]['exposure_time']} s")
            else:
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
        from .formats.export import why_not
        sys.exit(why_not(printer))
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
    from .formats.export import write_print_file
    meshes = [o.transformed() for o in scene.objects] + scene.support_meshes()
    write_print_file(out, result, meshes)
    print(f"wrote {out}: {len(result.layers)} layers, {result.volume_mm3() / 1000:.2f} mL, "
          f"~{result.print_time_s() / 60:.0f} min")
    if result.islands:
        where = ", ".join(f"{i.first}-{i.last}" if i.last > i.first else str(i.first) for i in result.islands)
        print(f"warning: {len(result.islands)} unsupported island(s) at layers {where}", file=sys.stderr)
    return 0


def _inspect_binary(path: str) -> int:
    from .formats.pwbinary import read_pwbinary
    f = read_pwbinary(path)
    h = f.header
    print(f"Photon Workshop file version {f.version}{': ' + f.machine_name if f.machine_name else ''}, "
          f"{h['res_x']}x{h['res_y']} px, {h['pixel_um']:g} um pixels")
    print(f"layers: {len(f.layers)} x {h['layer_height']:g} mm, exposure {h['exposure']:g} s "
          f"({h['bottom_layers']:g} bottom layers {h['bottom_exposure']:g} s), lift {h['lift_height']:g} mm "
          f"at {h['lift_speed']:g} mm/s, anti-aliasing {h['anti_aliasing']}")
    print(f"print time {h['print_time']} s, volume {h['volume_ml']:.3f} mL, "
          f"{h['currency']}{h['price']:.2f}")
    for i in (0, len(f.layers) // 2, len(f.layers) - 1) if f.layers else ():
        d = f.layers[i]
        print(f"  layer {i}: {d.lit_pixels} lit pixels, {d.length} bytes, exposure {d.exposure:g} s, "
              f"{d.thickness:g} mm")
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
