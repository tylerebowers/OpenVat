"""Convert Anycubic PhotonWorkshop printer files into OpenVat presets.

PhotonWorkshop describes every printer in a ``.pwsp`` JSON file (and embeds
the same JSON as ``anycubic_photon_resins.pwsp`` inside each export).  It
holds the machine specs and the factory resin profiles for that printer.

    printer_from_pwsp(pwsp)  -> PrinterProfile
    resins_from_pwsp(pwsp)   -> [ResinProfile, ...]
    import_presets(path)     -> writes <profiles>/presets/<printer>.json: the printer
                                and its resins in one file (folder=RESOURCES/printers:
                                the shipped presets, for contributors)

The original ``machine_type`` block and firmware parameters are kept in
``PrinterProfile.photonworkshop`` so exports carry exactly what the printer
expects (file extension, timing constants, preview sizes, ...).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ..core.profiles import (PrinterProfile, ResinProfile, LayerSettings, PRESETS, printer_file_name,
                             profiles_dir, write_printer_file)

VECTOR_FORMAT = "pwszImg"            # the only layer format OpenVat can write so far

_ACRONYMS = {"Abs": "ABS", "Hd": "HD", "Dlp": "DLP", "Diy": "DIY", "Pro2": "Pro 2", "Uv": "UV"}


def load_pwsp(path: str | Path) -> dict:
    """Read a .pwsp file, or the .pwsp embedded in a .pwsz-style export."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("anycubic_photon_resins.pwsp"))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# printers

def printer_from_pwsp(pwsp: dict) -> PrinterProfile:
    mt = pwsp["machine_type"]
    ext = pwsp.get("machine_extern", {})
    px = float(mt["xy_pixel"])
    py = float(mt.get("xy_pixel_y") or px)            # older printers: square pixels
    layer_format = mt.get("key_image_format", VECTOR_FORMAT)
    suffix = str(mt.get("key_suffix", "pwsz")).split(",")[0].strip()
    keep = {k: ext[k] for k in ("firmware_calc_print_time", "firmware_calc_print_time_paras",
                                 "firmware_calc_exp_time_paras") if k in ext}
    return PrinterProfile(
        name=mt["name"],
        output_type="anycubic_pwsz" if layer_format == VECTOR_FORMAT else "anycubic_bitmap",
        res_x=int(mt["res_x"]), res_y=int(mt["res_y"]),
        print_x=round(float(mt["print_xsize"]), 4), print_y=round(float(mt["print_ysize"]), 4),
        print_z=round(float(mt["print_zsize"]), 4),
        pixel_x_um=round(px, 4), pixel_y_um=round(py, 4),
        file_extension=suffix, layer_format=layer_format,
        photonworkshop={"machine_type": mt, **keep},
    )


# --------------------------------------------------------------------------
# resins

def _pretty(raw: str) -> str:
    words = raw.replace("_", " ").title().split()
    return " ".join(_ACRONYMS.get(w, w) for w in words)


def resin_from_photonworkshop(entry: dict) -> ResinProfile:
    """Convert one resin entry of a .pwsp (factory or user resin)."""
    prop, sp, ext = entry["property"], entry["slicepara"], entry.get("slice_extpara", {})
    brand = prop.get("brand_name") or "Anycubic"
    film = prop.get("film_name", "")
    # fast_print -> "Fast", high_quality_print_orange_red -> "High quality orange red"
    setting = prop.get("setting_name", "").replace("_print", "").replace("_", " ").strip().capitalize()
    thick = f"{round(sp['zthick'], 3):g} mm"
    details = ", ".join(x for x in (film, f"{setting} {thick}".strip()) if x)
    name = f"{brand} {_pretty(prop.get('resin_name') or prop.get('name') or 'resin')} ({details})"

    normal = LayerSettings(
        thickness=round(sp["zthick"], 4), exposure=round(sp["exposure_time"], 3),
        off_time=round(sp["off_time"], 3),
        wait_before_lift=round(sp.get("wait_before_lift", 0.0), 3),
        wait_after_lift=round(sp.get("wait_after_lift", 0.0), 3),
        lift_distance=round(sp["zup_height"], 3), lift_speed=round(sp["zup_speed"], 3),
        retract_speed=round(sp["zdown_speed"], 3))
    bottom = LayerSettings(
        thickness=normal.thickness, exposure=round(sp["bott_time"], 3),
        off_time=round(sp.get("bott_off_time", sp["off_time"]), 3),
        wait_before_lift=round(sp.get("bott_wait_before_lift", 0.0), 3),
        wait_after_lift=round(sp.get("bott_wait_after_lift", 0.0), 3),
        lift_distance=normal.lift_distance, lift_speed=normal.lift_speed,
        retract_speed=normal.retract_speed)
    # PhotonWorkshop's two-stage lift (multi_state_paras) is not modelled: its
    # own layers_controller.conf uses zup_height / zup_speed for every layer.
    # transition_layercount only applies when transition_type != 0 (a sample
    # export with count 15 / type 0 jumps straight to the normal exposure).
    transition = int(ext.get("transition_layercount", 0)) if ext.get("transition_type", 0) else 0
    scale = ext.get("material_scale_xyz", {"x": 1.0, "y": 1.0, "z": 1.0})
    return ResinProfile(
        name=name, brand=brand, density=round(prop.get("density", 1.1), 3),
        price_per_liter=round(prop.get("price", 0.0) * 1000.0 / max(prop.get("volume", 1000.0), 1.0), 2),
        currency=prop.get("currency", "$"), normal=normal, bottom=bottom,
        bottom_layers=int(sp["bott_layers"]), transition_layers=transition,
        anti_aliasing=max(int(sp.get("anti_count", 1)), 1),
        shrink_x=round(scale["x"], 5), shrink_y=round(scale["y"], 5), shrink_z=round(scale["z"], 5))


def resins_from_pwsp(pwsp: dict, include_user: bool = False) -> list[ResinProfile]:
    ext = pwsp.get("machine_extern", {})
    entries = list(ext.get("factory_resins", []))
    if include_user:
        entries += ext.get("user_resins", [])
    out, seen = [], {}
    for e in entries:
        r = resin_from_photonworkshop(e)
        n = seen.get(r.name, 0) + 1                  # keep names (= file names) unique
        seen[r.name] = n
        if n > 1:
            r.name = f"{r.name} #{n}"
        out.append(r)
    return out


# --------------------------------------------------------------------------
# writing presets

def import_presets(path: str | Path, folder: Path | None = None, include_user: bool = False
                   ) -> tuple[PrinterProfile, list[ResinProfile]]:
    """Write the printer preset found in ``path``, with its resin presets,
    as one printer file into ``folder`` (default: the presets folder in the
    user's profiles folder, see profiles.profiles_dir)."""
    folder = Path(folder) if folder is not None else profiles_dir() / PRESETS
    pwsp = load_pwsp(path)
    printer = printer_from_pwsp(pwsp)
    resins = resins_from_pwsp(pwsp, include_user)
    write_printer_file(folder / printer_file_name(printer), printer, resins)
    return printer, resins
