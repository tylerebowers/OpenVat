"""Anycubic .pwsz writer (and a reader for inspection).

A .pwsz file is a plain ZIP archive.  The layer "images" are *vector*
outlines (pwszImg) the printer rasterizes itself - or, when the resin asks
for anti-aliasing, bitmaps (pw0Img, as in the binary files, see pwbinary.py)
with grey edges: vectors cannot carry those.  Photon Workshop does the same
(its P1 resins have 16 anti-aliasing levels, so its .pp1 files hold
bitmaps), and turns the bitmaps half a turn for printers whose data say
rotate_z = 180.  The complete reverse-engineered layout is documented in
docs/FORMAT_PWSZ.md; in short:

    anycubic_photon_resins.pwsp   JSON  printer + resin profiles
    layers_controller.conf        JSON  per-layer exposure / thickness / lift
    lcd_function.json             JSON  model list, RERF (exposure test) settings
    print_info.json               JSON  time / volume / cost estimates
    software_info.conf            JSON  which slicer wrote the file
    scene.slice                   binary per-layer summary (z, area, bbox)
    calc_layer_volumes.data       binary volume per slab of ~0.2-0.25 mm
    layer_images/layer_N.pwszImg  binary outline segments of layer N
                  (or layer_N.pw0Img: the run-length encoded bitmap)
    preview_images/preview_N.png  thumbnails (224x168, 336x252, 800x600)

Coordinates in layer files are millimeters relative to the center of the
build plate, X along the long axis of the screen.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from .. import __version__
from ..core.profiles import PrinterProfile, ResinProfile
from ..core.slicer import Layer, SliceResult
from .preview import render_preview
from .pwbinary import encode_pw0, encode_pw0_band, rasterize_band, turned, decode_pw0

ProgressFn = Callable[[int, int], None]

# --------------------------------------------------------------------------
# layer image encoding


def layer_edges(geometry: MultiPolygon) -> tuple[np.ndarray, np.ndarray, int]:
    """Turn a layer's polygons into the edge table a .pwszImg stores.

    Returns (edges (N, 4) float32 x0 y0 x1 y1, flags (N,) uint8, contours).

    The printer fills layers with a scanline algorithm, so (verified against
    480 PhotonWorkshop layers):
      * every edge is stored bottom-to-top (y0 <= y1),
      * flag 1 = the outline originally ran that way (outer rings counter-
        clockwise, holes clockwise), flag 0 = it was flipped for storage,
      * horizontal edges keep their original direction and flag 1,
      * edges are sorted by y0 (stable, so ties keep outline order).
    """
    parts, contours = [], 0
    for poly in geometry.geoms:
        poly = orient(poly, 1.0)         # exterior CCW, interiors CW
        for ring in (poly.exterior, *poly.interiors):
            c = np.asarray(ring.coords, dtype=np.float32)
            if len(c) < 4:
                continue
            contours += 1
            parts.append(np.hstack([c[:-1], c[1:]]))
    if not parts:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.uint8), 0
    e = np.vstack(parts)
    e = e[(e[:, 0] != e[:, 2]) | (e[:, 1] != e[:, 3])]          # zero-length after float32
    down = e[:, 3] < e[:, 1]
    e[down] = e[down][:, [2, 3, 0, 1]]                           # store bottom-to-top
    flags = np.where(down, 0, 1).astype(np.uint8)
    order = np.argsort(e[:, 1], kind="stable")
    return e[order], flags[order], contours


def encode_layer_image(geometry: MultiPolygon) -> bytes:
    """Encode one layer's polygons as a .pwszImg blob.

    Layout (little endian):
        "{==\0"  f32 2*area  f32 xmin ymin xmax ymax  i32 0  i32 contour_count
        "[--\0"  i32 0  i32 edge_count  i32 1
            edge_count x ( f32 x0 y0 x1 y1 , u8 flag )      see layer_edges()
        "--]\0"  "==}\0"
    """
    edges, flags, contours = layer_edges(geometry)
    if len(edges):
        xmin, ymin = edges[:, [0, 2]].min(), edges[:, 1].min()
        xmax, ymax = edges[:, [0, 2]].max(), edges[:, 3].max()
        # 2*area as the printer computes it: signed shoelace over the stored edges
        x0, y0, x1, y1 = edges.astype(np.float64).T
        area2 = float(np.sum(np.where(flags == 1, 1.0, -1.0) * (x0 * y1 - x1 * y0)))
    else:
        xmin = ymin = xmax = ymax = 0.0
        area2 = 0.0

    out = io.BytesIO()
    out.write(b"{==\0")
    out.write(struct.pack("<f4fii", area2, xmin, ymin, xmax, ymax, 0, contours))
    out.write(b"[--\0")
    out.write(struct.pack("<iii", 0, len(edges), 1))
    rec = np.zeros(len(edges), dtype=np.dtype([("e", "<f4", 4), ("f", "u1")]))
    rec["e"], rec["f"] = edges, flags
    out.write(rec.tobytes())
    out.write(b"--]\0")
    out.write(b"==}\0")
    return out.getvalue()


def rasterize_layer_image(img: "LayerImage", pixel_x: float, pixel_y: float,
                          res_x: int, res_y: int) -> np.ndarray:
    """Fill a decoded layer the way a scanline printer would: only edges with
    y0 <= y < y1 are active on a row, each adds +1 (flag 1) or -1 (flag 0)
    to the winding number, and pixels with non-zero winding are exposed.
    Returns a bool (res_y, res_x) image, row 0 at the bottom.  Used to check
    files: anything not stored this way fills wrongly or not at all."""
    out = np.zeros((res_y, res_x), dtype=bool)
    if len(img.segments) == 0:
        return out
    x0, y0, x1, y1 = img.segments.astype(np.float64).T
    w = np.where(img.flags == 1, 1, -1)
    ys = (np.arange(res_y) - res_y / 2 + 0.5) * pixel_y          # pixel-center rows, mm
    xs = (np.arange(res_x) - res_x / 2 + 0.5) * pixel_x
    lo, hi = np.searchsorted(ys, [y0.min(), y1.max()])
    for r in range(max(lo, 0), min(hi + 1, res_y)):
        y = ys[r]
        act = (y0 <= y) & (y < y1)
        if not act.any():
            continue
        t = (y - y0[act]) / (y1[act] - y0[act])
        xc = x0[act] + t * (x1[act] - x0[act])
        order = np.argsort(xc)
        xc, wc = xc[order], np.cumsum(w[act][order])
        for k in range(len(xc) - 1):
            if wc[k] != 0:
                a, b = np.searchsorted(xs, [xc[k], xc[k + 1]])
                out[r, a:b] = True
    return out


@dataclass
class LayerImage:
    area2: float
    bbox: tuple[float, float, float, float]
    contours: int
    segments: np.ndarray      # (N, 4) float32 x0 y0 x1 y1
    flags: np.ndarray         # (N,) uint8


def decode_layer_image(data: bytes) -> LayerImage:
    assert data[:4] == b"{==\0", "not a pwszImg blob"
    area2, xmin, ymin, xmax, ymax, _zero, contours = struct.unpack_from("<f4fii", data, 4)
    p = 32
    segs, flags = [], []
    while data[p:p + 4] == b"[--\0":
        _a, n, _c = struct.unpack_from("<iii", data, p + 4)
        p += 16
        for _ in range(n):
            segs.append(struct.unpack_from("<4f", data, p))
            flags.append(data[p + 16])
            p += 17
        assert data[p:p + 4] == b"--]\0"
        p += 4
    assert data[p:p + 4] == b"==}\0"
    return LayerImage(area2, (xmin, ymin, xmax, ymax), contours,
                      np.asarray(segs, dtype=np.float32).reshape(-1, 4),
                      np.asarray(flags, dtype=np.uint8))


# --------------------------------------------------------------------------
# binary side files


def encode_scene_slice(layers: list[Layer], model_bounds: np.ndarray) -> bytes:
    """scene.slice: fixed header + one 64-byte record per layer."""
    out = io.BytesIO()
    out.write(b"ANYCUBIC-PWSZ".ljust(16, b"\0"))
    out.write(b"AnycubicPhotonWorkshop Exports".ljust(64, b"\0"))
    out.write(struct.pack("<4i", 3, 3, 0, 0))
    out.write(struct.pack("<fi", 1.0, len(layers)))
    lo, hi = model_bounds
    out.write(struct.pack("<6f", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]))
    out.write(b"\0" * (0x184 - out.tell()))
    out.write(b"<---")
    out.write(struct.pack("<i", len(layers)))
    for i, layer in enumerate(layers):
        b = layer.geometry.bounds if not layer.geometry.is_empty else (0, 0, 0, 0)
        n_contours = sum(1 + len(p.interiors) for p in layer.geometry.geoms)
        # third float: PhotonWorkshop only fills it on the very last layer
        last_area = layer.area if i == len(layers) - 1 else 0.0
        record = struct.pack("<2f4f2if", layer.z_center, layer.area, *b, n_contours, 0, last_area)
        out.write(record.ljust(64, b"\0"))
    out.write(b"--->")
    return out.getvalue()


def encode_layer_volumes(layers: list[Layer]) -> bytes:
    """calc_layer_volumes.data: resin volume (mm^3) per slab of height SLAB.
    PhotonWorkshop mixes 0.2 and 0.25 mm slabs; a uniform 0.2 mm works."""
    SLAB = 0.2
    slabs = []
    z = 0.0
    top = layers[-1].z_top if layers else 0.0
    thickness = layers[0].thickness if layers else 0.05
    while z < top - 1e-6:
        h = min(SLAB, top - z)
        # PhotonWorkshop stores int(z / thickness) computed in float32
        # PhotonWorkshop: float32 inputs, double division, truncation (0.2/0.05 -> 3)
        first = min(int(float(np.float32(z)) / float(np.float32(thickness))), len(layers) - 1)
        vol = sum(l.area * max(0.0, min(l.z_top, z + h) - max(l.z_bottom, z))
                  for l in layers if l.z_top > z and l.z_bottom < z + h)
        slabs.append((first, z, SLAB, vol))
        z += SLAB
    out = io.BytesIO()
    out.write(b"$\x10% ")
    out.write(struct.pack("<fii", SLAB, 1, len(slabs)))
    for first, z, h, vol in slabs:
        out.write(struct.pack("<iff", first, z, h))
        out.write(b"\0" * 32)
        out.write(struct.pack("<f", vol))
    return out.getvalue()


# --------------------------------------------------------------------------
# JSON side files

# Firmware timing constants copied from a PhotonWorkshop export for the
# Photon Mono M7 Pro.  They only affect the printer's own time estimate.
_FIRMWARE_TIME_PARAS = {
    "version": "2",
    "MACHINE_AXIS_STEPS_PER_UNIT": [100.0, 100.0, 3200.0, 94.0],
    "MACHINE_BLOCK_BUFFER_SIZE": 32,
    "MACHINE_DEFAULT_ACCELERATION": 1000.0,
    "MACHINE_DEFAULT_MINSEGMENTTIME": 20000,
    "MACHINE_DEFAULT_XYJERK": 20.0,
    "MACHINE_DEFAULT_ZJERK": 0.2,
    "MACHINE_GENERATE_FRAME_TIME": 450.0,
    "MACHINE_MAX_ACCELERATION": [1000, 1000, 160, 1000],
    "MACHINE_MAX_FEEDRATE": [200.0, 200.0, 20.0, 45.0],
    "MACHINE_MAX_STEP_FREQUENCY": 256000,
    "MACHINE_MINIMUM_PLANNER_SPEED": 0.05,
    "MACHINE_NOR_LAYER_DOWN_HEIGHT_DIV": 0.25,
    "MACHINE_NOR_LAYER_DOWN_SPEED_DIV": 0.5,
    "MACHINE_NOR_LAYER_UP_HEIGHT_DIV": 0.25,
    "MACHINE_NOR_LAYER_UP_SPEED_DIV": 0.5,
    "MACHINE_STEP_MUL": 1,
    "MACHINE_TIME_COMPENSATE": 0.0,
    "MACHINE_TIM_PRES": 30.0,
    "MACHINE_TIM_RCC_CLK": 60.0,
    "FUNCTION": 1,
    "MACHINE_MODE_ACCELERATION": [0, 0, 0, 0],
    "LAYER_COMPENSATE": [0, 0, 0, 0],
    "HEIGHT_COMPENSATE": [0, 0, 0, 0],
    "TIMES_COMPENSATE": [0, 0, 0, 0],
}
_FIRMWARE_EXP_PARAS = {
    "precision_range_branch": [0.0, 5.0, 25.0],
    "precision_per_volume": 5.0,
    "precision_coeff_value": [0.024, 0.01, -0.2],
    "energy_coeff": 0.0,
    "machine_exposure_ton": 0.4,
}
_TEMPERATURE_COEFFS = [
    {"temperature": 25.0, "x_coefficient": 197.78, "y_compensation": 1803.2},
    {"temperature": 30.0, "x_coefficient": 197.78, "y_compensation": 1803.2},
    {"temperature": 35.0, "x_coefficient": 197.78, "y_compensation": 1803.2},
    {"temperature": 40.0, "x_coefficient": 197.78, "y_compensation": 1803.2},
    {"temperature": 45.0, "x_coefficient": 197.78, "y_compensation": 1803.2},
]


def machine_type_json(printer: PrinterProfile) -> dict:
    """PhotonWorkshop's machine block.  Printers made from presets carry the
    original block; the editable specs from the profile are written over it."""
    original = printer.photonworkshop.get("machine_type")
    if original:
        mt = dict(original)
        mt.update({
            "name": printer.name, "key_suffix": printer.file_extension,
            "key_image_format": printer.layer_format,
            "res_x": printer.res_x, "res_y": printer.res_y,
            "xy_pixel": printer.pixel_x_um, "xy_pixel_y": printer.pixel_y_um,
            "print_xsize": printer.print_x, "print_ysize": printer.print_y,
            "print_zsize": printer.print_z,
        })
        mt.setdefault("rotate_z", 0.0)                 # newer PhotonWorkshop versions add these
        mt.setdefault("print_platform_zheights", [0.0])
        if "child_screen" in mt:
            mt["child_screen"] = [{"x": 0, "y": 0, "width": printer.res_x, "height": printer.res_y}]
        return mt
    return {
        "version": "3",
        "name": printer.name,
        "key_suffix": printer.file_extension,
        "key_image_format": printer.layer_format,
        "res_x": printer.res_x,
        "res_y": printer.res_y,
        "xy_pixel": printer.pixel_x_um,
        "xy_pixel_y": printer.pixel_y_um,
        "rotate_z": 0.0,
        "max_samples": 16,
        "property": 119,
        "print_xsize": printer.print_x,
        "print_ysize": printer.print_y,
        "print_zsize": printer.print_z,
        "max_file_version": 518,
        "prev_back_color": [0.0078125, 0.28125, 0.390625],
        "prev_model_color": [0.8046875, 0.8046875, 0.8046875],
        "prev_supports_color": [0.07421875, 0.92578125, 0.9296875],
        "prev_image_size": [224, 168],
        "child_screen": [{"x": 0, "y": 0, "width": printer.res_x, "height": printer.res_y}],
        "prev2_back_color": [0.07843, 0.10588, 0.16078],
        "prev2_image_size": [336, 252],
        "raster_segments_capacity": 100000,
        "raster_antialiasing": 4,
        "cloudprev_back_color": [0.93333, 0.94118, 0.96471],
        "cloudprev_imag_size": [800, 600],
        "print_platform_zheights": [0.0],
        "print_exposure_calibration_func": 3,
    }


def resin_json(resin: ResinProfile) -> dict:
    n, b = resin.normal, resin.bottom
    full_name = f"{resin.brand}@{resin.name}@ACF@OpenVat"
    return {
        "version": "2",
        "property": {
            "version": "4",
            "code": "10",
            "currency": resin.currency,
            "price": resin.price_per_liter,
            "type": "Standard",
            "volume": 1000.0,
            "subfunc_code": 0,
            "density": resin.density,
            "target_temperature": 25.0,
            "brand_name": resin.brand,
            "resin_name": resin.name,
            "film_name": "ACF",
            "setting_name": "OpenVat",
            "name": full_name,
        },
        "depth_penetration_curve": {
            "zthick_min": 0.01, "zthick_max": 0.2, "light_intensity": 9000.0,
            "safety_coefficient": 1.6, "current_tempcurve_selector": 0,
            "temperature_coefficients": _TEMPERATURE_COEFFS,
        },
        "slice_extpara": {
            "version": "3",
            "multi_state_used": 0,
            "transition_layercount": resin.transition_layers,
            "transition_type": 0,
            "multi_state_paras": {
                "bott_0": {"down_speed": b.retract_speed, "height": b.lift_distance, "up_speed": b.lift_speed},
                "bott_1": {"down_speed": b.retract_speed, "height": b.lift_distance, "up_speed": b.lift_speed},
                "normal_0": {"down_speed": n.retract_speed, "height": n.lift_distance, "up_speed": n.lift_speed},
                "normal_1": {"down_speed": n.retract_speed, "height": n.lift_distance, "up_speed": n.lift_speed},
            },
            "exposure_compensate": 0.0,
            "intelli_mode": 1,
            "max_acceleration": 20.0,
            "separate_support_exposure_delayed": 0.0,
            "material_scale_coeffs": [],
            "material_scale_coeffs_mode": 0,
            "material_scale_xyz": {"x": resin.shrink_x, "y": resin.shrink_y, "z": resin.shrink_z},
        },
        "slicepara": {
            "anti_count": max(int(resin.anti_aliasing), 1),
            "blur_level": 0,
            "bott_layers": resin.bottom_layers,
            "bott_time": b.exposure,
            "bott_time_dual": b.exposure,
            "exposure_time": n.exposure,
            "gray_level": 0,
            "off_time": n.off_time,
            "use_indivi_layerpara": 0,
            "use_random_erode": 0,
            "zdown_speed": n.retract_speed,
            "zthick": n.thickness,
            "zup_height": n.lift_distance,
            "zup_speed": n.lift_speed,
            "wait_before_lift": n.wait_before_lift,
            "wait_after_lift": n.wait_after_lift,
            "bott_off_time": b.off_time,
            "bott_wait_before_lift": b.wait_before_lift,
            "bott_wait_after_lift": b.wait_after_lift,
        },
    }


def pwsp_json(printer: PrinterProfile, resin: ResinProfile) -> dict:
    r = resin_json(resin)
    return {
        "version": "3",
        "machine_type": machine_type_json(printer),
        "machine_extern": {
            "version": "3",
            "alias": printer.name,
            "picture": "",
            "cloud_property": 0,
            "device_cn_code": "",
            "factory_resins": [],
            "user_resins": [r],
            "active_resins": [r["property"]["name"]],
            "firmware_calc_print_time": printer.photonworkshop.get("firmware_calc_print_time", 1),
            "firmware_calc_print_time_paras": printer.photonworkshop.get(
                "firmware_calc_print_time_paras", _FIRMWARE_TIME_PARAS),
            "firmware_calc_exp_time_paras": printer.photonworkshop.get(
                "firmware_calc_exp_time_paras", _FIRMWARE_EXP_PARAS),
        },
    }


def layers_controller_json(layers: list[Layer]) -> dict:
    return {
        "count": len(layers),
        "paras": [
            {
                "exposure_time": l.exposure,
                "layer_index": l.index,
                "layer_minheight": l.z_bottom,
                "layer_thickness": l.thickness,
                "zup_height": l.lift_distance,
                "zup_speed": l.lift_speed,
            }
            for l in layers
        ],
    }


def lcd_function_json(object_names: list[str]) -> dict:
    return {
        "models_processed_info": {
            "models": [
                {"auto_support": 0, "hollow": 0, "makeronline_sourceid": -1,
                 "manual_support": 0, "name": n, "punch": 0, "source_from": 0}
                for n in object_names
            ],
            "scene_models_from": 0,
            "slice_paras_process": 2,
            "software_version": f"OpenVat_V{__version__}",
        },
        "rerf_function": {
            "enable": False, "model_name": "", "model_type": 1,
            "partition_exposure_array": [], "partition_num": 0,
        },
    }


def print_info_json(result: SliceResult) -> dict:
    volume_ml = result.volume_mm3() / 1000.0
    weight_g = volume_ml * result.resin.density
    cost = volume_ml / 1000.0 * result.resin.price_per_liter
    seconds = int(result.print_time_s())
    n = len(result.layers)
    sub = {
        "estimated_cost": cost, "estimated_cost_currency": result.resin.currency,
        "estimated_time": seconds, "estimated_volume": volume_ml,
        "estimated_weight": weight_g, "model_layers_count": n,
    }
    return {
        "cost": cost, "currency": result.resin.currency, "model_layers_count": n,
        "options_infos": None, "print_time": seconds, "sub_estimated_infos": [sub],
        "volume": volume_ml, "weight": weight_g,
    }


def software_info_json() -> dict:
    return {"mark": "AC-PC", "opengl": "3.3-CoreProfile", "os": "OpenVat",
            "version": f"OpenVat {__version__}"}


# --------------------------------------------------------------------------
# writer


def bitmap_levels(resin: ResinProfile) -> int:
    """Anti-aliasing levels the layers are written with: 1 = vector layers
    (pwszImg), 2..16 = pw0Img bitmaps with that many grey levels."""
    return int(max(1, min(resin.anti_aliasing, 16)))


def write_pwsz(path: str | Path, result: SliceResult, preview_meshes: list[trimesh.Trimesh],
               progress: ProgressFn | None = None) -> Path:
    """Write the slice result to a .pwsz archive. ``preview_meshes`` are the
    world-space meshes used only for the thumbnail images."""
    path = Path(path)
    layers = result.layers
    total = len(layers) + 4

    def dump(obj) -> str:
        return json.dumps(obj, indent=4, ensure_ascii=False)

    # Entry order copied from PhotonWorkshop 4.2 exports.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("anycubic_photon_resins.pwsp", dump(pwsp_json(result.printer, result.resin)))
        zf.writestr("print_info.json", dump(print_info_json(result)))
        zf.writestr("software_info.conf", dump(software_info_json()))
        previews = [
            ((224, 168), (0.0078, 0.2813, 0.3906), (0.80, 0.80, 0.80)),
            ((336, 252), (0.0784, 0.1059, 0.1608), (0.80, 0.80, 0.80)),
            ((800, 600), (0.9333, 0.9412, 0.9647), (0.17, 0.60, 0.62)),
        ]
        for i, (size, bg, fg) in enumerate(previews):
            img = render_preview(preview_meshes, size, background=bg, model_color=fg)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(f"preview_images/preview_{i}.png", buf.getvalue())
        zf.writestr("layers_controller.conf", dump(layers_controller_json(layers)))
        zf.writestr("lcd_function.json", dump(lcd_function_json(result.object_names)))
        if progress:
            progress(2, total)

        printer = result.printer
        aa = bitmap_levels(result.resin)
        turn = turned(printer)
        for i, layer in enumerate(layers):
            if aa > 1:                       # anti-aliased: bitmaps (see the module docstring)
                r0, band = rasterize_band(layer.geometry, printer, aa, turn)
                blob = (encode_pw0_band(band, r0, printer.res_y) if len(band)
                        else encode_pw0(np.zeros((printer.res_y, printer.res_x), np.uint8)))
                zf.writestr(f"layer_images/layer_{layer.index}.pw0Img", blob)
            else:
                zf.writestr(f"layer_images/layer_{layer.index}.pwszImg", encode_layer_image(layer.geometry))
            if progress and i % 20 == 0:
                progress(2 + i, total)

        zf.writestr("calc_layer_volumes.data", encode_layer_volumes(layers))
        zf.writestr("scene.slice", encode_scene_slice(layers, result.model_bounds))
        if progress:
            progress(total, total)
    return path


# --------------------------------------------------------------------------
# reader (for the layer viewer / tests)


@dataclass
class PwszFile:
    printer_name: str
    res: tuple[int, int]
    pixel_um: tuple[float, float]
    print_size: tuple[float, float, float]
    layer_paras: list[dict]
    layer_images: list[LayerImage]           # vector layers (empty for a bitmap file)
    print_info: dict
    layer_bitmaps: list[bytes] = field(default_factory=list)   # pw0Img layers (empty for a vector file)

    @property
    def bitmap(self) -> bool:
        return bool(self.layer_bitmaps)

    def layer_polygons(self, i: int) -> list[np.ndarray]:
        """Return the layer's segments as (N,4) arrays - handy for drawing."""
        return [self.layer_images[i].segments]

    def layer_bitmap(self, i: int) -> np.ndarray:
        """A pw0Img layer's grey levels 0..15, shape (res_y, res_x)."""
        return decode_pw0(self.layer_bitmaps[i], *self.res)


def read_pwsz(path: str | Path) -> PwszFile:
    with zipfile.ZipFile(path) as zf:
        pwsp = json.loads(zf.read("anycubic_photon_resins.pwsp"))
        ctrl = json.loads(zf.read("layers_controller.conf"))
        info = json.loads(zf.read("print_info.json"))
        mt = pwsp["machine_type"]
        names = set(zf.namelist())
        images, bitmaps = [], []
        for i in range(ctrl["count"]):
            if f"layer_images/layer_{i}.pw0Img" in names:
                bitmaps.append(zf.read(f"layer_images/layer_{i}.pw0Img"))
            else:
                images.append(decode_layer_image(zf.read(f"layer_images/layer_{i}.pwszImg")))
    return PwszFile(
        printer_name=mt["name"], res=(mt["res_x"], mt["res_y"]),
        pixel_um=(mt["xy_pixel"], mt.get("xy_pixel_y", mt["xy_pixel"])),
        print_size=(mt["print_xsize"], mt["print_ysize"], mt["print_zsize"]),
        layer_paras=ctrl["paras"], layer_images=images, print_info=info, layer_bitmaps=bitmaps,
    )


def segments_to_polygons(img: LayerImage) -> MultiPolygon:
    """Rebuild shapely polygons from a decoded layer image."""
    from ..core.contours import segments_to_multipolygon
    if len(img.segments) == 0:
        return MultiPolygon()
    return segments_to_multipolygon(img.segments.reshape(-1, 2, 2).astype(np.float64))
