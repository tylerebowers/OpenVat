"""Photon Workshop binary print files, file version 1 with pw0Img layers:
.pwx (Anycubic Photon X) and .pw0 (Photon Zero).  Writer, and a reader for
inspection and tests.

Unlike .pwsz these hold bitmaps: every layer is the whole screen, run-length
encoded, so the slicer rasterizes (and anti-aliases) itself.  The layout was
taken from a Photon Workshop file (3DBenchy for the Photon X), little endian
throughout:

    0    "ANYCUBIC" + 4 zero bytes, u32 version = 1, u32 areas = 4
         u32 header address, 0, preview address, 0,
         layer-definition address, 0, layer-image address        (48 bytes)
    48   "HEADER" (12 bytes, zero padded), u32 length = 80, then 20 fields:
         f32 pixel size (um), layer height (mm), exposure (s), light-off time (s),
             bottom exposure (s), bottom layer count, lift height (mm),
             lift speed (mm/s), retract speed (mm/s), volume (mL)
         u32 anti-aliasing, resolution x, resolution y
         f32 weight (g), price
         u32 currency (a character code), per-layer override (0/1),
             print time (s), transition layer count, 0x3FF00000 (as Photon
             Workshop writes it)
    144  "PREVIEW", u32 length (= the data below + 16), u32 width 224,
         u32 'x', u32 height 168, then width*height u16 pixels: blue in bits
         11-15, green 5-10, red 0-4
    ...  "LAYERDEF", u32 length (= 4 + 32 * layers), u32 layer count, then per
         layer: u32 data address, u32 data length, f32 lift height, f32 lift
         speed, f32 exposure, f32 layer thickness, u32 lit pixel count, u32 0
    ...  the layers' pw0Img data, back to back (no section header)

pw0Img: the screen row by row (rows run on into each other), as runs of one
grey level 0..15.  A run of black (0) or white (15) is two bytes, the level
in the high nibble and a 12-bit length (up to 4095) in the low nibble + next
byte; a run of a grey level in between is one byte with a 4-bit length (up
to 15).  Longer runs are split.

Pixel column c is x = (c + 0.5 - res_x / 2) * pixel size and row r is
y = (r + 0.5 - res_y / 2) * pixel size, with the origin at the centre of
the plate: row 0 is the -Y (front) edge, so the image read top-down shows
the model seen from below - the 3DBenchy's "CT3D.xyz" reads right that way.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import MultiPolygon

from ..core.profiles import PrinterProfile
from ..core.slicer import SliceResult
from .preview import render_preview

ProgressFn = Callable[[int, int], None]

MAGIC = b"ANYCUBIC\0\0\0\0"
VERSION = 1
LAYER_FORMAT = "pw0Img"
EXTENSIONS = ("pwx", "pw0")
HEADER_TAIL = 0x3FF00000           # the header's last field, as Photon Workshop writes it
PREVIEW_SIZE = (224, 168)
DEFAULT_PREVIEW_COLORS = ((0.0078, 0.2813, 0.3906), (0.80, 0.80, 0.80))     # background, model
SUPERSAMPLE = 4                    # sub-rows per pixel row for anti-aliasing
FILE_MARK = struct.Struct("<12sII7I")
HEADER = struct.Struct("<10f3I2f4II")
LAYER_DEF = struct.Struct("<IIffffII")


def supports(printer: PrinterProfile) -> bool:
    """True if the printer takes this file type (bitmap layers in pw0Img,
    Photon Workshop file version 1)."""
    return (printer.output_type == "anycubic_bitmap" and printer.layer_format == LAYER_FORMAT
            and printer.file_version == VERSION)


def _section(name: bytes, length: int) -> bytes:
    return struct.pack("<12sI", name, length)


# --------------------------------------------------------------------------
# rasterizing


def _edges(geometry: MultiPolygon) -> np.ndarray:
    """All ring edges of the layer as (N, 4) x0 y0 x1 y1 in mm."""
    parts = []
    for poly in getattr(geometry, "geoms", [geometry]):
        if poly.is_empty:
            continue
        for ring in (poly.exterior, *poly.interiors):
            c = np.asarray(ring.coords, dtype=np.float64)
            if len(c) >= 4:
                parts.append(np.hstack([c[:-1, :2], c[1:, :2]]))
    return np.vstack(parts) if parts else np.zeros((0, 4))


def _crossings(edges_px: np.ndarray, rows: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the edges cross the scan lines v = (j + 0.5) * step, j < rows:
    (line index, u) pairs sorted by line then u.  A line meets an edge when
    min(v0, v1) <= v < max(v0, v1), so a shared vertex counts once."""
    u0, v0, u1, v1 = edges_px.T
    lo, hi = np.minimum(v0, v1), np.maximum(v0, v1)
    first = np.ceil(lo / step - 0.5).astype(np.int64)
    last = np.ceil(hi / step - 0.5).astype(np.int64)            # exclusive
    first, last = np.clip(first, 0, rows), np.clip(last, 0, rows)
    count = np.maximum(last - first, 0)
    keep = count > 0
    if not keep.any():
        return np.zeros(0, np.int64), np.zeros(0)
    u0, v0, u1, v1, first, count = u0[keep], v0[keep], u1[keep], v1[keep], first[keep], count[keep]
    idx = np.repeat(np.arange(len(count)), count)
    line = first[idx] + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
    v = (line + 0.5) * step
    t = (v - v0[idx]) / (v1[idx] - v0[idx])
    u = u0[idx] + t * (u1[idx] - u0[idx])
    order = np.lexsort((u, line))
    return line[order], u[order]


def rasterize(geometry: MultiPolygon, printer: PrinterProfile, anti_aliasing: int = 1) -> np.ndarray:
    """The layer as grey levels 0..15, shape (res_y, res_x), row 0 at -Y.

    Without anti-aliasing a pixel is lit when its centre is inside (even-odd
    rule, so holes and islands in holes come out right).  With it, coverage
    is measured on SUPERSAMPLE sub-rows per row, exactly along each row, and
    rounded to ``anti_aliasing`` levels (2..16)."""
    w, h = printer.res_x, printer.res_y
    out = np.zeros((h, w), np.uint8)
    e = _edges(geometry)
    if not len(e):
        return out
    px, py = printer.pixel_x_mm, printer.pixel_y_mm
    e = np.column_stack([e[:, 0] / px + w / 2, e[:, 1] / py + h / 2, e[:, 2] / px + w / 2, e[:, 3] / py + h / 2])
    e = e[e[:, 1] != e[:, 3]]                                    # horizontal edges never cross a line
    levels = int(max(1, min(anti_aliasing, 16)))
    s = 1 if levels <= 1 else SUPERSAMPLE
    line, u = _crossings(e, h * s, 1.0 / s)
    if not len(line):
        return out
    row = line[0::2] // s                       # spans: crossing pairs on each scan line
    r0, r1 = int(row.min()), int(row.max()) + 1  # only the rows the layer covers
    size = (r1 - r0) * (w + 1)
    base = (row - r0) * (w + 1)
    if s == 1:
        a = np.clip(np.ceil(u[0::2] - 0.5), 0, w).astype(np.int64)   # first lit column (centre inside)
        b = np.clip(np.ceil(u[1::2] - 0.5), 0, w).astype(np.int64)   # first dark column after it
        diff = np.bincount(base + a, minlength=size) - np.bincount(base + b, minlength=size)
        lit = np.cumsum(diff.reshape(r1 - r0, w + 1)[:, :w], axis=1) > 0
        out[r0:r1][lit] = 15
        return out

    # coverage: whole pixels inside a span count 1, the pixels at its ends the covered part
    ua, ub = np.clip(u[0::2], 0, w), np.clip(u[1::2], 0, w)
    ca, cb = np.ceil(ua).astype(np.int64), np.floor(ub).astype(np.int64)
    one = np.floor(ua).astype(np.int64)
    same = ca > cb                                               # both ends in one pixel
    whole = ~same & (cb > ca)
    cov = np.bincount(base[whole] + ca[whole], minlength=size) - np.bincount(base[whole] + cb[whole], minlength=size)
    cov = np.cumsum(cov.reshape(r1 - r0, w + 1), axis=1).astype(np.float64).ravel()
    left = ~same & (ca > ua)
    right = ~same & (ub > cb) & (cb < w)
    idx = np.concatenate([base[same] + one[same], base[left] + one[left], base[right] + cb[right]])
    wts = np.concatenate([(ub - ua)[same], (ca - ua)[left], (ub - cb)[right]])
    cov += np.bincount(idx, weights=wts, minlength=size)
    coverage = cov.reshape(r1 - r0, w + 1)[:, :w] / s
    q = np.round(np.clip(coverage, 0, 1) * (levels - 1)) / (levels - 1)
    out[r0:r1] = np.round(q * 15).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# pw0Img


def encode_pw0(levels: np.ndarray) -> bytes:
    """Run-length encode grey levels 0..15 (whole screen, row by row)."""
    flat = np.ascontiguousarray(levels, dtype=np.uint8).ravel()
    if flat.size == 0:
        return b""
    starts = np.concatenate([[0], np.flatnonzero(flat[1:] != flat[:-1]) + 1])
    lengths = np.diff(np.concatenate([starts, [flat.size]]))
    codes = flat[starts].astype(np.int64)
    full = (codes == 0) | (codes == 15)
    cap = np.where(full, 4095, 15)
    chunks = (lengths + cap - 1) // cap
    c_code, c_full = np.repeat(codes, chunks), np.repeat(full, chunks)
    c_len = np.repeat(cap, chunks)
    last = np.cumsum(chunks) - 1
    c_len[last] = lengths - (chunks - 1) * cap
    size = np.where(c_full, 2, 1)
    pos = np.cumsum(size) - size
    out = np.zeros(int(size.sum()), np.uint8)
    out[pos] = (c_code << 4) | np.where(c_full, c_len >> 8, c_len)
    out[pos[c_full] + 1] = c_len[c_full] & 0xFF
    return out.tobytes()


def decode_pw0(data: bytes, width: int, height: int) -> np.ndarray:
    """Grey levels 0..15, shape (height, width)."""
    buf = np.frombuffer(data, np.uint8)
    out = np.zeros(width * height, np.uint8)
    codes, lengths = [], []
    i, n = 0, len(buf)
    while i < n:
        b = int(buf[i])
        code, length = b >> 4, b & 0xF
        if code in (0, 15):
            length = (length << 8) | int(buf[i + 1])
            i += 2
        else:
            i += 1
        codes.append(code)
        lengths.append(length)
    ends = np.cumsum(lengths)
    if len(ends) and ends[-1] != width * height:
        raise ValueError(f"layer holds {ends[-1]} pixels, the screen {width * height}")
    starts = ends - np.asarray(lengths)
    for code, a, b in zip(codes, starts, ends):
        if code:
            out[a:b] = code
    return out.reshape(height, width)


# --------------------------------------------------------------------------
# preview


def encode_preview(img: Image.Image) -> bytes:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint16)
    v = (rgb[..., 0] >> 3) | ((rgb[..., 1] >> 2) << 5) | ((rgb[..., 2] >> 3) << 11)
    return v.astype("<u2").tobytes()


def decode_preview(data: bytes, width: int, height: int) -> Image.Image:
    v = np.frombuffer(data, "<u2", width * height).reshape(height, width).astype(np.uint32)
    rgb = np.dstack([(v & 0x1F) << 3, ((v >> 5) & 0x3F) << 2, (v >> 11) << 3]).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


# --------------------------------------------------------------------------
# writer


def write_pwx(path: str | Path, result: SliceResult, preview_meshes: list[trimesh.Trimesh],
              progress: ProgressFn | None = None) -> Path:
    """Write the slice result as a version-1 Photon Workshop file (.pwx /
    .pw0).  ``preview_meshes`` are the world-space meshes for the thumbnail."""
    path = Path(path)
    printer, resin, layers = result.printer, result.resin, result.layers
    if not supports(printer):
        raise ValueError(f"{printer.name} does not take .{'/.'.join(EXTENSIONS)} files")
    total = len(layers) + 2
    aa = int(max(1, min(resin.anti_aliasing, 16)))

    blobs, lit = [], []
    for i, layer in enumerate(layers):
        img = rasterize(layer.geometry, printer, aa)
        blobs.append(encode_pw0(img))
        lit.append(int(np.count_nonzero(img)))
        if progress and i % 10 == 0:
            progress(i, total)

    bottom_n = min(resin.bottom_layers, len(layers))
    normal = layers[bottom_n] if len(layers) > bottom_n else (layers[-1] if layers else None)
    lift_h = normal.lift_distance if normal else resin.normal.lift_distance
    lift_v = normal.lift_speed if normal else resin.normal.lift_speed
    retract_v = normal.retract_speed if normal else resin.normal.retract_speed
    # per-layer values only count when the header says so: when they differ
    # from what the header implies (transition layers, other bottom lifts)
    plain = all(abs(l.exposure - (resin.bottom.exposure if i < bottom_n else resin.normal.exposure)) < 1e-6
                and abs(l.lift_distance - lift_h) < 1e-6 and abs(l.lift_speed - lift_v) < 1e-6
                for i, l in enumerate(layers))
    volume_ml = result.volume_mm3() / 1000.0
    currency = ord(resin.currency[0]) if resin.currency else ord("$")
    header = HEADER.pack(
        printer.pixel_x_um, resin.normal.thickness, resin.normal.exposure, resin.normal.off_time,
        resin.bottom.exposure, float(bottom_n), lift_h, lift_v, retract_v, volume_ml,
        aa, printer.res_x, printer.res_y,
        volume_ml * resin.density, volume_ml / 1000.0 * resin.price_per_liter,
        currency, 0 if plain else 1, int(round(result.print_time_s())), 0, HEADER_TAIL)

    mt = printer.photonworkshop.get("machine_type", {})
    size = tuple(mt.get("prev_image_size") or PREVIEW_SIZE)
    back = tuple(mt.get("prev_back_color") or DEFAULT_PREVIEW_COLORS[0])
    model = tuple(mt.get("prev_model_color") or DEFAULT_PREVIEW_COLORS[1])
    preview = encode_preview(render_preview(preview_meshes, size, background=back, model_color=model))
    if progress:
        progress(len(layers) + 1, total)

    header_at = FILE_MARK.size
    preview_at = header_at + 16 + HEADER.size
    preview_body = struct.pack("<III", size[0], ord("x"), size[1]) + preview
    layerdef_at = preview_at + 16 + len(preview_body)
    layerdef_len = 4 + LAYER_DEF.size * len(layers)
    images_at = layerdef_at + 16 + layerdef_len

    out = io.BytesIO()
    out.write(FILE_MARK.pack(MAGIC, VERSION, 4, header_at, 0, preview_at, 0, layerdef_at, 0, images_at))
    out.write(_section(b"HEADER", HEADER.size) + header)
    out.write(_section(b"PREVIEW", len(preview_body) + 16) + preview_body)
    out.write(_section(b"LAYERDEF", layerdef_len) + struct.pack("<I", len(layers)))
    address = images_at
    for layer, blob, count in zip(layers, blobs, lit):
        out.write(LAYER_DEF.pack(address, len(blob), layer.lift_distance, layer.lift_speed,
                                 layer.exposure, layer.thickness, count, 0))
        address += len(blob)
    for blob in blobs:
        out.write(blob)
    path.write_bytes(out.getvalue())
    if progress:
        progress(total, total)
    return path


# --------------------------------------------------------------------------
# reader


@dataclass
class PwxLayer:
    address: int
    length: int
    lift_height: float
    lift_speed: float
    exposure: float
    thickness: float
    lit_pixels: int


@dataclass
class PwxFile:
    version: int
    header: dict
    preview: Image.Image
    layers: list[PwxLayer] = field(default_factory=list)
    data: bytes = b""

    @property
    def res(self) -> tuple[int, int]:
        return self.header["res_x"], self.header["res_y"]

    def layer_image(self, i: int) -> np.ndarray:
        """Grey levels 0..15, shape (res_y, res_x), row 0 at -Y."""
        d = self.layers[i]
        return decode_pw0(self.data[d.address:d.address + d.length], *self.res)


HEADER_FIELDS = ("pixel_um", "layer_height", "exposure", "off_time", "bottom_exposure", "bottom_layers",
                 "lift_height", "lift_speed", "retract_speed", "volume_ml", "anti_aliasing", "res_x", "res_y",
                 "weight_g", "price", "currency", "per_layer_override", "print_time", "transition_layers",
                 "tail")


def is_pwx(path: str | Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == MAGIC[:8]


def read_pwx(path: str | Path) -> PwxFile:
    data = Path(path).read_bytes()
    mark = FILE_MARK.unpack_from(data, 0)
    if mark[0] != MAGIC:
        raise ValueError("not a Photon Workshop file")
    version, header_at, preview_at, layerdef_at = mark[1], mark[3], mark[5], mark[7]
    if version != VERSION:
        raise ValueError(f"Photon Workshop file version {version}: only version {VERSION} is read")
    header = dict(zip(HEADER_FIELDS, HEADER.unpack_from(data, header_at + 16)))
    header["currency"] = chr(header["currency"])
    w, _x, h = struct.unpack_from("<III", data, preview_at + 16)
    preview = decode_preview(data[preview_at + 28:preview_at + 28 + w * h * 2], w, h)
    count = struct.unpack_from("<I", data, layerdef_at + 16)[0]
    layers = [PwxLayer(*LAYER_DEF.unpack_from(data, layerdef_at + 20 + LAYER_DEF.size * i)[:7])
              for i in range(count)]
    return PwxFile(version, header, preview, layers, data)
