"""Photon Workshop binary print files: the "ANYCUBIC" container with bitmap
layers - file version 1 (.pws Photon / Photon S in pwsImg; .pwx Photon X,
.pw0 Photon Zero in pw0Img) and versions 515, 516, 517 and 518 in pw0Img
(.pwmo Mono, .pwms Mono SE, .pmsq Mono SQ, .dlp Ultra, .pm3m Photon M3 Max,
.pwmx Mono X, .dl2p D2, .pwmb, .m5sp Mono M5s Pro, .pm5s and the other
printers of those versions).  Writer, and a reader for inspection
and tests.  Also the pw0Img codec and the rasterizer, which the .pwsz family
uses for its bitmap layers too (see pwsz.py).

The layouts come from Photon Workshop's own files - 3DBenchy for the
Photon and Photon S (.pws), Photon X (version 1), Photon Mono, Mono SE,
Mono SQ and Ultra (515), Photon M3 Max (516), D2 (517) and Mono M5s Pro
(518).  Reading one of them and writing it out again
gives the same bytes.  Little endian throughout; "section" = a 12-byte
zero-padded name + u32 length:

    file mark   "ANYCUBIC" + 4 zero bytes, u32 version, u32 area count
                (4 / 5 / 8 / 9 / 11), then the addresses (48 / 48 / 52 / 56 /
                64 bytes in all):
                  v1:   header, 0, preview, 0, layer table, 0, layers
                  v515: header, 0, preview, colour table, layer table, 0,
                        layers
                  v516: header, 0, preview, colour table, layer table, extra,
                        machine, layers
                  v517: header, software, preview, colour table, layer table,
                        extra, machine, layers, model
                  v518: as 517, then sub-images, preview 2
    HEADER      section, length = its fields: f32 pixel size (um, the width),
                layer height (mm), exposure (s), light-off time (s), bottom
                exposure (s), bottom layer count, lift height (mm), lift
                speed (mm/s), retract speed (mm/s), volume (mL); u32
                anti-aliasing, resolution x, y; f32 weight (g), price; u32
                currency (a character), per-layer override (0/1), print time
                (s), transition layer count, then
                  v1:   0x3FF00000 (Photon Workshop leaves stray values here)
                  v515: 0
                  v516: transition type, advanced mode
                  v517: transition type, advanced mode, grey + blur level
                        (2 x u16), resin code (10)
                  v518: as 517, then 1 (the resins' "intelli mode")
    PREVIEW     section, length = data + 16: u32 width 224, 'x', height 168,
                then u16 pixels - blue in bits 11-15, green 5-10, red 0-4
    colour      (515+, no section header) u32 0, u32 16, 16 bytes: the
    table       intensity of each anti-aliasing step (63, 127, 191, 255 for
                4 steps), the rest 255 - all 255 without anti-aliasing; u32 0
    LAYERDEF    section, length = data: u32 layer count, then per layer u32
                data address, u32 data length, f32 lift height, lift speed,
                exposure, layer thickness, u32 lit pixel count, u32 0
    EXTRA       (516+) section, length always 24 though 56 bytes follow:
                u32 2, then bottom-layer lift in two stages: f32 height 1,
                lift speed 1, retract speed 1, height 2, lift speed 2,
                retract speed 2; u32 2 and the same for the other layers
    MACHINE     (516+) section, length = data + 16: name (96 bytes), layer
                format "pw0Img" (16), u32 anti-aliasing levels (16), u32
                property flags, f32 screen width, height (mm), print height
                (mm), u32 file version, u32 background colour (r, g, b, 0);
                518 adds f32 pixel width, height (um), 32 unused bytes, u32
                screen count and up to 3 screens of u16 x, y, width, height
    software    (517+, no section header) mark "AC-PC" (32 bytes), u32 164
                (this block's size), version (32), OS (64), OpenGL (32)
    MODEL       (517+) section, length 0 (52 in 518): f32 model bounds x0 y0
                z0 x1 y1 z1, u32 0, u32 0
    SUBIMGS     (518) section, length = data + 16: u32 layer count, u32 1,
                then per layer the data address, length and lit pixel count
                again, and 8 u32 0
    PREVIEW2    (518) section, length = data + 16: u32 width 320, u32 the
                background as red-high RGB565, u32 height 190, then pixels as
                in PREVIEW
    layers      each layer's bitmap, back to back (no section header)

pw0Img: the screen row by row (rows run on into each other), as runs of one
grey level 0..15.  A run of black (0) or white (15) takes two bytes - the
level in the high nibble, a 12-bit length (up to 4095) in the low nibble and
the next byte; a run of a grey level in between one byte with a 4-bit
length (up to 15).  Longer runs are split.  With n anti-aliasing steps a
pixel k/n covered gets level 16k/n - 1 (4 steps: 0, 3, 7, 11, 15).

pwsImg (the Photon and Photon S): the same, black and white only, one byte
per run - bit 7 lit, bits 0-6 the length (up to 125).

Pixel column c is x = (c + 0.5 - res_x / 2) * pixel width and row r is
y = (r + 0.5 - res_y / 2) * pixel height, the origin at the centre of the
plate: row 0 is the -Y edge, so the image read top-down shows the model from
below (the 3DBenchy's "CT3D.xyz" reads right).  Printers whose Photon
Workshop data say rotate_z = 180 (the P1) have their bitmaps turned half a
turn: column 0 is +X, row 0 +Y.
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

from .. import __version__
from ..core.profiles import PrinterProfile
from ..core.slicer import SliceResult
from .preview import render_preview

ProgressFn = Callable[[int, int], None]

MAGIC = b"ANYCUBIC\0\0\0\0"
LAYER_FORMAT = "pw0Img"
PWS_FORMAT = "pwsImg"
VERSIONS = (1, 515, 516, 517, 518)  # written (and read); each checked against a Photon Workshop file
AREAS = {1: 4, 515: 5, 516: 8, 517: 9, 518: 11}
ADDRESSES = {1: 7, 515: 7, 516: 8, 517: 9, 518: 11}
HEADER_TAIL_V1 = 0x3FF00000         # version 1's last header field, as Photon Workshop's .pwx has it
RESIN_CODE = 10                     # version 517+ header; Photon Workshop's code for its standard resins
INTELLI_MODE = 1                    # version 518 header; the resins' "intelli_mode"
PREVIEW_SIZE = (224, 168)
PREVIEW2_SIZE = (320, 190)
DEFAULT_PREVIEW_COLORS = ((0.0078, 0.2813, 0.3906), (0.80, 0.80, 0.80))     # background, model
DEFAULT_PREVIEW2_COLORS = ((0.0784, 0.1059, 0.1608), (0.30, 0.90, 0.90))
SUPERSAMPLE = 4                     # sub-rows per pixel row for anti-aliasing
BAND_ROWS = 256                     # rows rasterized at a time (bounds memory on big screens)
LAYER_DEF = struct.Struct("<IIffffII")
HEADER_COMMON = "<10f3I2f4I"        # the 19 fields every version has
HEADER_TAIL = {1: "I", 515: "I", 516: "2I", 517: "4I", 518: "5I"}
MACHINE = struct.Struct("<96s16s2I3f2I")
MACHINE_518 = struct.Struct("<96s16s2I3f2I2f32sI12H")
SOFTWARE = struct.Struct("<32sI32s64s32s")
MODEL = struct.Struct("<6f2I")
MODEL_LENGTH_FIELD = {517: 0, 518: 52}
SUBIMG = struct.Struct("<3I8I")
EXTRA_LENGTH_FIELD = 24             # what the EXTRA section's length says (56 bytes follow)
SOFTWARE_MARK = b"AC-PC"


def supports(printer: PrinterProfile) -> bool:
    """True if OpenVat can write the printer's files: bitmap layers in pw0Img
    (Photon Workshop file version 1, 515, 516, 517 or 518) or pwsImg (version 1)."""
    if printer.output_type != "anycubic_bitmap":
        return False
    if printer.layer_format == PWS_FORMAT:
        return printer.file_version == 1
    return printer.layer_format == LAYER_FORMAT and printer.file_version in VERSIONS


def turned(printer: PrinterProfile) -> bool:
    """Bitmaps for this printer are turned 180 degrees (Photon Workshop's
    rotate_z, e.g. the Photon P1)."""
    try:
        return abs(float(printer.photonworkshop.get("machine_type", {}).get("rotate_z") or 0) - 180.0) < 1e-6
    except (AttributeError, TypeError, ValueError):
        return False


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
    first = np.clip(np.ceil(lo / step - 0.5).astype(np.int64), 0, rows)
    last = np.clip(np.ceil(hi / step - 0.5).astype(np.int64), 0, rows)     # exclusive
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


def _fill(row: np.ndarray, ua: np.ndarray, ub: np.ndarray, rows: int, w: int, s: int) -> np.ndarray:
    """Coverage of ``rows`` pixel rows by spans [ua, ub) on sub-row ``row``
    (``s`` sub-rows per pixel row): 0..1 per pixel, or 0/1 when s == 1
    (then a pixel counts when its centre is inside)."""
    size = rows * (w + 1)
    base = (row // s) * (w + 1)
    if s == 1:
        a = np.clip(np.ceil(ua - 0.5), 0, w).astype(np.int64)    # first column whose centre is inside
        b = np.clip(np.ceil(ub - 0.5), 0, w).astype(np.int64)    # first one after the span
        diff = np.bincount(base + a, minlength=size) - np.bincount(base + b, minlength=size)
        return np.cumsum(diff.reshape(rows, w + 1)[:, :w], axis=1) > 0
    ua, ub = np.clip(ua, 0, w), np.clip(ub, 0, w)
    ca, cb = np.ceil(ua).astype(np.int64), np.floor(ub).astype(np.int64)
    one = np.floor(ua).astype(np.int64)
    same = ca > cb                                               # both ends in one pixel
    whole = ~same & (cb > ca)
    cov = np.bincount(base[whole] + ca[whole], minlength=size) - np.bincount(base[whole] + cb[whole], minlength=size)
    cov = np.cumsum(cov.reshape(rows, w + 1), axis=1).astype(np.float64).ravel()
    left = ~same & (ca > ua)
    right = ~same & (ub > cb) & (cb < w)
    idx = np.concatenate([base[same] + one[same], base[left] + one[left], base[right] + cb[right]])
    wts = np.concatenate([(ub - ua)[same], (ca - ua)[left], (ub - cb)[right]])
    cov += np.bincount(idx, weights=wts, minlength=size)
    return cov.reshape(rows, w + 1)[:, :w] / s


def rasterize_band(geometry: MultiPolygon, printer: PrinterProfile, anti_aliasing: int = 1,
                   turn: bool = False) -> tuple[int, np.ndarray]:
    """The rows the layer covers: (first row, grey levels 0..15 of shape
    (rows, res_x)).  See ``rasterize``."""
    w, h = printer.res_x, printer.res_y
    e = _edges(geometry)
    if not len(e):
        return 0, np.zeros((0, w), np.uint8)
    sign = -1.0 if turn else 1.0
    px, py = printer.pixel_x_mm, printer.pixel_y_mm
    e = np.column_stack([sign * e[:, 0] / px + w / 2, sign * e[:, 1] / py + h / 2,
                         sign * e[:, 2] / px + w / 2, sign * e[:, 3] / py + h / 2])
    e = e[e[:, 1] != e[:, 3]]                                    # horizontal edges never cross a line
    levels = int(max(1, min(anti_aliasing, 16)))
    s = 1 if levels <= 1 else SUPERSAMPLE
    line, u = _crossings(e, h * s, 1.0 / s)
    if not len(line):
        return 0, np.zeros((0, w), np.uint8)
    line, ua, ub = line[0::2], u[0::2], u[1::2]                  # spans: crossing pairs on a scan line
    r0, r1 = int(line[0] // s), int(line[-1] // s) + 1
    # only the columns the layer reaches are worked on
    x0 = int(np.clip(np.floor(ua.min()) - 1, 0, w))
    x1 = int(np.clip(np.ceil(ub.max()) + 1, x0, w))
    ua, ub = np.clip(ua, x0, x1) - x0, np.clip(ub, x0, x1) - x0
    band = np.zeros((r1 - r0, w), np.uint8)
    for c0 in range(r0, r1, BAND_ROWS):
        c1 = min(c0 + BAND_ROWS, r1)
        a, b = np.searchsorted(line, [c0 * s, c1 * s])
        if a == b:
            continue
        cov = _fill(line[a:b] - c0 * s, ua[a:b], ub[a:b], c1 - c0, x1 - x0, s)
        if s == 1:
            band[c0 - r0:c1 - r0, x0:x1][cov] = 15
        else:
            # as Photon Workshop: k = covered share in 1/levels steps, level 16k/levels - 1
            # (levels 4: 0, 3, 7, 11, 15)
            k = np.rint(np.clip(cov, 0.0, 1.0) * levels)
            band[c0 - r0:c1 - r0, x0:x1] = np.where(k > 0, k * (16 // levels) - 1, 0).astype(np.uint8)
    return r0, band


def rasterize(geometry: MultiPolygon, printer: PrinterProfile, anti_aliasing: int = 1,
              turn: bool = False) -> np.ndarray:
    """The layer as grey levels 0..15, shape (res_y, res_x), row 0 at -Y
    (at +Y when ``turn``, which also puts column 0 at +X).

    Without anti-aliasing a pixel is lit when its centre is inside (even-odd
    rule, so holes and islands in holes come out right).  With it, coverage
    is measured on SUPERSAMPLE sub-rows per row, exactly along each row, and
    rounded to ``anti_aliasing`` steps (2..16): a pixel k steps covered gets
    level 16k/steps - 1, as Photon Workshop does."""
    out = np.zeros((printer.res_y, printer.res_x), np.uint8)
    r0, band = rasterize_band(geometry, printer, anti_aliasing, turn)
    out[r0:r0 + len(band)] = band
    return out


# --------------------------------------------------------------------------
# pw0Img


def _encode_runs(codes: np.ndarray, lengths: np.ndarray) -> bytes:
    codes, lengths = codes.astype(np.int64), lengths.astype(np.int64)
    keep = lengths > 0
    codes, lengths = codes[keep], lengths[keep]
    if not len(codes):
        return b""
    full = (codes == 0) | (codes == 15)
    cap = np.where(full, 4095, 15)
    chunks = (lengths + cap - 1) // cap
    c_code, c_full = np.repeat(codes, chunks), np.repeat(full, chunks)
    c_len = np.repeat(cap, chunks)
    c_len[np.cumsum(chunks) - 1] = lengths - (chunks - 1) * cap
    size = np.where(c_full, 2, 1)
    pos = np.cumsum(size) - size
    out = np.zeros(int(size.sum()), np.uint8)
    out[pos] = (c_code << 4) | np.where(c_full, c_len >> 8, c_len)
    out[pos[c_full] + 1] = c_len[c_full] & 0xFF
    return out.tobytes()


def _runs(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if flat.size == 0:
        return np.zeros(0, np.uint8), np.zeros(0, np.int64)
    starts = np.concatenate([[0], np.flatnonzero(flat[1:] != flat[:-1]) + 1])
    return flat[starts], np.diff(np.concatenate([starts, [flat.size]]))


def encode_pw0(levels: np.ndarray) -> bytes:
    """Run-length encode grey levels 0..15 (whole screen, row by row)."""
    return _encode_runs(*_runs(np.ascontiguousarray(levels, dtype=np.uint8).ravel()))


def encode_pw0_band(band: np.ndarray, row0: int, height: int) -> bytes:
    """Encode a whole screen of ``height`` rows that is black except for
    ``band``, which starts at row ``row0`` (see rasterize_band)."""
    width = band.shape[1] if band.ndim == 2 else 0
    codes, lengths = _runs(np.ascontiguousarray(band, dtype=np.uint8).ravel())
    before, after = row0 * width, (height - row0 - len(band)) * width
    if width == 0:                                               # an empty layer
        return b""
    codes = np.concatenate([[0], codes, [0]])
    lengths = np.concatenate([[before], lengths, [after]])
    # join the black margins with black runs at the ends of the band
    merged_c, merged_l = [], []
    for c, n in zip(codes.tolist(), lengths.tolist()):
        if merged_c and merged_c[-1] == c:
            merged_l[-1] += n
        else:
            merged_c.append(c)
            merged_l.append(n)
    return _encode_runs(np.asarray(merged_c), np.asarray(merged_l))


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


def encode_pws(levels: np.ndarray) -> bytes:
    """pwsImg: runs of lit (level > 7) or dark pixels, one byte each - bit 7
    lit, bits 0-6 the length, at most 125."""
    codes, lengths = _runs((np.ascontiguousarray(levels).ravel() > 7).astype(np.uint8))
    return _encode_pws_runs(codes, lengths)


def _encode_pws_runs(codes: np.ndarray, lengths: np.ndarray) -> bytes:
    keep = lengths > 0
    codes, lengths = codes[keep].astype(np.int64), lengths[keep].astype(np.int64)
    chunks = (lengths + 124) // 125
    lit = np.repeat(codes, chunks)
    size = np.full(int(chunks.sum()), 125, np.int64)
    size[np.cumsum(chunks) - 1] = lengths - (chunks - 1) * 125
    return ((lit << 7) | size).astype(np.uint8).tobytes()


def encode_pws_band(band: np.ndarray, row0: int, height: int) -> bytes:
    """encode_pws for a screen that is dark outside ``band`` (see rasterize_band)."""
    width = band.shape[1]
    codes, lengths = _runs((np.ascontiguousarray(band).ravel() > 7).astype(np.uint8))
    codes = np.concatenate([[0], codes, [0]])
    lengths = np.concatenate([[row0 * width], lengths, [(height - row0 - len(band)) * width]])
    starts = np.concatenate([[True], codes[1:] != codes[:-1]])          # join the dark margins
    group = np.cumsum(starts) - 1
    return _encode_pws_runs(codes[starts], np.bincount(group, weights=lengths).astype(np.int64))


def decode_pws(data: bytes, width: int, height: int) -> np.ndarray:
    """Levels 0 / 15, shape (height, width)."""
    b = np.frombuffer(data, np.uint8)
    lengths = (b & 0x7F).astype(np.int64)
    if lengths.sum() != width * height:
        raise ValueError(f"layer holds {lengths.sum()} pixels, the screen {width * height}")
    return np.repeat(np.where(b >> 7, 15, 0).astype(np.uint8), lengths).reshape(height, width)


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
# the file


def _section(name: bytes, length: int) -> bytes:
    return struct.pack("<12sI", name, length)


def color_table(anti_aliasing: int) -> bytes:
    """The 515+ colour table: the intensity of each of the ``anti_aliasing``
    grey steps (63, 127, 191, 255 for 4), then 255 - all 255 without
    anti-aliasing.  As Photon Workshop writes it."""
    n = int(max(1, min(anti_aliasing, 16)))
    greys = [min(255, (i + 1) * 256 // n - 1) if i < n else 255 for i in range(16)]
    return struct.pack("<II16BI", 0, 16, *greys, 0)


@dataclass
class PwBinary:
    """Everything in a file, as stored - so a file read can be written back
    unchanged (``build``)."""
    version: int
    header: tuple                         # the header's fields (see HEADER_FIELDS)
    preview_size: tuple[int, int]
    preview: bytes                        # u16 pixels
    layers: list[tuple] = field(default_factory=list)      # LAYER_DEF fields without the address
    blobs: list[bytes] = field(default_factory=list)       # each layer's bitmap data
    color_table: bytes = b""              # 516+
    extra: tuple = ()                     # 516+: the 14 EXTRA fields
    machine: tuple = ()                   # 516+: MACHINE (or MACHINE_518) fields
    software: tuple = ()                  # 517+: SOFTWARE fields
    model: tuple = ()                     # 517+: MODEL fields
    subimages: tuple = (1, [])            # 518: SUBIMGS's second field, each layer's 8 trailing values
    preview2: tuple = ()                  # 518: width, background, height, u16 pixels
    spare: tuple = ()                     # unused address slots (v1: 3, v516: 1) - 0, or stray values

    def build(self) -> bytes:
        v = self.version
        header = struct.pack(HEADER_COMMON + HEADER_TAIL[v], *self.header)
        preview = struct.pack("<III", self.preview_size[0], ord("x"), self.preview_size[1]) + self.preview
        layer_table_len = 4 + LAYER_DEF.size * len(self.layers)
        machine = MACHINE_518 if v >= 518 else MACHINE
        at = {"header": 20 + 4 * ADDRESSES[v]}
        at["preview"] = at["header"] + 16 + len(header)
        pos = at["preview"] + 16 + len(preview)
        if v >= 515:
            at["colors"], pos = pos, pos + len(self.color_table)
        at["layerdef"] = pos
        pos += 16 + layer_table_len
        if v >= 516:
            at["extra"], pos = pos, pos + 16 + 56
            at["machine"], pos = pos, pos + 16 + machine.size
        if v >= 517:
            at["software"], pos = pos, pos + SOFTWARE.size
            at["model"], pos = pos, pos + 16 + MODEL.size
        if v >= 518:
            at["subimages"], pos = pos, pos + 16 + 8 + SUBIMG.size * len(self.layers)
            at["preview2"], pos = pos, pos + 16 + 12 + len(self.preview2[3])
        at["layers"] = pos
        addresses = [at[k] for k in ("header", "software", "preview", "colors", "layerdef", "extra", "machine",
                                     "layers", "model", "subimages", "preview2") if k in at]
        if v == 1:                                               # header, -, preview, -, layer table, -, layers
            s1, s2, s3 = self.spare or (0, 0, 0)
            addresses = [at["header"], s1, at["preview"], s2, at["layerdef"], s3, at["layers"]]
        elif v == 515:                                           # header, -, preview, colours, layer table, -, layers
            s1, s2 = self.spare or (0, 0)
            addresses = [at["header"], s1, at["preview"], at["colors"], at["layerdef"], s2, at["layers"]]
        elif v == 516:                                           # no software block: its slot is unused
            addresses.insert(1, (self.spare or (0,))[0])

        out = io.BytesIO()
        out.write(struct.pack(f"<12sII{len(addresses)}I", MAGIC, v, AREAS[v], *addresses))
        out.write(_section(b"HEADER", len(header)) + header)
        out.write(_section(b"PREVIEW", len(preview) + 16) + preview)
        if v >= 515:
            out.write(self.color_table)
        out.write(_section(b"LAYERDEF", layer_table_len) + struct.pack("<I", len(self.layers)))
        addresses_of_layers, address = [], at["layers"]
        for fields, blob in zip(self.layers, self.blobs):
            out.write(LAYER_DEF.pack(address, *fields))
            addresses_of_layers.append(address)
            address += len(blob)
        if v >= 516:
            out.write(_section(b"EXTRA", EXTRA_LENGTH_FIELD) + struct.pack("<I6fI6f", *self.extra))
            out.write(_section(b"MACHINE", machine.size + 16) + machine.pack(*self.machine))
        if v >= 517:
            out.write(SOFTWARE.pack(*self.software))
            out.write(_section(b"MODEL", MODEL_LENGTH_FIELD[v]) + MODEL.pack(*self.model))
        if v >= 518:
            per_screen, tails = self.subimages
            out.write(_section(b"SUBIMGS", 8 + SUBIMG.size * len(self.layers) + 16)
                      + struct.pack("<II", len(self.layers), per_screen))
            for i, (fields, a) in enumerate(zip(self.layers, addresses_of_layers)):
                tail = tails[i] if i < len(tails) else (0,) * 8
                out.write(SUBIMG.pack(a, fields[0], fields[5], *tail))
            w, background, h, pixels = self.preview2
            out.write(_section(b"PREVIEW2", 12 + len(pixels) + 16) + struct.pack("<III", w, background, h) + pixels)
        for blob in self.blobs:
            out.write(blob)
        return out.getvalue()


HEADER_FIELDS = ("pixel_um", "layer_height", "exposure", "off_time", "bottom_exposure", "bottom_layers",
                 "lift_height", "lift_speed", "retract_speed", "volume_ml", "anti_aliasing", "res_x", "res_y",
                 "weight_g", "price", "currency", "per_layer_override", "print_time", "transition_layers")
HEADER_MORE = {1: ("tail",), 515: ("tail",), 516: ("transition_type", "advanced_mode"),
               517: ("transition_type", "advanced_mode", "grey_blur", "resin_code"),
               518: ("transition_type", "advanced_mode", "grey_blur", "resin_code", "intelli_mode")}


def _two_stages(height: float, up: float, down: float) -> tuple:
    """A one-stage lift as Photon Workshop's two stages: the same speeds, a
    short first stage - the motion is the same."""
    first = min(2.0, height / 2)
    return (first, up, down, height - first, up, down)


def _text(value: str, size: int) -> bytes:
    return value.encode("ascii", "replace")[:size - 1].ljust(size, b"\0")


def _rgb_bytes(color) -> bytes:
    return bytes(int(max(0.0, min(float(c), 1.0)) * 255) for c in tuple(color)[:3])


def _screens(printer: PrinterProfile, mt: dict) -> tuple:
    """Version 518: the screen count and 3 slots of (x, y, width, height)."""
    screens = [s for s in (mt.get("child_screen") or []) if isinstance(s, dict)][:3] or \
        [{"x": 0, "y": 0, "width": printer.res_x, "height": printer.res_y}]
    slots = []
    for i in range(3):
        sc = screens[i] if i < len(screens) else {}
        slots += [int(sc.get(k, 0)) & 0xFFFF for k in ("x", "y", "width", "height")]
    return (len(screens), *slots)


def write_pwbinary(path: str | Path, result: SliceResult, preview_meshes: list[trimesh.Trimesh],
                   progress: ProgressFn | None = None) -> Path:
    """Write the slice result in the printer's Photon Workshop binary file
    (version 1, 515, 516, 517 or 518 - whichever the printer reads; pwsImg layers
    for the Photon and Photon S).  ``preview_meshes`` are the world-space
    meshes for the thumbnails."""
    path = Path(path)
    printer, resin, layers = result.printer, result.resin, result.layers
    if not supports(printer):
        raise ValueError(f"{printer.name}: Photon Workshop file version {printer.file_version} "
                         f"({printer.layer_format}) is not one OpenVat writes")
    version = printer.file_version
    pws = printer.layer_format == PWS_FORMAT
    total = len(layers) + 2
    # pwsImg holds black and white only
    aa = 1 if pws else int(max(1, min(resin.anti_aliasing, 16)))
    turn = turned(printer)

    blobs, lit = [], []
    for i, layer in enumerate(layers):
        r0, band = rasterize_band(layer.geometry, printer, aa, turn)
        if not len(band):
            r0, band = 0, np.zeros((1, printer.res_x), np.uint8)
        blobs.append(encode_pws_band(band, r0, printer.res_y) if pws else encode_pw0_band(band, r0, printer.res_y))
        lit.append(int(np.count_nonzero(band > 7 if pws else band)))
        if progress and i % 10 == 0:
            progress(i, total)

    bottom_n = min(resin.bottom_layers, len(layers))
    normal = layers[bottom_n] if len(layers) > bottom_n else (layers[-1] if layers else None)
    lift_h = normal.lift_distance if normal else resin.normal.lift_distance
    lift_v = normal.lift_speed if normal else resin.normal.lift_speed
    retract_v = normal.retract_speed if normal else resin.normal.retract_speed

    def expected(i: int) -> tuple[float, float, float]:
        """exposure, lift height, lift speed the header (and EXTRA) imply"""
        if i < bottom_n:
            if version >= 516:              # EXTRA has the bottom lift
                return resin.bottom.exposure, resin.bottom.lift_distance, resin.bottom.lift_speed
            return resin.bottom.exposure, lift_h, lift_v
        return resin.normal.exposure, lift_h, lift_v

    # per-layer values only count when the header says so: needed when they
    # differ from what the header implies (transition layers, other lifts)
    plain = all(np.allclose((l.exposure, l.lift_distance, l.lift_speed), expected(i), atol=1e-6)
                for i, l in enumerate(layers))
    volume_ml = result.volume_mm3() / 1000.0
    common = (printer.pixel_x_um, resin.normal.thickness, resin.normal.exposure, resin.normal.off_time,
              resin.bottom.exposure, float(bottom_n), lift_h, lift_v, retract_v, volume_ml,
              aa, printer.res_x, printer.res_y, volume_ml * resin.density,
              volume_ml / 1000.0 * resin.price_per_liter, ord(resin.currency[0]) if resin.currency else ord("$"),
              0 if plain else 1, int(round(result.print_time_s())), resin.transition_layers)
    tail = {1: (HEADER_TAIL_V1,), 515: (0,), 516: (0, 0), 517: (0, 0, 0, RESIN_CODE),
            518: (0, 0, 0, RESIN_CODE, INTELLI_MODE)}[version]

    mt = printer.photonworkshop.get("machine_type", {})
    size = tuple(mt.get("prev_image_size") or PREVIEW_SIZE)
    back = tuple(mt.get("prev_back_color") or DEFAULT_PREVIEW_COLORS[0])
    model_color = tuple(mt.get("prev_model_color") or DEFAULT_PREVIEW_COLORS[1])
    preview = encode_preview(render_preview(preview_meshes, size, background=back, model_color=model_color))
    if progress:
        progress(len(layers) + 1, total)

    f = PwBinary(version, common + tail, size, preview,
                 layers=[(len(blob), l.lift_distance, l.lift_speed, l.exposure, l.thickness, n, 0)
                         for l, blob, n in zip(layers, blobs, lit)],
                 blobs=blobs)
    if version >= 515:
        f.color_table = color_table(aa)
    if version >= 516:
        b, n = resin.bottom, resin.normal
        f.extra = (2, *_two_stages(b.lift_distance, b.lift_speed, b.retract_speed),
                   2, *_two_stages(n.lift_distance, n.lift_speed, n.retract_speed))
        background = struct.unpack("<I", _rgb_bytes(back) + b"\0")[0]
        f.machine = (_text(mt.get("name") or printer.name, 96), _text(LAYER_FORMAT, 16),
                     int(mt.get("max_samples") or 16), int(mt.get("property") or 0),
                     float(mt.get("print_xsize") or printer.print_x), float(mt.get("print_ysize") or printer.print_y),
                     float(mt.get("print_zsize") or printer.print_z), version, background)
        if version >= 518:
            f.machine += (printer.pixel_x_um, printer.pixel_y_um, b"\0" * 32, *_screens(printer, mt))
    if version >= 517:
        f.software = (_text(SOFTWARE_MARK.decode(), 32), SOFTWARE.size, _text(f"OpenVat {__version__}", 32),
                      _text("OpenVat", 64), _text("3.3-CoreProfile", 32))
        lo, hi = np.asarray(result.model_bounds, dtype=float)
        f.model = (*lo, *hi, 0, 0)
    if version >= 518:
        f.subimages = (1, [])
        size2 = tuple(mt.get("prev2_image_size") or PREVIEW2_SIZE)
        back2 = tuple(mt.get("prev2_back_color") or DEFAULT_PREVIEW2_COLORS[0])
        img2 = render_preview(preview_meshes, size2, background=back2, model_color=DEFAULT_PREVIEW2_COLORS[1])
        r, g, bl = _rgb_bytes(back2)
        f.preview2 = (size2[0], ((r >> 3) << 11) | ((g >> 2) << 5) | (bl >> 3), size2[1], encode_preview(img2))
    path.write_bytes(f.build())
    if progress:
        progress(total, total)
    return path


# --------------------------------------------------------------------------
# reader


@dataclass
class LayerEntry:
    address: int
    length: int
    lift_height: float
    lift_speed: float
    exposure: float
    thickness: float
    lit_pixels: int


@dataclass
class PwBinaryFile:
    raw: PwBinary
    header: dict
    preview: Image.Image
    layers: list[LayerEntry]
    data: bytes
    pws: bool = False                     # pwsImg layers (Photon, Photon S)

    @property
    def version(self) -> int:
        return self.raw.version

    @property
    def res(self) -> tuple[int, int]:
        return self.header["res_x"], self.header["res_y"]

    @property
    def machine_name(self) -> str:
        return self.raw.machine[0].split(b"\0")[0].decode(errors="replace") if self.raw.machine else ""

    def layer_image(self, i: int) -> np.ndarray:
        """Grey levels 0..15, shape (res_y, res_x)."""
        d = self.layers[i]
        blob = self.data[d.address:d.address + d.length]
        return decode_pws(blob, *self.res) if self.pws else decode_pw0(blob, *self.res)


def is_pwbinary(path: str | Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == MAGIC[:8]


def _looks_pws(blob: bytes, width: int, height: int) -> bool:
    """Version 1 files hold pwsImg (Photon, Photon S) or pw0Img (Photon X,
    Zero); only one of the two adds up to the screen."""
    b = np.frombuffer(blob, np.uint8)
    return int((b & 0x7F).astype(np.int64).sum()) == width * height


def read_pwbinary(path: str | Path) -> PwBinaryFile:
    data = Path(path).read_bytes()
    if data[:12] != MAGIC:
        raise ValueError("not a Photon Workshop binary file")
    version, _areas = struct.unpack_from("<II", data, 12)
    if version not in VERSIONS:
        raise ValueError(f"Photon Workshop file version {version}: OpenVat reads {', '.join(map(str, VERSIONS))}")
    addr = struct.unpack_from(f"<{ADDRESSES[version]}I", data, 20)
    header_at, preview_at, layerdef_at = addr[0], addr[2], addr[4]
    header_vals = struct.unpack_from(HEADER_COMMON + HEADER_TAIL[version], data, header_at + 16)
    header = dict(zip(HEADER_FIELDS + HEADER_MORE[version], header_vals))
    header["currency"] = chr(header["currency"])
    w, _x, h = struct.unpack_from("<III", data, preview_at + 16)
    preview_px = data[preview_at + 28:preview_at + 28 + w * h * 2]
    count = struct.unpack_from("<I", data, layerdef_at + 16)[0]
    entries = [LAYER_DEF.unpack_from(data, layerdef_at + 20 + LAYER_DEF.size * i) for i in range(count)]
    raw = PwBinary(version, header_vals, (w, h), preview_px,
                   layers=[e[1:] for e in entries], blobs=[data[e[0]:e[0] + e[1]] for e in entries])
    if version == 1:
        raw.spare = (addr[1], addr[3], addr[5])
    if version == 515:
        raw.spare = (addr[1], addr[5])
    if version == 516:
        raw.spare = (addr[1],)
    if version >= 515:
        raw.color_table = data[addr[3]:addr[3] + 28]
    if version >= 516:
        raw.extra = struct.unpack_from("<I6fI6f", data, addr[5] + 16)
        raw.machine = (MACHINE_518 if version >= 518 else MACHINE).unpack_from(data, addr[6] + 16)
    if version >= 517:
        raw.software = SOFTWARE.unpack_from(data, addr[1])
        raw.model = MODEL.unpack_from(data, addr[8] + 16)
    if version >= 518:
        _n, per_screen = struct.unpack_from("<II", data, addr[9] + 16)
        raw.subimages = (per_screen, [SUBIMG.unpack_from(data, addr[9] + 24 + SUBIMG.size * i)[3:]
                                      for i in range(count)])
        w2, back2, h2 = struct.unpack_from("<III", data, addr[10] + 16)
        raw.preview2 = (w2, back2, h2, data[addr[10] + 28:addr[10] + 28 + w2 * h2 * 2])
    layers = [LayerEntry(*e[:7]) for e in entries]
    pws = version == 1 and bool(raw.blobs) and _looks_pws(raw.blobs[0], header["res_x"], header["res_y"])
    return PwBinaryFile(raw, header, decode_preview(preview_px, w, h), layers, data, pws)
