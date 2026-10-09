# Anycubic `.pwsz` file format

Reverse-engineered from a PhotonWorkshop 4.2 export for the Photon Mono M7 Pro
(`01_xyzCalibration_cube.pwsz`, 400 layers).  Everything below was verified
against that file; where a field is a best guess it says so.

A `.pwsz` is an ordinary **ZIP archive** (deflate).  Unlike the older
`.pwmx/.pwma/.pm3` formats, layers are **not bitmaps**: each layer is a list
of 2D line segments (the polygon outline) in millimeters and the printer
rasterizes them itself.  This is why files are tiny.

All binary values are **little-endian**; floats are IEEE-754 single precision.
All coordinates are millimeters with the origin at the **center of the build
plate**, X along the long axis of the screen, Y along the short axis.

```
01_xyzCalibration_cube.pwsz
├── anycubic_photon_resins.pwsp   JSON   printer + resin profiles
├── layers_controller.conf        JSON   per-layer exposure / thickness / lift
├── lcd_function.json             JSON   list of models, RERF settings
├── print_info.json               JSON   time / volume / weight / cost
├── software_info.conf            JSON   slicer version
├── scene.slice                   binary per-layer summary table
├── calc_layer_volumes.data       binary volume per ~0.2 mm slab
├── layer_images/
│   ├── layer_0.pwszImg           binary outline of layer 0
│   └── layer_N.pwszImg ...
└── preview_images/
    ├── preview_0.png             224 x 168  RGBA
    ├── preview_1.png             336 x 252  RGBA
    └── preview_2.png             800 x 600  RGBA
```

## `layer_images/layer_N.pwszImg`

```
offset  size  type    meaning
0       4     char    "{==\0"                    magic
4       4     f32     2 * area of the layer (mm²) – the raw shoelace sum
8       16    4×f32   bounding box: xmin, ymin, xmax, ymax (mm)
24      4     i32     0  (unknown, always 0)
28      4     i32     number of closed contours (outer rings + holes)
32      4     char    "[--\0"                    start of segment block
36      4     i32     0  (unknown, always 0)
40      4     i32     segment count S
44      4     i32     1  (unknown, always 1)
48      17*S         segments, each:
                        f32 x0, f32 y0, f32 x1, f32 y1, u8 flag
...     4     char    "--]\0"                    end of segment block
...     4     char    "==}\0"                    end of file
```

The segments form an **edge table for a scanline fill** (verified on 480
PhotonWorkshop layers from two different models):

* every edge is stored **bottom-to-top**: `y0 <= y1`, always;
* the **flag** says which way the outline originally ran: `1` = from
  `(x0,y0)` to `(x1,y1)`, `0` = the other way (the edge was flipped for
  storage).  Outer boundaries run counter-clockwise and holes clockwise, so
  `sum(sign * (x0*y1 - x1*y0))` with `sign = +1/-1` for flag `1/0` equals the
  `2 * area` field in the header exactly;
* horizontal edges keep their original direction and always have flag `1`;
* edges are **sorted by `y0`** (a stable sort: ties keep outline order).

A printer fills a row at height `y` from the edges with `y0 <= y < y1`,
adding `+1` / `-1` winding per edge according to the flag.  Edges stored
top-to-bottom never satisfy `y0 <= y`, so they are silently ignored: an
earlier OpenVat build stored rings in drawing order (half the edges pointing
down, all flags `1`) and its layers filled to nothing - an empty build plate.
`openvat.formats.pwsz.rasterize_layer_image` emulates this fill; for
PhotonWorkshop files it reproduces the pixel area PhotonWorkshop stores in
`scene.slice`.

Only one `[--` block has ever been observed per layer, even for layers with
several contours.

## `scene.slice`

```
offset  size   meaning
0x000   16     "ANYCUBIC-PWSZ" zero padded
0x010   64     "AnycubicPhotonWorkshop Exports" zero padded
0x050   16     i32 3, i32 3, i32 0, i32 0        (version?)
0x060   4      f32 1.0                            (scale?)
0x064   4      i32 layer count
0x068   24     6×f32 model bounding box: xmin ymin zmin xmax ymax zmax
               (unscaled model, before shrinkage compensation)
0x080   260    zero padding
0x184   4      "<---"
0x188   4      i32 layer count (again)
0x18C   64×N   one record per layer (below)
end     4      "--->"
```

Layer record (64 bytes):

```
f32  z center of the layer (z_bottom + thickness/2)
f32  area (mm²) – PhotonWorkshop stores the *rasterized* pixel area here
4×f32 bounding box xmin ymin xmax ymax
i32  contour count
i32  0
f32  exact area – only non-zero on the last layer in the sample, 0 otherwise
28 bytes zero
```

## `calc_layer_volumes.data`

```
0    4   "$\x10% "          magic
4    4   f32 0.2            slab height
8    4   i32 1
12   4   i32 record count R
16   48×R records:
           i32  index of the first layer in the slab  (= int(z / thickness) in float32)
           f32  z start of slab (mm)
           f32  slab height (0.2 or 0.25 mm in the sample)
           8×f32 zeros
           f32  resin volume in the slab (mm³)
```

The sample mixes 0.2 and 0.25 mm slabs in an irregular pattern; OpenVat
writes uniform 0.2 mm slabs.  The values are only used for estimates.

## `layers_controller.conf`

```json
{"count": 400,
 "paras": [{"exposure_time": 20.0, "layer_index": 0, "layer_minheight": 0.0,
            "layer_thickness": 0.05, "zup_height": 8.0, "zup_speed": 6.0}, ...]}
```

`layer_minheight` is the bottom of the layer.  Per-layer thickness is allowed,
so bottom layers may be thicker than normal ones.

## `anycubic_photon_resins.pwsp`

A large JSON file.  The important parts:

* `machine_type` – resolution (`res_x`, `res_y`), pixel pitch (`xy_pixel`,
  `xy_pixel_y`, µm), build volume (`print_xsize/ysize/zsize`), preview sizes
  and colors, `raster_antialiasing`.
* `machine_extern.user_resins[]` – resin profiles.  `slicepara` holds the
  classic settings (`zthick`, `exposure_time`, `bott_layers`, `bott_time`,
  `off_time`, `zup_height`, `zup_speed`, `zdown_speed`, `wait_before_lift`,
  `wait_after_lift`, `bott_off_time`, `bott_wait_before_lift`,
  `bott_wait_after_lift`, `anti_count`).  `slice_extpara` holds the
  transition layer count and `material_scale_xyz` (shrinkage compensation).
* `machine_extern.active_resins` – the name of the resin in use.
* `firmware_calc_*` – constants for the printer's own time estimate; OpenVat
  copies the M7 Pro values.

### Shrinkage compensation

The active resin's `material_scale_xyz` is baked into the layers.  X/Y are
scaled about each model's own center (a 20 mm cube with 1.0111 is 20.222 mm
wide in the layer files).  Z is scaled up from the build plate: with 1.055 a
step at 2.0 mm moves to 2.11 mm (between layers 41 and 42 at 0.05 mm).
PhotonWorkshop 4.2 then still writes only as many layers as the *unscaled*
model needs, so the top ~5 % of the scaled model is never printed.  OpenVat
writes all layers of the scaled model instead.

## `print_info.json`, `lcd_function.json`, `software_info.conf`

Plain JSON; see `openvat/formats/pwsz.py` for the exact keys.  `volume` is in
milliliters, `print_time` in seconds, `weight` in grams.
