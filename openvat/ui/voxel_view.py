"""Voxel preview page: the model rebuilt from its sliced layers.

One voxel = one printer pixel by one layer - exactly what the printer
cures.  Building the volume and its faces runs in a background thread,
once per slice; everything after that happens on the GPU:

* each face (a merged rectangle of voxel sides) is 16 bytes in a texture
  buffer - grid coordinates and a direction - and the vertex shader builds
  its two triangles from ``gl_VertexID`` (no vertex arrays at all), about a
  tenth of the memory a normal triangle mesh needs;
* faces are sorted into chunks per direction, and every frame only the
  chunks inside the view, facing the camera and below the layer cut are
  drawn, nearest first, in one ``glMultiDrawArrays`` call;
* faces overlap their neighbours by 2 % of a voxel, which closes the
  single-pixel cracks merged faces otherwise leave where they meet;
* "Show layers up to" moves a clip plane and draws a small cap over the
  cut, so dragging the slider costs a few milliseconds instead of a new
  mesh.
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt, Signal, QObject, QThread, Slot
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                               QSpinBox, QProgressBar)

from ..core.scene import Scene
from ..core.slicer import SliceResult
from ..core.voxels import build_volume, VoxelVolume, VoxelFaces, pack_faces
from .layer_view import PageButtons, separator, ISLAND_RED
from .view_cube import ViewCube
from .viewport3d import Viewport3D, GL_INFO, build_program, software_renderer

VOXEL_VERT = """
#version 330 core
uniform usamplerBuffer u_faces;   // per face: i0 | j0 << 16, i1 | j1 << 16, k0 | k1 << 16, direction
uniform samplerBuffer u_z;        // z (mm) of every layer edge
uniform mat4 u_vp;
uniform vec2 u_origin;            // world XY of grid corner (0, 0)
uniform vec2 u_voxel;             // voxel size in X and Y (mm)
uniform float u_zcut;             // show what lies below this height
uniform float u_seam;             // faces grow this much (mm) in their plane: no hairline gaps
flat out vec3 v_normal;
out vec3 v_world;
// two triangles per face, corners (0,0) (1,0) (1,1) (0,1) along the face's two axes
const vec2 UV[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));
void main() {
    int face = gl_VertexID / 6;
    vec2 uv = UV[gl_VertexID - face * 6];
    uvec4 d = texelFetch(u_faces, face);
    int dir = int(d.w);
    int axis = dir >> 1;                      // 0 x, 1 y, 2 z
    bool negative = (dir & 1) == 1;
    if (negative) uv = uv.yx;                 // reverse the winding: counter-clockwise seen from outside
    float z0 = texelFetch(u_z, int(d.z & 0xFFFFu)).r;
    float z1 = texelFetch(u_z, int(d.z >> 16)).r;
    vec3 lo = vec3(u_origin + vec2(float(d.x & 0xFFFFu), float(d.x >> 16)) * u_voxel, z0);
    vec3 hi = vec3(u_origin + vec2(float(d.y & 0xFFFFu), float(d.y >> 16)) * u_voxel, z1);
    ivec3 xyz = ivec3(0, 1, 2);
    vec3 ea = vec3(equal(ivec3(axis), xyz));
    vec3 eb = vec3(equal(ivec3((axis + 1) % 3), xyz));
    vec3 ec = vec3(equal(ivec3((axis + 2) % 3), xyz));
    // merged faces meet neighbours mid-edge (T-junctions), where rounding can leave
    // single-pixel holes; a tiny overlap of the coplanar, same-coloured faces closes them
    vec3 grow = (eb * (2.0 * uv.x - 1.0) + ec * (2.0 * uv.y - 1.0)) * u_seam;
    vec3 p = lo + (hi - lo) * (eb * uv.x + ec * uv.y) + grow;
    v_normal = negative ? -ea : ea;
    v_world = p;
    gl_Position = u_vp * vec4(p, 1.0);
    // the layer cut; faces on it that point down are the bottoms of the next layer
    gl_ClipDistance[0] = (dir == 5 && z0 >= u_zcut - 1e-4) ? -1.0 : u_zcut + 1e-4 - p.z;
}
"""

VOXEL_FRAG = """
#version 330 core
flat in vec3 v_normal;
in vec3 v_world;
uniform vec4 u_color;
uniform vec3 u_eye;
out vec4 frag;
void main() {
    vec3 n = v_normal;
    vec3 view = normalize(u_eye - v_world);
    vec3 l1 = normalize(vec3(0.4, -0.6, 1.0));
    vec3 l2 = normalize(vec3(-0.6, 0.5, 0.3));
    float diff = 0.30 + 0.55 * max(dot(n, l1), 0.0) + 0.25 * max(dot(n, l2), 0.0);
    vec3 h = normalize(l1 + view);
    float spec = pow(max(dot(n, h), 0.0), 32.0) * 0.15;
    frag = vec4(u_color.rgb * diff + spec, u_color.a);
}
"""

_UNIFORMS = ("u_faces", "u_z", "u_vp", "u_origin", "u_voxel", "u_zcut", "u_seam", "u_color", "u_eye")
SEAM = 0.02                       # overlap of neighbouring faces, in voxels


def _texture_buffer(data: np.ndarray, fmt) -> tuple[int, int]:
    """Upload ``data`` as a buffer texture -> (buffer, texture)."""
    buf = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, buf)
    GL.glBufferData(GL.GL_TEXTURE_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_BUFFER, tex)
    GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, fmt, buf)
    GL.glBindTexture(GL.GL_TEXTURE_BUFFER, 0)
    GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)
    return buf, tex


def _release(buffer_texture: tuple[int, int] | None) -> None:
    if buffer_texture is not None:
        GL.glDeleteTextures(1, [buffer_texture[1]])
        GL.glDeleteBuffers(1, [buffer_texture[0]])


class VoxelViewport(Viewport3D):
    """The build plate plus the voxel faces, drawn straight from the GPU."""

    def __init__(self, parent=None):
        super().__init__(Scene(), parent, interactive=False)
        self.volume: VoxelVolume | None = None
        self.faces: VoxelFaces | None = None
        self.z_cut = np.inf
        self.drawn = 0                             # faces drawn in the last frame
        self._cap = np.zeros((0, 7), dtype=np.int32)
        self._dirty = self._cap_dirty = False      # waiting to be uploaded
        self._prog = 0
        self._u: dict[str, int] = {}
        self._vao = 0
        self._parts: list[tuple[int, int, tuple[int, int]]] = []   # (first face, end, (buffer, texture))
        self._ztex: tuple[int, int] | None = None
        self._captex: tuple[int, int] | None = None

    # ------------------------------------------------------------ data
    def set_voxels(self, volume: VoxelVolume | None, faces: VoxelFaces | None) -> None:
        self.volume, self.faces = volume, faces
        self._dirty = True
        self.set_cut(volume.nz - 1 if volume is not None else 0)

    def set_cut(self, max_layer: int) -> None:
        """Show layers 0..max_layer."""
        vol = self.volume
        if vol is None:
            self.z_cut, self._cap = np.inf, np.zeros((0, 7), dtype=np.int32)
        else:
            max_layer = int(np.clip(max_layer, 0, vol.nz - 1))
            self.z_cut = float(vol.z_edges[max_layer + 1])
            self._cap = vol.cap(max_layer)
        self._cap_dirty = True
        self.update()

    def content_bounds(self) -> np.ndarray | None:
        f = self.faces
        if f is None or not len(f.start):
            return None
        return np.vstack([f.lo.min(axis=0), f.hi.max(axis=0)])

    # ------------------------------------------------------------ GL
    def initializeGL(self) -> None:
        super().initializeGL()
        self._prog = build_program(VOXEL_VERT, VOXEL_FRAG)
        self._u = {name: GL.glGetUniformLocation(self._prog, name) for name in _UNIFORMS}
        self._vao = GL.glGenVertexArrays(1)        # no attributes, but core profile wants one bound
        self._parts, self._ztex, self._captex = [], None, None
        self._dirty = self._cap_dirty = True       # a new context: upload again

    def _upload(self) -> None:
        for _p0, _p1, bt in self._parts:
            _release(bt)
        _release(self._ztex)
        self._parts, self._ztex = [], None
        self._dirty = False
        if self.faces is None or not len(self.faces):
            return
        packed = self.faces.packed()
        limit = max(int(GL_INFO.get("max_texture_buffer", 65536)), 65536)
        for p0 in range(0, len(packed), limit):     # one texture holds everything on any real GPU
            part = np.ascontiguousarray(packed[p0:p0 + limit])
            self._parts.append((p0, p0 + len(part), _texture_buffer(part, GL.GL_RGBA32UI)))
        self._ztex = _texture_buffer(self.volume.z_edges.astype(np.float32), GL.GL_R32F)

    def _upload_cap(self) -> None:
        _release(self._captex)
        self._captex = None
        self._cap_dirty = False
        if len(self._cap):
            self._captex = _texture_buffer(pack_faces(self._cap), GL.GL_RGBA32UI)

    def _draw_contents(self, vp: np.ndarray) -> None:
        if self._dirty:
            self._upload()
        if self._cap_dirty:
            self._upload_cap()
        self.drawn = 0
        if not self._parts or self._ztex is None:
            return
        vol, u = self.volume, self._u
        eye = self.camera.position()
        GL.glUseProgram(self._prog)
        GL.glUniformMatrix4fv(u["u_vp"], 1, GL.GL_TRUE, vp.astype(np.float32))
        GL.glUniform3fv(u["u_eye"], 1, eye.astype(np.float32))
        GL.glUniform4f(u["u_color"], *self.colors["model"])
        GL.glUniform2f(u["u_origin"], float(vol.origin[0]), float(vol.origin[1]))
        GL.glUniform2f(u["u_voxel"], float(vol.vx), float(vol.vy))
        GL.glUniform1f(u["u_zcut"], float(min(self.z_cut, 1e9)))
        GL.glUniform1f(u["u_seam"], float(SEAM * min(vol.vx, vol.vy)))
        GL.glUniform1i(u["u_faces"], 0)
        GL.glUniform1i(u["u_z"], 1)
        GL.glBindVertexArray(self._vao)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glEnable(GL.GL_CLIP_DISTANCE0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self._ztex[1])
        GL.glActiveTexture(GL.GL_TEXTURE0)

        groups = self.faces.visible(eye, vp, self.z_cut)
        starts, ends = self.faces.start[groups], self.faces.end[groups]
        for p0, p1, (_buf, tex) in self._parts:
            s, e = np.maximum(starts, p0), np.minimum(ends, p1)
            keep = e > s
            if not keep.any():
                continue
            s, e = s[keep], e[keep]
            GL.glBindTexture(GL.GL_TEXTURE_BUFFER, tex)
            firsts = np.ascontiguousarray((s - p0) * 6, dtype=np.int32)
            counts = np.ascontiguousarray((e - s) * 6, dtype=np.int32)
            GL.glMultiDrawArrays(GL.GL_TRIANGLES, firsts, counts, len(firsts))
            self.drawn += int((e - s).sum())
        if self._captex is not None:
            GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self._captex[1])
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6 * len(self._cap))
            self.drawn += len(self._cap)

        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, 0)
        GL.glDisable(GL.GL_CLIP_DISTANCE0)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindVertexArray(0)


class _Worker(QObject):
    """Builds the volume (if needed) and its faces off the GUI thread."""
    progress = Signal(str, int, int)
    finished = Signal(object, object, object)     # (SliceResult, VoxelVolume, VoxelFaces)
    failed = Signal(str)

    def __init__(self, result, volume):
        super().__init__()
        self.result, self.volume = result, volume

    @Slot()
    def run(self) -> None:
        try:
            vol = self.volume or build_volume(self.result, 1, lambda d, t: self.progress.emit("Layers", d, t))
            faces = VoxelFaces.build(vol.faces(progress=lambda d, t: self.progress.emit("Faces", d, t)), vol)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(self.result, vol, faces)


class VoxelViewPage(QWidget):
    back_requested = Signal()
    layers_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: SliceResult | None = None
        self.volume: VoxelVolume | None = None
        self._thread: QThread | None = None
        self._again = False                        # build again when the running build ends
        self._fit_after = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 0)

        body = QHBoxLayout()
        self.viewport = VoxelViewport()
        self.view_cube = ViewCube(self.viewport.camera, self.viewport)
        self.view_cube.view_direction.connect(self._look_from)
        self.view_cube.orbit_requested.connect(self._orbit_by)
        self.view_cube.home_requested.connect(self._home)
        self.viewport.overlay = self.view_cube
        self.viewport.installEventFilter(self)
        self.viewport.gl_started.connect(self._update_summary)
        body.addWidget(self.viewport, stretch=1)
        self.slider = QSlider(Qt.Vertical)
        body.addWidget(self.slider)
        root.addLayout(body, stretch=1)

        # the bottom bar: layers shown | cut height | the voxels (or progress) | warnings, then buttons
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("Show layers up to"))
        self.spin = QSpinBox()
        bottom.addWidget(self.spin)
        self.layer_max = QLabel("/ 0")
        bottom.addWidget(self.layer_max)
        bottom.addWidget(separator())
        self.layer_info = QLabel("")                    # z of the cut
        bottom.addWidget(self.layer_info)
        bottom.addWidget(separator())
        self.summary = QLabel("")                       # voxels · solid · faces
        bottom.addWidget(self.summary)
        self.busy = QProgressBar()
        self.busy.setMaximumWidth(200)
        self.busy.setVisible(False)
        bottom.addWidget(self.busy)
        self.slow_sep = separator()
        bottom.addWidget(self.slow_sep)
        self.slow = QLabel("")
        self.slow.setStyleSheet(f"color: {ISLAND_RED}; font-weight: bold;")
        bottom.addWidget(self.slow)
        bottom.addLayout(PageButtons([("← Back to 3D view", self.back_requested),
                                      ("← Layers", self.layers_requested),
                                      ("Export…", self.export_requested)]), stretch=1)
        root.addLayout(bottom)

        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self._layer_changed)

    def eventFilter(self, obj, event):
        if obj is self.viewport and event.type() == event.Type.Resize:
            self.view_cube.move(self.viewport.width() - self.view_cube.width() - 6, 6)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------ data
    def set_result(self, result: SliceResult) -> None:
        self.result = result
        self.volume = None
        self.viewport.set_printer(result.printer)
        self.viewport.set_voxels(None, None)
        n = len(result.layers)
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.spin.setRange(0, n - 1)
        self.layer_max.setText(f"/ {n - 1}")
        self.layer_info.setText(f"z ≤ {result.layers[-1].z_top:.2f} mm")
        self.slider.setValue(n - 1)
        self.spin.setValue(n - 1)
        self.slider.blockSignals(False)
        self._fit_after = True
        self.summary.setText("")
        self._build()

    def _build(self) -> None:
        if self.result is None:
            return
        if self._thread is not None and self._thread.isRunning():
            self._again = True                     # sliced again while building: next when this one ends
            return
        self.busy.setVisible(True)
        self.busy.setRange(0, 100)
        self.busy.setValue(0)
        self._thread = QThread(self)
        self._worker = _Worker(self.result, self.volume)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._done, Qt.QueuedConnection)
        self._worker.failed.connect(self._failed, Qt.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_done)
        self._thread.start()

    @Slot()
    def _thread_done(self) -> None:
        if self._again:
            self._again = False
            self._build()

    def _layer_changed(self, value: int) -> None:
        if self.result is not None:
            layer = self.result.layers[value]
            self.layer_info.setText(f"z ≤ {layer.z_top:.2f} mm")
        self.viewport.set_cut(value)              # a clip plane and a small cap: no new mesh
        self._update_summary()

    @Slot(str, int, int)
    def _progress(self, phase: str, done: int, total: int) -> None:
        self.busy.setFormat(f"{phase} %p%")
        self.busy.setValue(int(100 * done / max(total, 1)))

    @Slot(object, object, object)
    def _done(self, result: SliceResult, volume: VoxelVolume, faces: VoxelFaces) -> None:
        if result is not self.result:              # sliced again meanwhile: _thread_done starts over
            self._again = True
            return
        self.busy.setVisible(False)
        self.volume = volume
        self.viewport.set_voxels(volume, faces)
        self.viewport.set_cut(self.slider.value())
        if self._fit_after:
            self._fit_after = False
            self.viewport.fit_view()
        self._update_summary()

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.busy.setVisible(False)
        self.summary.setText(f"Voxel preview failed: {message}")

    def _update_summary(self) -> None:
        vol, faces = self.volume, self.viewport.faces
        if vol is not None and faces is not None:
            self.summary.setText(f"{vol.nx} × {vol.ny} × {vol.nz} voxels of {vol.vx * 1000:.1f} × "
                                 f"{vol.vy * 1000:.1f} µm · {vol.solid_up_to(self.slider.value()):,} solid · "
                                 f"{len(faces):,} faces")
        slow = software_renderer()
        self.slow_sep.setVisible(bool(slow))
        self.slow.setVisible(bool(slow))
        self.slow.setText("⚠ software OpenGL" if slow else "")
        self.slow.setToolTip(f"OpenGL runs in software ({slow}): the view will be slow - check the "
                             "graphics driver (Help → About)" if slow else "")

    # ------------------------------------------------------------ camera
    def _home(self) -> None:
        self.viewport.camera.set_view("iso")
        self.viewport.fit_view()

    def _look_from(self, direction) -> None:
        self.viewport.camera.look_from(direction)
        self.viewport.update()

    def _orbit_by(self, dyaw: float, dpitch: float) -> None:
        self.viewport.camera.orbit_by(dyaw, dpitch)
        self.viewport.update()
