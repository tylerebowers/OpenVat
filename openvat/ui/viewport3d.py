"""The 3D build-plate view (QOpenGLWidget + PyOpenGL, OpenGL 3.3 core).

Responsibilities:
    * draw the build plate, the objects and the supports
    * orbit / pan / zoom with the mouse
    * pick objects and supports with the mouse and drag them across the plate
    * "add support" mode: clicking on a model adds a support at that point

Mouse: left = select / drag, right = orbit, middle (or shift+left) = pan,
wheel = zoom.
"""

from __future__ import annotations

import ctypes
import os
import sys

import numpy as np
import trimesh
import OpenGL
# PyOpenGL asks the driver for errors after every call unless told not to:
# that doubles the cost of each call.  OPENVAT_GL_DEBUG=1 turns it back on.
OpenGL.ERROR_CHECKING = bool(os.environ.get("OPENVAT_GL_DEBUG"))
from OpenGL import GL  # noqa: E402  (after the flag above)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget, QApplication

from ..core.hollow import hole_world, hole_depth, lattice_segments, cavity_clip, tubes
from ..core.mesh import MeshObject
from ..core.profiles import PrinterProfile
from ..core.raycast import ray_mesh_closest, ray_mesh_first
from ..core.scene import Scene
from ..core.placement import snap_support
from ..core.supports import Support
from .camera import OrbitCamera, ray_plane_z
from .glplatform import default_surface_format  # noqa: F401  (re-exported)
from .theme_colors import DEFAULT_THEME_COLORS, DEFAULT_COLORS  # noqa: F401

VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_world;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = mat3(u_model) * in_normal;
    v_world = (u_model * vec4(in_pos, 1.0)).xyz;
}
"""

FRAG_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_world;
uniform vec4 u_color;
uniform vec3 u_eye;
uniform int u_lit;
out vec4 frag;
void main() {
    if (u_lit == 0) { frag = u_color; return; }
    vec3 n = normalize(v_normal);
    vec3 view = normalize(u_eye - v_world);
    if (dot(n, view) < 0.0) n = -n;           // light back faces too
    vec3 l1 = normalize(vec3(0.4, -0.6, 1.0));
    vec3 l2 = normalize(vec3(-0.6, 0.5, 0.3));
    float diff = 0.30 + 0.55 * max(dot(n, l1), 0.0) + 0.25 * max(dot(n, l2), 0.0);
    vec3 h = normalize(l1 + view);
    float spec = pow(max(dot(n, h), 0.0), 32.0) * 0.15;
    frag = vec4(u_color.rgb * diff + spec, u_color.a);
}
"""

COLOR_OUTSIDE = (0.90, 0.30, 0.30, 1.0)
COLOR_HOLE = (0.93, 0.36, 0.26, 1.0)       # drain holes
XRAY_ALPHA = 0.28                          # hollow models with "Show inside"
HOLE_MARK_OUT = 0.3                        # mm a hole marker stands out of the surface
PLATE_ALPHA = 0.55                 # the build plate is see-through (models show from below)


def hex_to_rgba(value: str, alpha: float = 1.0) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (alpha,)


# --------------------------------------------------------------------------
# which OpenGL implementation draws the views

GL_INFO: dict = {}           # vendor, renderer, version - filled when the first view starts

# renderers that run on the CPU (Mesa, Windows' fallbacks, Google's, Apple's)
_SOFTWARE = ("llvmpipe", "softpipe", "swrast", "software rasterizer", "gdi generic",
             "microsoft basic render", "swiftshader", "lavapipe", "apple software renderer")


def software_renderer() -> str:
    """The renderer's name if OpenGL runs in software (the views will be
    slow), otherwise an empty string."""
    name = GL_INFO.get("renderer", "")
    return name if any(s in name.lower() for s in _SOFTWARE) else ""


def gl_summary() -> str:
    if not GL_INFO:
        return "not started yet"
    return f"{GL_INFO['renderer']} ({GL_INFO['vendor']}), OpenGL {GL_INFO['version']}"


def probe_gl() -> None:
    """Ask for an OpenGL context before any window exists, to learn which
    renderer this machine gives us (fills GL_INFO).  On a software renderer,
    multisampling is switched off: it more than doubles the cost of every
    frame there.  Never fails - the views check again when they start."""
    try:
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
        ctx = QOpenGLContext()
        ctx.setFormat(QSurfaceFormat.defaultFormat())
        if not ctx.create():
            return
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if surface.isValid() and ctx.makeCurrent(surface):
            try:
                _read_gl_info()
            finally:
                ctx.doneCurrent()
        surface.destroy()
    except Exception as exc:                                    # pragma: no cover
        print(f"OpenVat: could not probe OpenGL ({exc})", file=sys.stderr)
        return
    if software_renderer():
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setSamples(0)
        QSurfaceFormat.setDefaultFormat(fmt)


def _read_gl_info() -> None:
    def text(name) -> str:
        try:
            value = GL.glGetString(name)
            return value.decode(errors="replace") if value else "?"
        except Exception:
            return "?"
    GL_INFO.update(vendor=text(GL.GL_VENDOR), renderer=text(GL.GL_RENDERER), version=text(GL.GL_VERSION))
    try:
        GL_INFO["max_texture_buffer"] = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_BUFFER_SIZE))
    except Exception:
        GL_INFO["max_texture_buffer"] = 65536                      # the minimum OpenGL 3.3 promises
    print(f"OpenVat: OpenGL renderer {gl_summary()}"
          + (" - software rendering, the 3D views will be slow" if software_renderer() else ""),
          file=sys.stderr)


def build_program(vertex_src: str, fragment_src: str) -> int:
    """Compile and link a shader program (raises with the GL log on errors)."""
    def compile_shader(src: str, kind) -> int:
        sh = GL.glCreateShader(kind)
        GL.glShaderSource(sh, src)
        GL.glCompileShader(sh)
        if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(sh).decode())
        return sh

    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, compile_shader(vertex_src, GL.GL_VERTEX_SHADER))
    GL.glAttachShader(prog, compile_shader(fragment_src, GL.GL_FRAGMENT_SHADER))
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
    return prog


class GLBuffer:
    """A vertex array on the GPU.  Each vertex is (x, y, z, nx, ny, nz)."""

    def __init__(self, data: np.ndarray, mode=GL.GL_TRIANGLES):
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, 6)
        self.count = len(data)
        self.mode = mode
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        stride = 24
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)

    @classmethod
    def triangles(cls, vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> "GLBuffer":
        """De-indexed triangles with flat (per-face) normals."""
        tri = np.asarray(vertices)[np.asarray(faces)].reshape(-1, 3)
        nrm = np.repeat(np.asarray(face_normals), 3, axis=0)
        return cls(np.hstack([tri, nrm]))

    @classmethod
    def from_trimesh(cls, m: trimesh.Trimesh) -> "GLBuffer":
        return cls.triangles(m.vertices, m.faces, m.face_normals)

    @classmethod
    def from_trimeshes(cls, meshes) -> "GLBuffer | None":
        """All meshes in one buffer: one draw call instead of one per mesh."""
        parts = []
        for m in meshes:
            if len(m.faces):
                tri = np.asarray(m.vertices, dtype=np.float32)[np.asarray(m.faces)].reshape(-1, 3)
                parts.append(np.hstack([tri, np.repeat(np.asarray(m.face_normals, dtype=np.float32), 3, axis=0)]))
        return cls(np.vstack(parts)) if parts else None

    @classmethod
    def lines(cls, points: np.ndarray) -> "GLBuffer":
        """Line list: consecutive point pairs."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return cls(np.hstack([pts, np.zeros_like(pts)]), mode=GL.GL_LINES)

    def draw(self) -> None:
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(self.mode, 0, self.count)
        GL.glBindVertexArray(0)

    def release(self) -> None:
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteVertexArrays(1, [self.vao])


class Viewport3D(QOpenGLWidget):
    selection_changed = Signal()
    object_moved = Signal()
    support_requested = Signal(object, object)   # (MeshObject, hit point)
    hole_requested = Signal(object, object, object)   # (MeshObject, hit point, outward face normal)
    gl_started = Signal()                        # OpenGL is up; GL_INFO is filled

    def __init__(self, scene: Scene, parent=None, interactive: bool = True):
        super().__init__(parent)
        self.scene = scene
        self.interactive = interactive          # False: camera only, no picking
        self.cull_back_faces = False            # skip faces pointing away (closed, outward-wound meshes)
        self.camera = OrbitCamera()
        self.colors = {k: hex_to_rgba(v) for k, v in DEFAULT_COLORS.items()}
        self.printer: PrinterProfile = PrinterProfile()
        self.add_support_mode = False
        self.add_hole_mode = False              # clicks on a model place / remove drain holes
        self.show_inside = False                # hollow models see-through with their internal supports
        self._hole_buffers: dict[int, tuple[tuple, GLBuffer | None]] = {}
        self._lattice_buffers: dict[int, tuple[tuple, GLBuffer | None, np.ndarray]] = {}
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(False)
        self.setAcceptDrops(False)

        self._program = 0
        self._uniforms: dict[str, int] = {}
        self._meshes: dict[int, tuple[tuple, GLBuffer, bool]] = {}   # obj id -> (cache key, buffer, closed)
        self._support_meshes: dict[int, tuple[tuple, GLBuffer]] = {}
        self._extra_buffer: GLBuffer | None = None               # all supports, braces and raft
        self._extras_version = -1
        self._plate: GLBuffer | None = None
        self._plate_fill: GLBuffer | None = None
        self._plate_key = None

        self._last_pos = QPoint()
        self._drag_target: MeshObject | Support | None = None
        self._drag_plane_z = 0.0
        self._drag_offset = np.zeros(3)
        self._pressed_button = None
        self._moved = False
        self._undo_pushed = False
        self._overlay: QWidget | None = None

        scene.listeners.append(self.update)

    @property
    def overlay(self) -> QWidget | None:
        """The navigation cube: a child widget that takes the mouse, painted by
        us into each frame (see view_cube)."""
        return self._overlay

    @overlay.setter
    def overlay(self, widget: QWidget | None) -> None:
        self._overlay = widget
        if widget is not None and hasattr(widget, "paint_on"):
            widget.drawn_by_view = True

    # ------------------------------------------------------------ printer
    def set_printer(self, printer: PrinterProfile) -> None:
        self.printer = printer
        self._plate_key = None
        self.update()

    def set_colors(self, colors: dict) -> None:
        """Change the user colors (#rrggbb by name, see DEFAULT_THEME_COLORS)."""
        for key, value in colors.items():
            if key in DEFAULT_COLORS:
                try:
                    self.colors[key] = hex_to_rgba(value)
                except (ValueError, TypeError):
                    pass                                   # a bad value keeps the old color
        self.update()

    def _plate_colors(self) -> tuple[tuple, tuple]:
        """See-through plate fill, and grid lines a step lighter (dark plate)
        or darker (light plate) than it."""
        r, g, b, _a = self.colors["plate"]
        toward = 1.0 if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else 0.0
        grid = tuple(c + (toward - c) * 0.28 for c in (r, g, b)) + (0.9,)
        return (r, g, b, PLATE_ALPHA), grid

    def content_bounds(self) -> np.ndarray | None:
        """What "fit" frames (subclasses drawing something else override this)."""
        return self.scene.bounds()

    def fit_view(self) -> None:
        self.camera.fit(self.content_bounds(), (self.printer.print_x, self.printer.print_y))
        self.update()

    def set_view(self, name: str) -> None:
        self.camera.set_view(name)
        self.update()

    # ------------------------------------------------------------ GL setup
    def initializeGL(self) -> None:
        if not GL_INFO:
            _read_gl_info()
        self._program = build_program(VERT_SHADER, FRAG_SHADER)
        for name in ("u_mvp", "u_model", "u_color", "u_eye", "u_lit"):
            self._uniforms[name] = GL.glGetUniformLocation(self._program, name)
        self.gl_started.emit()

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, w, h)
        self.camera.aspect = w / max(h, 1)

    # ------------------------------------------------------------ drawing
    @staticmethod
    def _set_gl_state() -> None:
        """Our drawing state - set every frame, because the navigation cube's
        QPainter leaves its own behind."""
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glFrontFace(GL.GL_CCW)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glEnable(GL.GL_BLEND)
        # blend colour, but keep the framebuffer opaque (alpha 1) where the
        # see-through plate is drawn, or the window could show through it
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_MULTISAMPLE)

    def paintGL(self) -> None:
        self._set_gl_state()
        GL.glClearColor(*self.colors["background"])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self._program)
        vp = self.camera.projection_matrix() @ self.camera.view_matrix()
        GL.glUniform3fv(self._uniforms["u_eye"], 1, self.camera.position().astype(np.float32))

        self._draw_contents(vp)
        # the plate goes last: it is see-through, so whatever is below it (seen
        # from underneath) must already be drawn; it writes no depth itself
        GL.glUseProgram(self._program)
        self._sync_plate()
        fill, grid = self._plate_colors()
        GL.glDepthMask(GL.GL_FALSE)
        self._draw(self._plate_fill, np.eye(4), vp, fill, lit=False)
        GL.glDepthMask(GL.GL_TRUE)
        self._draw(self._plate, np.eye(4), vp, grid, lit=False)
        GL.glUseProgram(0)
        GL.glBindVertexArray(0)
        cube = self._overlay
        if cube is not None and getattr(cube, "drawn_by_view", False) and cube.isVisible():
            p = QPainter(self)                    # into this frame: no raster widget to repaint
            p.translate(cube.pos())
            cube.paint_on(p)
            p.end()

    def _draw_contents(self, vp: np.ndarray) -> None:
        """The objects and supports (the main program is in use).  Closed
        models are drawn without their back faces (the GPU skips the half of
        the triangles that faces away)."""
        GL.glCullFace(GL.GL_BACK)
        fits = self.scene.fits(self.printer)
        see_through = []                          # hollow models with "Show inside": drawn last
        for obj in self.scene.objects:
            if not obj.visible:
                continue
            glm, closed = self._mesh_for(obj)
            if obj is self.scene.selected:
                color = self.colors["selected"]
            elif not fits.get(obj.id, True):
                color = COLOR_OUTSIDE
            else:
                color = self.colors["model"]
            if self.show_inside and obj.hollow is not None:
                see_through.append((glm, obj, color))
                continue
            model = obj.matrix()
            if closed or self.cull_back_faces:
                GL.glEnable(GL.GL_CULL_FACE)
                GL.glFrontFace(GL.GL_CW if np.linalg.det(model[:3, :3]) < 0 else GL.GL_CCW)   # mirrored
            self._draw(glm, model, vp, color)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glFrontFace(GL.GL_CCW)

        # the selected support first: the same triangles drawn again in the
        # support colour then fail the depth test, so its highlight stays
        sel = self.scene.selected_support
        if sel is not None:
            self._draw(self._support_mesh_for(sel), np.eye(4), vp, self.colors["support_selected"])
        self._draw(self._support_buffer(), np.eye(4), vp, self.colors["support"])

        for obj in self.scene.objects:            # drain holes, and inside: the internal supports
            if not obj.visible:
                continue
            self._draw(self._holes_for(obj), np.eye(4), vp, COLOR_HOLE)
            if self.show_inside and obj.hollow is not None and obj.hollow.lattice is not None:
                buf, shift = self._lattice_for(obj)
                if buf is not None:
                    model = np.eye(4)
                    model[:3, 3] = shift
                    r, g, b, _a = self.colors["model"]
                    self._draw(buf, model, vp, (r * 0.75, g * 0.75, b * 0.75, 1.0))
        if see_through:                           # back faces, then front faces, blended, no depth writes
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_CULL_FACE)
            for face in (GL.GL_FRONT, GL.GL_BACK):
                GL.glCullFace(face)
                for glm, obj, color in see_through:
                    self._draw(glm, obj.matrix(), vp, (*color[:3], XRAY_ALPHA))
            GL.glDepthMask(GL.GL_TRUE)
            GL.glCullFace(GL.GL_BACK)
        GL.glDisable(GL.GL_CULL_FACE)

    # ------------------------------------------------------------ hollowing
    def _holes_for(self, obj: MeshObject) -> GLBuffer | None:
        """Drain hole markers: a short stub standing out of the surface (and,
        with Show inside, the whole hole through the wall)."""
        if not obj.holes:
            entry = self._hole_buffers.pop(obj.id, None)
            if entry and entry[1] is not None:
                entry[1].release()
            return None
        deep = self.show_inside and obj.hollow is not None
        depth = hole_depth(obj.hollow.thickness) if deep else 0.6
        m = obj.matrix()
        key = (obj.holes, m.tobytes(), depth)
        entry = self._hole_buffers.get(obj.id)
        if entry is None or entry[0] != key:
            if entry and entry[1] is not None:
                entry[1].release()
            ends = [hole_world(m, h) for h in obj.holes]
            p0 = np.array([c + n * HOLE_MARK_OUT for c, n in ends])
            p1 = np.array([c - n * depth for c, n in ends])
            mesh = tubes(p0, p1, np.array([h.diameter / 2 for h in obj.holes]), sides=24)
            self._hole_buffers[obj.id] = (key, GLBuffer.from_trimesh(mesh) if len(mesh.faces) else None)
        return self._hole_buffers[obj.id][1]

    def _lattice_for(self, obj: MeshObject) -> tuple[GLBuffer | None, np.ndarray]:
        """The internal supports as tubes, clipped to the cavity (computed once
        per shape; moving the model only shifts them)."""
        m = obj.matrix()
        key = (id(obj.mesh), np.round(m[:3, :3], 9).tobytes(), obj.hollow)
        entry = self._lattice_buffers.get(obj.id)
        if entry is None or entry[0] != key:
            if entry and entry[1] is not None:
                entry[1].release()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                world = obj.transformed()
                lo, hi = world.bounds
                lat = obj.hollow.lattice
                p0, p1 = lattice_segments(lo, hi, lat)
                p0, p1 = cavity_clip(world, obj.hollow.thickness, p0, p1)
                mesh = tubes(p0, p1, lat.diameter / 2, sides=8)
                buf = GLBuffer.from_trimesh(mesh) if len(mesh.faces) else None
            finally:
                QApplication.restoreOverrideCursor()
            entry = (key, buf, m[:3, 3].copy())
            self._lattice_buffers[obj.id] = entry
        return entry[1], m[:3, 3] - entry[2]

    def _draw(self, glm: GLBuffer | None, model: np.ndarray, vp: np.ndarray, color, lit=True) -> None:
        if glm is None:
            return
        mvp = (vp @ model).astype(np.float32)
        GL.glUniformMatrix4fv(self._uniforms["u_mvp"], 1, GL.GL_TRUE, mvp)
        GL.glUniformMatrix4fv(self._uniforms["u_model"], 1, GL.GL_TRUE, model.astype(np.float32))
        GL.glUniform4f(self._uniforms["u_color"], *color)
        GL.glUniform1i(self._uniforms["u_lit"], 1 if lit else 0)
        glm.draw()

    # ------------------------------------------------------------ GPU caches
    def _mesh_for(self, obj: MeshObject) -> tuple[GLBuffer, bool]:
        """The object's GPU buffer, and whether the mesh is closed with its
        faces pointing out (then back faces can be skipped)."""
        key = (id(obj.mesh), len(obj.mesh.faces))
        entry = self._meshes.get(obj.id)
        if entry is None or entry[0] != key:
            if entry:
                entry[1].release()
            try:
                closed = bool(obj.mesh.is_volume)
            except Exception:                                    # pragma: no cover
                closed = False
            self._meshes[obj.id] = (key, GLBuffer.from_trimesh(obj.mesh), closed)
        return self._meshes[obj.id][1], self._meshes[obj.id][2]

    def _support_mesh_for(self, sup: Support) -> GLBuffer:
        """Buffer for one support (used to highlight the selected one)."""
        settings = self.scene.support_settings
        key = (tuple(np.round(sup.tip, 4)), tuple(np.round(sup.lean, 6)), sup.kind, sup.foot, sup.width,
               sup.angle, sup.stretch, sup.route, sup.trunk, tuple(sorted(settings.to_dict().items())))
        entry = self._support_meshes.get(sup.id)
        if entry is None or entry[0] != key:
            if entry:
                entry[1].release()
            self._support_meshes[sup.id] = (key, GLBuffer.from_trimesh(sup.to_mesh(settings)))
        return self._support_meshes[sup.id][1]

    def _support_buffer(self) -> GLBuffer | None:
        """One buffer with all support geometry (pillars, braces, raft): one
        draw call however many supports there are.  Rebuilt whenever the
        scene rebuilds its support meshes (``extras_version``); object ids
        cannot be used as keys because Python reuses them."""
        meshes = self.scene.support_meshes()
        if self._extras_version != self.scene.extras_version:
            if self._extra_buffer is not None:
                self._extra_buffer.release()
            self._extra_buffer = GLBuffer.from_trimeshes(meshes)
            self._extras_version = self.scene.extras_version
        return self._extra_buffer

    def invalidate_object(self, obj: MeshObject) -> None:
        """Call after the source mesh changed (mirror / repair)."""
        entry = self._meshes.pop(obj.id, None)
        if entry:
            self.makeCurrent()
            entry[1].release()
            self.doneCurrent()
        self.update()

    def _sync_plate(self) -> None:
        p = self.printer
        key = (p.print_x, p.print_y, p.print_z)
        if key == self._plate_key:
            return
        self._plate_key = key
        hx, hy, hz = p.print_x / 2, p.print_y / 2, p.print_z
        lines = []
        step = 10.0
        for x in np.arange(-hx, hx + 1e-6, step):
            lines += [[x, -hy, 0], [x, hy, 0]]
        for y in np.arange(-hy, hy + 1e-6, step):
            lines += [[-hx, y, 0], [hx, y, 0]]
        # build volume outline
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            lines += [[x0, y0, 0], [x1, y1, 0], [x0, y0, hz], [x1, y1, hz], [x0, y0, 0], [x0, y0, hz]]
        self._plate = GLBuffer.lines(np.array(lines, dtype=np.float32))
        quad = np.array([[-hx, -hy, -0.05], [hx, -hy, -0.05], [hx, hy, -0.05], [-hx, hy, -0.05]])
        self._plate_fill = GLBuffer.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), np.array([[0, 0, 1.0]] * 2))

    # ------------------------------------------------------------ picking
    def _pick(self, pos: QPoint):
        """Return (item, world point) under the cursor, item being a
        MeshObject, a Support or None."""
        origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
        best, best_t, best_pt = None, np.inf, None
        settings = self.scene.support_settings
        for sup in self.scene.supports:
            m = sup.to_mesh(settings)
            if len(m.faces) == 0:
                continue
            hit = _ray_hit(m, origin, direction)
            if hit is not None and hit[0] < best_t:
                best, best_t, best_pt = sup, hit[0], hit[1]
        for obj in self.scene.objects:
            if not obj.visible:
                continue
            hit = _ray_hit(obj.transformed(), origin, direction)
            if hit is not None and hit[0] < best_t:
                best, best_t, best_pt = obj, hit[0], hit[1]
        return best, best_pt

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:
        self._last_pos = e.position().toPoint()
        self._pressed_button = e.button()
        self._moved = False
        self._drag_target = None
        if e.button() == Qt.LeftButton and self.interactive and not (e.modifiers() & Qt.ShiftModifier):
            item, pt = self._pick(self._last_pos)
            if self.add_support_mode and isinstance(item, MeshObject):
                self.support_requested.emit(item, pt)
                return
            if self.add_hole_mode and isinstance(item, MeshObject):
                origin, direction = self.camera.ray(self._last_pos.x(), self._last_pos.y(), self.width(), self.height())
                world = item.transformed()
                hit = ray_mesh_first(world, origin, direction)
                if hit is not None:
                    normal = np.asarray(world.face_normals[hit[2]], float)
                    if normal @ direction > 0:          # the face we see points at us, whatever the winding
                        normal = -normal
                    self.hole_requested.emit(item, hit[1], normal)
                return
            if isinstance(item, MeshObject):
                self.scene.selected, self.scene.selected_support = item, None
            elif isinstance(item, Support):
                self.scene.selected_support, self.scene.selected = item, None
            else:
                self.scene.selected = self.scene.selected_support = None
            self.selection_changed.emit()
            if item is not None:
                self._drag_target = item
                self._undo_pushed = False
                self._drag_plane_z = float(pt[2])
                anchor = item.position if isinstance(item, MeshObject) else item.tip
                self._drag_offset = anchor[:2] - pt[:2]
            self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        pos = e.position().toPoint()
        dx, dy = pos.x() - self._last_pos.x(), pos.y() - self._last_pos.y()
        self._last_pos = pos
        self._moved = True
        btn = e.buttons()
        if btn & Qt.RightButton:
            self.camera.orbit(-dx, dy)
        elif btn & Qt.MiddleButton or (btn & Qt.LeftButton and e.modifiers() & Qt.ShiftModifier):
            self.camera.pan(dx, dy)
        elif btn & Qt.LeftButton and self._drag_target is not None:
            if not self._undo_pushed:          # one undo step per drag
                self.scene.push_undo()
                self._undo_pushed = True
            origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
            hit = ray_plane_z(origin, direction, self._drag_plane_z)
            if hit is not None:
                target = self._drag_target
                new_xy = hit[:2] + self._drag_offset
                if isinstance(target, MeshObject):
                    self.scene.move_object(target, [new_xy[0], new_xy[1], target.position[2]])
                else:
                    self._drag_support(target, new_xy)
                self.object_moved.emit()
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._drag_target is not None and self._moved:
            self.scene.changed()
        self._drag_target = None

    def wheelEvent(self, e: QWheelEvent) -> None:
        self.camera.zoom(e.angleDelta().y() / 120.0)
        self.update()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        self.fit_view()

    def _drag_support(self, sup: Support, xy) -> None:
        """Move a dragged support's contact to ``xy`` on the underside of its
        model and place it again; where there is no room for a support it
        stays at its last good spot."""
        obj = self.scene.find(sup.owner_id) if sup.owner_id else None
        if obj is None:
            return
        before = sup.state()
        sup.tip[:2] = xy
        if not snap_support(sup, obj, self.scene.support_settings):
            sup.restore(before)


def _ray_hit(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray):
    """Closest intersection of a ray with a mesh -> (distance, point) or None."""
    return ray_mesh_closest(mesh, origin, direction)
