"""The 3D build-plate view (QOpenGLWidget + PyOpenGL, OpenGL 3.3 core).

Responsibilities:
    * draw the build plate, the objects and the supports
    * orbit / pan / zoom with the mouse
    * pick objects and supports with the mouse and drag them across the plate
    * "add support" mode: clicking on a model adds a support at that point
    * on the selected model: move arrows and rotate rings (see gizmos.py),
      and "lay on face", which highlights the faces it can lie on

Mouse: left = select / drag, right = orbit, middle (or shift+left) = pan,
wheel = zoom.  Dragging an arrow moves the model along that axis, a ring
turns it about that axis (Ctrl: whole mm / 15 degree steps).
"""

from __future__ import annotations

import ctypes
import math
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
from ..core.orient import DOWN, flat_faces_of, rotated_about
from ..core.paint import ENFORCE, BLOCK, brushed
from ..core.profiles import PrinterProfile
from ..core.raycast import ray_mesh_closest, ray_mesh_first, ray_mesh_hits
from ..core.scene import Scene
from ..core.placement import snap_support
from ..core.supports import Support
from .camera import OrbitCamera, ray_plane_z
from . import gizmos
from .glplatform import default_surface_format  # noqa: F401  (re-exported)
from .lighting import Lighting
from .theme_colors import DEFAULT_THEME_COLORS, DEFAULT_COLORS  # noqa: F401

VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform vec4 u_clip;          // section view: only where dot(u_clip, (world, 1)) >= 0 is drawn
out vec3 v_normal;
out vec3 v_world;
out float gl_ClipDistance[1];
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = mat3(u_model) * in_normal;
    v_world = (u_model * vec4(in_pos, 1.0)).xyz;
    gl_ClipDistance[0] = dot(u_clip, vec4(v_world, 1.0));
}
"""

FRAG_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_world;
uniform vec4 u_color;
uniform vec3 u_eye;
uniform int u_lit;
uniform int u_cap;            // section view: what is seen from behind is the cut face
uniform vec3 u_key;           // toward the key light (see lighting.py)
uniform vec3 u_fill;          // toward the fill light
uniform vec4 u_light;         // ambient, key, fill, shine
uniform float u_gloss;
out vec4 frag;
void main() {
    if (u_cap == 1 && !gl_FrontFacing) { frag = vec4(u_color.rgb * 0.55, u_color.a); return; }
    if (u_lit == 0) { frag = u_color; return; }
    vec3 n = normalize(v_normal);
    vec3 view = normalize(u_eye - v_world);
    if (dot(n, view) < 0.0) n = -n;           // light back faces too
    float diff = u_light.x + u_light.y * max(dot(n, u_key), 0.0) + u_light.z * max(dot(n, u_fill), 0.0);
    vec3 h = normalize(u_key + view);
    float spec = pow(max(dot(n, h), 0.0), u_gloss) * u_light.w;
    frag = vec4(u_color.rgb * diff + spec, u_color.a);
}
"""

COLOR_OUTSIDE = (0.90, 0.30, 0.30, 1.0)
COLOR_HOLE = (0.93, 0.36, 0.26, 1.0)       # drain holes
PAINT_COLORS = {ENFORCE: (0.24, 0.52, 1.0, 0.70), BLOCK: (0.95, 0.27, 0.27, 0.70)}   # support paint
XRAY_ALPHA = 0.28                          # hollow models with "Show inside"
SEE_THROUGH_ALPHA = 0.5                    # View -> Semi-transparent models
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
    lay_face_requested = Signal(object, object)  # (MeshObject, world normal of the face to put down)
    gl_started = Signal()                        # OpenGL is up; GL_INFO is filled

    def __init__(self, scene: Scene, parent=None, interactive: bool = True):
        super().__init__(parent)
        self.scene = scene
        self.interactive = interactive          # False: camera only, no picking
        self.cull_back_faces = False            # skip faces pointing away (closed, outward-wound meshes)
        self.camera = OrbitCamera()
        self.colors = {k: hex_to_rgba(v) for k, v in DEFAULT_COLORS.items()}
        self.lighting = Lighting()              # Edit -> Preferences -> Lighting
        self.printer: PrinterProfile = PrinterProfile()
        self.add_support_mode = False
        self.add_hole_mode = False              # clicks on a model place / remove drain holes
        self.show_inside = False                # hollow models see-through with their internal supports
        self._hole_buffers: dict[int, tuple[tuple, GLBuffer | None]] = {}
        self._lattice_buffers: dict[int, tuple[tuple, GLBuffer | None, np.ndarray]] = {}
        self.show_move = False                  # move arrows on the selected model
        self.show_rotate = False                # rotate rings on the selected model
        self.lay_face_mode = False              # its flat faces highlighted: a click lays it on one
        self._gizmo_buffers: tuple[GLBuffer, GLBuffer] | None = None    # unit arrow, unit ring
        self._flat_buffers: dict[int, tuple] = {}   # id(mesh) -> (mesh, [GLBuffer]) of its flat faces
        self.show_plate = True                  # View -> Build plate (the volume's outline always shows)
        self.section: tuple[int, float, bool] | None = None    # section view: (axis, position, flip)
        self.see_through = False                # View -> Semi-transparent models
        self._hover = None                      # ("move" | "rotate", axis) or ("face", index) under the mouse
        self._gizmo_drag: dict | None = None    # while an arrow or ring is dragged
        self.paint_mode = False                 # dragging on a model paints support paint
        self.paint_value = ENFORCE
        self.brush_radius = 2.0                 # mm
        self._stroke: MeshObject | None = None  # the model being painted (mouse down)
        self._stroke_changed = False
        self._brush_at: tuple | None = None     # (point, normal) under the mouse in paint mode
        self._paint_buffers: dict[int, tuple] = {}   # obj id -> (paint, mesh, {value: GLBuffer})
        self._brush_ring: GLBuffer | None = None
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(False)
        self.setAcceptDrops(False)

        self._program = 0
        self._uniforms: dict[str, int] = {}
        self._meshes: dict[int, tuple] = {}     # id(mesh) -> (mesh, buffer, closed): clones share one
        self._support_meshes: dict[int, tuple[tuple, GLBuffer]] = {}
        self._extra_buffer: GLBuffer | None = None               # all supports, braces and raft
        self._extras_version = -1
        self._plate: GLBuffer | None = None       # grid lines on the plate
        self._plate_fill: GLBuffer | None = None
        self._volume: GLBuffer | None = None      # the build volume's outline
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

    def set_lighting(self, lighting: Lighting) -> None:
        self.lighting = lighting.clamped()
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

    def snapshot(self, width: int = 480, height: int = 300):
        """A picture (QImage) of the build plate and what is on it, from the
        usual corner and without the tools or the selection - for project
        thumbnails.  The view itself stays as it was.  None when OpenGL has
        not started yet."""
        if not self.isValid() or self.width() < 8 or self.height() < 8:
            return None
        cam = self.camera
        kept = (cam.target.copy(), cam.distance, cam.yaw, cam.pitch, self.show_move, self.show_rotate,
                self.lay_face_mode, self._brush_at, self.section, self.scene.selected,
                self.scene.selected_support, self._hover, self._overlay)
        try:
            self.show_move = self.show_rotate = self.lay_face_mode = False
            self._brush_at, self.section, self._hover, self._overlay = None, None, None, None
            self.scene.selected = self.scene.selected_support = None
            px, py = self.printer.print_x / 2, self.printer.print_y / 2
            box = np.array([[-px, -py, 0.0], [px, py, 0.0]])
            content = self.content_bounds()
            if content is not None:
                box = np.vstack([np.minimum(box[0], content[0]), np.maximum(box[1], content[1])])
            cam.set_view("iso")
            cam.fit(box, (self.printer.print_x, self.printer.print_y))
            cam.distance *= 0.8                         # the plate fills the picture
            image = self.grabFramebuffer()
        finally:
            (cam.target, cam.distance, cam.yaw, cam.pitch, self.show_move, self.show_rotate,
             self.lay_face_mode, self._brush_at, self.section, self.scene.selected,
             self.scene.selected_support, self._hover, self._overlay) = kept
            self.update()
        if image.isNull():
            return None
        w, h = image.width(), image.height()                # crop to the thumbnail's shape, then scale
        aspect = width / height
        cw, ch = (w, round(w / aspect)) if w / h < aspect else (round(h * aspect), h)
        image = image.copy((w - cw) // 2, (h - ch) // 2, cw, ch)
        return image.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    # ------------------------------------------------------------ GL setup
    def initializeGL(self) -> None:
        if not GL_INFO:
            _read_gl_info()
        self._program = build_program(VERT_SHADER, FRAG_SHADER)
        for name in ("u_mvp", "u_model", "u_color", "u_eye", "u_lit", "u_clip", "u_cap",
                     "u_key", "u_fill", "u_light", "u_gloss"):
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
        GL.glUniform4f(self._uniforms["u_clip"], 0.0, 0.0, 0.0, 1.0)     # nothing cut (see _section_on)
        GL.glUniform1i(self._uniforms["u_cap"], 0)
        light = self.lighting
        key, fill = light.directions(self.camera.yaw)
        GL.glUniform3fv(self._uniforms["u_key"], 1, key)
        GL.glUniform3fv(self._uniforms["u_fill"], 1, fill)
        GL.glUniform4f(self._uniforms["u_light"], light.ambient, light.key, light.fill, light.shine)
        GL.glUniform1f(self._uniforms["u_gloss"], light.gloss)

        self._draw_contents(vp)
        # the plate goes last: it is see-through, so whatever is below it (seen
        # from underneath) must already be drawn; it writes no depth itself
        GL.glUseProgram(self._program)
        self._sync_plate()
        fill, grid = self._plate_colors()
        if self.show_plate:
            GL.glDepthMask(GL.GL_FALSE)
            self._draw(self._plate_fill, np.eye(4), vp, fill, lit=False)
            GL.glDepthMask(GL.GL_TRUE)
            self._draw(self._plate, np.eye(4), vp, grid, lit=False)
        self._draw(self._volume, np.eye(4), vp, grid, lit=False)
        if self._gizmo_object() is not None and (self.show_move or self.show_rotate):
            self._draw_gizmos(vp)
        self._draw_brush(vp)
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
        cut = self._section_on()
        fits = self.scene.fits(self.printer)
        see_through = []                          # see-through models (and hollow ones with "Show inside"): last
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
            if self.see_through:
                see_through.append((glm, obj, (*color[:3], SEE_THROUGH_ALPHA)))
                continue
            if self.show_inside and obj.hollow is not None:
                see_through.append((glm, obj, (*color[:3], XRAY_ALPHA)))
                continue
            model = obj.matrix()
            if (closed or self.cull_back_faces) and not cut:      # cut open: the inside shows
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
                    self._draw(glm, obj.matrix(), vp, color)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glCullFace(GL.GL_BACK)
        GL.glDisable(GL.GL_CULL_FACE)
        if self.lay_face_mode and self._gizmo_object() is not None:
            self._draw_flat_faces(self._gizmo_object(), vp)
        if self.interactive and any(o.paint is not None for o in self.scene.objects):
            self._draw_paint(vp)
        if cut:
            self._section_off()

    # ------------------------------------------------------------ gizmos
    def set_tools(self, move: bool | None = None, rotate: bool | None = None,
                  lay_face: bool | None = None) -> None:
        """Show / hide the move arrows and rotate rings, and turn "lay on
        face" on or off (None: leave as it is)."""
        if move is not None:
            self.show_move = move
        if rotate is not None:
            self.show_rotate = rotate
        if lay_face is not None:
            self.lay_face_mode = lay_face
        self._hover = None
        self._update_tracking()
        self.update()

    def _update_tracking(self) -> None:
        """Mouse moves without a button pressed matter while a tool shows (hover highlight)."""
        self.setMouseTracking(self.show_move or self.show_rotate or self.lay_face_mode or self.paint_mode)

    # ------------------------------------------------------------ support paint
    def set_paint(self, on: bool | None = None, value: int | None = None, radius: float | None = None) -> None:
        """Paint mode: dragging on a model paints ``value`` (see core/paint.py)
        on its faces within ``radius`` mm of the cursor."""
        if on is not None:
            self.paint_mode = on
            self._brush_at = None
        if value is not None:
            self.paint_value = int(value)
        if radius is not None:
            self.brush_radius = float(radius)
        self._update_tracking()
        self.update()

    def _model_hit(self, pos: QPoint, only: MeshObject | None = None):
        """(model, world point, outward world normal, face index) under the mouse, or None."""
        origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
        best, best_t = None, np.inf
        for obj in ([only] if only is not None else self.scene.objects):
            if not obj.visible or not _ray_meets_box(origin, direction, obj.world_bounds(), best_t):
                continue
            inv = np.linalg.inv(obj.matrix())
            o, d = inv[:3, :3] @ origin + inv[:3, 3], inv[:3, :3] @ direction
            if self._section_plane() is not None:            # the first hit on the side that shows
                t = self._first_shown(obj.mesh, o, d, origin, direction)
                hit = None if t is None else ray_mesh_first(obj.mesh, o + d * (t - 1e-4), d)
                if hit is not None:
                    hit = (t, hit[1], hit[2])
            else:
                hit = ray_mesh_first(obj.mesh, o, d)
            if hit is not None and hit[0] < best_t:
                n = np.asarray(obj.mesh.face_normals[hit[2]], float) @ inv[:3, :3]
                n = n / (np.linalg.norm(n) or 1.0)
                best = (obj, origin + direction * hit[0], -n if n @ direction > 0 else n, hit[2])
                best_t = hit[0]
        return best

    def _paint_at(self, point, face: int | None = None) -> None:
        obj = self._stroke
        new = brushed(obj, point, self.brush_radius, self.paint_value, face)
        if new is not None:
            obj.paint = new
            self._stroke_changed = True

    def _paint_buffers_for(self, obj: MeshObject) -> dict:
        """GPU buffers of the object's painted faces, per paint value."""
        entry = self._paint_buffers.get(obj.id)
        if entry is None or entry[0] is not obj.paint or entry[1] is not obj.mesh:
            if entry:
                for b in entry[2].values():
                    b.release()
            mesh, bufs = obj.mesh, {}
            if obj.paint is not None and len(obj.paint) == len(mesh.faces):
                for value in PAINT_COLORS:
                    f = np.flatnonzero(obj.paint == value)
                    if len(f):
                        bufs[value] = GLBuffer.triangles(mesh.vertices, mesh.faces[f], mesh.face_normals[f])
            entry = self._paint_buffers[obj.id] = (obj.paint, mesh, bufs)
        return entry[2]

    def _draw_paint(self, vp: np.ndarray) -> None:
        """Painted faces over the models (and the brush under the mouse)."""
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(-1.0, -4.0)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        live = {o.id for o in self.scene.objects if o.paint is not None}
        for gone in [k for k in self._paint_buffers if k not in live]:     # deleted or cleared models
            for b in self._paint_buffers.pop(gone)[2].values():
                b.release()
        for obj in self.scene.objects:
            if obj.visible and obj.paint is not None:
                for value, buf in self._paint_buffers_for(obj).items():
                    self._draw(buf, obj.matrix(), vp, PAINT_COLORS[value], lit=False)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

    def _draw_brush(self, vp: np.ndarray) -> None:
        if not (self.paint_mode and self._brush_at is not None):
            return
        if self._brush_ring is None:
            self._brush_ring = GLBuffer.from_trimesh(gizmos.ring_mesh(tube=0.05, segments=48, sides=6))
        point, normal = self._brush_at
        m = np.eye(4)
        m[:3, :3] = trimesh.geometry.align_vectors([0, 0, 1], normal)[:3, :3] * self.brush_radius
        m[:3, 3] = point + normal * 0.05
        color = PAINT_COLORS.get(self.paint_value, (0.85, 0.85, 0.85, 1.0))
        GL.glDisable(GL.GL_DEPTH_TEST)                    # always visible, on curved surfaces too
        self._draw(self._brush_ring, m, vp, (*color[:3], 1.0), lit=False)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def set_view_options(self, plate: bool | None = None, see_through: bool | None = None) -> None:
        """View menu: show the build plate, draw the models semi-transparent."""
        if plate is not None:
            self.show_plate = plate
        if see_through is not None:
            self.see_through = see_through
        self.update()

    # ------------------------------------------------------------ section view
    def set_section(self, axis: int | None, position: float = 0.0, flip: bool = False) -> None:
        """Cut the view at ``position`` mm on world axis ``axis`` (0 = X, 1 = Y,
        2 = Z; None: no cut).  The part below / left / in front of the cut
        shows, or the other part with ``flip``; the cut faces of models and
        supports show darker."""
        self.section = None if axis is None else (int(axis), float(position), bool(flip))
        self.update()

    def _section_plane(self) -> tuple | None:
        """(a, b, c, d): what shows has a*x + b*y + c*z + d >= 0."""
        if self.section is None or not self.interactive:
            return None
        axis, pos, flip = self.section
        n = [0.0, 0.0, 0.0]
        n[axis] = 1.0 if flip else -1.0
        return (*n, -pos if flip else pos)

    def section_shows(self, point) -> bool:
        plane = self._section_plane()
        return plane is None or float(np.dot(plane[:3], point) + plane[3]) >= -1e-9

    def _section_on(self) -> bool:
        plane = self._section_plane()
        if plane is None:
            return False
        GL.glUniform4f(self._uniforms["u_clip"], *plane)
        GL.glUniform1i(self._uniforms["u_cap"], 1)
        GL.glEnable(GL.GL_CLIP_DISTANCE0)
        return True

    def _section_off(self) -> None:
        GL.glUseProgram(self._program)
        GL.glDisable(GL.GL_CLIP_DISTANCE0)
        GL.glUniform4f(self._uniforms["u_clip"], 0.0, 0.0, 0.0, 1.0)
        GL.glUniform1i(self._uniforms["u_cap"], 0)

    def _gizmo_object(self) -> MeshObject | None:
        """The model the tools work on: the selected one, if shown."""
        obj = self.scene.selected
        if not self.interactive or obj is None or not obj.visible or obj not in self.scene.objects:
            return None
        return obj

    def _draw_gizmos(self, vp: np.ndarray) -> None:
        """Arrows and rings at the selected model's center, over everything
        else (the depth buffer is cleared first, so they still hide each
        other correctly)."""
        obj = self._gizmo_object()
        center = np.asarray(obj.position, float)
        if self._gizmo_buffers is None:
            self._gizmo_buffers = (GLBuffer.from_trimesh(gizmos.arrow_mesh()),
                                   GLBuffer.from_trimesh(gizmos.ring_mesh()))
        arrow, ring = self._gizmo_buffers
        active = self._gizmo_drag["handle"] if self._gizmo_drag else self._hover
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        for kind, buf, px, shown in (("rotate", ring, gizmos.RING_PX, self.show_rotate),
                                     ("move", arrow, gizmos.ARROW_PX, self.show_move)):
            if not shown:
                continue
            size = gizmos.world_size(self.camera, center, px, self.height())
            for axis in range(3):
                color = gizmos.HOVER_COLOR if active == (kind, axis) else gizmos.AXIS_COLORS[axis]
                self._draw(buf, gizmos.model_matrix(center, axis, size), vp, color)

    def _flat_faces_of(self, obj: MeshObject) -> list:
        """The faces the model can be laid on (worked out once per mesh)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return flat_faces_of(obj)
        finally:
            QApplication.restoreOverrideCursor()

    def lay_candidates(self, obj: MeshObject) -> list[tuple[int, np.ndarray]]:
        """(index, world normal) of the faces to offer: all but the one the
        model already lies on."""
        m = obj.matrix()[:3, :3]
        to_world = np.linalg.inv(m).T                     # normals follow the inverse transpose
        out = []
        for i, f in enumerate(self._flat_faces_of(obj)):
            n = to_world @ f.normal
            n = n / np.linalg.norm(n)
            if n @ DOWN < 1 - 1e-6:
                out.append((i, n))
        return out

    def _draw_flat_faces(self, obj: MeshObject, vp: np.ndarray) -> None:
        faces = self._flat_faces_of(obj)
        entry = self._flat_buffers.get(id(obj.mesh))
        if entry is None or entry[0] is not obj.mesh:
            if entry:
                for b in entry[1]:
                    b.release()
            if len(self._flat_buffers) >= 8:                   # the last few models are plenty
                for b in self._flat_buffers.pop(next(iter(self._flat_buffers)))[1]:
                    b.release()
            mesh = obj.mesh
            entry = (mesh, [GLBuffer.triangles(mesh.vertices, mesh.faces[f.faces], mesh.face_normals[f.faces])
                            for f in faces])
            self._flat_buffers[id(mesh)] = entry
        model = obj.matrix()
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)            # drawn on the model's own surface
        GL.glPolygonOffset(-1.0, -4.0)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        for i, _n in self.lay_candidates(obj):
            color = gizmos.FACE_HOVER_COLOR if self._hover == ("face", i) else gizmos.FACE_COLOR
            self._draw(entry[1][i], model, vp, color, lit=False)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

    def _view_projection(self) -> np.ndarray:
        return self.camera.projection_matrix() @ self.camera.view_matrix()

    def _handle_at(self, pos: QPoint):
        """The arrow or ring under the mouse, as ("move" | "rotate", axis)."""
        obj = self._gizmo_object()
        if obj is None or not (self.show_move or self.show_rotate):
            return None
        return gizmos.pick(self._view_projection(), self.camera, np.asarray(obj.position, float),
                           self.width(), self.height(), (pos.x(), pos.y()), self.show_move, self.show_rotate)

    def _face_at(self, pos: QPoint) -> tuple[int, np.ndarray] | None:
        """In "lay on face" mode: the highlighted face under the mouse, as
        (index, world normal).  A face is only seen from outside the model
        (the whole model is behind its plane), so nothing of it can hide it."""
        obj = self._gizmo_object()
        if obj is None or not self.lay_face_mode:
            return None
        origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
        faces = self._flat_faces_of(obj)
        m = obj.matrix()
        best, best_t = None, np.inf
        for i, n in self.lay_candidates(obj):
            if n @ direction >= 0:
                continue
            local = obj.mesh.vertices[obj.mesh.faces[faces[i].faces]].reshape(-1, 3)
            t = gizmos.ray_triangles(origin, direction, trimesh.transform_points(local, m).reshape(-1, 3, 3))
            if t is not None and t < best_t:
                best, best_t = (i, n), t
        return best

    def _update_hover(self, pos: QPoint) -> None:
        hover = self._handle_at(pos)
        if hover is None and self.lay_face_mode:
            face = self._face_at(pos)
            hover = ("face", face[0]) if face is not None else None
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, _e) -> None:
        if self._hover is not None and self._gizmo_drag is None:
            self._hover = None
            self.update()

    def _start_gizmo_drag(self, handle, pos: QPoint) -> None:
        obj = self._gizmo_object()
        kind, axis = handle
        e = np.eye(3)[axis]
        drag = dict(handle=handle, obj=obj, undo=False, moved=False,
                    start_pos=obj.position.copy(), start_rot=obj.rotation.copy())
        if kind == "move":
            origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
            drag["t0"] = gizmos.axis_parameter(origin, direction, drag["start_pos"], e)
            drag["bottom"] = float(obj.world_bounds()[0][2] - obj.position[2])   # lowest point vs center
        else:
            # follow the mouse around the ring in its own plane; seen nearly
            # edge-on, by the angle around the center on screen instead
            origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
            drag["turned"] = 0.0
            angle = gizmos.plane_angle(origin, direction, obj.position, axis)
            drag["in_plane"] = angle is not None and abs(direction @ e) > gizmos.EDGE_ON
            if drag["in_plane"]:
                drag["last"] = angle
            else:
                c_px = gizmos.project(self._view_projection(), [obj.position], self.width(), self.height())[0][0]
                drag["center_px"] = c_px
                drag["last"] = gizmos.screen_angle(c_px, (pos.x(), pos.y()))
                # counter-clockwise on screen is a positive turn when the axis points at the camera
                drag["sign"] = 1.0 if e @ (self.camera.position() - obj.position) > 0 else -1.0
        self._gizmo_drag = drag
        self.update()

    def _drag_gizmo(self, pos: QPoint, modifiers) -> None:
        d = self._gizmo_drag
        obj = d["obj"]
        if obj not in self.scene.objects:
            self._gizmo_drag = None
            return
        kind, axis = d["handle"]
        e = np.eye(3)[axis]
        fine = not (modifiers & Qt.ControlModifier)
        if kind == "move":
            if d["t0"] is None:                              # looking straight along the axis
                return
            origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
            t = gizmos.axis_parameter(origin, direction, d["start_pos"], e)
            if t is None:
                return
            target = d["start_pos"].copy()
            target[axis] += t - d["t0"]
            if not fine:
                target[axis] = round(target[axis])           # whole mm
            if axis == 2:
                target[2] = max(target[2], -d["bottom"])     # never below the plate
            if np.allclose(target, obj.position):
                return
            if not d["undo"]:
                self.scene.push_undo()
                d["undo"] = True
            self.scene.move_object(obj, target)
        else:
            if d["in_plane"]:
                origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
                a = gizmos.plane_angle(origin, direction, d["start_pos"], axis)
                if a is None:
                    return
                sign = 1.0
            else:
                a = gizmos.screen_angle(d["center_px"], (pos.x(), pos.y()))
                sign = d["sign"]
            d["turned"] += gizmos.unwrap(a - d["last"])
            d["last"] = a
            step = 1.0 if fine else 15.0
            degrees = round(math.degrees(d["turned"]) * sign / step) * step
            angles = rotated_about(d["start_rot"], e, degrees)
            if np.allclose(angles, obj.rotation):
                return
            if not d["undo"]:
                self.scene.push_undo()
                d["undo"] = True
            obj.rotation[:] = angles
            obj.drop_to_plate()
        d["moved"] = True
        self.object_moved.emit()
        self.update()

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
        entry = self._meshes.get(id(obj.mesh))
        if entry is None or entry[0] is not obj.mesh:
            if entry:
                entry[1].release()
            self._prune_meshes()
            try:
                closed = bool(obj.mesh.is_volume)
            except Exception:                                    # pragma: no cover
                closed = False
            entry = self._meshes[id(obj.mesh)] = (obj.mesh, GLBuffer.from_trimesh(obj.mesh), closed)
        return entry[1], entry[2]

    def _prune_meshes(self) -> None:
        """Free the GPU buffers of meshes no model uses any more (an undo
        that brings one back uploads it again)."""
        used = {id(o.mesh) for o in self.scene.objects}
        for key in [k for k in self._meshes if k not in used]:
            self._meshes.pop(key)[1].release()

    def _support_mesh_for(self, sup: Support) -> GLBuffer:
        """Buffer for one support (used to highlight the selected one)."""
        mesh = self.scene.support_mesh(sup)                    # the same object while unchanged
        entry = self._support_meshes.get(sup.id)
        if entry is None or entry[0] is not mesh:
            if entry:
                entry[1].release()
            if len(self._support_meshes) > 16:                  # only the selected one is ever drawn
                for key in [k for k in self._support_meshes if k != sup.id]:
                    self._support_meshes.pop(key)[1].release()
            self._support_meshes[sup.id] = (mesh, GLBuffer.from_trimesh(mesh))
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
        """Call after the source mesh changed (mirror / repair): the new mesh
        gets its own buffer at the next frame, the old one is freed."""
        if self._meshes and self.isValid():
            self.makeCurrent()
            self._prune_meshes()
            self.doneCurrent()
        self.update()

    def _sync_plate(self) -> None:
        p = self.printer
        key = (p.print_x, p.print_y, p.print_z)
        if key == self._plate_key:
            return
        self._plate_key = key
        for buf in (self._plate, self._plate_fill, self._volume):
            if buf is not None:
                buf.release()
        hx, hy, hz = p.print_x / 2, p.print_y / 2, p.print_z
        lines = []
        step = 10.0
        for x in np.arange(-hx, hx + 1e-6, step):
            lines += [[x, -hy, 0], [x, hy, 0]]
        for y in np.arange(-hy, hy + 1e-6, step):
            lines += [[-hx, y, 0], [hx, y, 0]]
        self._plate = GLBuffer.lines(np.array(lines, dtype=np.float32))
        box = []                                  # the build volume's outline
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            box += [[x0, y0, 0], [x1, y1, 0], [x0, y0, hz], [x1, y1, hz], [x0, y0, 0], [x0, y0, hz]]
        self._volume = GLBuffer.lines(np.array(box, dtype=np.float32))
        quad = np.array([[-hx, -hy, -0.05], [hx, -hy, -0.05], [hx, hy, -0.05], [-hx, hy, -0.05]])
        self._plate_fill = GLBuffer.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), np.array([[0, 0, 1.0]] * 2))

    # ------------------------------------------------------------ picking
    def _pick(self, pos: QPoint):
        """Return (item, world point) under the cursor, item being a
        MeshObject, a Support or None.  Only what the ray's bounding-box test
        lets through is tested triangle by triangle: the supports' meshes as
        the scene keeps them, the models in their own coordinates (no
        transformed copy)."""
        origin, direction = self.camera.ray(pos.x(), pos.y(), self.width(), self.height())
        best, best_t, best_pt = None, np.inf, None
        for sup in self.scene.supports:
            m = self.scene.support_mesh(sup)
            if len(m.faces) == 0 or not _ray_meets_box(origin, direction, m.bounds, best_t):
                continue
            t = self._first_shown(m, origin, direction, origin, direction)
            if t is not None and t < best_t:
                best, best_t, best_pt = sup, t, origin + direction * t
        for obj in self.scene.objects:
            if not obj.visible or not _ray_meets_box(origin, direction, obj.world_bounds(), best_t):
                continue
            inv = np.linalg.inv(obj.matrix())            # the ray in the model's own coordinates:
            t = self._first_shown(obj.mesh, inv[:3, :3] @ origin + inv[:3, 3], inv[:3, :3] @ direction,
                                  origin, direction)
            if t is not None and t < best_t:              # same t along it (an affine map keeps t)
                best, best_t, best_pt = obj, t, origin + direction * t
        return best, best_pt

    def _first_shown(self, mesh, o, d, world_o, world_d) -> float | None:
        """t of the first hit of the ray (o, d) on ``mesh`` whose world point
        (world_o + t * world_d) is not cut away by the section view."""
        if self._section_plane() is None:
            hit = _ray_hit(mesh, o, d)
            return None if hit is None else hit[0]
        for t in ray_mesh_hits(mesh, o, d):
            if self.section_shows(world_o + world_d * t):
                return float(t)
        return None

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:
        self._last_pos = e.position().toPoint()
        self._pressed_button = e.button()
        self._moved = False
        self._drag_target = None
        if e.button() == Qt.LeftButton and self.interactive and not (e.modifiers() & Qt.ShiftModifier):
            handle = self._handle_at(self._last_pos)
            if handle is not None:                           # an arrow or ring of the selected model
                self._start_gizmo_drag(handle, self._last_pos)
                return
            if self.paint_mode:                              # paint on the model under the mouse
                hit = self._model_hit(self._last_pos)
                if hit is not None:
                    obj = hit[0]
                    if obj is not self.scene.selected:
                        self.scene.selected, self.scene.selected_support = obj, None
                        self.selection_changed.emit()
                    self.scene.push_undo()
                    self._stroke, self._stroke_changed = obj, False
                    self._paint_at(hit[1], hit[3])
                    self._brush_at = (hit[1], hit[2])
                    self.update()
                    return
            face = self._face_at(self._last_pos)
            if face is not None:                             # lay the model on this face
                self._hover = None
                self.lay_face_requested.emit(self.scene.selected, face[1])
                return
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
        if self._gizmo_drag is not None and e.buttons() & Qt.LeftButton:
            self._last_pos = pos
            self._drag_gizmo(pos, e.modifiers())
            return
        if self._stroke is not None and e.buttons() & Qt.LeftButton:
            self._last_pos = pos
            hit = self._model_hit(pos, only=self._stroke)
            if hit is not None:
                self._paint_at(hit[1], hit[3])
                self._brush_at = (hit[1], hit[2])
            self.update()
            return
        if not e.buttons():                                  # hovering (tracking is on while tools show)
            self._last_pos = pos
            if self.paint_mode:
                hit = self._model_hit(pos)
                self._brush_at = None if hit is None else (hit[1], hit[2])
                self.update()
            self._update_hover(pos)
            return
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
        if self._stroke is not None:
            changed = self._stroke_changed
            self._stroke = None
            if changed:
                self.scene.changed()
            return
        if self._gizmo_drag is not None:
            moved = self._gizmo_drag["moved"]
            self._gizmo_drag = None
            self._update_hover(e.position().toPoint())
            if moved:
                self.scene.changed()
            self.update()
            return
        if self._drag_target is not None and self._moved:
            self.scene.changed()
        self._drag_target = None

    def wheelEvent(self, e: QWheelEvent) -> None:
        self.camera.zoom(e.angleDelta().y() / 120.0)
        self.update()

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


def _ray_meets_box(origin, direction, bounds, max_t: float = np.inf) -> bool:
    """Whether the ray passes through the box [[x0,y0,z0],[x1,y1,z1]] before max_t."""
    lo, hi = np.asarray(bounds, float)
    t0, t1 = 0.0, float(max_t)
    for a in range(3):
        d = float(direction[a])
        if abs(d) < 1e-15:
            if origin[a] < lo[a] or origin[a] > hi[a]:
                return False
            continue
        ta, tb = (lo[a] - origin[a]) / d, (hi[a] - origin[a]) / d
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return True
