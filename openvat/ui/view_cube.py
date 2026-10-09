"""Navigation cube overlay for the 3D view.

A chamfered cube (a rhombicuboctahedron: 6 main faces, 12 edge faces at
45°, 8 corner faces) drawn with the same orientation as the camera, so it
rotates with the scene.  Clicking any of its 26 faces moves the camera to
look at the model from that direction.  Around it: a home button, four
orbit arrows (45° steps), two roll-free "spin" arrows (90° about Z) and a
"from below" button.  A small XYZ axis gizmo sits in the lower-left corner.

Over a 3D view the cube is painted by the view itself, into its OpenGL
frame (``paint_on``), and this widget only takes the mouse.  A raster
widget repainted on every frame over an OpenGL view makes Qt repaint the
whole window - every panel, button and label - each time the camera moves.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPolygonF, QColor, QPen, QFont, QTransform, QPainterPath
from PySide6.QtWidgets import QWidget

from .camera import OrbitCamera

_C = 1.0 + 1.0 / math.sqrt(2.0)  # main faces (width 2) are twice as wide as the 45° edge faces

MAIN_FACES = {                 # normal -> (label, face right vector, face up vector)
    (0, 0, 1): ("TOP", (1, 0, 0), (0, 1, 0)),
    (0, 0, -1): ("BOTTOM", (1, 0, 0), (0, -1, 0)),
    (0, -1, 0): ("FRONT", (1, 0, 0), (0, 0, 1)),
    (0, 1, 0): ("BACK", (-1, 0, 0), (0, 0, 1)),
    (1, 0, 0): ("RIGHT", (0, 1, 0), (0, 0, 1)),
    (-1, 0, 0): ("LEFT", (0, -1, 0), (0, 0, 1)),
}


def _build_faces() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(unit normal, (k,3) vertices)] for all 26 faces."""
    verts = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-_C, _C):
                verts += [(c, a, b), (a, c, b), (a, b, c)]
    verts = np.unique(np.array(verts, dtype=float), axis=0)
    faces = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == y == z == 0:
                    continue
                n = np.array([x, y, z], dtype=float)
                n /= np.linalg.norm(n)
                d = verts @ n
                ring = verts[np.isclose(d, d.max())]
                # order the vertices around the normal
                u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u)
                v = np.cross(n, u)
                ang = np.arctan2(ring @ v, ring @ u)
                faces.append((n, ring[np.argsort(ang)]))
    return faces


FACES = _build_faces()


class ViewCube(QWidget):
    view_direction = Signal(object)       # unit vector the camera should look from
    orbit_requested = Signal(float, float)  # (yaw delta, pitch delta) in degrees
    home_requested = Signal()

    SIZE = 150

    def __init__(self, camera: OrbitCamera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self._hover: int | None = None           # face index
        self._hover_btn: str | None = None
        self._face_polys: list[tuple[int, QPolygonF]] = []
        self._buttons: dict[str, QPainterPath] = {}
        self.drawn_by_view = False               # True: the 3D view paints us (paint_on), we only take the mouse

    # ------------------------------------------------------------ geometry
    def _projection(self):
        """Screen-space basis for the current camera orientation."""
        r, u, f = self.camera.right(), self.camera.up(), self.camera.forward()
        return r, u, f

    def _project(self, p: np.ndarray, r, u, scale, center) -> QPointF:
        return QPointF(center[0] + (p @ r) * scale, center[1] - (p @ u) * scale)

    def _layout_buttons(self) -> None:
        s = self.SIZE
        m = 10
        tri = 12
        self._buttons = {}

        def triangle(cx, cy, dx, dy):
            path = QPainterPath()
            px, py = -dy, dx
            path.moveTo(cx + dx * tri, cy + dy * tri)
            path.lineTo(cx + px * tri, cy + py * tri)
            path.lineTo(cx - px * tri, cy - py * tri)
            path.closeSubpath()
            return path

        self._buttons["left"] = triangle(m + tri, s / 2, -1, 0)
        self._buttons["right"] = triangle(s - m - tri, s / 2, 1, 0)
        self._buttons["up"] = triangle(s / 2, m + tri, 0, -1)
        self._buttons["down"] = triangle(s / 2, s - m - tri, 0, 1)

        home = QPainterPath()
        hx, hy = m + 14, m + 12
        home.moveTo(hx, hy - 12); home.lineTo(hx + 13, hy); home.lineTo(hx + 8, hy)
        home.lineTo(hx + 8, hy + 11); home.lineTo(hx - 8, hy + 11); home.lineTo(hx - 8, hy)
        home.lineTo(hx - 13, hy); home.closeSubpath()
        self._buttons["home"] = home

        spin = QPainterPath()
        spin.addEllipse(QRectF(s - m - 30, m, 26, 20))
        self._buttons["spin"] = spin

        below = QPainterPath()
        below.addRoundedRect(QRectF(s - m - 30, s - m - 18, 28, 16), 3, 3)
        self._buttons["below"] = below

    # ------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:
        if self.drawn_by_view:
            return
        p = QPainter(self)
        self.paint_on(p)

    def paint_on(self, p: QPainter) -> None:
        """Paint the cube, its buttons and the axes with ``p`` (whose origin is
        this widget's top-left corner)."""
        self._layout_buttons()
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        self._paint_buttons(p)
        self._paint_cube(p)
        self._paint_axes(p)

    def moveEvent(self, e) -> None:
        super().moveEvent(e)
        if self.drawn_by_view:
            self._repaint()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._repaint()

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self._repaint()

    def _repaint(self) -> None:
        parent = self.parentWidget()
        (parent if self.drawn_by_view and parent is not None else self).update()

    def _paint_buttons(self, p: QPainter) -> None:
        for name, path in self._buttons.items():
            hot = name == self._hover_btn
            p.setPen(QPen(QColor(230, 230, 230) if hot else QColor(150, 150, 150), 1.2))
            p.setBrush(QColor(255, 255, 255, 60) if hot else QColor(255, 255, 255, 20))
            if name == "spin":
                r = path.boundingRect()
                p.drawArc(r, 30 * 16, 300 * 16)
                p.drawLine(QPointF(r.right() - 3, r.center().y() - 2), QPointF(r.right() - 7, r.center().y() + 5))
                p.drawLine(QPointF(r.right() - 3, r.center().y() - 2), QPointF(r.right() + 3, r.center().y() + 3))
            elif name == "below":
                r = path.boundingRect()
                p.drawLine(QPointF(r.left(), r.top() + 5), QPointF(r.right(), r.top() + 5))
                p.drawLine(QPointF(r.left(), r.top() + 10), QPointF(r.right(), r.top() + 10))
                p.drawPath(path)
            else:
                p.drawPath(path)

    def _paint_cube(self, p: QPainter) -> None:
        r, u, f = self._projection()
        center = (self.SIZE / 2, self.SIZE / 2)
        scale = self.SIZE * 0.25 / _C
        self._face_polys = []
        order = []
        for i, (n, ring) in enumerate(FACES):
            facing = -(n @ f)              # > 0 when the face looks at the camera
            if facing <= 0.02:
                continue
            depth = ring.mean(axis=0) @ f
            order.append((depth, i, facing))
        order.sort()                        # far first (f points away from the camera)

        p.setFont(QFont("Sans", 8, QFont.Bold))
        for _depth, i, facing in order:
            n, ring = FACES[i]
            poly = QPolygonF([self._project(v, r, u, scale, center) for v in ring])
            self._face_polys.append((i, poly))
            key = tuple(int(round(c)) for c in n) if np.allclose(np.abs(n).max(), 1.0) else None
            is_main = key in MAIN_FACES
            shade = 120 + int(100 * facing)
            if i == self._hover:
                color = QColor(255, 200, 90)
            else:
                color = QColor(shade, shade, shade + 6) if is_main else QColor(shade - 15, shade - 15, shade - 10)
            p.setBrush(color)
            p.setPen(QPen(QColor(50, 50, 55), 1))
            p.drawPolygon(poly)
            if is_main and facing > 0.25:
                label, fr, fu = MAIN_FACES[key]
                self._draw_face_label(p, label, np.array(fr, float), np.array(fu, float),
                                      n * _C, r, u, scale, center)

    def _draw_face_label(self, p, text, fr, fu, face_center, r, u, scale, center) -> None:
        """Draw text lying flat on a face: map the face's 2D frame to screen."""
        c = self._project(face_center, r, u, scale, center)
        rx, ry = (fr @ r), -(fr @ u)
        ux, uy = (fu @ r), -(fu @ u)
        k = scale * 0.05
        t = QTransform(rx * k, ry * k, -ux * k, -uy * k, c.x(), c.y())
        p.save()
        p.setTransform(t, True)
        p.setPen(QColor(35, 35, 40))
        p.drawText(QRectF(-40, -10, 80, 20), Qt.AlignCenter, text)
        p.restore()

    def _paint_axes(self, p: QPainter) -> None:
        r, u, _ = self._projection()
        origin = (26.0, self.SIZE - 26.0)
        length = 22.0
        for axis, color in ((np.array([1.0, 0, 0]), QColor(220, 70, 70)),
                            (np.array([0, 1.0, 0]), QColor(70, 190, 70)),
                            (np.array([0, 0, 1.0]), QColor(80, 110, 240))):
            end = QPointF(origin[0] + (axis @ r) * length, origin[1] - (axis @ u) * length)
            p.setPen(QPen(color, 2))
            p.drawLine(QPointF(*origin), end)

    # ------------------------------------------------------------ mouse
    def _hit(self, pos: QPointF):
        for name, path in self._buttons.items():
            if path.boundingRect().adjusted(-3, -3, 3, 3).contains(pos):
                return ("btn", name)
        for i, poly in reversed(self._face_polys):   # front-most last drawn
            if poly.containsPoint(pos, Qt.OddEvenFill):
                return ("face", i)
        return None

    def mouseMoveEvent(self, e) -> None:
        hit = self._hit(e.position())
        hover = hit[1] if hit and hit[0] == "face" else None
        hover_btn = hit[1] if hit and hit[0] == "btn" else None
        if hover != self._hover or hover_btn != self._hover_btn:
            self._hover, self._hover_btn = hover, hover_btn
            self.setCursor(Qt.PointingHandCursor if hit else Qt.ArrowCursor)
            self._repaint()

    def leaveEvent(self, _e) -> None:
        self._hover = self._hover_btn = None
        self._repaint()

    def mousePressEvent(self, e) -> None:
        hit = self._hit(e.position())
        if hit is None:
            e.ignore()
            return
        kind, what = hit
        if kind == "face":
            self.view_direction.emit(FACES[what][0].copy())
        elif what == "home":
            self.home_requested.emit()
        elif what == "left":
            self.orbit_requested.emit(-45.0, 0.0)
        elif what == "right":
            self.orbit_requested.emit(45.0, 0.0)
        elif what == "up":
            self.orbit_requested.emit(0.0, 45.0)
        elif what == "down":
            self.orbit_requested.emit(0.0, -45.0)
        elif what == "spin":
            self.orbit_requested.emit(90.0, 0.0)
        elif what == "below":
            self.view_direction.emit(np.array([0.0, 0.0, -1.0]))
