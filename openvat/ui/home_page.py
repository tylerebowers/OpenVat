"""The home page: the logo with buttons to start under it, two rows of
cards - the model files opened lately and the projects saved or opened
lately (with a picture of their build plate) - and running totals (see
core/stats.py).  Click a card to open it; right-click for "Show in folder"
and "Remove from list".  Files that no longer exist are left out.  Like the
other pages, the button to move on ("Prepare →", the build plate) is at
the bottom right."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize, QRectF, QUrl, QEvent
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QPen, QColor, QFont, QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
                               QMenu, QSizePolicy, QGridLayout)

from .. import APP_NAME, __version__
from ..core.profiles import RESOURCES
from ..core.project import read_thumbnail
from ..core.stats import STATS, shown
from .nav import NavButton, nav_gap
from .zoom import relative_font, scaled

CARD_RADIUS = 10
FILE_CARD = QSize(240, 190)            # at 100 % zoom (``scaled`` follows the zoom)
PROJECT_CARD = QSize(340, 270)


def _scaled(size: QSize) -> QSize:
    return QSize(scaled(size.width()), scaled(size.height()))


def _ago(seconds: float) -> str:
    """'just now', '5 min ago', 'yesterday', '12 Oct 2026'."""
    d = time.time() - seconds
    if d < 90:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    if d < 86400:
        return f"{int(d // 3600)} h ago"
    if d < 2 * 86400:
        return "yesterday"
    if d < 7 * 86400:
        return f"{int(d // 86400)} days ago"
    return time.strftime("%d %b %Y", time.localtime(seconds))


class Card(QFrame):
    """A rounded, clickable card: a picture (or a badge) over a name and a
    detail line."""
    clicked = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, path: str, size: QSize, picture: QPixmap | None, badge: str, detail: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._hover = False
        self._picture = picture
        self._badge = badge
        self.setFixedSize(size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(path)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        self.art = QWidget()
        self.art.setAttribute(Qt.WA_TransparentForMouseEvents)
        lay.addWidget(self.art, 1)
        name = QLabel()
        relative_font(name, 1.0, True)
        name.setText(name.fontMetrics().elidedText(Path(path).stem, Qt.ElideRight, size.width() - 20))
        sub = QLabel()
        relative_font(sub, 0.85)
        sub.setText(sub.fontMetrics().elidedText(detail, Qt.ElideMiddle, size.width() - 20))
        sub.setStyleSheet("color: #888;")
        for w in (name, sub):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
            lay.addWidget(w)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        card = QPainterPath()
        card.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), CARD_RADIUS, CARD_RADIUS)
        p.fillPath(card, pal.base() if not self._hover else pal.alternateBase())
        p.setPen(QPen(pal.highlight().color() if self._hover else pal.mid().color(), 1.5 if self._hover else 1))
        p.drawPath(card)
        art = QRectF(self.art.geometry())
        if self._picture is not None and not self._picture.isNull():
            pix = self._picture.scaled(art.size().toSize() * self.devicePixelRatioF(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)
            pix.setDevicePixelRatio(self.devicePixelRatioF())
            w, h = pix.width() / pix.devicePixelRatio(), pix.height() / pix.devicePixelRatio()
            clip = QPainterPath()
            target = QRectF(art.center().x() - w / 2, art.center().y() - h / 2, w, h)
            clip.addRoundedRect(target, 6, 6)
            p.setClipPath(clip)
            p.drawPixmap(target.topLeft(), pix)
            p.setClipping(False)
        else:                                   # no picture: the file type in a rounded badge
            side = min(art.width(), art.height()) * 0.8
            badge = QRectF(art.center().x() - side / 2, art.center().y() - side / 2, side, side)
            fill = QColor(pal.highlight().color())
            fill.setAlphaF(0.18)
            path = QPainterPath()
            path.addRoundedRect(badge, 10, 10)
            p.fillPath(path, fill)
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(max(f.pointSizeF() * 1.4, 10))
            p.setFont(f)
            p.setPen(pal.highlight().color())
            p.drawText(badge, Qt.AlignCenter, self._badge)

    def enterEvent(self, _e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.path)

    def _menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Open", lambda: self.clicked.emit(self.path))
        menu.addAction("Show in folder",
                       lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.path).parent))))
        menu.addSeparator()
        menu.addAction("Remove from list", lambda: self.remove_requested.emit(self.path))
        menu.exec(self.mapToGlobal(pos))


class CardRow(QScrollArea):
    """One row of cards that scrolls sideways (the wheel scrolls it too)."""

    def __init__(self, card_size: QSize, empty_text: str, parent=None):
        super().__init__(parent)
        self.card_size = card_size
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(_scaled(card_size).height() + 22)
        self.setStyleSheet("QScrollArea { background: transparent; }")
        self.inner = QWidget()
        self.inner.setAutoFillBackground(False)
        self.row = QHBoxLayout(self.inner)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(12)
        self.empty = QLabel(empty_text)
        self.empty.setStyleSheet("color: #888;")
        self.setWidget(self.inner)
        self.viewport().setAutoFillBackground(False)
        self.cards: list[Card] = []

    def set_cards(self, cards: list[Card]) -> None:
        while self.row.count():
            item = self.row.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.empty:
                w.hide()                        # gone at once, not when the event loop gets to it
                w.setParent(None)
                w.deleteLater()
        self.cards = cards
        self.setFixedHeight(_scaled(self.card_size).height() + 22)       # (the zoom may have changed)
        for c in cards:
            self.row.addWidget(c)
        if not cards:
            self.row.addWidget(self.empty)
        self.empty.setVisible(not cards)
        self.row.addStretch(1)

    def wheelEvent(self, e) -> None:
        bar = self.horizontalScrollBar()
        if bar.maximum() == 0:
            e.ignore()                          # nothing to scroll sideways: the page scrolls
            return
        delta = e.angleDelta().y() or e.angleDelta().x()
        bar.setValue(bar.value() - delta)
        e.accept()


class StatTile(QFrame):
    """A total: the number in a larger font over what it counts."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(0)
        self.value = QLabel("0")
        relative_font(self.value, 1.6, True)
        self.caption = QLabel(label)
        self.caption.setStyleSheet("color: #888;")
        self.caption.setWordWrap(True)
        self.detail = QLabel("")
        relative_font(self.detail, 0.85)
        self.detail.setStyleSheet("color: #888;")
        self.detail.hide()
        for w in (self.value, self.caption, self.detail):
            lay.addWidget(w)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), CARD_RADIUS, CARD_RADIUS)
        p.fillPath(path, self.palette().base())
        p.setPen(QPen(self.palette().mid().color(), 1))
        p.drawPath(path)


class HomePage(QWidget):
    open_file = Signal(str)              # a recent model file was clicked
    open_project = Signal(str)           # a recent project was clicked
    new_project = Signal()
    open_model_requested = Signal()
    open_project_requested = Signal()
    back_requested = Signal()            # back to the build plate
    remove_recent = Signal(str, str)     # "files" / "projects", path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumbs: dict[tuple, QPixmap | None] = {}      # (path, mtime) -> picture
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(10)

        self.logo = QLabel()
        pix = QPixmap(str(RESOURCES / "logo.png"))
        if not pix.isNull():
            h = 72
            ratio = self.devicePixelRatioF()
            pix = pix.scaledToHeight(round(h * ratio), Qt.SmoothTransformation)
            pix.setDevicePixelRatio(ratio)
            self.logo.setPixmap(pix)
        else:
            self.logo.setText(f"<h1>{APP_NAME}</h1>")
        self.logo.setToolTip(f"{APP_NAME} {__version__}")
        lay.addWidget(self.logo, 0, Qt.AlignHCenter)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        for text, signal, tip in (("New project", self.new_project, "An empty build plate"),
                                  ("Open project…", self.open_project_requested, "Open an OpenVat project"),
                                  ("Open model…", self.open_model_requested, "Add a model to the build plate")):
            b = QPushButton(text)
            b.setMinimumHeight(32)
            b.setStyleSheet("padding: 4px 16px;")
            b.setToolTip(tip)
            b.clicked.connect(signal)
            buttons.addWidget(b)
        buttons.addStretch(1)
        lay.addLayout(buttons)

        lay.addSpacing(8)
        lay.addWidget(self._heading("Recent files"))
        self.files_row = CardRow(FILE_CARD, "No model files opened yet - Open model…, or drop one here.")
        lay.addWidget(self.files_row)
        lay.addSpacing(6)
        lay.addWidget(self._heading("Recent projects"))
        self.projects_row = CardRow(PROJECT_CARD, "No projects yet - File → Save project keeps the build "
                                                  "plate, its supports and settings to open again.")
        lay.addWidget(self.projects_row)
        lay.addSpacing(6)
        lay.addWidget(self._heading("Totals"))
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(12)
        self.tiles: dict[str, StatTile] = {key: StatTile(label) for key, label in STATS.items()}
        self._columns = 0
        self._place_tiles(4)
        lay.addLayout(self.grid)
        lay.addStretch(1)

        # like the other pages: the button to move on at the bottom right
        bar = self._bar = QHBoxLayout()
        self._fit_gap()
        bar.addStretch(1)
        self.prepare_btn = NavButton("Prepare →", bold=True)
        self.prepare_btn.setToolTip("The build plate: place, support and slice the models")
        self.prepare_btn.clicked.connect(self.back_requested)
        bar.addWidget(self.prepare_btn)
        outer.addLayout(bar)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        relative_font(label, 1.25, True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return label

    # ------------------------------------------------------------ content
    def set_recent(self, files: list[str], projects: list[str], file_thumbs: dict | None = None) -> None:
        """``file_thumbs``: path -> picture file of a model (optional)."""
        file_thumbs = file_thumbs or {}
        cards = []
        for path in files:
            if not os.path.isfile(path):
                continue
            pic = self._picture(path, file_thumbs.get(path))
            ext = Path(path).suffix.lstrip(".").upper() or "?"
            c = Card(path, _scaled(FILE_CARD), pic, ext, self._detail(path, folder=True))
            c.clicked.connect(self.open_file)
            c.remove_requested.connect(lambda p: self.remove_recent.emit("files", p))
            cards.append(c)
        self.files_row.set_cards(cards)
        cards = []
        for path in projects:
            if not os.path.isfile(path):
                continue
            c = Card(path, _scaled(PROJECT_CARD), self._picture(path), "OV", self._detail(path, folder=False))
            c.clicked.connect(self.open_project)
            c.remove_requested.connect(lambda p: self.remove_recent.emit("projects", p))
            cards.append(c)
        self.projects_row.set_cards(cards)

    def _fit_gap(self) -> None:
        """The Prepare button floats NAV_GAP off the bottom right corner."""
        g = nav_gap()
        self._bar.setContentsMargins(6, 4, g, g)

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == QEvent.FontChange:              # the zoom changed
            self._fit_gap()

    def _place_tiles(self, columns: int) -> None:
        """The totals in one row on a wide window, in two or four on narrower ones."""
        if columns == self._columns:
            return
        self._columns = columns
        for tile in self.tiles.values():
            self.grid.removeWidget(tile)
        for i, tile in enumerate(self.tiles.values()):
            self.grid.addWidget(tile, i // columns, i % columns)
        for col in range(len(self.tiles)):
            self.grid.setColumnStretch(col, 1 if col < columns else 0)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        w = self.width()
        self._place_tiles(len(self.tiles) if w >= scaled(1180) else 4 if w >= scaled(640) else 2)

    def set_totals(self, totals: dict) -> None:
        """The running totals (core/stats.py)."""
        for key, tile in self.tiles.items():
            tile.value.setText(shown(key, totals.get(key, 0)))
        usb = int(totals.get("usb", 0))
        self.tiles["exports"].detail.setText(f"{usb:,} to a USB drive" if usb else "")
        for tile in self.tiles.values():
            tile.detail.setVisible(bool(tile.detail.text()))

    @staticmethod
    def _detail(path: str, folder: bool) -> str:
        try:
            when = _ago(os.path.getmtime(path))
        except OSError:
            when = ""
        if folder:
            return f"{Path(path).parent.name or Path(path).parent} · {when}"
        return f"saved {when}"

    def _picture(self, path: str, picture_file: str | None = None) -> QPixmap | None:
        """A project's build-plate picture, or a model's picture file."""
        src = picture_file or path
        try:
            key = (src, os.path.getmtime(src))
        except OSError:
            return None
        if key not in self._thumbs:
            pix = QPixmap()
            if picture_file is not None:
                ok = pix.load(picture_file)
            else:
                data = read_thumbnail(path)
                ok = bool(data) and pix.loadFromData(data)
            self._thumbs[key] = pix if ok else None
        return self._thumbs[key]
