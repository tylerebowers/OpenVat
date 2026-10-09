"""Floating, collapsible side panels.

``CollapsibleSection`` is a card: an opaque rounded box with a clickable
header (folds the content away and shows a one-line summary instead) and
the content.  Cards holding lists get a grip on their bottom edge to make
them taller or shorter.

``SectionColumn`` lays its cards out itself - no scroll area - and clips
itself to exactly the cards' rounded shapes in the same step (a widget
mask), so the cards look and act like separate boxes floating over the 3D
view: the gaps between them, their rounded corners and whatever is below
the last one belong to the view (it shows there and gets the clicks and
the wheel).  When the cards are taller than the window the column scrolls
(wheel over a card, or drag the slim pill on the outer edge; keyboard
focus scrolls to the focused field).  Dragging a card's inner edge makes
the column wider or narrower.

Nothing here is see-through except the 1-pixel anti-aliased rim of the
corners, so how the window system composes widgets over OpenGL does not
matter.

``ViewportHost`` holds the 3D view and floats the two columns over its
edges, the navigation cube, and along the bottom a hint line (``HintLine``:
outlined text, no background) and an action button (Slice).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QPoint, QRect, QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath, QFont, QFontMetrics, QRegion
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QLabel, QSizePolicy, QApplication

CARD_RADIUS = 10          # px, rounded corners of the cards
CARD_GAP = 8              # px between cards
GUTTER = 8                # px kept on a column's outer side for the scroll pill
PILL_WIDTH = 4            # px
EDGE_WIDTH = 6            # px, the drag zone on a card's inner edge (resizes the column)
WHEEL_STEP = 48           # px scrolled per wheel notch
MIN_COLUMN_WIDTH = 200    # px
MAX_COLUMN_SHARE = 0.45   # a column may cover at most this share of the view's width


def _rounded_region(rect: QRect, radius: float) -> QRegion:
    """The pixels of a rounded rectangle, rim included (anti-aliased
    painting touches pixels half a pixel outside the outline)."""
    path = QPainterPath()
    path.addRoundedRect(QRectF(rect).adjusted(-0.5, -0.5, 0.5, 0.5), radius + 0.5, radius + 0.5)
    return QRegion(path.toFillPolygon().toPolygon())


class _ResizeGrip(QWidget):
    """A thin bar at the bottom of a card; drag it to change the card's
    content height."""

    def __init__(self, section: "CollapsibleSection"):
        super().__init__(section)
        self.section = section
        self.setFixedHeight(8)
        self.setCursor(Qt.SizeVerCursor)
        self._start_y = None
        self._start_h = 0

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setPen(QPen(self.palette().mid().color(), 1))
        cx, cy = self.width() / 2, self.height() / 2
        for dx in (-6, 0, 6):
            p.drawPoint(int(cx + dx), int(cy))

    def mousePressEvent(self, e) -> None:
        self._start_y = e.globalPosition().y()
        self._start_h = self.section.content.height()

    def mouseMoveEvent(self, e) -> None:
        if self._start_y is not None:
            h = int(self._start_h + e.globalPosition().y() - self._start_y)
            self.section.set_content_height(max(h, self.section.min_content_height))

    def mouseReleaseEvent(self, _e) -> None:
        self._start_y = None


class CollapsibleSection(QWidget):
    toggled = Signal(bool)
    resized = Signal()

    def __init__(self, title: str, content: QWidget, parent=None, expanded: bool = True,
                 resizable: bool = False, content_height: int = 150):
        super().__init__(parent)
        self.content = content
        self._expanded = expanded
        self.min_content_height = 60
        self.setAttribute(Qt.WA_StyledBackground, False)

        self.header = QToolButton()
        self.header.setText(title)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # flat and bold without a style sheet, so its text follows the theme
        self.header.setAutoRaise(True)
        bold = QFont(self.header.font())
        bold.setBold(True)
        self.header.setFont(bold)
        self.header.clicked.connect(lambda: self.set_expanded(not self._expanded))

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
        self.summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        head_row = QWidget()
        hl = QHBoxLayout(head_row)
        hl.setContentsMargins(6, 3, 10, 1)
        hl.addWidget(self.header)
        hl.addWidget(self.summary, stretch=1)
        head_row.setCursor(Qt.PointingHandCursor)
        head_row.mousePressEvent = lambda e: self.set_expanded(not self._expanded)

        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(10, 2, 10, 8 if not resizable else 0)
        bl.addWidget(content)
        if resizable:
            content.setFixedHeight(content_height)
            self.grip = _ResizeGrip(self)
            bl.addWidget(self.grip)
        else:
            self.grip = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)
        layout.addWidget(head_row)
        layout.addWidget(self.body)
        self.set_expanded(expanded)

    # ------------------------------------------------------------ look
    def paintEvent(self, _e) -> None:
        """An opaque rounded box with a hairline border."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), CARD_RADIUS, CARD_RADIUS)
        p.fillPath(path, self.palette().window())
        p.setPen(QPen(self.palette().mid().color(), 1))
        p.drawPath(path)

    # ------------------------------------------------------------ state
    def set_summary(self, text: str) -> None:
        self.summary.setText(text)

    def set_content_height(self, h: int) -> None:
        self.content.setFixedHeight(h)
        self.updateGeometry()
        self.resized.emit()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)
        self.summary.setVisible(not expanded)
        self.updateGeometry()
        self.toggled.emit(expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def height_for(self, width: int) -> int:
        """The card's height at ``width`` (wrapped labels grow taller)."""
        h = self.heightForWidth(width) if self.hasHeightForWidth() else -1
        if h < 0:
            h = self.sizeHint().height()
        return max(h, self.minimumSizeHint().height())


class _EdgeHandle(QWidget):
    """The inner edge of a card: drag it sideways to make the whole column
    wider or narrower.  It sits on the card's own opaque background and
    shows a thin accent line while hovered or dragged."""

    def __init__(self, card: CollapsibleSection, column: "SectionColumn"):
        super().__init__(card)
        self.column = column
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip("Drag to resize the panel")
        self.setAttribute(Qt.WA_NoSystemBackground)
        self._hover = False
        self._x0: float | None = None
        self._w0 = 0

    def enterEvent(self, _e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = False
        self.update()

    def paintEvent(self, _e) -> None:
        if not (self._hover or self._x0 is not None):
            return
        p = QPainter(self)
        x = self.width() - 3 if self.column.inner_on_right else 1
        p.fillRect(QRect(x, CARD_RADIUS, 2, max(self.height() - 2 * CARD_RADIUS, 0)),
                   self.palette().highlight())

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._x0 = e.globalPosition().x()
            self._w0 = self.column.width()

    def mouseMoveEvent(self, e) -> None:
        if self._x0 is not None:
            dx = e.globalPosition().x() - self._x0
            self.column.resize_requested.emit(int(self._w0 + (dx if self.column.inner_on_right else -dx)))

    def mouseReleaseEvent(self, _e) -> None:
        if self._x0 is not None:
            self._x0 = None
            self.update()
            self.column.resize_finished.emit()


class _ScrollPill(QWidget):
    """The slim scroll indicator on a column's outer edge; drag it to scroll."""

    def __init__(self, column: "SectionColumn"):
        super().__init__(column)
        self.column = column
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setCursor(Qt.ArrowCursor)
        self._hover = False
        self._y0: float | None = None
        self._off0 = 0

    def enterEvent(self, _e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = False
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = self.palette().highlight() if (self._hover or self._y0 is not None) else self.palette().mid()
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self.width() / 2, self.width() / 2)
        p.fillPath(path, color)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._y0 = e.globalPosition().y()
            self._off0 = self.column.offset

    def mouseMoveEvent(self, e) -> None:
        if self._y0 is not None:
            self.column.scroll_by_pill(self._off0, e.globalPosition().y() - self._y0)

    def mouseReleaseEvent(self, _e) -> None:
        self._y0 = None
        self.update()


class HintLine(QWidget):
    """A line of text drawn straight over the 3D view: no background, and
    every letter outlined in a contrasting color so it reads over the plate,
    the model and the background alike.  ``set_parts`` takes
    (text, color, bold) pieces."""

    PAD = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parts: list[tuple[str, QColor, bool]] = []
        self._halo = QColor(0, 0, 0, 170)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_parts(self, parts: list[tuple[str, QColor, bool]], halo: QColor) -> None:
        self._parts, self._halo = list(parts), QColor(halo)
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return "".join(t for t, _c, _b in self._parts)

    def _font(self, bold: bool) -> QFont:
        f = QFont(self.font())
        f.setBold(bold)
        return f

    def sizeHint(self) -> QSize:
        w = sum(QFontMetrics(self._font(b)).horizontalAdvance(t) for t, _c, b in self._parts)
        return QSize(w + 2 * self.PAD, self.fontMetrics().height() + 2 * self.PAD)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        fm = self.fontMetrics()
        base = (self.height() + fm.ascent() - fm.descent()) / 2
        halo = QPainterPath()
        x = float(self.PAD)
        runs = []
        for text, color, bold in self._parts:
            font = self._font(bold)
            halo.addText(x, base, font, text)
            runs.append((x, text, color, font))
            x += QFontMetrics(font).horizontalAdvance(text)
        p.strokePath(halo, QPen(self._halo, 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for x, text, color, font in runs:
            p.setFont(font)
            p.setPen(color)
            p.drawText(QPointF(x, base), text)


class SectionColumn(QWidget):
    """A vertical stack of floating CollapsibleSection cards.

    ``layout_changed`` fires when its natural height may have changed (the
    host then gives it a new height); ``resize_requested(width)`` /
    ``resize_finished`` come from dragging a card's inner edge."""
    layout_changed = Signal()
    resize_requested = Signal(int)
    resize_finished = Signal()

    def __init__(self, width: int, parent=None):
        super().__init__(parent)
        self.setFixedWidth(width)
        self.setAutoFillBackground(False)
        self.sections: dict[str, CollapsibleSection] = {}
        self._cards: list[CollapsibleSection] = []
        self._handles: dict[CollapsibleSection, _EdgeHandle] = {}
        self.inner_on_right = True              # a column on the window's left: inner edge right
        self.offset = 0                         # px scrolled
        self._content = 0                       # total height of the cards
        self._regions: dict[tuple[int, int], QRegion] = {}
        self._pill = _ScrollPill(self)
        self._pill.hide()
        QApplication.instance().focusChanged.connect(self._focus_changed)

    # ------------------------------------------------------------ building
    def set_outer_side(self, left: bool) -> None:
        """``left``: the column sits at the window's left edge (scroll pill
        on the left, resize edge on the right of its cards)."""
        self.inner_on_right = left
        self._layout()

    def add(self, key: str, title: str, content: QWidget, expanded: bool = True,
            stretch: int = 0) -> CollapsibleSection:
        """``stretch`` > 0 marks a card that holds a list: it gets a resize grip."""
        sec = CollapsibleSection(title, content, self, expanded=expanded, resizable=stretch > 0)
        sec.toggled.connect(lambda _on: self.relayout())
        sec.resized.connect(self.relayout)
        sec.installEventFilter(self)            # its size hint changes -> lay out again
        self._handles[sec] = _EdgeHandle(sec, self)
        self._cards.append(sec)
        self.sections[key] = sec
        sec.show()
        return sec

    def finish(self) -> None:
        self.relayout()

    def relayout(self) -> None:
        self.layout_changed.emit()              # the host may give the column a new height ...
        self._layout()                          # ... and the cards are placed and clipped right away

    # ------------------------------------------------------------ sizes
    def card_width(self) -> int:
        return max(self.width() - GUTTER, 10)

    def natural_height(self) -> int:
        cards = self._shown()
        cw = self.card_width()
        return sum(c.height_for(cw) for c in cards) + CARD_GAP * max(len(cards) - 1, 0)

    def minimum_width(self) -> int:
        return max([c.minimumSizeHint().width() for c in self._cards] + [0]) + GUTTER

    def _shown(self) -> list[CollapsibleSection]:
        return [c for c in self._cards if not c.isHidden()]

    # ------------------------------------------------------------ layout + mask
    def _layout(self) -> None:
        """Place the cards (scrolled by ``offset``) and clip the column to
        them - one step, so what is drawn and what is clickable always agree."""
        w, h = self.width(), self.height()
        cards = self._shown()
        cw = self.card_width()
        x0 = GUTTER if self.inner_on_right else 0           # the pill gutter is on the outer side
        heights = [c.height_for(cw) for c in cards]
        self._content = sum(heights) + CARD_GAP * max(len(cards) - 1, 0)
        self.offset = max(0, min(self.offset, self._content - h))
        region = QRegion()
        y = -self.offset
        for card, ch in zip(cards, heights):
            card.setGeometry(x0, y, cw, ch)
            handle = self._handles[card]
            handle.setGeometry(cw - EDGE_WIDTH if self.inner_on_right else 0, 0, EDGE_WIDTH, ch)
            handle.raise_()
            if y + ch > 0 and y < h:
                key = (cw, ch)
                if key not in self._regions:
                    if len(self._regions) > 64:
                        self._regions.clear()
                    self._regions[key] = _rounded_region(QRect(0, 0, cw, ch), CARD_RADIUS)
                region += self._regions[key].translated(x0, y)
            y += ch + CARD_GAP
        region &= QRegion(self.rect())
        if self._content > h > 0:                            # scrolling: show the pill
            th = max(24, h * h // self._content)
            ty = round((h - th) * self.offset / (self._content - h))
            px = (GUTTER - PILL_WIDTH) // 2 if self.inner_on_right else w - GUTTER + (GUTTER - PILL_WIDTH) // 2
            self._pill.setGeometry(px, ty, PILL_WIDTH, th)
            self._pill.show()
            self._pill.raise_()
            region += QRegion(self._pill.geometry())
        else:
            self._pill.hide()
        self.setMask(region if not region.isEmpty() else QRegion(0, 0, 1, 1))  # empty would mean "no mask"

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._layout()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.LayoutRequest and obj in self._handles:
            self.relayout()                     # a card's contents changed size
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------ scrolling
    def scroll_to(self, offset: int) -> None:
        offset = max(0, min(int(offset), self._content - self.height()))
        if offset != self.offset:
            self.offset = offset
            self._layout()

    def scroll_by_pill(self, start_offset: int, dy: float) -> None:
        h = self.height()
        track = h - self._pill.height()
        if track > 0:
            self.scroll_to(start_offset + dy * (self._content - h) / track)

    def wheelEvent(self, e) -> None:
        """Wheel over a card that did not use it (a spin box does): scroll
        the column if it does not fit."""
        if self._content <= self.height():
            e.ignore()
            return
        pixels = e.pixelDelta().y()
        dy = -pixels if pixels else -e.angleDelta().y() / 120 * WHEEL_STEP
        self.scroll_to(self.offset + dy)
        e.accept()

    def _focus_changed(self, _old, new) -> None:
        """Keyboard focus moved into this column: scroll it into view."""
        if new is None or not self.isAncestorOf(new) or self._content <= self.height():
            return
        top = new.mapTo(self, QPoint(0, 0)).y()
        bottom = top + new.height()
        if top < 0:
            self.scroll_to(self.offset + top - CARD_GAP)
        elif bottom > self.height():
            self.scroll_to(self.offset + bottom - self.height() + CARD_GAP)


class ViewportHost(QWidget):
    """Holds the 3D view and floats over it: the two card columns at its
    left and right edges, the navigation cube left of the right column, and
    along the bottom a hint line (transparent, clicks go through to the
    view) and an action button.  ``widths_changed`` fires when the user has
    resized a column."""

    MARGIN = 8                # px around the floating things
    EDGE = 2                  # px from the window edge to a column (its gutter adds the rest)
    widths_changed = Signal()

    def __init__(self, viewport: QWidget, left: SectionColumn, right: SectionColumn,
                 overlay: QWidget | None = None, hint: QWidget | None = None,
                 action: QWidget | None = None, parent=None):
        super().__init__(parent)
        self.viewport, self.left, self.right, self.overlay = viewport, left, right, overlay
        self.hint, self.action = hint, action
        viewport.setParent(self)
        for col in (left, right):
            col.setParent(self)
            col.layout_changed.connect(self._place)
            col.resize_requested.connect(lambda w, c=col: self.set_column_width(c, w))
            col.resize_finished.connect(self.widths_changed)
            col.raise_()
        left.set_outer_side(True)
        right.set_outer_side(False)
        # the width the user picked; a narrow window may show a column narrower
        # for a while, it gets this width back when the window grows again
        self._wanted = {left: left.width(), right: right.width()}
        if hint is not None:
            hint.setParent(self)
            hint.setAttribute(Qt.WA_TransparentForMouseEvents)     # clicks reach the 3D view
            hint.setAutoFillBackground(False)
        if action is not None:
            action.setParent(self)
        for w in (hint, action):
            if w is not None:
                w.raise_()

    def set_column_width(self, column: SectionColumn, width: int) -> None:
        """Resize a column within sensible limits (also used to restore saved
        widths before the window is shown)."""
        self._wanted[column] = self._clamp(column, int(width))
        column.setFixedWidth(self._wanted[column])
        self._place()

    def column_width(self, column: SectionColumn) -> int:
        """The width the user picked for a column."""
        return self._wanted[column]

    def _clamp(self, column: SectionColumn, width: int) -> int:
        low = max(MIN_COLUMN_WIDTH, column.minimum_width())
        # not laid out yet (Qt's default size): don't cut a restored width down
        high = int(self.width() * MAX_COLUMN_SHARE) if self.isVisible() else 10_000
        return max(low, min(width, max(high, low)))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        for col in (self.left, self.right):           # keep the columns within the new size
            w = self._clamp(col, self._wanted[col])
            if col.width() != w:
                col.setFixedWidth(w)
        self._place()

    def relayout(self) -> None:
        """Re-place the floating widgets (e.g. after the hint text changed)."""
        self._place()

    def _place(self) -> None:
        m = self.MARGIN
        w, h = self.width(), self.height()
        self.viewport.setGeometry(0, 0, w, h)
        band = 0                                      # room kept free along the bottom
        if self.action is not None:
            ah = self.action.sizeHint().height()
            aw = self.action.sizeHint().width()
            self.action.setGeometry(w - m - aw, h - m - ah, aw, ah)
            band = ah + m
        if self.hint is not None:
            hh = self.hint.sizeHint().height()
            right = (w - m - self.action.sizeHint().width() - m) if self.action is not None else w - m
            self.hint.setGeometry(m + 4, h - m - max(hh, band - m), max(right - m - 4, 10), max(hh, band - m))
            band = max(band, hh + m)
        avail = max(h - 2 * m - band, 50)
        for col, x in ((self.left, self.EDGE), (self.right, w - self.right.width() - self.EDGE)):
            col.setGeometry(x, m, col.width(), min(col.natural_height(), avail))
            col._layout()
        if self.overlay is not None:          # navigation cube: left of the right column's cards
            self.overlay.move(w - self.EDGE - self.right.width() - m - self.overlay.width(), m)
