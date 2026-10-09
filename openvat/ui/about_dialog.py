"""Help → About: logo, authors, project link, version and the OpenGL renderer."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox

from .. import APP_NAME, AUTHORS, GITHUB_URL, __version__
from ..core.profiles import RESOURCES
from .viewport3d import gl_summary, software_renderer

SECTION_SPACING = 18
LOGO_WIDTH = 360          # px (the image has a margin round the lettering)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(0)

        # logo: scaled for the screen's pixel density and never squeezed by the layout
        logo = QLabel()
        pix = QPixmap(str(RESOURCES / "logo.png"))
        if not pix.isNull():
            ratio = self.devicePixelRatioF()
            pix = pix.scaledToWidth(round(LOGO_WIDTH * ratio), Qt.SmoothTransformation)
            pix.setDevicePixelRatio(ratio)
            logo.setPixmap(pix)
            logo.setFixedSize(round(pix.width() / ratio), round(pix.height() / ratio))
        else:
            logo.setText(f"<h1>{APP_NAME}</h1>")
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo, alignment=Qt.AlignHCenter)
        layout.addSpacing(SECTION_SPACING)

        # authors, one per line
        layout.addWidget(self._heading("Authors"))
        layout.addSpacing(4)
        for name in AUTHORS:
            layout.addWidget(self._centered(name))
        layout.addSpacing(SECTION_SPACING)

        # project link
        layout.addWidget(self._heading("Source code"))
        layout.addSpacing(4)
        link = self._centered(f'<a href="{GITHUB_URL}">{GITHUB_URL}</a>')
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(link)
        layout.addSpacing(SECTION_SPACING)

        # version
        layout.addWidget(self._heading("Version"))
        layout.addSpacing(4)
        layout.addWidget(self._centered(__version__))
        layout.addSpacing(SECTION_SPACING)

        # which OpenGL implementation draws the 3D views
        layout.addWidget(self._heading("Graphics"))
        layout.addSpacing(4)
        gl = self._centered(gl_summary())
        gl.setWordWrap(True)
        gl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(gl)
        if software_renderer():
            note = self._centered(
                "<span style='color:#e05a5a'>This is a software renderer: the 3D views run on the CPU "
                "and will be slow.<br>Install or update the graphics driver; on laptops with two GPUs, "
                "pick the dedicated one for OpenVat.</span>")
            note.setWordWrap(True)
            layout.addSpacing(4)
            layout.addWidget(note)
        layout.addSpacing(SECTION_SPACING)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(f"<b>{text}</b>")
        label.setAlignment(Qt.AlignCenter)
        return label

    @staticmethod
    def _centered(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        return label
