"""Edit → Preferences: light / dark theme (dark unless the user picks
light), the 3D view's colors for each theme, the interface zoom and the
folder the printer and resin profiles are kept in.

The theme switches the whole application between an explicit light and an
explicit dark palette (Fusion style for both, so it changes even when the
desktop itself is dark) together with that theme's 3D view colors.  Changes
apply at once (Cancel puts the old ones back) and are saved in
``~/.openvat/settings.json``.  The zoom is applied through Qt's
QT_SCALE_FACTOR, which has to be set before the QApplication exists - so
changing it offers to restart OpenVat.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from pathlib import Path

from PySide6.QtCore import QProcess, Qt, Signal, QUrl
from PySide6.QtGui import QColor, QPalette, QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QComboBox, QLabel, QCheckBox,
                               QVBoxLayout, QHBoxLayout, QMessageBox, QApplication, QPushButton, QColorDialog,
                               QGroupBox, QGridLayout, QStyleFactory, QLineEdit, QFileDialog)

from ..core.profiles import ProfileStore, PRINTERS
from .theme_colors import DEFAULT_THEME_COLORS

ZOOM_LEVELS = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
THEMES = ("light", "dark")
COLOR_LABELS = {                       # the order shown in the dialog
    "background": "Background",
    "plate": "Build plate",
    "model": "Model",
    "selected": "Selected model",
    "support": "Supports",
    "support_selected": "Selected support",
}


def apply_ui_scale(store: ProfileStore) -> None:
    """Call before creating the QApplication.  An explicit QT_SCALE_FACTOR in
    the environment wins over the setting."""
    scale = float(store.settings.get("ui_scale", 1.0))
    if "QT_SCALE_FACTOR" not in os.environ and abs(scale - 1.0) > 1e-3:
        os.environ["QT_SCALE_FACTOR"] = f"{scale:g}"


def restart_application() -> None:
    """Start a fresh OpenVat with the same arguments and quit this one."""
    if os.environ.get("APPIMAGE"):                       # running from an AppImage
        program, args = os.environ["APPIMAGE"], sys.argv[1:]
    elif getattr(sys, "frozen", False):                  # PyInstaller build
        program, args = sys.executable, sys.argv[1:]
    elif sys.argv[0].endswith("__main__.py"):            # python -m openvat
        program, args = sys.executable, ["-m", "openvat", *sys.argv[1:]]
    else:                                                # python run.py / console script
        program, args = sys.executable, sys.argv
    os.environ.pop("QT_SCALE_FACTOR", None)              # let the new setting apply
    QProcess.startDetached(program, args)
    QApplication.quit()


# --------------------------------------------------------------------------
# preferences in settings.json

def saved_colors(store: ProfileStore) -> dict:
    """The 3D view colors of both themes - {"light": {...}, "dark": {...}}:
    the defaults, overridden by what the user picked."""
    picked = store.settings.get("colors", {})
    if picked and not any(isinstance(v, dict) for v in picked.values()):
        picked = {"dark": picked}                       # settings from before there were two themes
    out = {}
    for theme in THEMES:
        colors = dict(DEFAULT_THEME_COLORS[theme])
        mine = picked.get(theme, {}) if isinstance(picked.get(theme), dict) else {}
        colors.update({k: v for k, v in mine.items() if k in colors})
        out[theme] = colors
    return out


def theme_name(dark: bool) -> str:
    return "dark" if dark else "light"


def dark_mode(store: ProfileStore) -> bool:
    """Dark mode is on unless the user switched it off."""
    return bool(store.settings.get("dark_mode", True))


def apply_dark_mode(on: bool) -> None:
    """Switch the whole application to the dark or the light theme.  Both
    are explicit palettes on the Fusion style, so switching always shows,
    whatever the desktop's own theme is."""
    app = QApplication.instance()
    if app is None:
        return
    if app.style().name().lower() != "fusion":
        app.setStyle("Fusion")
    app.setPalette(dark_palette() if on else light_palette())


def light_palette() -> QPalette:
    style = QStyleFactory.create("Fusion")
    return QPalette(style.standardPalette()) if style is not None else QPalette()


def dark_palette() -> QPalette:
    p = QPalette()
    text, dim = QColor(222, 224, 228), QColor(128, 131, 138)
    for role, color in ((QPalette.Window, QColor(44, 46, 51)), (QPalette.WindowText, text),
                        (QPalette.Base, QColor(31, 33, 37)), (QPalette.AlternateBase, QColor(39, 41, 46)),
                        (QPalette.ToolTipBase, QColor(52, 54, 60)), (QPalette.ToolTipText, text),
                        (QPalette.PlaceholderText, dim), (QPalette.Text, text),
                        (QPalette.Button, QColor(54, 56, 62)), (QPalette.ButtonText, text),
                        (QPalette.BrightText, QColor(255, 90, 90)), (QPalette.Light, QColor(72, 75, 82)),
                        (QPalette.Midlight, QColor(62, 65, 71)), (QPalette.Mid, QColor(84, 87, 95)),
                        (QPalette.Dark, QColor(26, 27, 31)), (QPalette.Shadow, QColor(12, 12, 14)),
                        (QPalette.Highlight, QColor(52, 120, 198)), (QPalette.HighlightedText, QColor(255, 255, 255)),
                        (QPalette.Link, QColor(98, 165, 235)), (QPalette.LinkVisited, QColor(160, 130, 230))):
        p.setColor(role, color)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, dim)
    return p


# --------------------------------------------------------------------------
# the dialog

class ColorButton(QPushButton):
    """A button showing a color swatch; clicking it opens a color picker."""
    changed = Signal(str)

    def __init__(self, color: str, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setMinimumWidth(110)
        self.clicked.connect(self._pick)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        self.color = QColor(color).name()
        text = "#000000" if QColor(self.color).lightnessF() > 0.55 else "#ffffff"
        self.setText(self.color)
        self.setStyleSheet(f"QPushButton {{ background: {self.color}; color: {text}; border: 1px solid #777;"
                           f" border-radius: 4px; padding: 4px 10px; }}")

    def _pick(self) -> None:
        picked = QColorDialog.getColor(QColor(self.color), self, self.title)
        if picked.isValid() and picked.name() != self.color:
            self.set_color(picked.name())
            self.changed.emit(self.color)


class PreferencesDialog(QDialog):
    """``preview(dark_mode, colors)`` is called whenever something changes
    (``colors`` holds both themes, see ``saved_colors``), so the main window
    can show it right away - and again with the old values on Cancel."""

    def __init__(self, store: ProfileStore, preview: Callable[[bool, dict], None], parent=None):
        super().__init__(parent)
        self.store, self.preview = store, preview
        self._old = (dark_mode(store), saved_colors(store))
        self.profiles_folder = store.profiles_dir          # applied on OK
        self.setWindowTitle("Preferences")
        root = QVBoxLayout(self)

        look = QGroupBox("Appearance")
        form = QFormLayout(look)
        self.dark = QCheckBox("Dark mode")
        self.dark.setChecked(self._old[0])
        self.dark.toggled.connect(self._changed)
        form.addRow("", self.dark)
        self.zoom = QComboBox()
        for z in ZOOM_LEVELS:
            self.zoom.addItem(f"{z * 100:.0f} %", z)
        current = float(store.settings.get("ui_scale", 1.0))
        self.zoom.setCurrentIndex(min(range(len(ZOOM_LEVELS)), key=lambda i: abs(ZOOM_LEVELS[i] - current)))
        form.addRow("Interface zoom", self.zoom)
        note = QLabel("<i>Zoom scales all text, buttons and panels; it takes effect after a restart.</i>")
        note.setWordWrap(True)
        form.addRow(note)
        root.addWidget(look)

        colors = QGroupBox("3D view colors")
        grid = QGridLayout(colors)
        grid.setHorizontalSpacing(10)
        self.headers: dict[str, QLabel] = {}
        for col, theme in enumerate(THEMES, start=1):
            head = QLabel(f"<b>{theme.capitalize()} mode</b>")
            head.setAlignment(Qt.AlignCenter)
            grid.addWidget(head, 0, col)
            self.headers[theme] = head
        self.buttons: dict[str, dict[str, ColorButton]] = {theme: {} for theme in THEMES}
        for row, (key, label) in enumerate(COLOR_LABELS.items(), start=1):
            grid.addWidget(QLabel(label), row, 0)
            for col, theme in enumerate(THEMES, start=1):
                b = ColorButton(self._old[1][theme][key], f"{label} ({theme} mode)")
                b.changed.connect(self._changed)
                grid.addWidget(b, row, col)
                self.buttons[theme][key] = b
        reset = QPushButton("Default colors")
        reset.clicked.connect(self._default_colors)
        grid.addWidget(reset, len(COLOR_LABELS) + 1, 1, 1, 2)
        root.addWidget(colors)

        where = QGroupBox("Printer and resin profiles")
        lay = QVBoxLayout(where)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setMinimumWidth(360)
        lay.addWidget(self.folder_edit)
        row = QHBoxLayout()
        change = QPushButton("Change…")
        change.clicked.connect(self._pick_folder)
        default = QPushButton("Default")
        default.setToolTip(f"Back to {store.home}")
        default.clicked.connect(lambda: self._set_folder(store.home))
        show = QPushButton("Open folder")
        show.clicked.connect(self._open_folder)
        for b in (change, default, show):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.folder_note = QLabel()
        self.folder_note.setWordWrap(True)
        lay.addWidget(self.folder_note)
        root.addWidget(where)
        self._set_folder(self.profiles_folder)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)
        self._mark_active()

    def values(self) -> tuple[bool, dict]:
        return self.dark.isChecked(), {theme: {k: b.color for k, b in buttons.items()}
                                       for theme, buttons in self.buttons.items()}

    def _mark_active(self) -> None:
        """Underline the column the 3D view uses now."""
        active = theme_name(self.dark.isChecked())
        for theme, head in self.headers.items():
            text = f"{theme.capitalize()} mode"
            head.setText(f"<b><u>{text}</u></b> ◂ in use" if theme == active else f"<b>{text}</b>")

    def _changed(self, *_args) -> None:
        self._mark_active()
        self.preview(*self.values())

    # -- the profiles folder (applied on OK) -----------------------------
    def _set_folder(self, folder: Path) -> None:
        self.profiles_folder = Path(folder)
        self.folder_edit.setText(str(self.profiles_folder))
        moved = self.profiles_folder != self.store.profiles_dir
        self.folder_note.setText(
            f"<i>Each printer is one file in <b>{PRINTERS}/</b> here, with its resins.  Settings stay in "
            f"{self.store.home}.  A shared or synced folder lets several computers use the same profiles."
            + ("<br><b>Changes when you press OK.</b>" if moved else "") + "</i>")

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder for printer and resin profiles",
                                                  str(self.profiles_folder))
        if folder:
            self._set_folder(Path(folder))

    def _open_folder(self) -> None:
        folder = self.profiles_folder
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _move_profiles(self) -> bool:
        """Switch to the picked folder, copying the profiles there if the user
        wants.  False: stay in the dialog (cancelled, or it failed)."""
        new = self.profiles_folder
        if new == self.store.profiles_dir:
            return True
        there = ProfileStore.printers_in(new)
        mine = len(self.store.printers)
        box = QMessageBox(QMessageBox.Question, "Profiles folder",
                          f"Keep printer and resin profiles in\n{new}\nfrom now on?", parent=self)
        detail = (f"That folder has {len(there)} printer(s) already: {', '.join(there[:6])}"
                  + ("…" if len(there) > 6 else "") + "." if there else "That folder has no printers yet.")
        box.setInformativeText(f"{detail}\n\nCopy your {mine} printer(s) with their resins there? "
                               "Same-named ones are replaced; the old folder keeps its copy.")
        copy = box.addButton("Copy mine there", QMessageBox.AcceptRole) if mine else None
        use = box.addButton("Use the folder as it is", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(copy or use)
        box.exec()
        if box.clickedButton() not in (copy, use) or box.clickedButton() is None:
            return False
        try:
            self.store.set_profiles_dir(new, copy=box.clickedButton() is copy)
        except OSError as exc:
            QMessageBox.warning(self, "Profiles folder", f"Could not use {new}:\n{exc}")
            return False
        return True

    def _default_colors(self) -> None:
        for theme, buttons in self.buttons.items():
            for key, b in buttons.items():
                b.set_color(DEFAULT_THEME_COLORS[theme][key])
        self._changed()

    def reject(self) -> None:
        self.preview(*self._old)
        super().reject()

    def accept(self) -> None:
        if not self._move_profiles():
            return
        dark, colors = self.values()
        self.store.settings["dark_mode"] = dark
        self.store.settings["colors"] = {
            theme: {k: v for k, v in colors[theme].items() if v != DEFAULT_THEME_COLORS[theme][k]}
            for theme in THEMES}
        new = float(self.zoom.currentData())
        old = float(self.store.settings.get("ui_scale", 1.0))
        self.store.settings["ui_scale"] = new
        self.store.save_settings()
        super().accept()
        if abs(new - old) > 1e-3 and QMessageBox.question(
                self.parent(), "Interface zoom",
                f"Restart OpenVat now to apply {new * 100:.0f} % zoom?\n"
                "(Unsaved work in the scene will be lost.)") == QMessageBox.Yes:
            restart_application()
