"""Main window: menu bar, the 3D page and the layer-view page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from PySide6.QtGui import QAction, QKeySequence, QColor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QStackedWidget,
                               QFileDialog, QMessageBox, QProgressDialog, QPushButton,
                               QApplication)

from .. import APP_NAME, __version__
from ..core.loaders import load_file, FILE_FILTER, ALL_EXTENSIONS
from ..core.profiles import ProfileStore, PrinterProfile
from ..core.scene import Scene
from ..core.slicer import slice_scene, SliceResult
from ..core.supports import SupportSettings
from ..core.hollow import HollowSettings
from ..formats.export import write_print_file, why_not
from .about_dialog import AboutDialog
from .app_settings_dialog import PreferencesDialog, apply_dark_mode, dark_mode, saved_colors, theme_name
from .layer_view import LayerViewPage
from .voxel_view import VoxelViewPage
from .side_panel import SidePanel, add_first_printer
from .collapsible import ViewportHost, HintLine
from .tools_panel import ToolsPanel
from .view_cube import ViewCube
from .viewport3d import Viewport3D, software_renderer

HINT = ("Left-drag: move · Right-drag: orbit · Middle/Shift-drag: pan · "
        "Wheel: zoom · Double-click: fit · Drop files here to import")


class SliceWorker(QObject):
    """Runs the slicer in a background thread.  It only *emits* signals;
    all UI updates happen in the main thread via queued connections."""
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, scene, printer, resin):
        super().__init__()
        self.scene, self.printer, self.resin = scene, printer, resin

    @Slot()
    def run(self) -> None:
        try:
            result = slice_scene(self.scene, self.printer, self.resin, self.progress.emit)
        except Exception as exc:  # surfaced in the UI
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, store: ProfileStore | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self.scene = Scene()
        self.store = store or ProfileStore()
        if not self.store.printers and not add_first_printer(self.store):
            self.store.printers.append(PrinterProfile())     # unsaved stand-in
        if "supports" in self.store.settings:
            self.scene.support_settings = SupportSettings.from_dict(self.store.settings["supports"])
        if "hollowing" in self.store.settings:
            self.scene.hollow_settings = HollowSettings.from_dict(self.store.settings["hollowing"])
        self.result: SliceResult | None = None
        self._thread: QThread | None = None
        self._worker: SliceWorker | None = None
        self.progress: QProgressDialog | None = None

        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.pages.addWidget(self._build_3d_page())
        self.layer_page = LayerViewPage()
        self.layer_page.back_requested.connect(lambda: self.pages.setCurrentIndex(0))
        self.layer_page.export_requested.connect(self.export_print_file)
        self.layer_page.voxels_requested.connect(self._show_voxels)
        self.pages.addWidget(self.layer_page)
        self.voxel_page = VoxelViewPage()
        self.voxel_page.back_requested.connect(lambda: self.pages.setCurrentIndex(0))
        self.voxel_page.layers_requested.connect(lambda: self.pages.setCurrentIndex(1))
        self.voxel_page.export_requested.connect(self.export_print_file)
        self.pages.addWidget(self.voxel_page)

        self._build_menus()
        self._set_view_colors(saved_colors(self.store)[theme_name(dark_mode(self.store))])
        self._restore_panel_widths()
        self._printer_changed(self.side.current_printer())
        self.scene.listeners.append(self._update_hint)
        self._update_hint()

    # ------------------------------------------------------------ layout
    def _build_3d_page(self) -> QWidget:
        """The 3D view fills the page; the tool and profile panels float over
        its left and right edges (ViewportHost, they can be resized), the
        mouse hint and the Slice button over its bottom edge."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.viewport = Viewport3D(self.scene)
        self.tools = ToolsPanel(self.scene)
        self.side = SidePanel(self.store, self.scene)
        self.side.printer_changed.connect(self._printer_changed)
        self.tools.layer_height = self._layer_height        # Automatic looks for islands at this

        # the navigation cube floats over the view, left of the right panel
        self.view_cube = ViewCube(self.viewport.camera, self.viewport)
        self.view_cube.view_direction.connect(self._look_from)
        self.view_cube.orbit_requested.connect(self._orbit_by)
        self.view_cube.home_requested.connect(self._home_view)
        self.viewport.overlay = self.view_cube
        self.viewport.gl_started.connect(lambda: self._update_hint())   # the renderer is known now

        # the mouse hint floats over the bottom of the view (transparent; it also
        # warns about objects outside the build volume), the Slice button at its right
        self.hint = HintLine()
        slice_btn = QPushButton("Slice ▶")
        slice_btn.setMinimumHeight(32)
        slice_btn.setStyleSheet("font-weight: bold; padding: 4px 18px;")
        slice_btn.clicked.connect(self.slice)

        self.host = ViewportHost(self.viewport, self.tools, self.side, overlay=self.view_cube,
                                 hint=self.hint, action=slice_btn)
        self.host.widths_changed.connect(self._save_panel_widths)
        layout.addWidget(self.host, stretch=1)

        self.tools.mesh_edited.connect(self.viewport.invalidate_object)
        self.tools.arrange_requested.connect(self.arrange)
        self.tools.support_mode_toggled.connect(self._set_support_mode)
        self.tools.settings_changed.connect(self._save_support_settings)
        self.tools.hole_mode_toggled.connect(self._set_hole_mode)
        self.tools.show_inside_toggled.connect(self._set_show_inside)
        self.tools.hollow_settings_changed.connect(self._save_hollow_settings)
        self.viewport.hole_requested.connect(self.tools.add_hole_at)
        self.viewport.support_requested.connect(self.tools.add_support_at)
        self.viewport.selection_changed.connect(self.scene.changed)
        self.viewport.object_moved.connect(self.tools.refresh)
        return page

    def _build_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._action(file_menu, "&Open model…", self.open_models, QKeySequence.Open)
        self._action(file_menu, "&Export print file…", self.export_print_file, "Ctrl+E")
        file_menu.addSeparator()
        self._action(file_menu, "Clear scene", self.scene.clear)
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, QKeySequence.Quit)

        edit_menu = bar.addMenu("&Edit")
        self._action(edit_menu, "&Undo", self.undo, "Ctrl+Z")
        self._action(edit_menu, "&Redo", self.redo, "Ctrl+Shift+Z")
        edit_menu.addSeparator()
        self._action(edit_menu, "Clone", self.tools.clone_selected, "Ctrl+D")
        self._action(edit_menu, "Delete", self.tools.delete_key, QKeySequence.Delete)
        self._action(edit_menu, "Repair mesh", self.tools.repair_selected)
        self._action(edit_menu, "Auto-arrange", self.arrange, "Ctrl+A")
        edit_menu.addSeparator()
        self._action(edit_menu, "Delete all profiles…", self.reset_profiles)
        edit_menu.addSeparator()
        self._action(edit_menu, "Preferences…", self.open_preferences, QKeySequence.Preferences)

        view_menu = bar.addMenu("&View")
        for label, name, key in (("Isometric", "iso", "0"), ("Front", "front", "1"), ("Back", "back", "2"),
                                 ("Left", "left", "3"), ("Right", "right", "4"),
                                 ("Top", "top", "5"), ("Bottom", "bottom", "6")):
            self._action(view_menu, label, lambda _=False, n=name: self.viewport.set_view(n), key)
        view_menu.addSeparator()
        self._action(view_menu, "Fit to scene", self.viewport.fit_view, "F")
        self._action(view_menu, "3D view", lambda: self.pages.setCurrentIndex(0), "Ctrl+1")
        self._action(view_menu, "Layer view", self._show_layers, "Ctrl+2")
        self._action(view_menu, "Voxel preview", self._show_voxels, "Ctrl+3")

        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "About", self.about)

    def _action(self, menu, text, slot, shortcut=None) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, event) -> None:
        if any(self._is_model(u.toLocalFile()) for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if self._is_model(path):
                self.load_model(path)
        event.acceptProposedAction()

    @staticmethod
    def _is_model(path: str) -> bool:
        return Path(path).suffix.lower() in ALL_EXTENSIONS

    # ------------------------------------------------------------ slots
    def open_models(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open 3D model", "", FILE_FILTER)
        for p in paths:
            self.load_model(p)

    def load_model(self, path: str) -> None:
        try:
            objects = load_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open model", f"{Path(path).name}\n\n{exc}")
            return
        for obj in objects:
            self.scene.add(obj)
        if len(self.scene.objects) > 1:
            self.scene.arrange(self.side.current_printer())
        self.viewport.fit_view()
        self.pages.setCurrentIndex(0)

    def undo(self) -> None:
        if self.scene.undo_stack.undo():
            self._refresh_meshes()
            self._save_support_settings()     # undo may have changed them (e.g. Z lift)
            self._save_hollow_settings()

    def redo(self) -> None:
        if self.scene.undo_stack.redo():
            self._refresh_meshes()
            self._save_support_settings()
            self._save_hollow_settings()

    def _refresh_meshes(self) -> None:
        for obj in self.scene.objects:          # meshes may have been swapped back
            self.viewport.invalidate_object(obj)

    def arrange(self) -> None:
        self.scene.arrange(self.side.current_printer())

    def _layer_height(self) -> float:
        resin = self.side.current_resin()
        return float(resin.normal.thickness) if resin is not None else 0.05

    def _printer_changed(self, printer) -> None:
        self.viewport.set_printer(printer)
        self._update_hint()

    def _save_support_settings(self) -> None:
        self.store.settings["supports"] = self.scene.support_settings.to_dict()
        self.store.save_settings()

    def _save_hollow_settings(self) -> None:
        self.store.settings["hollowing"] = self.scene.hollow_settings.to_dict()
        self.store.save_settings()

    def _set_hole_mode(self, on: bool) -> None:
        """Clicking the model places drain holes (not supports)."""
        if on and self.tools.add_support_btn.isChecked():
            self.tools.add_support_btn.setChecked(False)
        self.viewport.add_hole_mode = on
        self.viewport.setCursor(Qt.CrossCursor if on or self.viewport.add_support_mode else Qt.ArrowCursor)

    def _set_show_inside(self, on: bool) -> None:
        self.viewport.show_inside = on
        self.viewport.update()

    def _set_support_mode(self, on: bool) -> None:
        if on and self.tools.hole_btn.isChecked():
            self.tools.hole_btn.setChecked(False)
        self.viewport.add_support_mode = on
        self.viewport.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _look_from(self, direction) -> None:
        self.viewport.camera.look_from(direction)
        self.viewport.update()

    def _orbit_by(self, dyaw: float, dpitch: float) -> None:
        self.viewport.camera.orbit_by(dyaw, dpitch)
        self.viewport.update()

    def _home_view(self) -> None:
        self.viewport.camera.set_view("iso")
        self.viewport.fit_view()

    # ------------------------------------------------------------ hint line
    def _update_hint(self) -> None:
        """The line over the bottom of the 3D view: the mouse controls and a
        warning when objects stick out of the build volume, in a color that
        stands out from the view's background."""
        r, g, b, _a = self.viewport.colors["background"]
        if QColor.fromRgbF(r, g, b).lightnessF() > 0.55:          # light background: dark text
            text, warn, halo = QColor(20, 22, 26, 220), QColor(198, 40, 40), QColor(255, 255, 255, 190)
        else:
            text, warn, halo = QColor(235, 237, 240, 200), QColor(255, 107, 107), QColor(0, 0, 0, 170)
        parts = [(HINT, text, False)]
        n_out = sum(not ok for ok in self.scene.fits(self.side.current_printer()).values())
        if n_out:
            s = "s" if n_out != 1 else ""
            parts += [("   ·   ", text, False), (f"⚠ {n_out} object{s} outside the build volume", warn, True)]
        if software_renderer():
            parts += [("   ·   ", text, False),
                      ("⚠ OpenGL runs in software - the 3D view will be slow (Help → About)", warn, True)]
        self.hint.set_parts(parts, halo)
        self.host.relayout()

    # ------------------------------------------------------------ preferences
    def open_preferences(self) -> None:
        before = self.store.profiles_dir
        PreferencesDialog(self.store, self._apply_preferences, self).exec()
        if self.store.profiles_dir != before:
            self._profiles_moved(before)

    def _profiles_moved(self, before) -> None:
        """The profiles folder changed (Preferences): show its printers - or,
        if it has none, ask for one; without one, go back to the old folder."""
        if not self.store.printers and not add_first_printer(self.store, self):
            self.store.set_profiles_dir(before)
            QMessageBox.information(self, "Profiles folder",
                                    f"OpenVat needs a printer, so the profiles stay in\n{before}")
        self.side.reload_profiles()

    def _apply_preferences(self, dark: bool, colors: dict) -> None:
        """``colors`` holds both themes; the 3D views use the one in force."""
        apply_dark_mode(dark)
        self._set_view_colors(colors[theme_name(dark)])

    def _set_view_colors(self, colors: dict) -> None:
        for view in (self.viewport, self.voxel_page.viewport):
            view.set_colors(colors)
        self.layer_page.canvas.set_background(colors["background"])
        self._update_hint()

    def _restore_panel_widths(self) -> None:
        widths = self.store.settings.get("panel_widths")
        if isinstance(widths, (list, tuple)) and len(widths) == 2:
            for column, width in zip((self.tools, self.side), widths):
                try:
                    self.host.set_column_width(column, int(width))
                except (TypeError, ValueError):
                    pass

    def _save_panel_widths(self) -> None:
        self.store.settings["panel_widths"] = [self.host.column_width(self.tools),
                                               self.host.column_width(self.side)]
        self.store.save_settings()

    def reset_profiles(self) -> None:
        if QMessageBox.question(
                self, "Delete all profiles",
                "Delete all your printer and resin profiles?\n\n"
                "Presets are kept; you will be asked to add a printer again.") != QMessageBox.Yes:
            return
        self.store.reset_to_defaults()
        if not add_first_printer(self.store, self):
            self.close()                      # OpenVat needs a printer to work
            return
        self.side.reload_profiles()

    def _show_layers(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Layer view", "Slice the scene first.")
            return
        self.pages.setCurrentIndex(1)

    def _show_voxels(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Voxel preview", "Slice the scene first.")
            return
        if self.voxel_page.result is not self.result:
            self.voxel_page.set_result(self.result)
        self.pages.setCurrentIndex(2)

    # ------------------------------------------------------------ slicing
    def slice(self) -> None:
        if not self.scene.objects:
            QMessageBox.information(self, "Slice", "Open a model first (File → Open model).")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        printer, resin = self.side.current_printer(), self.side.current_resin()
        if resin is None:
            QMessageBox.information(self, "Slice", f"There is no resin profile for {printer.name} yet.\n"
                                    "Pick one from the presets (or enter your own) in the next window.")
            if not self.side.add_resin():
                return
            resin = self.side.current_resin()
        if any(not ok for ok in self.scene.fits(printer).values()):
            if QMessageBox.question(self, "Slice", "Some objects are outside the build volume. "
                                    "Slice anyway?") != QMessageBox.Yes:
                return

        self.progress = QProgressDialog("Slicing…", None, 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)

        self._thread = QThread(self)
        self._worker = SliceWorker(self.scene, printer, resin)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Queued connections: the slots below run in the GUI thread.
        self._worker.progress.connect(self._slice_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._slice_done, Qt.QueuedConnection)
        self._worker.failed.connect(self._slice_failed, Qt.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    @Slot(int, int)
    def _slice_progress(self, done: int, total: int) -> None:
        if self.progress is not None:
            self.progress.setValue(int(100 * done / max(total, 1)))

    @Slot(object)
    def _slice_done(self, result: SliceResult) -> None:
        self._close_progress()
        self.result = result
        self.layer_page.set_result(result)
        self.pages.setCurrentIndex(1)

    @Slot(str)
    def _slice_failed(self, message: str) -> None:
        self._close_progress()
        QMessageBox.critical(self, "Slicing failed", message)

    def _close_progress(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress.deleteLater()
            self.progress = None

    def export_print_file(self) -> None:
        """Save the sliced result in the printer's own file type: the .pwsz
        family (.pwsz, .pp1, .pm7, ...) or Photon Workshop's binary files
        (.pws, .pwx, .pwmo, .dlp, .pm3m, .dl2p, .pwmx, .m5sp, ...)."""
        if self.result is None:
            QMessageBox.information(self, "Export", "Slice the scene first.")
            return
        printer = self.result.printer
        if not printer.can_export:
            QMessageBox.information(
                self, "Export",
                why_not(printer) + "\n\nSlicing and the previews work for every printer.")
            return
        ext = printer.file_extension or "pwsz"
        default = (self.scene.objects[0].name if self.scene.objects else "print") + f".{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Export print file", default,
                                              f"{printer.name} (*.{ext});;All files (*)")
        if not path:
            return
        if not path.lower().endswith(f".{ext}"):
            path += f".{ext}"
        meshes = [o.transformed() for o in self.scene.objects if o.visible]
        meshes += self.scene.support_meshes()
        progress = QProgressDialog(f"Writing {Path(path).name}…", None, 0, 100, self)
        progress.setWindowTitle("Export")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(400)

        def step(done: int, total: int) -> None:
            progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            write_print_file(path, self.result, meshes, step)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()
            progress.deleteLater()
        QMessageBox.information(self, "Export", f"Saved {Path(path).name}\n\n{path}")

    def about(self) -> None:
        AboutDialog(self).exec()
