"""Main window: menu bar, the home page, the 3D page, the layer view and
the voxel preview.

It also keeps the project: the file the plate was opened from or saved to
(the title shows its name, with a * while there are unsaved changes -
``Scene.revision`` differs from the one saved), and the recent files and
projects the home page lists."""

from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot, QUrl, QTimer, QBuffer, QIODevice
from PySide6.QtGui import QAction, QKeySequence, QColor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QStackedWidget,
                               QFileDialog, QMessageBox, QProgressDialog, QApplication)

from .. import APP_NAME, __version__
from ..core.loaders import load_file, FILE_FILTER, ALL_EXTENSIONS
from ..core.profiles import ProfileStore, PrinterProfile
from ..core.scene import Scene
from ..core.slicer import slice_scene, SliceResult
from ..core.supports import SupportSettings
from ..core.hollow import HollowSettings
from ..core import stats, usb
from ..core.problems import find_problems
from ..core.project import EXTENSION as PROJECT_EXT, FILE_FILTER as PROJECT_FILTER, is_project, load_project, \
    save_project
from ..formats.export import write_print_file, why_not
from ..formats.preview import render_preview
from .about_dialog import AboutDialog
from .app_settings_dialog import (PreferencesDialog, apply_dark_mode, dark_mode, saved_colors, saved_lighting,
                                  theme_name)
from .layer_view import LayerViewPage
from .voxel_view import VoxelViewPage
from .side_panel import SidePanel, add_first_printer
from .collapsible import ViewportHost, HintLine
from .tools_panel import ToolsPanel
from .view_cube import ViewCube
from .viewport3d import Viewport3D, software_renderer
from .usb_export import UsbWatcher
from .home_page import HomePage
from .nav import NavButton
from .zoom import UiZoom

HINT = ("Left-drag: move · Right-drag: orbit · Middle/Shift-drag: pan · "
        "Wheel: zoom · F: fit · Drop files here to import")
PAGE_3D, PAGE_LAYERS, PAGE_VOXELS, PAGE_HOME = range(4)
RECENT_MAX = 12                     # entries kept in each recent list


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
        self.ui_zoom = UiZoom(self.store, self)    # Ctrl + / Ctrl - / Ctrl + wheel (zoom.py)
        self.result: SliceResult | None = None
        self._result_revision: int | None = None    # the plate's revision when it was sliced
        self.project_path: str | None = None         # the project file the plate belongs to
        self._saved_revision = self.scene.revision   # unsaved changes: the revision moved on from this
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
        for page in (self.layer_page, self.voxel_page):
            page.usb_btn.export_to.connect(self.export_to_usb)
        self.home_page = HomePage()
        self.home_page.open_file.connect(self.load_model)
        self.home_page.open_project.connect(self.open_project)
        self.home_page.new_project.connect(self.new_project)
        self.home_page.open_model_requested.connect(self.open_models)
        self.home_page.open_project_requested.connect(self.open_project_dialog)
        self.home_page.back_requested.connect(lambda: self.pages.setCurrentIndex(PAGE_3D))
        self.home_page.remove_recent.connect(self._forget_recent)
        self.pages.addWidget(self.home_page)

        self._build_menus()
        self.usb = UsbWatcher(self)                # USB drives come and go: the Export to USB buttons
        self.usb.changed.connect(self._drives_changed)
        self._set_view_colors(saved_colors(self.store)[theme_name(dark_mode(self.store))])
        self._set_lighting(saved_lighting(self.store))
        self._restore_panel_widths()
        self._restore_view_tools()
        self._printer_changed(self.side.current_printer())
        self.scene.listeners.append(self._update_hint)
        self._update_hint()

        # unsaved changes (the title's *) and the problems list follow every edit
        self._problems_timer = QTimer(self)
        self._problems_timer.setSingleShot(True)
        self._problems_timer.setInterval(120)
        self._problems_timer.timeout.connect(self._refresh_problems)
        self.scene.listeners.append(self._scene_changed)
        self.tools.settings_changed.connect(self._settings_touched)       # changes without an undo step
        self.tools.hollow_settings_changed.connect(self._settings_touched)
        self.side.printer_changed.connect(lambda _p: self._settings_touched())
        self.side.resin_changed.connect(lambda _r: self._settings_touched())
        self.side.problems.activated.connect(self._problem_clicked)
        self.tools.supports_made.connect(lambda n: self._count(supports=n))
        self._saved_revision = self.scene.revision
        self.ui_zoom.changed.connect(self._zoom_changed)
        self._zoom_seen = self.ui_zoom.relative()
        self._update_title()
        self._refresh_problems()
        self._refresh_home()
        if self.store.settings.get("home_at_start", True):
            self.pages.setCurrentIndex(PAGE_HOME)

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
        self.side.section_changed.connect(lambda st: self.viewport.set_section(*(st or (None,))))
        self.tools.layer_height = self._layer_height        # Automatic looks for islands at this
        self.tools.printer = self.side.current_printer      # Clone… arranges the plate of this one
        self.tools.page_changed.connect(self._save_tools_tab)

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
        slice_btn = NavButton("Slice ▶", bold=True)
        slice_btn.clicked.connect(self.slice)
        home_btn = NavButton("⌂ Home")
        home_btn.setToolTip("The home page: recent files and projects (Ctrl+0)")
        home_btn.clicked.connect(self.show_home)

        self.host = ViewportHost(self.viewport, self.tools, self.side, overlay=self.view_cube,
                                 hint=self.hint, action=slice_btn, left_action=home_btn)
        self.host.widths_changed.connect(self._save_panel_widths)
        layout.addWidget(self.host, stretch=1)

        self.tools.mesh_edited.connect(self.viewport.invalidate_object)
        self.tools.arrange_requested.connect(self.arrange)
        self.tools.support_mode_toggled.connect(self._set_support_mode)
        self.tools.settings_changed.connect(self._save_support_settings)
        self.tools.hole_mode_toggled.connect(self._set_hole_mode)
        self.tools.show_inside_toggled.connect(self._set_show_inside)
        self.tools.hollow_settings_changed.connect(self._save_hollow_settings)
        self.tools.move_tool_toggled.connect(lambda on: self._set_view_tool("move", on))
        self.tools.rotate_tool_toggled.connect(lambda on: self._set_view_tool("rotate", on))
        self.tools.move_snap_toggled.connect(lambda on: self._set_snap("snap_move", on))
        self.tools.rotate_snap_toggled.connect(lambda on: self._set_snap("snap_rotate", on))
        self.tools.lay_face_toggled.connect(self._set_lay_face_mode)
        self.tools.paint_mode_toggled.connect(self._set_paint_mode)
        self.tools.paint_options_changed.connect(self._paint_options_changed)
        self.viewport.lay_face_requested.connect(self.tools.lay_on_face)
        self.viewport.hole_requested.connect(self.tools.add_hole_at)
        self.viewport.support_requested.connect(self.tools.add_support_at)
        self.viewport.selection_changed.connect(self.scene.changed)
        self.viewport.object_moved.connect(self.tools.refresh)
        return page

    def _build_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._action(file_menu, "&New project", self.new_project, QKeySequence.New)
        self._action(file_menu, "Open &project…", self.open_project_dialog, "Ctrl+Shift+O")
        self._action(file_menu, "&Open model…", self.open_models, QKeySequence.Open)
        file_menu.addSeparator()
        self._action(file_menu, "&Save project", self.save_project, QKeySequence.Save)
        self._action(file_menu, "Save project &as…", lambda: self.save_project(as_new=True), "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._action(file_menu, "&Export print file…", self.export_print_file, "Ctrl+E")
        self.usb_menu = file_menu.addMenu("Export to &USB")
        self._drives_changed([])
        file_menu.addSeparator()
        self._action(file_menu, "Clear scene", self.scene.clear)
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, QKeySequence.Quit)

        edit_menu = bar.addMenu("&Edit")
        self._action(edit_menu, "&Undo", self.undo, "Ctrl+Z")
        self._action(edit_menu, "&Redo", self.redo, "Ctrl+Shift+Z")
        edit_menu.addSeparator()
        self._action(edit_menu, "Clone", self.tools.clone_selected, "Ctrl+D")
        self._action(edit_menu, "Clone several…", self.tools.clone_many_selected, "Ctrl+Shift+D")
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
        self.plate_action = self._toggle(view_menu, "Build plate", "plate", True,
                                         lambda on: self._set_view_option(plate=on))
        self.see_through_action = self._toggle(view_menu, "Semi-transparent models", "see_through", False,
                                               lambda on: self._set_view_option(see_through=on))
        for act, box in ((self.plate_action, self.side.view.plate),          # the View card's check boxes
                         (self.see_through_action, self.side.view.see_through)):
            box.setChecked(act.isChecked())
            box.toggled.connect(act.setChecked)
            act.toggled.connect(box.setChecked)
        view_menu.addSeparator()
        self._action(view_menu, "Fit to scene", self.viewport.fit_view, "F")
        view_menu.addSeparator()
        zoom_in = self._action(view_menu, "Zoom in", lambda: self.ui_zoom.step(1), QKeySequence.ZoomIn)
        zoom_in.setShortcuts([QKeySequence.ZoomIn, QKeySequence("Ctrl+=")])     # + without Shift too
        self._action(view_menu, "Zoom out", lambda: self.ui_zoom.step(-1), QKeySequence.ZoomOut)
        self._action(view_menu, "Actual size", self.ui_zoom.reset)
        view_menu.addSeparator()
        self._action(view_menu, "Home page", self.show_home, "Ctrl+0")
        self._action(view_menu, "3D view", lambda: self.pages.setCurrentIndex(0), "Ctrl+1")
        self._action(view_menu, "Layer view", self._show_layers, "Ctrl+2")
        self._action(view_menu, "Voxel preview", self._show_voxels, "Ctrl+3")

        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "About", self.about)

    def _toggle(self, menu, text: str, key: str, default: bool, slot) -> QAction:
        """A checkable View menu entry, remembered in settings.json ("view")."""
        act = QAction(text, self)
        act.setCheckable(True)
        saved = self.store.settings.get("view")
        on = bool(saved.get(key, default)) if isinstance(saved, dict) else default
        act.setChecked(on)
        act.toggled.connect(slot)
        menu.addAction(act)
        slot(on)
        return act

    def _set_view_option(self, plate: bool | None = None, see_through: bool | None = None) -> None:
        """View menu: the build plate (in both 3D views), see-through models."""
        if plate is not None:
            for view in (self.viewport, self.voxel_page.viewport):
                view.set_view_options(plate=plate)
            self._remember_view("plate", plate)
        if see_through is not None:
            self.viewport.set_view_options(see_through=see_through)
            self._remember_view("see_through", see_through)

    def _remember_view(self, key: str, on: bool) -> None:
        saved = self.store.settings.get("view")
        saved = dict(saved) if isinstance(saved, dict) else {}
        if saved.get(key) != on:
            saved[key] = on
            self.store.settings["view"] = saved
            self.store.save_settings()

    def _action(self, menu, text, slot, shortcut=None) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, event) -> None:
        if any(self._is_model(u.toLocalFile()) or is_project(u.toLocalFile()) for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if is_project(path) or self._is_model(path):
                self.open_path(path)
        event.acceptProposedAction()

    @staticmethod
    def _is_model(path: str) -> bool:
        return Path(path).suffix.lower() in ALL_EXTENSIONS

    # ------------------------------------------------------------ slots
    def open_models(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open 3D model", "", FILE_FILTER)
        for p in paths:
            self.load_model(p)

    def open_path(self, path: str) -> None:
        """A project or a model file (the command line, a drop)."""
        if is_project(path):
            self.open_project(path)
        else:
            self.load_model(path)

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
        self.pages.setCurrentIndex(PAGE_3D)
        self._count(models=1)
        self._remember_recent("files", path)
        meshes = [o.transformed() for o in objects]
        QTimer.singleShot(0, lambda: self._make_model_thumbnail(path, meshes))

    # ------------------------------------------------------------ projects
    def new_project(self) -> None:
        """An empty plate (the printer, resin and settings stay)."""
        if not self._maybe_save("starting a new project"):
            return
        self.scene.replace_all([], [])
        self._project_loaded(None)

    def open_project_dialog(self) -> None:
        start = str(Path(self.project_path).parent) if self.project_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open project", start, f"{PROJECT_FILTER};;All files (*)")
        if path:
            self.open_project(path)

    def open_project(self, path: str) -> None:
        """Put a project on the plate: its models, supports and settings, and
        its printer and resin (added to the profiles if they are not there)."""
        if not self._maybe_save("opening another project"):
            return
        try:
            proj = load_project(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))
            return
        added = self._use_profiles(proj.printer, proj.resin)
        self.scene.replace_all(proj.objects, proj.supports, proj.support_settings, proj.hollow_settings)
        self._save_support_settings()                 # what the panels and the next Automatic use
        self._save_hollow_settings()
        self._project_loaded(path)
        if added:
            QMessageBox.information(self, "Open project", "Added from the project to your profiles: "
                                    + ", ".join(added) + ".")

    def _project_loaded(self, path: str | None) -> None:
        self.result, self._result_revision = None, None
        self.project_path = str(path) if path else None
        self._saved_revision = self.scene.revision
        if path:
            self._remember_recent("projects", path)
        self.viewport.fit_view()
        self.pages.setCurrentIndex(PAGE_3D)
        self._update_title()

    def _use_profiles(self, printer, resin) -> list[str]:
        """Select the project's printer and resin - the same-named profile
        if there is one, else the project's own copy, added.  Returns the
        names added."""
        if printer is None:
            return []
        added = []
        mine = self.store.find_printer(printer.name)
        if mine is None:
            self.store.printers.append(printer)
            mine = printer
            added.append(printer.name)
        res = None
        if resin is not None:
            res = self.store.find_resin(mine, resin.name)
            if res is None:
                self.store.resins_for(mine).append(resin)
                res = resin
                added.append(resin.name)
        if added:
            self.store.save()
        self.side.select_profiles(mine, res)
        return added

    def save_project(self, as_new: bool = False) -> bool:
        """Save the plate to its project file (asking for one the first time,
        or with ``as_new``).  False when it was not saved."""
        path = self.project_path
        if as_new or not path:
            name = Path(path).name if path else self._project_name() + PROJECT_EXT
            start = str(Path(path).parent / name) if path else name
            path, _ = QFileDialog.getSaveFileName(self, "Save project", start, PROJECT_FILTER)
            if not path:
                return False
            if not path.lower().endswith(PROJECT_EXT):
                path += PROJECT_EXT
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            save_project(path, self.scene, self.side.current_printer(), self.side.current_resin(),
                         self._project_thumbnail())
        except OSError as exc:
            QMessageBox.critical(self, "Could not save project", f"{path}\n\n{exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.project_path = path
        self._saved_revision = self.scene.revision
        self._remember_recent("projects", path)
        self._update_title()
        return True

    def _project_name(self) -> str:
        name = self.scene.objects[0].name if self.scene.objects else "project"
        return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .") or "project"

    def _project_thumbnail(self) -> bytes | None:
        """PNG of the build plate: the 3D view's picture, or (before OpenGL
        has started) a plain rendering of the models."""
        image = self.viewport.snapshot(480, 300)
        if image is not None:
            buf = QBuffer()
            buf.open(QIODevice.WriteOnly)
            image.save(buf, "PNG")
            return bytes(buf.data())
        meshes = [o.transformed() for o in self.scene.objects if o.visible] + self.scene.support_meshes()
        if not meshes:
            return None
        out = io.BytesIO()
        render_preview(meshes, (480, 300), background=self.viewport.colors["background"][:3],
                       model_color=self.viewport.colors["model"][:3]).save(out, "PNG")
        return out.getvalue()

    def _dirty(self) -> bool:
        return self.scene.revision != self._saved_revision

    def _update_title(self) -> None:
        name = Path(self.project_path).stem if self.project_path else "Untitled"
        self.setWindowTitle(f"{name}[*] - {APP_NAME} {__version__}")
        self.setWindowModified(self._dirty())

    def _scene_changed(self) -> None:
        self._update_title()
        self._problems_timer.start()

    def _settings_touched(self) -> None:
        """A setting saved with the project changed (no undo step for it)."""
        self.scene.touch()
        self._update_title()
        self._problems_timer.start()

    def _maybe_save(self, doing: str) -> bool:
        """With "Ask to save" on (Preferences) and unsaved changes: Save /
        Don't save / Cancel.  False: cancelled (or saving failed)."""
        if not self.store.settings.get("ask_save_on_close", False) or not self._dirty():
            return True
        if not self.scene.objects and not self.project_path:
            return True                              # nothing on an unsaved plate: nothing to lose
        name = Path(self.project_path).stem if self.project_path else "this project"
        box = QMessageBox(QMessageBox.Question, "Save project?",
                          f"Save the changes to {name} before {doing}?", parent=self)
        box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Save)
        answer = box.exec()
        if answer == QMessageBox.Save:
            return self.save_project()
        return answer == QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if not self._maybe_save("closing"):
            event.ignore()
            return
        self.voxel_page.stop()                         # no thread may outlive the window
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(30000)
        event.accept()

    # ------------------------------------------------------------ home page
    def show_home(self) -> None:
        self._refresh_home()
        self.pages.setCurrentIndex(PAGE_HOME)

    def _recent(self, kind: str) -> list[str]:
        items = self.store.settings.get(f"recent_{kind}", [])
        return [str(p) for p in items] if isinstance(items, list) else []

    def _remember_recent(self, kind: str, path: str) -> None:
        path = str(Path(path).resolve())
        items = [path] + [p for p in self._recent(kind) if p != path]
        self.store.settings[f"recent_{kind}"] = items[:RECENT_MAX]
        self.store.save_settings()
        self._refresh_home()

    def _forget_recent(self, kind: str, path: str) -> None:
        self.store.settings[f"recent_{kind}"] = [p for p in self._recent(kind) if p != path]
        self.store.save_settings()
        self._refresh_home()

    def _thumbnail_file(self, path: str) -> Path:
        return self.store.home / "thumbnails" / (hashlib.sha1(path.encode("utf-8")).hexdigest()[:20] + ".png")

    def _make_model_thumbnail(self, path: str, meshes: list) -> None:
        """A picture of a model file for its card on the home page (kept in
        the OpenVat folder; made again when the file changes)."""
        target = self._thumbnail_file(str(Path(path).resolve()))
        try:
            if target.exists() and target.stat().st_mtime >= os.path.getmtime(path):
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            render_preview(meshes, (360, 240), background=self.viewport.colors["background"][:3],
                           model_color=self.viewport.colors["model"][:3]).save(target, "PNG")
        except (OSError, ValueError):
            return
        self._refresh_home()

    def _count(self, **amounts) -> None:
        """Add to the running totals the home page shows."""
        stats.add(self.store.settings, **amounts)
        self.store.save_settings()
        self.home_page.set_totals(stats.totals(self.store.settings))

    def _refresh_home(self) -> None:
        self.home_page.set_totals(stats.totals(self.store.settings))
        files = self._recent("files")
        thumbs = {}
        for p in files:
            t = self._thumbnail_file(p)
            if t.exists():
                thumbs[p] = str(t)
        self.home_page.set_recent(files, self._recent("projects"), thumbs)

    # ------------------------------------------------------------ problems
    def _refresh_problems(self) -> None:
        printer, resin = self.side.current_printer(), self.side.current_resin()
        current = self.result is not None and self._result_revision == self.scene.revision
        note = ""
        if printer is not None and not printer.can_export:
            note = f"{printer.name}: slicing and the previews work, but OpenVat cannot write its print files yet"
        self.side.set_problems(find_problems(self.scene, printer, resin, self.result if current else None, note))

    def _problem_clicked(self, problem) -> None:
        """Show what a problem is about: its layer (islands) or its model."""
        if problem.layer is not None and self.result is not None:
            self.pages.setCurrentIndex(PAGE_LAYERS)
            self.layer_page.slider.setValue(problem.layer)
            return
        obj = self.scene.find(problem.object_id) if problem.object_id is not None else None
        if obj is not None:
            self.scene.selected, self.scene.selected_support = obj, None
            self.scene.changed()
            self.pages.setCurrentIndex(PAGE_3D)

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

    def _set_view_tool(self, kind: str, on: bool) -> None:
        """The move arrows / rotate rings (the eyes on the Position and
        Rotation cards); remembered for the next start."""
        self.viewport.set_tools(**{kind: on})
        self._save_view_tools()

    def _set_snap(self, kind: str, on: bool) -> None:
        """The grid / angle buttons: 10 mm and 15 degree steps (remembered)."""
        setattr(self.viewport, kind, on)
        self._save_view_tools()

    def _save_view_tools(self) -> None:
        v = self.viewport
        self.store.settings["view_tools"] = {"move": v.show_move, "rotate": v.show_rotate,
                                             "snap_move": v.snap_move, "snap_rotate": v.snap_rotate}
        self.store.save_settings()
        self._update_hint()

    def _restore_view_tools(self) -> None:
        saved = self.store.settings.get("view_tools")
        if isinstance(saved, dict):
            self.tools.move_eye.setChecked(bool(saved.get("move")))
            self.tools.rotate_eye.setChecked(bool(saved.get("rotate")))
            self.tools.move_snap.setChecked(bool(saved.get("snap_move")))
            self.tools.rotate_snap.setChecked(bool(saved.get("snap_rotate")))
        tab = self.store.settings.get("tools_tab")
        if isinstance(tab, int):
            self.tools.show_page(tab)

    def _save_tools_tab(self, index: int) -> None:
        if self.store.settings.get("tools_tab") != index:
            self.store.settings["tools_tab"] = index
            self.store.save_settings()

    def _only_mode(self, keep) -> None:
        """Clicks on the model do one thing at a time: turning on one of the
        click modes (lay on face, drain holes, add supports, support paint)
        turns the others off."""
        for btn in (self.tools.lay_btn, self.tools.hole_btn, self.tools.add_support_btn, self.tools.paint_btn):
            if btn is not keep and btn.isChecked():
                btn.setChecked(False)

    def _update_cursor(self) -> None:
        v = self.viewport
        v.setCursor(Qt.CrossCursor if v.add_support_mode or v.add_hole_mode or v.paint_mode else Qt.ArrowCursor)

    def _set_lay_face_mode(self, on: bool) -> None:
        """Clicks on the selected model's highlighted faces lay it down (no
        supports or holes are placed meanwhile)."""
        if on:
            self._only_mode(self.tools.lay_btn)
        self.viewport.set_tools(lay_face=on)
        self._update_hint()

    def _set_hole_mode(self, on: bool) -> None:
        """Clicking the model places drain holes (not supports)."""
        if on:
            self._only_mode(self.tools.hole_btn)
        self.viewport.add_hole_mode = on
        self._update_cursor()

    def _paint_options_changed(self) -> None:
        self.viewport.set_paint(**self.tools.paint_settings())
        self._update_hint()

    def _set_paint_mode(self, on: bool) -> None:
        """Dragging on a model paints where supports must go / must not go."""
        if on:
            self._only_mode(self.tools.paint_btn)
        self.viewport.set_paint(on=on, **self.tools.paint_settings())
        self._update_cursor()
        self._update_hint()

    def _set_show_inside(self, on: bool) -> None:
        self.viewport.show_inside = on
        self.viewport.update()

    def _set_support_mode(self, on: bool) -> None:
        if on:
            self._only_mode(self.tools.add_support_btn)
        self.viewport.add_support_mode = on
        self._update_cursor()

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
        sep = ("   ·   ", text, False)
        if self.viewport.lay_face_mode:
            parts = [(self._lay_face_hint(), text, True), ("   ·   ", text, False)] + parts
        elif self.viewport.paint_mode:
            what = {1: "where supports go", 2: "where no supports go", 0: "to erase paint"}.get(
                self.viewport.paint_value, "")
            parts = [(f"Support paint: drag on a model {what}", text, True), ("   ·   ", text, False)] + parts
        elif (self.viewport.show_move or self.viewport.show_rotate) and self.scene.selected is not None:
            parts += [("   ·   ", text, False), (self._snap_hint(), text, False)]
        n_out = sum(not ok for ok in self.scene.fits(self.side.current_printer()).values())
        if n_out:
            s = "s" if n_out != 1 else ""
            parts += [("   ·   ", text, False), (f"⚠ {n_out} object{s} outside the build volume", warn, True)]
        if software_renderer():
            parts += [("   ·   ", text, False),
                      ("⚠ OpenGL runs in software - the 3D view will be slow (Help → About)", warn, True)]
        compact = [p for p in parts if p[0] != HINT]              # no room: the mouse controls go first
        while compact and compact[0] == sep:
            compact = compact[1:]
        compact = [p for i, p in enumerate(compact) if not (p == sep and i + 1 < len(compact) and compact[i + 1] == sep)]
        if compact and compact[-1] == sep:
            compact = compact[:-1]
        self.hint.set_parts(parts, halo, compact)
        self.host.relayout()

    def _snap_hint(self) -> str:
        v = self.viewport
        snapping = [w for w, on in (("10 mm", v.snap_move), ("15°", v.snap_rotate)) if on]
        if not snapping:
            return "Ctrl-drag an arrow / ring: 1 mm / 15° steps"
        return f"Snapping to {' and '.join(snapping)} (Ctrl: 1 mm / {'1°' if v.snap_rotate else '15°'} steps)"

    def _lay_face_hint(self) -> str:
        obj = self.scene.selected
        if obj is None:
            return "Lay on face: select a model"
        if not self.viewport.lay_candidates(obj):
            return "Lay on face: no other flat face it can lie on"
        return "Lay on face: click a highlighted face to put the model down on it"

    # ------------------------------------------------------------ preferences
    def open_preferences(self) -> None:
        before = self.store.profiles_dir
        dlg = PreferencesDialog(self.store, self._apply_preferences, self)
        if dlg.exec() and getattr(dlg, "chosen_zoom", None) is not None:
            self.ui_zoom.set_zoom(dlg.chosen_zoom)
        if self.store.profiles_dir != before:
            self._profiles_moved(before)

    def _zoom_changed(self, _zoom: float) -> None:
        """The interface zoom changed: the panels keep their share of the
        room, the home page's cards grow with the text."""
        rel = self.ui_zoom.relative()
        factor, self._zoom_seen = rel / self._zoom_seen, rel
        for column in (self.tools, self.side):
            self.host.set_column_width(column, round(self.host.column_width(column) * factor))
        self.host.relayout()
        self._refresh_home()
        self._update_hint()

    def _profiles_moved(self, before) -> None:
        """The profiles folder changed (Preferences): show its printers - or,
        if it has none, ask for one; without one, go back to the old folder."""
        if not self.store.printers and not add_first_printer(self.store, self):
            self.store.set_profiles_dir(before)
            QMessageBox.information(self, "Profiles folder",
                                    f"OpenVat needs a printer, so the profiles stay in\n{before}")
        self.side.reload_profiles()

    def _apply_preferences(self, dark: bool, colors: dict, lighting=None) -> None:
        """``colors`` holds both themes; the 3D views use the one in force."""
        apply_dark_mode(dark)
        self._set_view_colors(colors[theme_name(dark)])
        if lighting is not None:
            self._set_lighting(lighting)

    def _set_lighting(self, lighting) -> None:
        for view in (self.viewport, self.voxel_page.viewport):
            view.set_lighting(lighting)

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
                    self.host.set_column_width(column, round(float(width) * self.ui_zoom.relative()))
                except (TypeError, ValueError):
                    pass

    def _save_panel_widths(self) -> None:
        """Kept as widths at the start-up zoom (the next start scales them all)."""
        rel = self.ui_zoom.relative()
        self.store.settings["panel_widths"] = [round(self.host.column_width(self.tools) / rel),
                                               round(self.host.column_width(self.side) / rel)]
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

        self._slicing_revision = self.scene.revision
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
        self._result_revision = getattr(self, "_slicing_revision", None)
        self._count(slices=1, layers=len(result.layers))
        self.layer_page.set_result(result)
        self.pages.setCurrentIndex(1)
        self._refresh_problems()

    @Slot(str)
    def _slice_failed(self, message: str) -> None:
        self._close_progress()
        QMessageBox.critical(self, "Slicing failed", message)

    def _close_progress(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress.deleteLater()
            self.progress = None

    def _can_export(self) -> bool:
        if self.result is None:
            QMessageBox.information(self, "Export", "Slice the scene first.")
            return False
        printer = self.result.printer
        if not printer.can_export:
            QMessageBox.information(
                self, "Export",
                why_not(printer) + "\n\nSlicing and the previews work for every printer.")
            return False
        return True

    def _export_name(self) -> str:
        """'<first model>.<printer extension>', safe on a FAT-formatted stick."""
        ext = self.result.printer.file_extension or "pwsz"
        name = self.scene.objects[0].name if self.scene.objects else "print"
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .") or "print"
        return f"{name}.{ext}"

    def export_print_file(self) -> None:
        """Save the sliced result in the printer's own file type: the .pwsz
        family (.pwsz, .pp1, .pm7, ...) or Photon Workshop's binary files
        (.pws, .pwx, .pwmo, .dlp, .pm3m, .dl2p, .pwmx, .m5sp, ...).  USB
        drives that are plugged in are listed in the dialog's side bar."""
        if not self._can_export():
            return
        printer = self.result.printer
        ext = printer.file_extension or "pwsz"
        dlg = QFileDialog(self, "Export print file", self._export_name())
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilters([f"{printer.name} (*.{ext})", "All files (*)"])
        dlg.setDefaultSuffix(ext)
        drives = [d for d in self.usb.refresh() if d.mounted]
        if drives:                                  # Qt's own dialog: the drives in the side bar
            urls = list(dlg.sidebarUrls())
            urls += [QUrl.fromLocalFile(d.path) for d in drives if QUrl.fromLocalFile(d.path) not in urls]
            dlg.setSidebarUrls(urls)
        if not dlg.exec() or not dlg.selectedFiles():
            return
        path = dlg.selectedFiles()[0]
        if not path.lower().endswith(f".{ext}"):
            path += f".{ext}"
        if self._write_export(path):
            QMessageBox.information(self, "Export", f"Saved {Path(path).name}\n\n{path}")

    def export_to_usb(self, drive: "usb.Drive") -> None:
        """Save the print file to the root of a USB drive (mounting it first
        if it is not yet), then offer to eject it."""
        if not self._can_export():
            return
        if not drive.mounted:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                drive = usb.mount(drive)
            except OSError as exc:
                QMessageBox.critical(self, "Export to USB", str(exc))
                return
            finally:
                QApplication.restoreOverrideCursor()
        path = os.path.join(drive.path, self._export_name())
        if os.path.exists(path):
            answer = QMessageBox.question(self, "Export to USB",
                                          f"{Path(path).name} is already on {drive.label}. Replace it?")
            if answer != QMessageBox.Yes:
                return
        if not self._write_export(path):
            return
        self._count(usb=1)
        box = QMessageBox(QMessageBox.Information, "Export to USB",
                          f"Saved {Path(path).name} to {drive.label}.", parent=self)
        eject = box.addButton("Eject", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Ok)
        box.setDefaultButton(eject)
        box.exec()
        if box.clickedButton() is eject:                  # eject and that's it (a message only if it fails)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                ok, why = usb.eject(drive)
            finally:
                QApplication.restoreOverrideCursor()
            self.usb.refresh()
            if not ok:
                QMessageBox.warning(self, "Export to USB", f"Could not eject {drive.label}.\n\n{why}")

    def _write_export(self, path: str) -> bool:
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
            with open(path, "rb+") as f:              # on the drive, not in a cache, before it is pulled out
                os.fsync(f.fileno())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()
            progress.deleteLater()
        self._count(exports=1, resin_ml=self.result.volume_mm3() / 1000, print_s=self.result.print_time_s())
        return True

    def _drives_changed(self, drives: list) -> None:
        """USB drives came or went: the Export to USB buttons and menu."""
        for page in (getattr(self, "layer_page", None), getattr(self, "voxel_page", None)):
            if page is not None:
                page.usb_btn.set_drives(drives)
        self.usb_menu.clear()
        for d in drives:
            self.usb_menu.addAction(d.describe(), lambda _=False, d=d: self.export_to_usb(d))
        if not drives:
            self.usb_menu.addAction("No USB drive found").setEnabled(False)

    def about(self) -> None:
        AboutDialog(self).exec()
