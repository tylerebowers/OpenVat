"""Printer and resin profiles.

Profiles are plain dataclasses that round-trip to JSON.  A printer and the
resins made for it live together in one file named after the printer:

    {"printer": {...}, "resins": [{...}, {...}, ...]}

    openvat/resources/printers/<printer>.json   presets that ship with the program (only read)
    <profiles>/printers/<printer>.json          your printers, each with its resins
    <profiles>/presets/<printer>.json           presets you imported (openvat-slice --import-presets);
                                                they join the shipped ones
    ~/.openvat/settings.json                    preferences, last printer / resin, support and
                                                hollowing settings, panel widths - and <profiles>

<profiles> is ~/.openvat unless the user picked another folder (Edit ->
Preferences, saved as "profiles_folder" in settings.json); ~/.openvat is
$OPENVAT_HOME when that is set.  The same for every way OpenVat is run -
from source, AppImage, Windows installer or macOS app.  A new folder holds no
printers: the GUI asks for one on the first start.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import TypeVar, Type

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
DATA_ENV = "OPENVAT_HOME"            # another folder instead of ~/.openvat (tests, portable use)


SETTINGS_FILE = "settings.json"
PROFILES_KEY = "profiles_folder"      # settings.json: where printers/ and presets/ are (unset: ~/.openvat)
PRINTERS = "printers"                 # <folder>/printers/<printer>.json - a printer and its resins
PRESETS = "presets"                   # <profiles>/presets/<printer>.json - imported presets, same format


def data_dir() -> Path:
    """The user's OpenVat folder: ``~/.openvat``, or $OPENVAT_HOME if set.
    settings.json is always here; the profiles are too, unless moved."""
    custom = os.environ.get(DATA_ENV, "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".openvat"


def _read_settings(home: Path) -> dict:
    try:
        with open(home / SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def profiles_dir(home: Path | None = None, settings: dict | None = None) -> Path:
    """Where the user's printers (and imported presets) are: the folder
    picked in the preferences, else the OpenVat folder itself."""
    home = Path(home) if home is not None else data_dir()
    settings = _read_settings(home) if settings is None else settings
    custom = str(settings.get(PROFILES_KEY) or "").strip()
    return Path(custom).expanduser() if custom else home


OUTPUT_TYPES = {
    "anycubic_pwsz": "Anycubic, vector layers (pwszImg)",
    "anycubic_bitmap": "Anycubic, bitmap layers (export not supported yet)",
}
SUPPORTED_OUTPUT_TYPES = {"anycubic_pwsz"}


@dataclass
class PrinterProfile:
    name: str = "Anycubic Photon Mono M7 Pro"
    output_type: str = "anycubic_pwsz"
    res_x: int = 13312          # LCD resolution in pixels
    res_y: int = 5120
    print_x: float = 223.642    # build volume in mm
    print_y: float = 126.48
    print_z: float = 230.0
    pixel_x_um: float = 16.8    # pixel pitch in micrometers
    pixel_y_um: float = 24.8
    file_extension: str = "pwsz"      # e.g. pwsz, pm7, pm7m, pm4u, pp1
    layer_format: str = "pwszImg"     # PhotonWorkshop key_image_format
    # PhotonWorkshop's own machine block + firmware constants (from presets),
    # written back unchanged into exports.  Empty for hand-made printers.
    photonworkshop: dict = field(default_factory=dict)

    @property
    def can_export(self) -> bool:
        return self.output_type in SUPPORTED_OUTPUT_TYPES

    @property
    def pixel_x_mm(self) -> float:
        return self.pixel_x_um / 1000.0

    @property
    def pixel_y_mm(self) -> float:
        return self.pixel_y_um / 1000.0


@dataclass
class LayerSettings:
    """Exposure / motion settings for one class of layer (bottom or normal)."""
    thickness: float = 0.05          # mm
    exposure: float = 2.0            # seconds
    off_time: float = 0.5            # seconds (light-off / rest time)
    wait_before_lift: float = 0.0    # seconds
    wait_after_lift: float = 0.0     # seconds
    lift_distance: float = 8.0       # mm
    lift_speed: float = 6.0          # mm/s
    retract_speed: float = 6.0       # mm/s


@dataclass
class ResinProfile:
    name: str = "Standard Resin"
    brand: str = "Generic"
    density: float = 1.13           # g/cm^3, for weight estimates
    price_per_liter: float = 25.0   # for cost estimates
    currency: str = "$"

    normal: LayerSettings = field(default_factory=lambda: LayerSettings(
        thickness=0.05, exposure=1.8, off_time=0.5))
    bottom: LayerSettings = field(default_factory=lambda: LayerSettings(
        thickness=0.05, exposure=20.0, off_time=30.0,
        wait_before_lift=15.0, wait_after_lift=15.0))
    bottom_layers: int = 3
    elephant_foot_mm: float = 0.0    # bottom layers: every edge moved inward by this (polygon offset)

    transition_layers: int = 0       # layers that ramp bottom -> normal exposure
    anti_aliasing: int = 1           # 1 = off; the printer firmware does the AA
    shrink_x: float = 1.0            # shrinkage compensation scale factors
    shrink_y: float = 1.0
    shrink_z: float = 1.0

    def total_bottom_layers(self) -> int:
        return self.bottom_layers


# --------------------------------------------------------------------------
# (de)serialization helpers

T = TypeVar("T")


def _from_dict(cls: Type[T], data: dict) -> T:
    """Build a dataclass from a dict, ignoring unknown keys and recursing
    into nested dataclasses (LayerSettings)."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name in ("normal", "bottom") and isinstance(value, dict):
            value = _from_dict(LayerSettings, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def printer_from_dict(d: dict) -> PrinterProfile:
    return _from_dict(PrinterProfile, d)


def resin_from_dict(d: dict) -> ResinProfile:
    return _from_dict(ResinProfile, d)


def safe_filename(name: str) -> str:
    """Profile name -> file/folder name: only characters that Windows, macOS
    or Linux forbid in file names are replaced."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .") or "profile"


# --------------------------------------------------------------------------
# printer files: {"printer": {...}, "resins": [...]}

def printer_file_data(printer: PrinterProfile, resins: list[ResinProfile]) -> dict:
    return {"printer": asdict(printer), "resins": [asdict(r) for r in resins]}


def read_printer_file(path: Path) -> tuple[PrinterProfile, list[ResinProfile]] | None:
    """A printer and its resins from ``path`` (None if it is not a printer
    file - e.g. a profile of the old one-file-per-profile layout)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("printer"), dict):
        return None
    resins = [resin_from_dict(r) for r in data.get("resins", []) if isinstance(r, dict)]
    return printer_from_dict(data["printer"]), resins


def write_printer_file(path: Path, printer: PrinterProfile, resins: list[ResinProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")              # never leave a half-written file behind
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(printer_file_data(printer, resins), fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def printer_file_name(printer: PrinterProfile | str) -> str:
    return f"{safe_filename(printer if isinstance(printer, str) else printer.name)}.json"


def read_printer_folder(folder: Path) -> list[tuple[PrinterProfile, list[ResinProfile]]]:
    """Every printer file in ``folder``, in file-name order."""
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        item = read_printer_file(path)
        if item is not None:
            out.append(item)
    return out


def write_printer_folder(folder: Path, printers: list[PrinterProfile], resins: dict[str, list[ResinProfile]],
                         prune: bool = True) -> None:
    """One file per printer with its resins; ``prune`` removes the files of
    printers that were renamed or deleted (only printer files are touched)."""
    keep = set()
    for printer in printers:
        name = printer_file_name(printer)
        keep.add(name)
        write_printer_file(folder / name, printer, resins.get(printer.name, []))
    if prune and folder.is_dir():
        for stale in folder.glob("*.json"):
            if stale.name not in keep and read_printer_file(stale) is not None:
                stale.unlink()


def _read_legacy_printers(folder: Path) -> list[tuple[PrinterProfile, list[ResinProfile], Path]]:
    """Profiles in the layout before printer files (one file per printer in
    printers/, its resins in resins/<printer>/), with the printer's file."""
    out = []
    for path in sorted((folder / PRINTERS).glob("*.json")) if (folder / PRINTERS).is_dir() else []:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "printer" in data or "name" not in data:
            continue
        printer = printer_from_dict(data)
        resins = []
        rdir = folder / "resins" / safe_filename(printer.name)
        for rpath in sorted(rdir.glob("*.json")) if rdir.is_dir() else []:
            try:
                with open(rpath, "r", encoding="utf-8") as fh:
                    resins.append(resin_from_dict(json.load(fh)))
            except (OSError, ValueError, TypeError):
                continue
        out.append((printer, resins, path))
    return out


class ProfileStore:
    """Printers and their resins (one file per printer, see the module
    docstring) plus the app settings.

    ``home`` holds settings.json (``~/.openvat``); ``profiles_dir`` holds
    printers/ and presets/ - the same folder unless the user picked
    another one.  Resin profiles belong to a printer: ``resins`` maps
    printer name -> list.  There is no built-in default printer and nothing
    is copied in automatically - not from the presets, not from folders older
    versions used: when ``printers`` is empty the GUI asks the user to add
    one.  Preset folders are only read (an imported preset with a shipped
    one's name replaces it).  Profiles saved in the older layout
    (printers/<printer>.json + resins/<printer>/<resin>.json) are read and
    rewritten as printer files.
    """

    def __init__(self, home: Path | None = None):
        self.home = Path(home) if home is not None else data_dir()
        self.profiles_dir = self.home
        self.printers: list[PrinterProfile] = []
        self.resins: dict[str, list[ResinProfile]] = {}
        self.printer_presets: list[PrinterProfile] = []
        self.resin_presets: dict[str, list[ResinProfile]] = {}
        self.settings: dict = {}          # preferences, last selection etc. (settings.json)
        self.load()

    # -- app settings ----------------------------------------------------
    def _settings_path(self) -> Path:
        return self.home / SETTINGS_FILE

    def load_settings(self) -> None:
        self.settings = _read_settings(self.home)

    def save_settings(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self._settings_path(), "w", encoding="utf-8") as fh:
            json.dump(self.settings, fh, indent=2)

    # -- where the profiles are ------------------------------------------
    def set_profiles_dir(self, folder: Path | str | None, copy: bool = False) -> None:
        """Keep the printers in ``folder`` from now on (None: back to the
        OpenVat folder).  ``copy`` first writes the current printers there
        (replacing same-named ones; nothing else in it is touched).  Then
        the profiles are read from the new folder.  Raises OSError when the
        folder cannot be written."""
        new = Path(folder).expanduser() if folder else self.home
        if copy:
            write_printer_folder(new / PRINTERS, self.printers, self.resins, prune=False)
        else:
            (new / PRINTERS).mkdir(parents=True, exist_ok=True)
        if new == self.home:
            self.settings.pop(PROFILES_KEY, None)
        else:
            self.settings[PROFILES_KEY] = str(new)
        self.save_settings()
        self.load()

    @staticmethod
    def printers_in(folder: Path | str) -> list[str]:
        """Names of the printers saved in a profiles folder."""
        folder = Path(folder).expanduser()
        return ([p.name for p, _r in read_printer_folder(folder / PRINTERS)]
                + [p.name for p, _r, _f in _read_legacy_printers(folder)])

    # -- lookups ---------------------------------------------------------
    def find_printer(self, name) -> PrinterProfile | None:
        return next((p for p in self.printers if p.name == name), None)

    def resins_for(self, printer: PrinterProfile | str) -> list[ResinProfile]:
        """The (mutable) resin list of a printer, created if missing."""
        name = printer if isinstance(printer, str) else printer.name
        return self.resins.setdefault(name, [])

    def resin_presets_for(self, printer: PrinterProfile | str) -> list[ResinProfile]:
        name = printer if isinstance(printer, str) else printer.name
        return self.resin_presets.get(name, [])

    def find_resin(self, printer, name) -> ResinProfile | None:
        return next((r for r in self.resins_for(printer) if r.name == name), None)

    def default_printer(self) -> PrinterProfile | None:
        if not self.printers:
            return None
        return self.find_printer(self.settings.get("printer")) or self.printers[0]

    def default_resin(self, printer: PrinterProfile | None = None) -> ResinProfile | None:
        printer = printer or self.default_printer()
        if printer is None:
            return None
        resins = self.resins_for(printer)
        chosen = self.settings.get("resins", {}).get(printer.name)
        return self.find_resin(printer, chosen) or (resins[0] if resins else None)

    def remember_selection(self, printer: PrinterProfile | None, resin: ResinProfile | None) -> None:
        if printer is None:
            return
        self.settings["printer"] = printer.name
        if resin is not None:
            self.settings.setdefault("resins", {})[printer.name] = resin.name
        self.settings.pop("resin", None)               # pre-0.1 key
        self.save_settings()

    # -- printer bookkeeping ---------------------------------------------
    def rename_printer(self, old: str, new: str) -> None:
        if old != new and old in self.resins:
            self.resins[new] = self.resins.pop(old)
            sel = self.settings.get("resins", {})
            if old in sel:
                sel[new] = sel.pop(old)

    def forget_printer(self, name: str) -> None:
        self.resins.pop(name, None)
        self.settings.get("resins", {}).pop(name, None)

    # -- loading ---------------------------------------------------------
    def load(self) -> None:
        self.load_settings()
        self.profiles_dir = profiles_dir(self.home, self.settings)
        mine = read_printer_folder(self.profiles_dir / PRINTERS)
        legacy = _read_legacy_printers(self.profiles_dir)
        self.printers = [p for p, _r in mine] + [p for p, _r, _f in legacy]
        self.resins = {p.name: list(r) for p, r in mine}
        self.resins.update({p.name: list(r) for p, r, _f in legacy})
        if legacy:                                     # rewrite in the current layout
            try:
                self.save()
                written = {printer_file_name(p) for p in self.printers}
                for _p, _r, path in legacy:
                    if path.name not in written:
                        path.unlink(missing_ok=True)
                shutil.rmtree(self.profiles_dir / "resins", ignore_errors=True)
            except OSError:
                pass                                   # read-only folder: keep using them as they are

        presets: dict[str, tuple[PrinterProfile, list[ResinProfile]]] = {}
        for folder in (RESOURCES / PRINTERS, self.profiles_dir / PRESETS):
            for printer, resins in read_printer_folder(folder):
                presets[printer.name] = (printer, resins)       # imported replaces shipped
        ordered = sorted(presets.values(), key=lambda item: item[0].name.casefold())
        self.printer_presets = [p for p, _r in ordered]
        self.resin_presets = {p.name: r for p, r in ordered if r}

    # -- saving ----------------------------------------------------------
    def save(self) -> None:
        write_printer_folder(self.profiles_dir / PRINTERS, self.printers, self.resins)

    def reset_to_defaults(self) -> None:
        """Delete all user printers and resins (the GUI then asks for a
        printer).  Imported presets and the other settings stay."""
        folder = self.profiles_dir / PRINTERS
        if folder.is_dir():
            for path in folder.glob("*.json"):
                path.unlink()
        shutil.rmtree(self.profiles_dir / "resins", ignore_errors=True)   # older layout
        self.settings.pop("resins", None)
        self.settings.pop("printer", None)
        self.save_settings()
        self.load()
