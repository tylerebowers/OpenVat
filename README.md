![OpenVat](docs/logos/OPENVAT_nbg.png)

A FOSS resin slicer with support for Anycubic **`.pwsz`** (`.pwsz`, `.pm7`, `.pm7m`, `.pm4u`, `.pp1`, `.pp1m`) and binary files: binary files of versions 1, 516 and 517 (**`.pwx`**, `.pw0`, **`.pm3m`**, `.pm3`, `.pwmx`, `.pwma`, **`.dl2p`**, `.pwmb`, `.pm3r`, `.pm3n`, `.pm4n`, `.pm5`, `.pmx2`, `.px6s`, ...).
Runs on Linux, Windows and macOS.

![home_view](docs/screenshots/home_view.png)

## Installing

See releases

## Running

Without installing (from source):

```
# download & extract OpenVat, then
pip install -r requirements.txt 
python run.py 
```

<details>

<summary>Install as a python module (optional)</summary>

```
pip install -e .
openvat # GUI
openvat-slice model.stl -o model.pwsz --printer "Anycubic Photon Mono M7 Pro" --resin "Anycubic Standard Resin (ACF, Normal 0.05 mm)"
openvat-slice --list-profiles
openvat-slice --import-presets "Anycubic Photon Mono M7.pwsp"
openvat-slice --inspect model.pwsz
```

STEP/IGES import needs CadQuery: `pip install cadquery` (large download).

</details>

## Building packages

```
make appimage   # Linux   -> dist/OpenVat-<ver>-x86_64.AppImage
make exe        # Windows -> dist/OpenVat-<ver>-setup.exe   (needs Inno Setup 6)
make dmg        # macOS   -> dist/OpenVat-<ver>.dmg
```


## What it does

* Opens STL, OBJ, PLY, 3MF, OFF, glTF (and STEP/IGES with CadQuery).
* 3D build-plate view with orbit / pan / zoom, navigation cube, click-to-select and drag to move models and supports, drag-and-drop import, undo/redo (Ctrl+Z / Ctrl+Shift+Z).
* Model editing: position, rotation, scale, mirroring, clone, mesh repair (fills holes, fixes normals), auto-arrange.
* Supports: manual or automatic, with a density preset and a settings dialog for the base, cross bars between pillars, the interface and more. Z lift raises models off the plate so supports can go underneath. Supports are selectable and draggable; Del removes the selected one.
* Printer profiles and resin profiles, all editable in the GUI. 
* Slices to vector layers and shows them in a per-layer viewer, then a voxel preview page that rebuilds the model from the sliced layers.
* Exports `.pwsz`, see [docs/FORMAT_PWSZ.md](docs/FORMAT_PWSZ.md) for the full format description.

## Notes for contributors

TBD
