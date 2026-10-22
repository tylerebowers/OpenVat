"""Project files (``.openvat``): the build plate saved to open again later.

A project is a zip archive with

* ``project.json`` - the printer and resin profile (whole, so the project
  opens on another computer too), the support and hollowing settings, every
  model (name, where it stands, how it is turned and scaled, Z lift, drain
  holes, hollowing, whether it shows) and every support (all that placement
  decided: tip, arm, trunk, detour, nook foot ...);
* ``meshes.npz`` - the models' meshes (vertices and faces exactly as they
  were - clones share one) and their support paint;
* ``thumbnail.png`` - a picture of the build plate, for the home page.

Qt-free: the 3D view's picture comes in as PNG bytes.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from .. import __version__
from .hollow import HollowSettings, HollowState, Hole, LatticeParams
from .mesh import MeshObject
from .profiles import PrinterProfile, ResinProfile, printer_from_dict, resin_from_dict
from .supports import Support, SupportSettings

EXTENSION = ".openvat"
FILE_FILTER = f"OpenVat project (*{EXTENSION})"
FORMAT = 1                              # project.json "format": bumped when old readers can't cope


@dataclass
class Project:
    """What a project file holds (see the module docstring)."""
    objects: list[MeshObject] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    support_settings: SupportSettings = field(default_factory=SupportSettings)
    hollow_settings: HollowSettings = field(default_factory=HollowSettings)
    printer: PrinterProfile | None = None
    resin: ResinProfile | None = None
    saved: float = 0.0                  # time.time() when it was saved
    app: str = ""                       # "OpenVat 0.3.0"


# --------------------------------------------------------------------------
# saving

def save_project(path: str | Path, scene, printer: PrinterProfile | None, resin: ResinProfile | None,
                 thumbnail: bytes | None = None) -> None:
    """Write the scene to ``path`` (written next to it first, then moved in
    place, so a failed save never leaves half a project behind)."""
    path = Path(path)
    meshes: dict[int, int] = {}          # id(mesh) -> index in meshes.npz
    arrays: dict[str, np.ndarray] = {}
    objects = []
    for obj in scene.objects:
        k = meshes.get(id(obj.mesh))
        if k is None:
            k = meshes[id(obj.mesh)] = len(meshes)
            arrays[f"v{k}"] = np.asarray(obj.mesh.vertices, dtype=np.float64)
            arrays[f"f{k}"] = np.asarray(obj.mesh.faces, dtype=np.int64)
        entry = {
            "id": obj.id, "name": obj.name, "mesh": k,
            "position": _floats(obj.position), "rotation": _floats(obj.rotation), "scale": _floats(obj.scale),
            "visible": bool(obj.visible), "elevation": float(obj.elevation),
            "holes": [{"point": _floats(h.point), "normal": _floats(h.normal), "diameter": float(h.diameter)}
                      for h in obj.holes],
            "hollow": _hollow_dict(obj.hollow),
            "paint": None,
        }
        if obj.paint is not None and len(obj.paint) == len(obj.mesh.faces) and obj.paint.any():
            entry["paint"] = f"p{len(objects)}"
            arrays[entry["paint"]] = np.asarray(obj.paint, dtype=np.int8)
        objects.append(entry)
    data = {
        "format": FORMAT,
        "app": f"OpenVat {__version__}",
        "saved": time.time(),
        "printer": asdict(printer) if printer is not None else None,
        "resin": asdict(resin) if resin is not None else None,
        "support_settings": scene.support_settings.to_dict(),
        "hollow_settings": scene.hollow_settings.to_dict(),
        "objects": objects,
        "supports": [_support_dict(s) for s in scene.supports],
    }
    npz = io.BytesIO()
    np.savez_compressed(npz, **arrays)
    tmp = path.with_name(path.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(data, indent=1))
        z.writestr("meshes.npz", npz.getvalue(), compress_type=zipfile.ZIP_STORED)   # already compressed
        if thumbnail:
            z.writestr("thumbnail.png", thumbnail, compress_type=zipfile.ZIP_STORED)
    tmp.replace(path)


def _floats(v) -> list[float]:
    return [float(x) for x in v]


def _hollow_dict(h: HollowState | None):
    if h is None:
        return None
    return {"thickness": float(h.thickness), "lattice": asdict(h.lattice) if h.lattice is not None else None}


def _support_dict(s: Support) -> dict:
    out = {"owner": s.owner_id}
    for k, v in s.state().items():
        if isinstance(v, np.ndarray):
            v = _floats(v)
        elif k in ("trunk", "foot") and v is not None:
            v = _floats(v)
        elif k == "route":
            v = [_floats(j) for j in v]
        out[k] = v
    return out


# --------------------------------------------------------------------------
# opening

def load_project(path: str | Path) -> Project:
    """Read a project file.  The models get new ids (the supports follow
    them).  Raises ValueError when the file is not a project OpenVat can
    read."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("project.json").decode("utf-8"))
            arrays = np.load(io.BytesIO(z.read("meshes.npz")), allow_pickle=False)
            arrays = {k: arrays[k] for k in arrays.files}
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise ValueError(f"{path.name} is not an OpenVat project ({exc})") from None
    if int(data.get("format", 0)) > FORMAT:
        raise ValueError(f"{path.name} was saved by a newer OpenVat ({data.get('app', '?')}) - update OpenVat "
                         "to open it")
    proj = Project(saved=float(data.get("saved", 0.0)), app=str(data.get("app", "")))
    proj.support_settings = SupportSettings.from_dict(data.get("support_settings") or {})
    proj.hollow_settings = HollowSettings.from_dict(data.get("hollow_settings") or {})
    if isinstance(data.get("printer"), dict):
        proj.printer = printer_from_dict(data["printer"])
    if isinstance(data.get("resin"), dict):
        proj.resin = resin_from_dict(data["resin"])

    meshes: dict[int, trimesh.Trimesh] = {}
    new_id: dict = {}
    for entry in data.get("objects", []):
        k = int(entry["mesh"])
        if k not in meshes:                  # process=False: the faces keep their order (paint, holes)
            meshes[k] = trimesh.Trimesh(arrays[f"v{k}"], arrays[f"f{k}"], process=False)
        obj = MeshObject(name=str(entry.get("name", "model")), mesh=meshes[k],
                         position=np.array(entry.get("position", [0, 0, 0]), float),
                         rotation=np.array(entry.get("rotation", [0, 0, 0]), float),
                         scale=np.array(entry.get("scale", [1, 1, 1]), float),
                         visible=bool(entry.get("visible", True)),
                         elevation=float(entry.get("elevation", 0.0)),
                         holes=tuple(Hole(tuple(h["point"]), tuple(h["normal"]), float(h["diameter"]))
                                     for h in entry.get("holes", [])),
                         hollow=_hollow_from(entry.get("hollow")))
        paint = entry.get("paint")
        if paint and paint in arrays and len(arrays[paint]) == len(obj.mesh.faces):
            obj.paint = np.asarray(arrays[paint], dtype=np.int8)
        new_id[entry.get("id")] = obj.id
        proj.objects.append(obj)
    for entry in data.get("supports", []):
        if entry.get("owner") not in new_id:
            continue
        proj.supports.append(_support_from(entry, new_id[entry["owner"]]))
    return proj


def _hollow_from(d) -> HollowState | None:
    if not isinstance(d, dict):
        return None
    lattice = LatticeParams(**d["lattice"]) if isinstance(d.get("lattice"), dict) else None
    return HollowState(float(d.get("thickness", 2.0)), lattice)


def _support_from(d: dict, owner: int) -> Support:
    state = {}
    for k in Support.STATE:
        if k not in d:
            continue
        v = d[k]
        if k in ("tip", "lean", "normal"):
            v = np.array(v, float)
        elif k in ("trunk", "foot"):
            v = tuple(float(x) for x in v) if v is not None else None
        elif k == "route":
            v = tuple(tuple(float(x) for x in j) for j in v)
        state[k] = v
    return Support(owner_id=owner, **state)


def read_thumbnail(path: str | Path) -> bytes | None:
    """The project's picture of its build plate (PNG), or None."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.read("thumbnail.png")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def is_project(path: str | Path) -> bool:
    return str(path).lower().endswith(EXTENSION)
