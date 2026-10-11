"""Run with:  pytest -q"""

import json
import zipfile

import numpy as np
import pytest
import shapely
import trimesh

from openvat.core.contours import segments_to_multipolygon
from openvat.core.mesh import MeshObject
from openvat.core.profiles import ProfileStore, ResinProfile, PrinterProfile
from openvat.core.raycast import ray_mesh_closest
from openvat.core.scene import Scene
from openvat.core.slicer import slice_scene, layer_schedule
from openvat.core.supports import Support
from openvat.formats.pwsz import (encode_layer_image, decode_layer_image, write_pwsz, read_pwsz,
                                  segments_to_polygons)


@pytest.fixture
def store(tmp_path):
    """Plain lists of fresh preset copies: M7 Pro first, its 0.05 mm
    standard resin first (tests may modify them freely)."""
    from types import SimpleNamespace
    st = ProfileStore(tmp_path)
    printers = sorted(st.printer_presets, key=lambda p: p.name != "Anycubic Photon Mono M7 Pro")
    resins = sorted(st.resin_presets_for(printers[0]), key=lambda r: r.name != "Anycubic Standard Resin (ACF, Normal 0.05 mm)")
    return SimpleNamespace(printers=printers, resins=resins)

def cube_scene(size=20.0):
    scene = Scene()
    obj = MeshObject("cube", trimesh.creation.box(extents=[size] * 3))
    obj.center_on_plate()
    obj.drop_to_plate()
    scene.add(obj)
    return scene, obj


def test_contours_square_with_hole():
    outer = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    inner = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    segs = []
    for ring in (outer, inner):
        for a, b in zip(ring, ring[1:] + ring[:1]):
            segs.append([a, b])
    mp = segments_to_multipolygon(np.array(segs, dtype=float))
    assert len(mp.geoms) == 1
    assert len(mp.geoms[0].interiors) == 1
    assert mp.area == pytest.approx(100 - 16)


def test_layer_schedule_bottom_and_transition():
    resin = ResinProfile(bottom_layers=2, transition_layers=2)
    resin.bottom.exposure, resin.normal.exposure = 10.0, 2.0
    sched = layer_schedule(0.3, resin)
    assert [round(p["exposure"], 2) for p in sched] == [10.0, 10.0, 7.33, 4.67, 2.0, 2.0]


def test_slice_cube_areas(store):
    scene, obj = cube_scene()
    resin = store.resins[0]
    result = slice_scene(scene, store.printers[0], resin)
    assert len(result.layers) == 400
    expected = (20 * resin.shrink_x) * (20 * resin.shrink_y)
    assert result.layers[200].area == pytest.approx(expected, rel=1e-4)
    assert result.layers[0].exposure == resin.bottom.exposure
    assert result.layers[-1].exposure == resin.normal.exposure


def test_elephant_foot_shrinks_bottom_layers(store):
    scene, _ = cube_scene()
    resin = store.resins[0]
    resin.elephant_foot_mm = 0.5
    resin.shrink_x = resin.shrink_y = 1.0
    result = slice_scene(scene, store.printers[0], resin)
    assert result.layers[0].area == pytest.approx(19 * 19)
    assert result.layers[-1].area == pytest.approx(20 * 20)


def test_layer_image_roundtrip():
    scene, _ = cube_scene()
    hole = trimesh.creation.cylinder(radius=3, height=40)
    scene.objects[0].mesh = scene.objects[0].mesh.difference(hole)
    resin = ResinProfile(shrink_x=1.0, shrink_y=1.0)
    result = slice_scene(scene, PrinterProfile(), resin)
    layer = result.layers[100]
    blob = encode_layer_image(layer.geometry)
    assert blob.startswith(b"{==\0") and blob.endswith(b"==}\0")
    img = decode_layer_image(blob)
    assert img.contours == 2
    # PhotonWorkshop's edge-table rules (see layer_edges)
    seg = img.segments
    assert (seg[:, 3] >= seg[:, 1]).all()                    # stored bottom-to-top
    assert (np.diff(seg[:, 1]) >= 0).all()                    # sorted by y0
    assert set(np.unique(img.flags)) == {0, 1}                # both directions present
    assert img.area2 / 2 == pytest.approx(layer.area, rel=1e-5)
    rebuilt = segments_to_polygons(img)
    assert rebuilt.area == pytest.approx(layer.area, rel=1e-4)


def test_write_and_read_pwsz(tmp_path, store):
    scene, obj = cube_scene()
    scene.add_support(Support(tip=np.array([15.0, 0.0, 5.0]), owner_id=obj.id))
    result = slice_scene(scene, store.printers[0], store.resins[0])
    out = write_pwsz(tmp_path / "cube.pwsz", result, [obj.transformed()])
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        for required in ("anycubic_photon_resins.pwsp", "layers_controller.conf", "lcd_function.json",
                         "print_info.json", "software_info.conf", "scene.slice",
                         "calc_layer_volumes.data", "preview_images/preview_2.png"):
            assert required in names
        assert sum(n.startswith("layer_images/") for n in names) == 400
        scene_slice = zf.read("scene.slice")
        assert scene_slice.startswith(b"ANYCUBIC-PWSZ") and scene_slice.endswith(b"--->")
        assert len(scene_slice) == 0x18C + 64 * 400 + 4
        pwsp = json.loads(zf.read("anycubic_photon_resins.pwsp"))
        assert pwsp["machine_type"]["res_x"] == store.printers[0].res_x
    f = read_pwsz(out)
    assert len(f.layer_images) == 400
    assert f.layer_paras[0]["exposure_time"] == store.resins[0].bottom.exposure


def test_mesh_edit_operations():
    scene, obj = cube_scene()
    obj.rotate(0, 45)
    assert obj.world_bounds()[0][2] == pytest.approx(0.0)
    obj.mirror(1)
    assert obj.mesh.is_watertight
    obj.set_uniform_scale(2.0)
    assert obj.size()[0] == pytest.approx(40, rel=1e-6)
    clone = scene.clone(obj)
    assert len(scene.objects) == 2 and clone.name != obj.name
    printer = PrinterProfile()
    scene.arrange(printer)
    assert all(scene.fits(printer).values())


def test_raycast():
    m = trimesh.creation.box(extents=[10, 10, 10])
    hit = ray_mesh_closest(m, np.array([0, 0, 50.0]), np.array([0, 0, -1.0]))
    assert hit is not None and hit[1][2] == pytest.approx(5.0)
    assert ray_mesh_closest(m, np.array([50, 0, 50.0]), np.array([0, 0, -1.0])) is None


def test_undo_redo():
    scene, obj = cube_scene()
    scene.push_undo()
    obj.rotate(2, 45)
    scene.push_undo()
    obj.mirror(0)
    mirrored = obj.mesh
    assert scene.undo_stack.undo() and obj.mesh is not mirrored
    assert scene.undo_stack.undo() and obj.rotation[2] == 0
    assert scene.undo_stack.redo() and obj.rotation[2] == 45
    assert scene.undo_stack.redo() and obj.mesh is mirrored
    scene.clone(obj)
    assert len(scene.objects) == 2
    assert scene.undo_stack.undo() and len(scene.objects) == 1


def test_auto_supports_braces_and_raft():
    from openvat.core.supports import generate_braces, raft_mesh
    scene = Scene()
    slab = trimesh.creation.box(extents=[30, 20, 3])
    obj = MeshObject("slab", slab)
    obj.center_on_plate(); obj.drop_to_plate(); obj.position[2] = 12
    scene.add(obj)
    scene.support_settings.raft = True
    n = scene.auto_supports()
    assert n > 10
    assert obj.world_bounds()[0][2] == pytest.approx(10.5)               # not moved (no Z lift)
    assert all(abs(s.tip[2] - 10.5) < 1e-6 for s in scene.supports)       # underside of the slab
    braces = generate_braces(scene.supports, scene.support_settings)
    raft = raft_mesh(scene.supports, scene.support_settings)
    assert len(braces.faces) > 0
    assert len(raft.split(only_watertight=False)) == 1 and raft.is_watertight   # one solid plate
    trunks = {sup.trunk for sup in scene.supports if sup.trunk}
    assert len(scene.support_meshes()) == n + len(trunks) + 2
    obj.position[2] = 1.5                                                 # on the plate: nothing to do
    scene.changed()
    assert scene.auto_supports() == 0


def test_supports_find_islands_and_follow_surfaces():
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports, find_contact_points
    st = SupportSettings()
    # a cube floating above a base plate: one island, supported from the base
    base = trimesh.creation.box(extents=[20, 20, 2]); base.apply_translation([0, 0, 1])
    floating = trimesh.creation.box(extents=[3, 3, 3]); floating.apply_translation([5, 5, 8])
    obj = MeshObject("t", trimesh.util.concatenate([base, floating]))
    obj.drop_to_plate()
    pts = find_contact_points(obj.transformed(), st)
    assert [k for *_xyz, k in pts] == ["island"]
    sups = generate_supports(obj, st)
    assert len(sups) == 1 and sups[0].is_nook and sups[0].base_z == pytest.approx(2.0)   # a nook on the base
    # a lifted sphere: arms tilt along the surface normal, never beyond the limit
    sphere = MeshObject("s", trimesh.creation.icosphere(subdivisions=4, radius=10))
    sphere.drop_to_plate(); sphere.position[2] += 5
    sups = generate_supports(sphere, st)
    assert len(sups) >= 8
    assert max(s.angle for s in sups) <= st.max_arm_angle + 1e-6
    assert max(s.angle for s in sups) > 20

def test_voxel_rebuild_matches_slices(store):
    from openvat.core.voxels import build_volume
    scene, _ = cube_scene(10.0)
    scene.objects[0].position[:2] = [0.123, -3.77]       # off the pixel grid on purpose
    resin = store.resins[0]
    resin.shrink_x = resin.shrink_y = 1.0
    result = slice_scene(scene, store.printers[0], resin)
    vol = build_volume(result, 1)
    px_area = vol.vx * vol.vy
    assert abs(vol.vx - store.printers[0].pixel_x_mm) < 1e-12        # one voxel = one pixel
    # every layer of a cube is identical, so every voxel layer must be too
    assert all(np.array_equal(vol.packed[0], p) for p in vol.packed)
    assert len(vol.groups()) == 1
    # >= 50 % coverage: area within one pixel row/column of the true area
    area = vol.solid_count[0] * px_area
    assert abs(area - 100.0) < 10.0 * (vol.vx + vol.vy)
    mesh = vol.mesh()
    assert abs(mesh.volume - sum(vol.solid_count) * px_area * resin.normal.thickness) < 1e-6


def test_voxel_faces_cut_and_culling(store):
    """Faces close the voxels exactly; clip + cap equals building only the
    lower layers; culling keeps what faces the camera and drops the rest."""
    from openvat.core.voxels import (build_volume, clip_faces, faces_to_mesh, pack_faces, VoxelFaces,
                                     PZ, NZ)
    scene = Scene()                          # a sphere (every layer differs) and a mushroom overhang
    ball = MeshObject("ball", trimesh.creation.icosphere(subdivisions=3, radius=2.0))
    ball.drop_to_plate(); ball.position[0] = -4.0
    cap = trimesh.creation.box(extents=[4.0, 4.0, 1.0]); cap.apply_translation([0, 0, 2.5])
    stem = trimesh.creation.box(extents=[1.0, 1.0, 2.0]); stem.apply_translation([0, 0, 1.0])
    mush = MeshObject("mushroom", trimesh.util.concatenate([stem, cap]))
    mush.position[:] = [4.0, 0.0, 1.5]
    scene.add(ball); scene.add(mush)
    resin = store.resins[0]
    resin.shrink_x = resin.shrink_y = 1.0
    vol = build_volume(slice_scene(scene, store.printers[0], resin), 1)
    voxel = vol.vx * vol.vy
    dz = np.diff(vol.z_edges)
    solid = lambda n: sum(c * dz[i] for i, c in enumerate(vol.solid_count[:n])) * voxel

    faces = vol.faces()
    mesh = faces_to_mesh(faces, vol)                          # merged faces meet at T-junctions, so the
    assert mesh.volume == pytest.approx(solid(vol.nz), rel=1e-9)   # signed volume checks closure and winding
    # the straight stem walls are one rectangle each however many layers they span
    assert faces[:, 6].max() - faces[:, 3].min() > 0 and (faces[:, 6] - faces[:, 3]).max() > 10
    for k in (1, vol.nz // 3, vol.nz // 2, vol.nz - 1):      # "show layers up to k - 1"
        cut = np.concatenate([clip_faces(faces, k), vol.cap(k - 1)])
        assert faces_to_mesh(cut, vol).volume == pytest.approx(solid(k), rel=1e-9)
        assert faces_to_mesh(vol.faces(k - 1), vol).volume == pytest.approx(solid(k), rel=1e-9)

    # GPU packing: 16 bytes per face, round trip
    p = pack_faces(faces)
    assert p.dtype == np.uint32 and p.shape == (len(faces), 4)
    back = np.stack([p[:, 3], p[:, 0] & 0xFFFF, p[:, 0] >> 16, p[:, 2] & 0xFFFF,
                     p[:, 1] & 0xFFFF, p[:, 1] >> 16, p[:, 2] >> 16], 1)
    assert np.array_equal(back, faces)

    vf = VoxelFaces.build(faces, vol, chunk=(16, 16, 8))
    assert vf.start[0] == 0 and vf.end[-1] == len(faces) and np.array_equal(vf.start[1:], vf.end[:-1])
    assert all(len(set(vf.faces[s:e, 0])) == 1 for s, e in zip(vf.start, vf.end))   # one direction per group
    lo, hi = vol.world(vf.faces)
    for g in range(len(vf.start)):                            # group bounds hold their faces
        s, e = vf.start[g], vf.end[g]
        assert (lo[s:e] >= vf.lo[g] - 1e-9).all() and (hi[s:e] <= vf.hi[g] + 1e-9).all()

    def view(eye, target):
        from openvat.ui.camera import OrbitCamera
        cam = OrbitCamera()
        d = np.asarray(eye, float) - target
        cam.target, cam.distance = np.asarray(target, float), float(np.linalg.norm(d))
        cam.look_from(d)
        return cam.position(), cam.projection_matrix() @ cam.view_matrix()

    center = (vf.lo.min(0) + vf.hi.max(0)) / 2
    eye, m = view(center + [0, 0, 200.0], center)            # from straight above
    seen = vf.visible(eye, m)
    assert len(seen) and not (vf.dir[seen] == NZ).any()       # undersides face away
    assert set(np.flatnonzero(vf.dir == PZ)) <= set(seen)     # every top is drawn
    d = ((vf.lo[seen] + vf.hi[seen]) / 2 - eye) ** 2
    assert (np.diff(d.sum(1)) >= -1e-9).all()                 # nearest first
    eye, m = view(center + [0, 0, 200.0], center + [0, 0, 400.0])   # looking away
    assert len(vf.visible(eye, m)) == 0
    eye, m = view(center + [0, 0, 200.0], center)            # cut low: groups above the cut go
    z_cut = float(vol.z_edges[2])
    assert (vf.lo[vf.visible(eye, m, z_cut), 2] <= z_cut + 1e-4).all()


def test_world_bounds_cache():
    obj = MeshObject("box", trimesh.creation.box(extents=[2.0, 4.0, 6.0]))
    def exact():
        v = trimesh.transform_points(obj.mesh.vertices, obj.matrix())
        return np.vstack([v.min(0), v.max(0)])
    assert np.allclose(obj.world_bounds(), exact())
    obj.position[:] = [5.0, -3.0, 1.0]                        # moved: the cached box shifts
    assert np.allclose(obj.world_bounds(), exact())
    obj.rotation[:] = [30.0, 0.0, 45.0]                       # rotated in place: recomputed
    assert np.allclose(obj.world_bounds(), exact())
    obj.scale[:] = [2.0, 1.0, 1.0]
    assert np.allclose(obj.world_bounds(), exact())
    obj.mirror(0)                                             # a new mesh: recomputed
    obj.mesh = trimesh.creation.box(extents=[10.0, 1.0, 1.0])
    assert np.allclose(obj.world_bounds(), exact())


def test_z_lift(store):
    """Z lift is applied by Automatic, not when the option is switched on."""
    scene, obj = cube_scene()
    scene.support_settings.z_lift = True
    scene.support_settings.z_lift_height = 5.0
    assert obj.world_bounds()[0][2] == pytest.approx(0.0)              # option alone moves nothing
    assert scene.auto_supports() > 0
    assert obj.world_bounds()[0][2] == pytest.approx(5.0)
    obj.rotate(0, 45)                                                   # rotation keeps the lift
    assert obj.world_bounds()[0][2] == pytest.approx(5.0)
    scene.clear_supports()                                              # back onto the plate
    assert obj.world_bounds()[0][2] == pytest.approx(0.0)
    scene.undo_stack.undo()
    assert obj.world_bounds()[0][2] == pytest.approx(5.0) and scene.supports

def test_profile_presets_and_user_folders(tmp_path):
    """One file per printer holding its resins - for the presets and for the
    user's own printers (printers/<printer>.json)."""
    from dataclasses import replace
    from openvat.core.profiles import read_printer_file
    res = tmp_path / "openvat-home"
    st = ProfileStore(res)
    assert st.printers == [] and st.default_printer() is None      # GUI asks for a printer
    assert len(st.printer_presets) == 30
    assert sum(len(r) for r in st.resin_presets.values()) == 575    # PhotonWorkshop factory resins (+1)
    m7 = next(p for p in st.printer_presets if p.name == "Anycubic Photon Mono M7 Pro")
    assert len(st.resin_presets_for(m7)) > 50
    assert "Standard Resin 0.05" in [r.name for r in st.resin_presets_for(m7)]

    st.printers.append(replace(m7, name="My M7"))
    st.resins_for("My M7").append(replace(st.resin_presets_for(m7)[0], name="My resin"))
    st.resins_for("My M7").append(replace(st.resin_presets_for(m7)[1], name="Other resin"))
    st.save()
    st.remember_selection(st.printers[0], st.resins_for("My M7")[0])
    printer, resins = read_printer_file(res / "printers" / "My M7.json")
    assert printer.name == "My M7" and [r.name for r in resins] == ["My resin", "Other resin"]
    assert sorted(p.name for p in res.iterdir()) == ["printers", "settings.json"]   # no resins/ folder

    # renaming a printer renames its file; deleting it removes the file
    st.printers[0] = replace(st.printers[0], name="Shop M7")
    st.rename_printer("My M7", "Shop M7")
    st.save()
    assert [p.name for p in (res / "printers").iterdir()] == ["Shop M7.json"]
    st2 = ProfileStore(res)
    assert [r.name for r in st2.resins_for("Shop M7")] == ["My resin", "Other resin"]
    st2.printers.clear(); st2.forget_printer("Shop M7"); st2.save()
    assert not any((res / "printers").iterdir())


def test_old_profile_layout_is_converted(tmp_path):
    """printers/<printer>.json + resins/<printer>/<resin>.json (the layout
    before printer files) are read and rewritten as printer files."""
    from dataclasses import asdict
    from openvat.core.profiles import read_printer_file
    (tmp_path / "printers").mkdir()
    (tmp_path / "printers" / "Old M7.json").write_text(json.dumps(asdict(PrinterProfile(name="Old M7"))))
    (tmp_path / "printers" / "renamed file.json").write_text(json.dumps(asdict(PrinterProfile(name="Second"))))
    (tmp_path / "resins" / "Old M7").mkdir(parents=True)
    for name in ("A", "B"):
        (tmp_path / "resins" / "Old M7" / f"{name}.json").write_text(json.dumps(asdict(ResinProfile(name=name))))
    st = ProfileStore(tmp_path)
    assert [p.name for p in st.printers] == ["Old M7", "Second"]
    assert [r.name for r in st.resins_for("Old M7")] == ["A", "B"]
    assert sorted(p.name for p in (tmp_path / "printers").iterdir()) == ["Old M7.json", "Second.json"]
    assert not (tmp_path / "resins").exists()
    assert [r.name for r in read_printer_file(tmp_path / "printers" / "Old M7.json")[1]] == ["A", "B"]
    assert [p.name for p in ProfileStore(tmp_path).printers] == ["Old M7", "Second"]      # stable


def test_user_folder_starts_empty(tmp_path, monkeypatch):
    """Everything goes to ~/.openvat (or $OPENVAT_HOME); a first start has no
    printers - nothing is copied in from the presets or from the folders
    older versions used (~/.config/OpenVat, the program's resources)."""
    from openvat.core.profiles import data_dir, RESOURCES
    monkeypatch.delenv("OPENVAT_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))                    # Path.home() on Windows
    old = tmp_path / ".config" / "OpenVat" / "printers"
    old.mkdir(parents=True)
    (old / "Old.json").write_text(json.dumps({"name": "Old"}))
    shipped_before = sorted(p.name for p in RESOURCES.rglob("*"))
    assert data_dir() == tmp_path / ".openvat"
    st = ProfileStore()
    assert st.home == st.profiles_dir == tmp_path / ".openvat"
    assert st.printers == [] and st.resins == {}
    assert len(st.printer_presets) == 30
    st.printers.append(st.printer_presets[0])
    st.save()
    st.remember_selection(st.printers[0], None)
    assert sorted(p.name for p in (tmp_path / ".openvat").iterdir()) == ["printers", "settings.json"]
    assert sorted(p.name for p in RESOURCES.rglob("*")) == shipped_before          # nothing written there
    monkeypatch.setenv("OPENVAT_HOME", str(tmp_path / "elsewhere"))
    assert data_dir() == tmp_path / "elsewhere" and ProfileStore().printers == []


def test_profiles_folder_can_move(tmp_path):
    """The printers can live in another folder; settings.json stays home."""
    from dataclasses import replace
    from openvat.core.profiles import profiles_dir, read_printer_file
    home, shared = tmp_path / "home", tmp_path / "Dropbox" / "OpenVat profiles"
    st = ProfileStore(home)
    st.printers.append(replace(st.printer_presets[3], name="Mine"))
    st.resins_for("Mine").append(ResinProfile(name="Goo"))
    st.save()
    assert ProfileStore.printers_in(shared) == []
    st.set_profiles_dir(shared, copy=True)
    assert st.profiles_dir == shared == profiles_dir(home)
    assert [p.name for p in st.printers] == ["Mine"] and [r.name for r in st.resins_for("Mine")] == ["Goo"]
    assert read_printer_file(shared / "printers" / "Mine.json")[1][0].name == "Goo"
    assert (home / "settings.json").exists() and not (shared / "settings.json").exists()
    st.printers.append(replace(st.printers[0], name="Second")); st.save()
    assert ProfileStore(home).profiles_dir == shared                       # remembered
    assert sorted(ProfileStore.printers_in(shared)) == ["Mine", "Second"]
    assert [p.name for p in ProfileStore(home).printers] == ["Mine", "Second"]
    st.set_profiles_dir(None)                                              # back home, without copying
    assert st.profiles_dir == home and [p.name for p in st.printers] == ["Mine"]
    assert "profiles_folder" not in json.loads((home / "settings.json").read_text())
    empty = tmp_path / "empty"
    st.set_profiles_dir(empty)                                             # an empty folder: no printers
    assert st.printers == [] and (empty / "printers").is_dir()


def test_imported_presets_go_to_the_user_folder(tmp_path):
    """Imported presets (presets/<printer>.json in the profiles folder) join
    the shipped ones and replace one of the same name."""
    from dataclasses import replace
    from openvat.core.profiles import PRESETS, write_printer_file
    st = ProfileStore(tmp_path)
    shipped = len(st.printer_presets)
    m7 = next(p for p in st.printer_presets if p.name == "Anycubic Photon Mono M7 Pro")
    write_printer_file(tmp_path / PRESETS / "Homebrew Printer.json", replace(m7, name="Homebrew Printer"),
                       [ResinProfile(name="Goo")])
    write_printer_file(tmp_path / PRESETS / "whatever.json", replace(m7, print_z=111.0), [])
    st = ProfileStore(tmp_path)
    assert len(st.printer_presets) == shipped + 1
    assert next(p for p in st.printer_presets if p.name == m7.name).print_z == 111.0
    assert [r.name for r in st.resin_presets_for("Homebrew Printer")] == ["Goo"]
    assert st.printers == []


def test_import_presets_cli_target(tmp_path, monkeypatch):
    from pathlib import Path
    from openvat.core.profiles import RESOURCES, PRESETS
    from openvat.formats.photonworkshop import import_presets
    monkeypatch.setenv("OPENVAT_HOME", str(tmp_path))
    shipped_before = sorted(p.name for p in RESOURCES.rglob("*"))
    export = Path(__file__).resolve().parents[1] / "examples" / "xyzCalibration_cube.pwsz"
    printer, resins = import_presets(export)                     # an export carries its printer file
    assert (tmp_path / PRESETS / f"{printer.name}.json").is_file()
    assert sorted(p.name for p in RESOURCES.rglob("*")) == shipped_before
    st = ProfileStore()
    assert st.profiles_dir == tmp_path and st.printers == []
    assert any(p.name == printer.name for p in st.printer_presets)


def test_shipped_presets_are_printer_files():
    from openvat.core.profiles import RESOURCES, PRINTERS
    files = sorted((RESOURCES / PRINTERS).glob("*.json"))
    assert len(files) == 30
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"printer", "resins"} and path.stem == data["printer"]["name"]


def test_photonworkshop_resin_import():
    from openvat.formats.photonworkshop import resin_from_photonworkshop
    entry = {"property": {"brand_name": "Anycubic", "resin_name": "standard_resin_v2", "film_name": "ACF",
                          "setting_name": "normal_print", "density": 1.1, "price": 30.0, "volume": 1000.0},
             "slicepara": {"zthick": 0.05, "exposure_time": 1.8, "off_time": 0.5, "bott_layers": 4,
                           "bott_time": 25.0, "zup_height": 4.0, "zup_speed": 3.0, "zdown_speed": 3.0},
             "slice_extpara": {"transition_layercount": 15, "transition_type": 0,
                               "material_scale_xyz": {"x": 1.0, "y": 1.0, "z": 1.0}}}
    r = resin_from_photonworkshop(entry)
    assert r.name == "Anycubic Standard Resin V2 (ACF, Normal 0.05 mm)"
    assert (r.normal.exposure, r.bottom.exposure, r.bottom_layers, r.transition_layers) == (1.8, 25.0, 4, 0)


def test_printer_presets_from_photonworkshop():
    from openvat.formats.pwsz import machine_type_json
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    def preset(name):
        return read_printer_file(RESOURCES / PRINTERS / f"{name}.json")[0]
    m7 = preset("Anycubic Photon Mono M7")
    assert (m7.file_extension, m7.layer_format, m7.can_export) == ("pm7", "pwszImg", True)
    assert machine_type_json(m7)["key_suffix"] == "pm7"
    mono_x = preset("Anycubic Photon Mono X")
    assert mono_x.file_version == 516 and mono_x.pixel_x_um == mono_x.pixel_y_um == 50.0
    assert mono_x.can_export and preset("Anycubic Photon Mono SE").can_export          # 516 and 515


def test_export_uses_printer_extension(tmp_path, store):
    import json, zipfile
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    m7max = read_printer_file(RESOURCES / PRINTERS / "Anycubic Photon Mono M7 Max.json")[0]
    scene, obj = cube_scene()
    result = slice_scene(scene, m7max, store.resins[0])
    out = write_pwsz(tmp_path / "cube.pm7m", result, [obj.transformed()])
    with zipfile.ZipFile(out) as zf:
        mt = json.loads(zf.read("anycubic_photon_resins.pwsp"))["machine_type"]
    assert (mt["name"], mt["key_suffix"], mt["res_x"]) == ("Anycubic Photon Mono M7 Max", "pm7m", 6480)



def test_layer_fills_like_the_printer():
    """Regression test for the empty print: run encoded layers through a
    scanline fill (what the firmware does) and check the exposed area."""
    from openvat.formats.pwsz import rasterize_layer_image
    from openvat.core.profiles import PrinterProfile
    scene = Scene()
    for o in __import__("openvat.core.loaders", fromlist=["load_file"]).load_file("examples/openvat-logo.stl"):
        o.scale[:] = [0.1, 0.1, 1.0]
        o.drop_to_plate()
        scene.add(o)
    printer = PrinterProfile()
    result = slice_scene(scene, printer, ResinProfile(shrink_x=1.0, shrink_y=1.0, shrink_z=1.0))
    for layer in (result.layers[0], result.layers[50]):
        img = decode_layer_image(encode_layer_image(layer.geometry))
        filled = rasterize_layer_image(img, printer.pixel_x_mm, printer.pixel_y_mm,
                                       printer.res_x, printer.res_y)
        exposed = filled.sum() * printer.pixel_x_mm * printer.pixel_y_mm
        assert exposed == pytest.approx(layer.area, rel=0.01)


def test_z_shrink_compensation():
    scene, _ = cube_scene(10.0)
    from openvat.core.profiles import PrinterProfile
    result = slice_scene(scene, PrinterProfile(), ResinProfile(shrink_x=1.0, shrink_y=1.0, shrink_z=1.1))
    assert result.height == pytest.approx(11.0, abs=0.05)     # 10 mm cube printed 10 % taller



def test_support_bulbs_flat_contacts_and_splitting():
    from openvat.core.placement import generate_supports
    from openvat.core.supports import SupportSettings, Support, arm_for_normal, columns
    st = SupportSettings()
    assert st.raft is True                                            # raft on by default
    # bulb at the pivot: a tilted arm and the pillar meet inside a sphere of rod diameter
    sup = Support(tip=np.array([0.0, 0.0, 10.0]), lean=np.array([1.0, 0.0]), angle=30.0)
    m = sup.to_mesh(st)
    p = sup.pivot(st)
    near = np.linalg.norm(m.vertices - p, axis=1)
    assert np.isclose(near.min(), 0.0) or (near < st.pillar_diameter / 2 + 1e-6).sum() >= 12
    # arms: straight on flat faces, perpendicular to tilted ones, capped
    assert arm_for_normal([0, 0, -1], 45, 10)[1] == 0.0
    assert arm_for_normal([np.sin(np.radians(5)), 0, -np.cos(np.radians(5))], 45, 10)[1] == 0.0
    assert arm_for_normal([np.sin(np.radians(30)), 0, -np.cos(np.radians(30))], 45, 10)[1] == pytest.approx(30)
    assert arm_for_normal([1, 0, 0], 45, 10)[1] == 45.0
    # a slab floating 20 mm up: flat underside -> all arms straight, pillars split
    slab = MeshObject("slab", trimesh.creation.box(extents=[30, 20, 3]))
    slab.drop_to_plate(); slab.position[2] += 20
    sups = generate_supports(slab, st)
    assert sups and all(s.angle == 0.0 for s in sups)
    split = [s for s in sups if s.trunk]
    assert split, "nearby contacts should share pillars"
    assert len(columns(sups, st)) < len(sups)
    for s in split:                                                    # branches within the angle
        dx = np.linalg.norm(s.pivot(st)[:2] - np.array(s.trunk[:2]))
        dz = s.pivot(st)[2] - s.trunk[2]
        assert np.degrees(np.arctan2(dx, dz)) <= st.branch_angle + 1e-6
    # no splitting when disabled
    st.branches = False
    assert all(s.trunk is None for s in generate_supports(slab, st))


def test_overhang_contacts_prefer_flat_faces():
    """A lifted cone (point down) is an island at its tip; elsewhere contacts
    should land on the flattest faces available, so arms stay mostly straight."""
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports
    st = SupportSettings()
    cone = trimesh.creation.cone(radius=10, height=4, sections=48)
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))   # tip down
    obj = MeshObject("cone", cone)
    obj.drop_to_plate(); obj.position[2] += 10
    sups = generate_supports(obj, st)
    assert sups
    tip_z = obj.world_bounds()[0][2]
    assert min(s.tip[2] for s in sups) == pytest.approx(tip_z, abs=0.15)   # island at the tip


def _mushroom() -> MeshObject:
    """A cap on a stem on a wide base standing on the plate (three touching shells)."""
    base = trimesh.creation.cylinder(radius=12, height=4, sections=48); base.apply_translation([0, 0, 2])
    stem = trimesh.creation.cylinder(radius=3, height=21, sections=24); stem.apply_translation([0, 0, 14.5])
    cap = trimesh.creation.cylinder(radius=15, height=2, sections=48); cap.apply_translation([0, 0, 26])
    obj = MeshObject("mushroom", trimesh.util.concatenate([base, stem, cap]))
    obj.drop_to_plate()
    return obj


def test_clearance_field():
    from openvat.core.clearance import ClearanceField, clearance_field
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=10)
    f = ClearanceField(sphere, cap=3.0)
    p = np.random.default_rng(0).uniform(-13, 13, (5000, 3))
    true = np.linalg.norm(p, axis=1) - 10
    d = f.distance(p)
    near = (true > 0) & (true < 2.5)
    assert np.abs(d[near] - true[near]).max() < 2 * f.cell
    assert (d[true < -f.cell] < 0).all()                              # inside is negative
    assert f.segment_clear([12, 0, -20], [12, 0, 20], 1.5)            # passes 2 mm from it
    assert not f.segment_clear([11, 0, -20], [11, 0, 20], 1.5)
    obj = MeshObject("s", sphere)                                      # moving reuses the field
    a = clearance_field(obj, 3.0)
    obj.position += [5, 0, 0]
    b = clearance_field(obj, 3.0)
    assert b.dist is a.dist
    assert b.distance([[17, 0, 0]])[0] == pytest.approx(a.distance([[12, 0, 0]])[0])


def test_supports_route_around_the_model():
    """Contacts under the mushroom's cap can't drop straight (the base is in
    the way): they detour around it to the plate, and every rod keeps clear."""
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports, model_space, _rod_clearance
    st = SupportSettings()
    obj = _mushroom()
    sups = generate_supports(obj, st)
    assert all(s.tip[2] > 20 for s in sups)          # nothing on faces buried inside the model
    routed = [s for s in sups if s.route]
    assert len(routed) >= 10
    field = model_space(obj, st).field
    for s in routed:
        pts = [s.pivot(st), *map(np.array, s.route), np.array([*s.pillar_xy(st), st.base_height])]
        for a, b in zip(pts, pts[1:]):
            d = b - a
            if np.allclose(d, 0):                                              # detour ends on the foot
                continue
            assert d[2] < 0                                                    # always downward
            assert np.degrees(np.arctan2(np.hypot(d[0], d[1]), -d[2])) <= st.branch_angle + 3.5
            assert field.segment_clear(a, b, _rod_clearance(st) - 0.05)
        assert np.hypot(*s.pillar_xy(st)) > 12 + st.base_diameter / 2           # foot beside the base
        assert s.status.startswith("routed")


def test_nook_supports_are_small_and_stand_on_the_part():
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports
    st = SupportSettings(routing=False)          # no detours: everything over the base is a nook
    sups = generate_supports(_mushroom(), st)
    nooks = [s for s in sups if s.is_nook]
    assert nooks and all(s.base_z == pytest.approx(4.0) for s in nooks)          # on the base
    for s in nooks:
        m = s.to_mesh(st)
        tip, foot = np.array(s.tip), np.array(s.foot)
        axis = (foot - tip) / np.linalg.norm(foot - tip)
        along = (m.vertices - tip) @ axis
        radial = np.linalg.norm(m.vertices - tip - along[:, None] * axis, axis=1)    # off the rod's axis
        assert np.degrees(np.arccos(-axis[2])) <= st.max_arm_angle + 1e-6
        assert radial.max() <= st.nook_diameter / 2 + 1e-6                         # thin all the way
        ends = (along < 0.2) | (along > np.linalg.norm(foot - tip) - 0.2)          # both contacts
        assert radial[ends].max() <= st.nook_contact_diameter / 2 + 1e-6
        assert s.status.startswith("nook support")


def _cave() -> MeshObject:
    """A block on the plate with a cave (10 x 6 x 4 mm, open at the -X side)."""
    def box(x0, x1, y0, y1, z0, z1):
        b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
        b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        return b
    parts = [box(-10, 10, -10, 10, 0, 4), box(-10, 10, -10, 10, 8, 12),           # floor, roof
             box(0, 10, -10, 10, 4, 8),                                            # back wall
             box(-10, 0, -10, -3, 4, 8), box(-10, 0, 3, 10, 4, 8)]                 # side walls
    obj = MeshObject("cave", trimesh.util.concatenate(parts))
    obj.drop_to_plate()
    return obj


def test_cave_gets_nook_supports_and_nothing_crosses_the_part():
    """Plan first (kind of every support), then route: deep in the cave the
    roof can only be held by nooks (an arm may reach out of the cave's mouth
    near it); nothing goes through the walls."""
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports, model_space, _rod_clearance
    st = SupportSettings(density="high")
    obj = _cave()
    report = {}
    sups = generate_supports(obj, st, report)
    nooks = [s for s in sups if s.is_nook]
    assert len(nooks) >= 6 and report["nook"] == len(nooks) and report["skipped"] == 0
    assert all(s.is_nook for s in sups if s.tip[0] > -8)                         # deep inside
    field = model_space(obj, st).field
    for s in sups:
        assert s.tip[2] == pytest.approx(8.0)                                    # the roof
        assert -10 < s.tip[0] < 0 and -3 < s.tip[1] < 3                          # inside the cave
        if s.is_nook:
            assert s.base_z == pytest.approx(4.0)                                # on the floor
            mid = (np.array(s.tip) + np.array(s.foot)) / 2
            assert field.distance([mid])[0] >= st.nook_diameter / 2 + st.model_clearance - 0.1
        else:                                                                    # out through the mouth
            pts = [s.pivot(st), *map(np.array, s.route), np.array([*s.pillar_xy(st), st.base_height])]
            assert all(field.segment_clear(a, b, _rod_clearance(st) - 0.05) for a, b in zip(pts, pts[1:])
                       if not np.allclose(a, b))
    # the plate-reach map: under the cave's roof nothing can get out; beside the part everything can
    reach = model_space(obj, st).reach
    assert not reach.contains([[-5, 0, 5.5]])[0]
    assert reach.contains([[-14, 0, 5.5]])[0]


def test_plate_reach_map_matches_the_mushroom():
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import model_space
    st = SupportSettings()
    reach = model_space(_mushroom(), st).reach
    # high under the cap above the base there is room to tilt out past the base; low there is not
    assert reach.contains([[8, 0, 20]])[0]
    assert not reach.contains([[8, 0, 6]])[0]
    assert reach.contains([[20, 0, 6]])[0]                                       # beside the base


def test_arm_turns_away_from_a_wall():
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import manual_support
    st = SupportSettings()
    slab = trimesh.creation.box(extents=[20, 20, 2]); slab.apply_translation([0, 0, 21])
    lip = trimesh.creation.box(extents=[1, 20, 5]); lip.apply_translation([9.5, 0, 17.5])   # hangs at x 9..10
    obj = MeshObject("lip", trimesh.util.concatenate([slab, lip]))
    obj.drop_to_plate(); obj.position[2] += 10
    z = obj.world_bounds()[1][2] - 2                                   # the slab's underside
    free = manual_support(obj, np.array([0.0, 0.0, z]), st)
    assert free.angle == 0 and free.status == "straight to the plate"
    near = manual_support(obj, np.array([8.5, 0.0, z]), st)           # 0.5 mm from the lip
    assert "arm turned" in near.status
    assert near.angle > 0 and near.lean[0] < 0                         # tilted away from the wall


def test_routes_move_undo_and_raft_keeps_off_the_model():
    scene = Scene()
    obj = _mushroom()
    scene.add(obj)
    scene.auto_supports()
    raft = scene.support_meshes()[-1]
    assert raft.bounds[1][2] == pytest.approx(scene.support_settings.base_height)
    # the base stands on the plate: the raft goes around it, not under it
    assert np.hypot(raft.vertices[:, 0], raft.vertices[:, 1]).min() > 12 + scene.support_settings.model_clearance
    routed = [s for s in scene.supports if s.route]
    before = [np.array(s.route) for s in routed]
    scene.move_object(obj, obj.position + [3, 0, 0])
    assert all(np.allclose(np.array(s.route) - b, [3, 0, 0]) for s, b in zip(routed, before))
    scene.clear_supports()                                            # undo brings detours back
    assert scene.undo_stack.undo()
    assert [np.array(s.route) for s in scene.supports if s.route][0] == pytest.approx(before[0] + [3, 0, 0])


def _rotated_xyz_cube() -> MeshObject:
    """The calibration cube turned +45 deg about X and Y (a corner points down)
    and lifted 5 mm - engraved letters on slanted faces."""
    obj = MeshObject("xyz", trimesh.load("examples/xyzCalibration_cube.stl"))
    obj.rotate(0, 45)
    obj.rotate(1, 45)
    obj.elevation = 5.0
    obj.drop_to_plate()
    return obj


def test_grooves_get_short_nooks_square_to_their_walls():
    """Overhanging walls inside the engraved letters get small nooks straight
    across the groove (like a rung), meeting both walls squarely, kept clear
    of the groove's bottom and ends - never into the part."""
    from openvat.core.supports import SupportSettings, nook_cone_length
    from openvat.core.placement import generate_supports
    st = SupportSettings()
    obj = _rotated_xyz_cube()
    report = {}
    sups = generate_supports(obj, st, report)
    nooks = [s for s in sups if s.is_nook]
    assert len(nooks) >= 4 and report["skipped"] == 0
    tri = obj.transformed().triangles
    for s in nooks:
        tip, foot = np.array(s.tip), np.array(s.foot)
        length = np.linalg.norm(foot - tip)
        axis = (foot - tip) / length
        n = s.normal / np.linalg.norm(s.normal)
        assert length <= st.nook_max_gap
        assert np.degrees(np.arccos(np.clip(axis @ n, -1, 1))) < 10                # square to the wall
        # the rod part of the nook (between its two contact cones) keeps clear of every face
        v = s.to_mesh(st).vertices
        along = (v - tip) @ axis
        cone = nook_cone_length(length, st)
        rod = v[(along > cone - 1e-6) & (along < length - cone + 1e-6)]
        d = [np.linalg.norm(trimesh.triangles.closest_point(tri, np.repeat([p], len(tri), 0)) - p, axis=1).min()
             for p in rod]
        assert min(d) >= st.model_clearance - 0.05


def test_lowest_point_is_anchored_by_several_supports():
    """A corner pointing down is an island that starts as a single point: it
    gets several supports spread around it, each on its own pillar."""
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports
    obj = MeshObject("box", trimesh.creation.box(extents=[20, 20, 20]))
    obj.rotate(0, 45)
    obj.rotate(1, 45)
    obj.elevation = 5.0
    obj.drop_to_plate()
    low = obj.world_bounds()[0][2]
    sups = generate_supports(obj, SupportSettings())
    bottom = [s for s in sups if s.tip[2] < low + 3.0]
    assert len(bottom) >= 4
    assert all(s.trunk is None and not s.is_nook for s in bottom)                 # own pillars
    xy = np.array([s.tip[:2] for s in bottom])
    assert np.ptp(xy[:, 0]) + np.ptp(xy[:, 1]) > 2.0                               # spread out
    one = generate_supports(obj, SupportSettings(island_anchors=1))                 # anchors off
    assert len([s for s in one if s.tip[2] < low + 2.0]) == 1


def test_overlapping_support_parts_add_up_when_sliced():
    """Support parts overlap (rods, bulbs, cones, crossing braces): slicing
    them as one mesh with even-odd nesting cancelled the overlaps.  Shell by
    shell they add up."""
    from openvat.core.slicer import slice_mesh_at, shell_labels
    a = trimesh.creation.box(extents=[10, 10, 10])
    b = trimesh.creation.box(extents=[4, 4, 4])                     # fully inside a
    c = trimesh.creation.box(extents=[10, 2, 2]); c.apply_translation([8, 0, 0])   # sticks out of a
    m = trimesh.util.concatenate([a, b, c])
    whole = slice_mesh_at(m, 0.0, shell_labels(m))
    assert len(set(shell_labels(m))) == 3
    assert whole.area == pytest.approx(100 + 8 * 2, rel=1e-6)      # union: no hole where b is
    assert len(whole.geoms) == 1 and not whole.geoms[0].interiors


def test_find_islands_follows_unheld_regions_up_the_stack():
    from openvat.core.slicer import Layer, find_islands
    from shapely.geometry import box
    def layer(i, *polys):
        from shapely.geometry import MultiPolygon
        return Layer(geometry=MultiPolygon(list(polys)), index=i, z_bottom=i * 0.05, thickness=0.05, exposure=2,
                     off_time=0, wait_before_lift=0, wait_after_lift=0, lift_distance=5, lift_speed=3,
                     retract_speed=3)
    base, far = box(0, 0, 5, 5), box(10, 0, 12, 2)
    joined = box(0, 0, 12, 5)
    layers = [layer(0, base), layer(1, base), layer(2, base, far), layer(3, base, far), layer(4, joined),
              layer(5, joined)]
    islands = find_islands(layers)
    assert len(islands) == 1
    assert (islands[0].first, islands[0].last) == (2, 3)
    assert islands[0].xy[0] > 9
    near = box(5.05, 0, 6, 1)                                         # within the tolerance: held
    assert find_islands([layer(0, base), layer(1, base, near)]) == []


def test_rotated_cube_prints_without_islands():
    """The engraved letters of a rotated calibration cube have pieces that
    start in mid-air for a few layers.  Found by slicing at the printed
    layer height, each is supported where it starts, so the sliced print
    (model + supports) holds everything."""
    from openvat.core.slicer import slice_scene
    scene = Scene()
    obj = _rotated_xyz_cube()
    scene.add(obj)
    scene.auto_supports(layer_height=0.1)
    resin = ResinProfile(shrink_x=1.0, shrink_y=1.0, shrink_z=1.0)
    resin.normal.thickness = resin.bottom.thickness = 0.1
    result = slice_scene(scene, PrinterProfile(), resin)
    assert result.islands == []
    assert scene.support_report["skipped"] == 0


def test_short_bridges_need_no_support():
    """A roof over a narrow channel bridges itself; over a wide one it needs supports."""
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports

    def channel(gap):
        walls = [trimesh.creation.box(extents=[3, 20, 6]), trimesh.creation.box(extents=[3, 20, 6])]
        walls[0].apply_translation([-(gap / 2 + 1.5), 0, 3])
        walls[1].apply_translation([gap / 2 + 1.5, 0, 3])
        roof = trimesh.creation.box(extents=[gap + 6, 20, 2]); roof.apply_translation([0, 0, 7])
        obj = MeshObject("channel", trimesh.util.concatenate(walls + [roof]))
        obj.drop_to_plate()
        return obj

    st = SupportSettings()
    narrow = channel(2.0)
    report = {}
    sups = generate_supports(narrow, st, report)
    inside = [s for s in sups if abs(s.tip[0] - narrow.position[0]) < 1.0]
    assert report["bridged"] > 0 and not inside
    wide = channel(8.0)
    report = {}
    sups = generate_supports(wide, st, report)
    assert len([s for s in sups if abs(s.tip[0] - wide.position[0]) < 4.0]) > 0


def test_overhang_bookkeeping_survives_broken_slivers():
    """Growing and shrinking thin overhang slivers can leave self-touching
    rings; set operations on them raised GEOS 'non-noded intersection'
    (seen with high density and 0.03 mm layers).  They are repaired now."""
    from shapely.geometry import Polygon as P
    from openvat.core.supports import SupportSettings
    from openvat.core.supports import _safe, _valid
    from openvat.core.placement import find_contact_points
    bowtie = P([(0, 0), (2, 2), (2, 0), (0, 2)])                     # self-intersecting ring
    assert not bowtie.is_valid and _valid(bowtie).is_valid
    assert _safe(shapely.difference, bowtie, P([(0, 0), (1, 0), (1, 1)])).is_valid
    obj = _rotated_xyz_cube()
    obj.position[:2] = [11.772959800209328, -12.436755059250771]       # exactly the case that crashed
    obj.drop_to_plate()
    pts = find_contact_points(obj.transformed(), SupportSettings(density="high"), layer_height=0.03)
    assert len(pts) > 20


def test_theme_colors(tmp_path):
    """Two colour sets (light, dark); colours saved by the one-set version
    carry over to the dark set."""
    from openvat.ui.app_settings_dialog import saved_colors, theme_name
    from openvat.ui.viewport3d import DEFAULT_THEME_COLORS
    st = ProfileStore(tmp_path)
    assert saved_colors(st) == DEFAULT_THEME_COLORS
    st.settings["colors"] = {"model": "#112233", "bogus": "#000000"}            # old flat format
    c = saved_colors(st)
    assert c["dark"]["model"] == "#112233" and "bogus" not in c["dark"]
    assert c["light"] == DEFAULT_THEME_COLORS["light"]
    st.settings["colors"] = {"light": {"background": "#ffffff"}, "dark": {}}   # current format
    c = saved_colors(st)
    assert c["light"]["background"] == "#ffffff" and c["dark"] == DEFAULT_THEME_COLORS["dark"]
    assert theme_name(True) == "dark" and theme_name(False) == "light"
    from openvat.ui.app_settings_dialog import dark_mode
    assert dark_mode(ProfileStore(tmp_path / "fresh"))                          # dark by default
    st.settings["dark_mode"] = False
    assert not dark_mode(st)
    assert set(DEFAULT_THEME_COLORS["light"]) == set(DEFAULT_THEME_COLORS["dark"])


def test_hollowing(store):
    """Walls, floor and ceiling keep the thickness; the drain hole opens the
    cavity; internal supports stay inside the model and print without islands."""
    import math
    from openvat.core.hollow import HollowState, LatticeParams, hole_from_world
    resin = store.resins[0]
    resin.shrink_x = resin.shrink_y = resin.shrink_z = 1.0
    resin.elephant_foot_mm = 0.0
    scene, obj = cube_scene(20.0)
    solid = slice_scene(scene, store.printers[0], resin)
    obj.holes = (hole_from_world(obj.matrix(), [0, 0, 20.0], [0, 0, 1], 3.0),)
    assert slice_scene(scene, store.printers[0], resin).volume_mm3() == pytest.approx(solid.volume_mm3())  # a marker only
    obj.hollow = HollowState(2.0)
    hollow = slice_scene(scene, store.printers[0], resin)
    layers = hollow.layers
    mid = layers[len(layers) // 2].geometry
    assert len(mid.geoms) == 1 and len(mid.geoms[0].interiors) == 1
    inner = shapely.Polygon(mid.geoms[0].interiors[0])
    wall = (20.0 - (inner.bounds[2] - inner.bounds[0])) / 2
    assert 2.0 <= wall <= 2.1                                         # never thinner than asked
    open_layers = [i for i, l in enumerate(layers) if any(p.interiors for p in l.geometry.geoms)]
    assert layers[open_layers[0]].z_bottom >= 2.0                     # floor
    top = layers[-1].geometry                                         # the hole goes through the top
    assert top.area == pytest.approx(400.0 - math.pi * 1.5 ** 2, rel=0.01)
    for layer in layers:                                              # ceiling: 2 mm solid but for the hole
        if layer.z_bottom >= 18.0:
            assert layer.area == pytest.approx(top.area, rel=0.01)
    assert hollow.volume_mm3() < 0.55 * solid.volume_mm3()
    assert not hollow.islands
    outline = shapely.box(-10.001, -10.001, 10.001, 10.001)
    for angle in (0.0, 45.0, 90.0):
        obj.hollow = HollowState(2.0, LatticeParams(0.0, angle, 0.0, 5.0, 1.2))
        r = slice_scene(scene, store.printers[0], resin)
        assert hollow.volume_mm3() < r.volume_mm3() < solid.volume_mm3()
        assert all(outline.contains(l.geometry) for l in r.layers if not l.geometry.is_empty)
        assert not r.islands, angle
    assert r.layers[-1].geometry.area == pytest.approx(top.area, rel=0.01)      # poles keep out of the hole


def test_lattice_geometry():
    import math
    from openvat.core.hollow import (lattice_segments, lattice_rotation, cylinder_sections, LatticeParams,
                                     HollowSettings)
    lo, hi = np.array([-20.0, -20, 0]), np.array([20.0, 20, 40])

    def directions(angles):
        p0, p1 = lattice_segments(lo, hi, LatticeParams(*angles, 8.0, 1.0))
        assert np.all(p0 >= lo - 1e-9) and np.all(p1 <= hi + 1e-9)
        d = (p1 - p0) / np.linalg.norm(p1 - p0, axis=1)[:, None]
        d *= np.where(d @ [1.0, 2.0, 3.0] < 0, -1.0, 1.0)[:, None]           # one sign per direction
        return np.unique(np.round(d, 6), axis=0)

    for angles in [(0, a, 0) for a in (0, 30, 45, 60, 90)] + [(20, 45, 10), (45, 35, 0), (90, 90, 90), (10, 0, 70)]:
        dirs = directions(angles)
        assert len(dirs) == 3                                               # three pole directions ...
        assert np.allclose(np.abs(dirs @ dirs.T), np.eye(3), atol=1e-6)      # ... at right angles
        assert all(np.isclose(np.abs(dirs @ ax), 1.0).any() for ax in lattice_rotation(*angles))
    for a in (0, 30, 45, 60, 90):                                           # about Y alone: X and Z tilt, Y stays level
        rises = sorted({round(math.degrees(math.asin(abs(z))), 3) for z in directions((0, a, 0))[:, 2]})
        assert rises == sorted({0.0, float(a), 90.0 - a}), (a, rises)
    x, y, z = lattice_rotation(0, 45, 0)                                    # the default: an X seen from the front
    assert np.allclose(x, [math.sqrt(0.5), 0, -math.sqrt(0.5)]) and np.allclose(z, [math.sqrt(0.5), 0, math.sqrt(0.5)])
    assert np.allclose(y, [0, 1, 0])
    corner = lattice_rotation(45, 35.264, 0)                                # on a corner: no pole lies flat
    assert min(math.degrees(math.asin(abs(v[2]))) for v in corner) > 35.0
    y_turned = lattice_rotation(0, 45, 30)[1]                               # Z turns the level Y poles in plan
    assert abs(y_turned[2]) < 1e-9 and np.isclose(math.degrees(math.atan2(-y_turned[0], y_turned[1])), 30.0)
    old = HollowSettings.from_dict({"lattice_angle": 30.0, "lattice_spacing": 6.0})   # one angle before: about Y
    assert (old.lattice_angle_x, old.lattice_angle_y, old.lattice_angle_z, old.lattice_spacing) == (0.0, 30.0, 0.0, 6.0)
    for rise in (20, 45, 70, 90):                                            # exact cross-section areas
        a = math.radians(rise)
        d = np.array([math.cos(a), 0, math.sin(a)])
        sec = cylinder_sections(np.array([[0, 0, 10.0]]) - 40 * d, np.array([[0, 0, 10.0]]) + 40 * d, 0.5, 10.0)
        assert sum(p.area for p in sec) == pytest.approx(math.pi * 0.25 / math.sin(a), rel=0.03)


def test_holes_follow_the_model():
    from openvat.core.hollow import hole_from_world, hole_world, HollowState
    scene, obj = cube_scene(20.0)
    p, n = np.array([10.0, 2.0, 5.0]), np.array([1.0, 0.0, 0.0])       # on the +X face
    obj.holes = (hole_from_world(obj.matrix(), p, n, 2.0),)
    obj.position[:] += [7.0, -3.0, 0.0]
    c, nn = hole_world(obj.matrix(), obj.holes[0])
    assert np.allclose(c, p + [7.0, -3.0, 0.0]) and np.allclose(nn, n)
    obj.rotation[:] = [0, 0, 90]                                           # +X face turns to +Y
    c, nn = hole_world(obj.matrix(), obj.holes[0])
    assert np.allclose(nn, [0, 1, 0], atol=1e-9)
    obj.rotation[:] = 0
    obj.scale[:] = [2.0, 1.0, 1.0]                                         # stays on the stretched face
    c, nn = hole_world(obj.matrix(), obj.holes[0])
    assert c[0] == pytest.approx(obj.world_bounds()[1][0]) and np.allclose(nn, n)
    obj.scale[:] = 1.0
    obj.mirror(0)                                                          # mirrored with the mesh
    c, nn = hole_world(obj.matrix(), obj.holes[0])
    assert c[0] == pytest.approx(obj.world_bounds()[0][0]) and np.allclose(nn, -n)
    # undo brings back holes and hollow state
    scene.push_undo()
    obj.hollow = HollowState(1.5)
    obj.holes = ()
    assert scene.undo_stack.undo()
    assert obj.hollow is None and len(obj.holes) == 1
    assert scene.undo_stack.redo()
    assert obj.hollow == HollowState(1.5) and obj.holes == ()


def _run_python(code: str, env: dict | None = None):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    full = dict(os.environ, **(env or {}))
    full.pop("PYOPENGL_PLATFORM", None)
    return subprocess.run([sys.executable, "-c", code], cwd=root, env=full,
                          capture_output=True, text=True, timeout=120)


def test_opengl_waits_for_qt():
    """PyOpenGL picks GLX or EGL when OpenGL.GL is first imported, so nothing
    the app imports before QApplication exists may pull it in."""
    r = _run_python("import sys, openvat.app, openvat.ui.app_settings_dialog, openvat.core.profiles\n"
                    "print(sorted(m for m in sys.modules if m.split('.')[0] == 'OpenGL'))")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_gl_interface_matches_qt():
    """A Wayland session with Qt on XWayland (as in the AppImage): PyOpenGL
    alone would choose EGL while Qt draws with GLX - a segfault at the first
    frame.  match_pyopengl makes it follow Qt."""
    import os
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X server (run under xvfb-run)")
    code = ("import os\n"
            "from PySide6.QtGui import QOpenGLContext\n"
            "from PySide6.QtWidgets import QApplication\n"
            "from openvat.ui.glplatform import match_pyopengl\n"
            "app = QApplication([])\n"
            "chosen = match_pyopengl()\n"
            "import OpenGL.GL, OpenGL.platform as P\n"
            "print(QOpenGLContext().create(), chosen, type(P.PLATFORM).__name__)\n")
    r = _run_python(code, {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "openvat-test-none",
                           "QT_QPA_PLATFORM": "xcb", "QT_XCB_GL_INTEGRATION": ""})
    assert r.returncode == 0, r.stderr
    has_gl, chosen, loaded = r.stdout.split()
    if has_gl != "True":
        pytest.skip("no OpenGL available")
    assert (chosen, loaded) == ("glx", "GLXPlatform")


def test_pw0_run_length_encoding():
    """pw0Img: black/white runs take two bytes and up to 4095 pixels, grey
    runs one byte and up to 15; the rows run on into each other."""
    from openvat.formats.pwbinary import encode_pw0, decode_pw0
    assert encode_pw0(np.zeros(5, np.uint8)) == b"\x00\x05"
    assert encode_pw0(np.full(4096, 15, np.uint8)) == b"\xff\xff\xf0\x01"
    assert encode_pw0(np.full(3, 7, np.uint8)) == b"\x73"
    assert encode_pw0(np.full(40, 9, np.uint8)) == b"\x9f\x9f\x9a"
    rng = np.random.default_rng(1)
    img = np.repeat(rng.choice([0, 15, 3, 12], size=400), rng.integers(1, 9000, size=400)).astype(np.uint8)
    img = img[:img.size // 50 * 50].reshape(-1, 50)
    assert np.array_equal(decode_pw0(encode_pw0(img), img.shape[1], img.shape[0]), img)


def test_pwx_rasterizer():
    """Pixel centres inside are lit; row 0 is the -Y edge and column 0 the -X
    edge; anti-aliasing gives grey levels that add up to the area."""
    from shapely.geometry import MultiPolygon, box, Point
    from openvat.core.profiles import PrinterProfile
    from openvat.formats.pwbinary import rasterize
    p = PrinterProfile(name="t", output_type="anycubic_bitmap", res_x=200, res_y=100, print_x=20.0, print_y=10.0,
                       pixel_x_um=100.0, pixel_y_um=100.0, file_extension="pwx", layer_format="pw0Img")
    img = rasterize(MultiPolygon([box(2.0, 1.0, 3.0, 2.0)]), p)          # +x, +y quadrant, 10 x 10 px
    rows, cols = np.nonzero(img)
    assert img.max() == 15 and len(rows) == 100
    assert (cols.min(), cols.max(), rows.min(), rows.max()) == (120, 129, 60, 69)
    ring = box(-5, -3, 5, 3).difference(box(-3, -2, 3, 2)).union(box(-1, -1, 1, 1))   # island in a hole
    img = rasterize(MultiPolygon(list(ring.geoms)), p)
    assert (img > 0).sum() == round(ring.area / 0.01)
    assert img[50, 100] == 15 and img[50, 75] == 0                     # island lit, hole dark
    disc = Point(0.33, -0.21).buffer(3.1, 256)
    for aa in (1, 4, 16):
        img = rasterize(MultiPolygon([disc]), p, aa)
        light = np.where(img > 0, (img.astype(float) + 1) / 16, 0.0)     # level 16k/n - 1 = k/n covered
        assert abs(light.sum() * 0.01 - disc.area) < 0.01 * disc.area
    assert set(np.unique(rasterize(MultiPolygon([disc]), p, 4)).tolist()) <= {0, 3, 7, 11, 15}
    assert len(np.unique(rasterize(MultiPolygon([disc]), p, 16))) > 8           # greys along the edge


def test_write_and_read_pwx(tmp_path):
    """A Photon X file has the layout of Photon Workshop's own (.pwx, file
    version 1, pw0Img layers)."""
    import struct
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    from openvat.formats.export import write_print_file, why_not
    from openvat.formats.pwbinary import read_pwbinary
    px, resins = read_printer_file(RESOURCES / PRINTERS / "Anycubic Photon X.json")
    zero = read_printer_file(RESOURCES / PRINTERS / "Anycubic Photon Zero.json")[0]
    odd = PrinterProfile(name="Odd", output_type="anycubic_bitmap", layer_format="pw9Img")
    assert px.can_export and zero.can_export and px.file_version == 1
    assert not odd.can_export and "pw9Img" in why_not(odd)
    resin = resins[0]
    scene, obj = cube_scene(10.0)
    obj.position[:2] = [20.0, -15.0]                                      # off centre: +x, -y
    result = slice_scene(scene, px, resin)
    out = write_print_file(tmp_path / "cube.pwx", result, [obj.transformed()])
    data = out.read_bytes()
    assert data[:12] == b"ANYCUBIC\0\0\0\0" and struct.unpack_from("<II7I", data, 12) == (
        1, 4, 48, 0, 144, 0, 75436, 0, 75436 + 20 + 32 * len(result.layers))
    f = read_pwbinary(out)
    h = f.header
    assert (h["res_x"], h["res_y"], h["pixel_um"], h["anti_aliasing"]) == (2560, 1600, 75.0, 1)
    assert (h["exposure"], h["bottom_exposure"], h["bottom_layers"]) == (
        resin.normal.exposure, resin.bottom.exposure, resin.bottom_layers)
    assert h["per_layer_override"] == 0 and h["tail"] == 0x3FF00000
    assert len(f.layers) == len(result.layers) == 200
    assert [d.exposure for d in f.layers[:resin.bottom_layers + 1]] == \
        [resin.bottom.exposure] * resin.bottom_layers + [resin.normal.exposure]
    assert f.preview.size == (224, 168)
    img = f.layer_image(100)
    assert np.count_nonzero(img) == f.layers[100].lit_pixels
    assert abs(f.layers[100].lit_pixels * 0.075 ** 2 - 100.0) < 1.5      # 10 x 10 mm
    rows, cols = np.nonzero(img)
    assert abs(cols.mean() - (1280 + 20 / 0.075)) < 1 and abs(rows.mean() - (800 - 15 / 0.075)) < 1
    assert f.raw.build() == data                                          # reads back to the same bytes


@pytest.mark.parametrize("name, version, first_layer_at", [("Anycubic Photon Mono SE", 515, 106204),
                                                           ("Anycubic Photon M3 Max", 516, 106440),
                                                           ("Anycubic Photon D2", 517, 106664),
                                                           ("Anycubic Photon Mono M5s Pro", 518, 270636)])
def test_write_binary_516_517(tmp_path, name, version, first_layer_at):
    """Versions 515 (.pwms), 516 (.pm3m), 517 (.dl2p) and 518 (.m5sp) as
    Photon Workshop lays them out: with 960 layers the layer data starts
    where it does in its own files."""
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    from openvat.formats.export import write_print_file
    from openvat.formats.pwbinary import read_pwbinary
    printer, resins = read_printer_file(RESOURCES / PRINTERS / f"{name}.json")
    resin = resins[0]
    resin.normal.thickness = resin.bottom.thickness = 0.01                # 960 layers of a 9.6 mm cube
    scene, obj = cube_scene(9.6)
    result = slice_scene(scene, printer, resin)
    assert len(result.layers) == 960
    out = write_print_file(tmp_path / f"cube.{printer.file_extension}", result, [obj.transformed()])
    f = read_pwbinary(out)
    assert f.version == version == printer.file_version and f.layers[0].address == first_layer_at
    assert f.raw.color_table[8:24] == b"\xff" * 16
    if version >= 516:
        assert f.machine_name == name and f.raw.machine[1].rstrip(b"\0") == b"pw0Img"
        assert f.raw.machine[4:7] == pytest.approx((printer.print_x, printer.print_y, printer.print_z), rel=1e-6)
        assert f.raw.machine[7] == version
        extra = f.raw.extra
        assert extra[0] == extra[7] == 2
        assert extra[1] + extra[4] == pytest.approx(resin.bottom.lift_distance)
        assert extra[8] + extra[11] == pytest.approx(resin.normal.lift_distance)
    else:
        assert not f.raw.machine and not f.raw.extra
    if version >= 517:
        assert f.raw.software[0].rstrip(b"\0") == b"AC-PC" and f.raw.software[1] == 164
        assert f.raw.model[:6] == pytest.approx((-4.8, -4.8, 0.0, 4.8, 4.8, 9.6), abs=1e-4)
        assert f.header["resin_code"] == 10
    if version == 518:
        assert f.header["intelli_mode"] == 1
        assert f.raw.machine[9:11] == pytest.approx((printer.pixel_x_um, printer.pixel_y_um), rel=1e-6)
        assert f.raw.machine[12:17] == (1, 0, 0, printer.res_x, printer.res_y)         # one screen
        assert f.raw.subimages[0] == 1 and f.raw.preview2[0::2][:2] == (320, 190)
    lit = f.layers[500].lit_pixels * (printer.pixel_x_mm * printer.pixel_y_mm)
    assert lit == pytest.approx(9.6 * 9.6, rel=0.02)
    assert f.raw.build() == out.read_bytes()


def test_bitmap_layers_in_pwsz_family(tmp_path):
    """With anti-aliasing the .pwsz family holds pw0Img bitmaps (as Photon
    Workshop's .pp1 files do), turned 180 degrees for the P1; without it,
    vector layers as before."""
    import zipfile
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    from openvat.formats.export import write_print_file
    from openvat.formats.pwsz import read_pwsz
    p1, resins = read_printer_file(RESOURCES / PRINTERS / "Anycubic Photon P1.json")
    resin = resins[0]
    assert resin.anti_aliasing == 16
    scene, obj = cube_scene(4.0)
    obj.position[:2] = [30.0, 20.0]                                       # +x, +y corner
    result = slice_scene(scene, p1, resin)
    out = write_print_file(tmp_path / "cube.pp1", result, [obj.transformed()])
    names = zipfile.ZipFile(out).namelist()
    assert "layer_images/layer_0.pw0Img" in names and not any(n.endswith(".pwszImg") for n in names)
    f = read_pwsz(out)
    assert f.bitmap and len(f.layer_bitmaps) == len(result.layers)
    img = f.layer_bitmap(len(result.layers) // 2)
    rows, cols = np.nonzero(img)
    # turned: +x is towards column 0, +y towards row 0
    assert abs(cols.mean() - (p1.res_x / 2 - 30.0 / p1.pixel_x_mm)) < 2
    assert abs(rows.mean() - (p1.res_y / 2 - 20.0 / p1.pixel_y_mm)) < 2
    assert len(np.unique(img)) > 3                                          # grey edges
    light = np.where(img > 0, (img.astype(float) + 1) / 16, 0.0)
    assert light.sum() * p1.pixel_x_mm * p1.pixel_y_mm == pytest.approx(16.0, rel=0.01)
    resin.anti_aliasing = 1
    out2 = write_print_file(tmp_path / "cube_vector.pp1", slice_scene(scene, p1, resin), [obj.transformed()])
    assert not read_pwsz(out2).bitmap


@pytest.mark.parametrize("name", ["Anycubic Photon", "Anycubic Photon S"])
def test_write_pws(tmp_path, name):
    """The Photon and Photon S: version 1 files with pwsImg layers - one byte
    per run, bit 7 lit, up to 125 pixels."""
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    from openvat.formats.export import write_print_file
    from openvat.formats.pwbinary import read_pwbinary, encode_pws, decode_pws
    assert encode_pws(np.full(300, 15, np.uint8)) == bytes([0xFD, 0xFD, 0xB2])
    assert encode_pws(np.zeros(126, np.uint8)) == bytes([0x7D, 0x01])
    printer, resins = read_printer_file(RESOURCES / PRINTERS / f"{name}.json")
    assert printer.can_export and printer.layer_format == "pwsImg"
    scene, obj = cube_scene(10.0)
    obj.position[:2] = [10.0, -20.0]
    result = slice_scene(scene, printer, resins[0])
    out = write_print_file(tmp_path / "cube.pws", result, [obj.transformed()])
    f = read_pwbinary(out)
    assert f.version == 1 and f.pws and f.res == (1440, 2560) and f.header["anti_aliasing"] == 1
    img = f.layer_image(100)
    assert set(np.unique(img)) == {0, 15} and np.count_nonzero(img) == f.layers[100].lit_pixels
    rows, cols = np.nonzero(img)
    px = printer.pixel_x_mm
    assert abs(cols.mean() - (720 + 10 / px)) < 1 and abs(rows.mean() - (1280 - 20 / px)) < 1
    assert np.count_nonzero(img) * px * px == pytest.approx(100.0, rel=0.02)
    blob = f.data[f.layers[100].address:f.layers[100].address + f.layers[100].length]
    assert max(b & 0x7F for b in blob) <= 125 and np.array_equal(decode_pws(blob, 1440, 2560), img)
    assert f.raw.build() == out.read_bytes()


def test_write_ultra_anti_aliased(tmp_path):
    """The Photon Ultra's resins use 4 anti-aliasing steps: grey levels
    0, 3, 7, 11, 15 and a colour table of 63, 127, 191, 255 - as in Photon
    Workshop's .dlp."""
    from openvat.core.profiles import RESOURCES, PRINTERS, read_printer_file
    from openvat.formats.export import write_print_file
    from openvat.formats.pwbinary import read_pwbinary
    printer, resins = read_printer_file(RESOURCES / PRINTERS / "Anycubic Photon Ultra.json")
    assert printer.file_version == 515 and resins[0].anti_aliasing == 4
    scene, obj = cube_scene(10.0)
    obj.rotate(2, 30)                                                     # slanted edges: greys
    out = write_print_file(tmp_path / "cube.dlp", slice_scene(scene, printer, resins[0]), [obj.transformed()])
    f = read_pwbinary(out)
    assert f.header["anti_aliasing"] == 4
    assert f.raw.color_table[8:24] == bytes([63, 127, 191, 255] + [255] * 12)
    img = f.layer_image(100)
    assert set(np.unique(img).tolist()) == {0, 3, 7, 11, 15}
    assert f.raw.build() == out.read_bytes()
