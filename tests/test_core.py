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


def unscaled(resin: ResinProfile) -> ResinProfile:
    """The resin without scale compensation (bottom and normal layers)."""
    for layer in (resin.bottom, resin.normal):
        layer.scale_x = layer.scale_y = layer.scale_z = 1.0
    return resin


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
    expected = (20 * resin.normal.scale_x) * (20 * resin.normal.scale_y)
    assert result.layers[200].area == pytest.approx(expected, rel=1e-4)
    assert result.layers[0].exposure == resin.bottom.exposure
    assert result.layers[-1].exposure == resin.normal.exposure


def test_bottom_transition_and_normal_layers_scale(store):
    """Each layer type has its own X/Y/Z scale; transition layers ramp from
    the bottom values to the normal ones (like the exposure)."""
    scene, obj = cube_scene()
    obj.position[:2] = [30.0, -10.0]                           # scaled about its own center
    resin = unscaled(store.resins[0])
    resin.bottom_layers, resin.transition_layers = 3, 1
    resin.bottom.scale_x = resin.bottom.scale_y = 0.95
    resin.normal.scale_x = 1.01
    result = slice_scene(scene, store.printers[0], resin)
    sizes = [(round(b[2] - b[0], 6), round(b[3] - b[1], 6)) for b in (l.geometry.bounds for l in result.layers)]
    assert sizes[:3] == [(19.0, 19.0)] * 3                     # bottom
    assert sizes[3] == (pytest.approx(19.6), pytest.approx(19.5))   # transition: halfway
    assert sizes[4] == sizes[-1] == (pytest.approx(20.2), 20.0)     # normal
    assert all(np.allclose(l.geometry.centroid.coords[0], [30.0, -10.0]) for l in result.layers)
    assert len(result.layers) == 400                           # Z unscaled


def test_layer_z_scale(store):
    """Z scale: a layer stands for thickness / Z of the model."""
    scene, _ = cube_scene(10.0)
    resin = unscaled(store.resins[0])
    resin.bottom_layers, resin.transition_layers = 4, 0
    resin.bottom.thickness = resin.normal.thickness = 0.05
    resin.bottom.scale_z = 2.0                     # 4 bottom layers cover 0.1 mm of the model
    result = slice_scene(scene, store.printers[0], resin)
    assert len(result.layers) == 202 and result.height == pytest.approx(10.1)
    # the same Z everywhere is the old global shrinkage compensation
    resin.bottom.scale_z = resin.normal.scale_z = 1.1
    assert slice_scene(scene, store.printers[0], resin).height == pytest.approx(11.0)


def test_scale_follows_each_model_and_its_supports():
    """Two models apart on the plate: each one's layers (supports included)
    scale about that model's own center, so supports stay under it."""
    import shapely.affinity
    scene = Scene()
    slabs = []
    for x in (-40.0, 35.0):
        slab = MeshObject("slab", trimesh.creation.box(extents=[20, 14, 3]))
        slab.drop_to_plate()
        slab.position[:] += [x, 5.0, 8.0]
        scene.add(slab)
        slabs.append(slab)
    scene.auto_supports()
    plain = unscaled(ResinProfile())
    big = unscaled(ResinProfile())
    big.bottom.scale_x = big.bottom.scale_y = big.normal.scale_x = big.normal.scale_y = 1.1
    a = slice_scene(scene, PrinterProfile(), plain)
    b = slice_scene(scene, PrinterProfile(), big)
    for i in (0, 60, len(a.layers) - 1):                       # raft, pillars, the slabs
        for slab in slabs:
            c = slab.world_bounds().mean(axis=0)[:2]
            box = shapely.box(c[0] - 30, c[1] - 30, c[0] + 30, c[1] + 30)
            want = shapely.affinity.scale(a.layers[i].geometry.intersection(box), 1.1, 1.1, origin=tuple(c))
            got = b.layers[i].geometry.intersection(box)
            assert got.symmetric_difference(want).area < 1e-6 * max(want.area, 1.0)


def test_old_resin_scale_and_elephant_foot_are_converted():
    from dataclasses import asdict
    from openvat.core.profiles import resin_from_dict
    old = {"name": "old", "shrink_x": 1.02, "shrink_y": 1.03, "shrink_z": 1.04, "elephant_foot_mm": 0.3,
           "normal": {"thickness": 0.03, "exposure": 2.5}, "bottom_layers": 5}
    r = resin_from_dict(old)
    assert (r.normal.thickness, r.normal.exposure, r.bottom_layers) == (0.03, 2.5, 5)
    assert (r.normal.scale_x, r.normal.scale_y, r.normal.scale_z) == (1.02, 1.03, 1.04)
    assert (r.bottom.scale_x, r.bottom.scale_y, r.bottom.scale_z) == (1.02, 1.03, 1.04)
    assert r.bottom.exposure == ResinProfile().bottom.exposure      # missing: the resin's own default
    assert not {"shrink_x", "elephant_foot_mm"} & set(asdict(r))
    new = dict(old, normal={"scale_x": 0.99})                         # new-style values win
    assert resin_from_dict(new).normal.scale_x == 0.99


def test_layer_image_roundtrip():
    scene, _ = cube_scene()
    hole = trimesh.creation.cylinder(radius=3, height=40)
    scene.objects[0].mesh = scene.objects[0].mesh.difference(hole)
    resin = ResinProfile()
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


def test_delete_any_support_or_object():
    """Supports and objects are found by identity: removing one that is not
    first in the list used to compare numpy arrays and raise."""
    scene = Scene()
    slab = MeshObject("slab", trimesh.creation.box(extents=[30, 20, 3]))
    slab.drop_to_plate()
    slab.position[2] += 10
    scene.add(slab)
    assert scene.auto_supports(slab) > 3
    n = len(scene.supports)
    third = scene.supports[2]
    scene.selected_support = third
    scene.remove_support(third)
    assert len(scene.supports) == n - 1 and all(s is not third for s in scene.supports)
    assert scene.selected_support is None
    assert scene.undo_stack.undo() and len(scene.supports) == n
    # objects too, even next to a twin with the same name and mesh
    twin = MeshObject("slab", slab.mesh, position=slab.position.copy())
    scene.add(twin)
    scene.remove(twin)
    assert scene.objects == [slab]


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
    resin = unscaled(store.resins[0])
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
    resin = unscaled(store.resins[0])
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
    export = Path(__file__).resolve().parents[1] / "tests" / "xyz_cube.pwsz"
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
                               "material_scale_xyz": {"x": 1.01111, "y": 1.01111, "z": 1.055}}}
    r = resin_from_photonworkshop(entry)
    assert r.name == "Anycubic Standard Resin V2 (ACF, Normal 0.05 mm)"
    assert (r.normal.exposure, r.bottom.exposure, r.bottom_layers, r.transition_layers) == (1.8, 25.0, 4, 0)
    for layer in (r.normal, r.bottom):                      # one set for every layer in PhotonWorkshop
        assert (layer.scale_x, layer.scale_y, layer.scale_z) == (1.01111, 1.01111, 1.055)


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
    for o in __import__("openvat.core.loaders", fromlist=["load_file"]).load_file("docs/logos/openvat-logo.stl"):
        o.scale[:] = [0.1, 0.1, 1.0]
        o.drop_to_plate()
        scene.add(o)
    printer = PrinterProfile()
    result = slice_scene(scene, printer, ResinProfile())
    for layer in (result.layers[0], result.layers[50]):
        img = decode_layer_image(encode_layer_image(layer.geometry))
        filled = rasterize_layer_image(img, printer.pixel_x_mm, printer.pixel_y_mm,
                                       printer.res_x, printer.res_y)
        exposed = filled.sum() * printer.pixel_x_mm * printer.pixel_y_mm
        assert exposed == pytest.approx(layer.area, rel=0.01)


def test_z_shrink_compensation():
    scene, _ = cube_scene(10.0)
    from openvat.core.profiles import PrinterProfile
    resin = ResinProfile()
    resin.bottom.scale_z = resin.normal.scale_z = 1.1
    result = slice_scene(scene, PrinterProfile(), resin)
    assert result.height == pytest.approx(11.0, abs=0.05)     # 10 mm cube printed 10 % taller


def test_z_stretch_prints_the_lowest_layers_taller():
    from dataclasses import asdict, replace
    from openvat.core.profiles import resin_from_dict
    from openvat.core.slicer import stretched_z, unstretched_z, slice_mesh_at
    plain = ResinProfile()
    plain.bottom.thickness = plain.normal.thickness = 0.05
    stretch = replace(plain, stretch_height=2.0, stretch_factor=1.1)
    # older profiles have no stretch settings: off
    old = resin_from_dict({"name": "old"})
    assert (old.stretch_height, old.stretch_factor) == (0.0, 1.0)
    assert resin_from_dict(asdict(stretch)).stretch_factor == 1.1
    # the mapping (model height -> print height) and its inverse
    z = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    assert stretched_z(z, stretch) == pytest.approx([0.0, 1.1, 2.2, 3.2, 10.2])
    assert unstretched_z(stretched_z(z, stretch), stretch) == pytest.approx(z)
    assert stretched_z(z, plain) == pytest.approx(z)
    # a cone, so every layer's outline depends on its height
    scene = Scene()
    cone = MeshObject("cone", trimesh.creation.cone(radius=8, height=10, sections=64))
    cone.center_on_plate()
    cone.drop_to_plate()
    scene.add(cone)
    a = slice_scene(scene, PrinterProfile(), plain)
    b = slice_scene(scene, PrinterProfile(), stretch)
    assert a.height == pytest.approx(10.0, abs=0.01)
    assert b.height == pytest.approx(10.2, abs=0.01)         # the lowest 2 mm print as 2.2 mm
    assert len(b.layers) == len(a.layers) + 4
    # the stretched part: 44 layers that cut the model at 1/1.1 of their height
    mesh = cone.transformed()
    for layer in b.layers[:44:5]:
        assert layer.area == pytest.approx(slice_mesh_at(mesh, layer.z_center / 1.1).area, rel=1e-6)
    # above it: the same layers as without stretch, 0.2 mm higher
    for la, lb in zip(a.layers[40:], b.layers[44:]):
        assert lb.z_bottom == pytest.approx(la.z_bottom + 0.2)
        assert lb.area == pytest.approx(la.area, rel=1e-6)



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
    from openvat.core.supports import SupportSettings, SIDES, _equal_area
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
        octagon = _equal_area(SIDES)                                              # corners of the octagon
        assert radial.max() <= st.nook_diameter / 2 * octagon + 1e-6               # thin all the way
        ends = (along < 0.2) | (along > np.linalg.norm(foot - tip) - 0.2)          # both contacts
        assert radial[ends].max() <= st.nook_contact_diameter / 2 * octagon + 1e-6
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
    """The 20 mm XYZ cube (10 mm letters engraved 2 mm deep) turned +45 deg
    about X and Y (a corner points down) and lifted 5 mm - engraved letters
    on slanted faces."""
    obj = MeshObject("xyz", trimesh.load("tests/xyz_cube.stl"))
    obj.rotate(0, 45)
    obj.rotate(1, 45)
    obj.elevation = 5.0
    obj.drop_to_plate()
    return obj


def test_grooves_get_short_nooks_square_to_their_walls():
    """Overhanging walls inside the engraved letters get small nooks straight
    across the groove (like a rung), meeting both walls squarely, kept clear
    of the groove's bottom and ends - never into the part.  A spot where the
    groove is a little too wide for that (where the Z's diagonal meets a bar)
    gets an ordinary nook standing on the part instead, also clear of it."""
    from openvat.core.supports import SupportSettings, nook_cone_length
    from openvat.core.placement import generate_supports
    st = SupportSettings()
    obj = _rotated_xyz_cube()
    report = {}
    sups = generate_supports(obj, st, report)
    nooks = [s for s in sups if s.is_nook]
    across = {s.id for s in nooks if "across" in s.status}            # straight across a groove
    assert len(across) >= 4 and report["skipped"] == 0
    tri = obj.transformed().triangles
    for s in nooks:
        tip, foot = np.array(s.tip), np.array(s.foot)
        length = np.linalg.norm(foot - tip)
        axis = (foot - tip) / length
        n = s.normal / np.linalg.norm(s.normal)
        if s.id in across:
            assert length <= st.nook_max_gap
            assert np.degrees(np.arccos(np.clip(axis @ n, -1, 1))) < 10            # square to the wall
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
    resin = ResinProfile()
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
    obj = _rotated_xyz_cube()                                          # the settings it was seen with
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
    resin = unscaled(store.resins[0])
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


def _bracket():
    """An L-shaped bracket: its two inside faces are concave."""
    return trimesh.boolean.union([trimesh.creation.box(extents=[30, 12, 6]).apply_translation([0, 0, 3]),
                                  trimesh.creation.box(extents=[6, 12, 24]).apply_translation([-12, 0, 12])])


def test_rotation_helpers():
    from openvat.core.orient import rotation_matrix, angles_of, rotated_about, laid_on, turn_onto
    obj = MeshObject("box", trimesh.creation.box(extents=[1, 2, 3]))
    obj.rotation[:] = [30, -50, 170]
    assert np.allclose(obj.matrix()[:3, :3], rotation_matrix(obj.rotation))   # same convention as the model
    assert angles_of(rotation_matrix([30, -50, 170])) == pytest.approx([30, -50, 170])
    assert angles_of(rotation_matrix([0, 90, 0])) == pytest.approx([0, 90, 0])
    # a turn about a world axis composes on the left
    turned = rotated_about([30, -50, 170], [0, 0, 1], 25)
    want = trimesh.transformations.rotation_matrix(np.radians(25), [0, 0, 1])[:3, :3] @ rotation_matrix([30, -50, 170])
    assert np.allclose(rotation_matrix(turned), want)
    # laying on a face turns that face's normal straight down
    for n in ([1, 0, 0], [0, 0, 1], [0, 0, -1], [0.3, -0.5, 0.8]):
        n = np.array(n, float) / np.linalg.norm(n)
        assert turn_onto(n) @ n == pytest.approx([0, 0, -1])
        start = [10, 20, 30]
        local = rotation_matrix(start).T @ n                       # the face in the model's own frame
        assert rotation_matrix(laid_on(start, n)) @ local == pytest.approx([0, 0, -1], abs=1e-6)


def test_flat_faces_leave_out_concave_and_curved_faces():
    from openvat.core.orient import flat_faces, laid_on
    bracket = _bracket()
    faces = flat_faces(bracket)
    normals = {tuple(np.round(f.normal, 6) + 0.0) for f in faces}
    assert normals == {(0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)}
    top = next(f for f in faces if f.normal[2] > 0.99)
    assert top.area == pytest.approx(6 * 12)                      # the tall leg's top, not the short leg's
    side = next(f for f in faces if f.normal[0] > 0.99)
    assert side.area == pytest.approx(6 * 12)                     # the long leg's end, not the inside wall
    for f in faces:                                               # laid on any of them: nothing below it
        obj = MeshObject("bracket", bracket)
        obj.rotation[:] = laid_on(obj.rotation, f.normal)
        obj.drop_to_plate()
        world = obj.transformed()
        assert world.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-6)
        assert world.area_faces[world.face_normals[:, 2] < -0.9999].sum() >= f.area - 1e-6
    # a cylinder offers its two ends, not the strips of its side; a ball offers nothing
    ends = flat_faces(trimesh.creation.cylinder(radius=5, height=10, sections=64))
    assert sorted(round(f.normal[2]) for f in ends) == [-1, 1]
    assert flat_faces(trimesh.creation.icosphere(3, 10)) == []


def test_gizmo_picking_and_drag_math():
    from openvat.ui import gizmos
    from openvat.ui.camera import OrbitCamera
    cam = OrbitCamera()
    cam.aspect = 800 / 600
    cam.target = np.array([0.0, 0.0, 10.0])
    vp = cam.projection_matrix() @ cam.view_matrix()
    c = np.array([0.0, 0.0, 10.0])
    size = gizmos.world_size(cam, c, gizmos.ARROW_PX, 600)
    for axis in range(3):                                         # the middle of each arrow grabs it
        mid, _ = gizmos.project(vp, [c + np.eye(3)[axis] * size * 0.6], 800, 600)
        assert gizmos.pick(vp, cam, c, 800, 600, mid[0], True, False) == ("move", axis)
    r = gizmos.world_size(cam, c, gizmos.RING_PX, 600)
    on_z, _ = gizmos.project(vp, [c + r * np.array([np.cos(0.6), np.sin(0.6), 0])], 800, 600)
    assert gizmos.pick(vp, cam, c, 800, 600, on_z[0], False, True) == ("rotate", 2)
    assert gizmos.pick(vp, cam, c, 800, 600, (5, 5), True, True) is None
    # dragging: where the mouse ray passes the axis, and the angle around a ring
    assert gizmos.axis_parameter([0, -10, 3], [0, 1, 0], [0, 0, 0], [0, 0, 1]) == pytest.approx(3)
    assert gizmos.axis_parameter([0, 0, 50], [0, 0, -1], [0, 0, 0], [0, 0, 1]) is None
    assert gizmos.plane_angle([1, 1, 5], [0, 0, -1], [0, 0, 0], 2) == pytest.approx(np.pi / 4)
    assert gizmos.unwrap(np.radians(350)) == pytest.approx(np.radians(-10))


def test_view_tools_in_the_3d_view():
    """Drag a move arrow and a rotate ring, and lay the model on a face,
    with real mouse events on the 3D view."""
    import os
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X server (run under xvfb-run)")
    code = '''
import numpy as np, trimesh
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
app = QApplication([])
from openvat.core.scene import Scene
from openvat.core.mesh import MeshObject
from openvat.ui.viewport3d import Viewport3D
from openvat.ui import gizmos
from tests.test_core import _bracket
scene = Scene()
obj = MeshObject("bracket", _bracket()); obj.drop_to_plate(); scene.add(obj)
v = Viewport3D(scene); v.resize(800, 600); v.show()
v.camera.set_view("iso"); v.camera.fit(scene.bounds(), (100, 100)); v.camera.distance *= 1.6
v.set_tools(move=True, rotate=True)
for _ in range(10): app.processEvents(); QTest.qWait(10)
def at(p):
    xy, _ = gizmos.project(v._view_projection(), [p], v.width(), v.height())
    return QPoint(int(round(xy[0][0])), int(round(xy[0][1])))
def drag(a, b):
    QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, a)
    for k in range(1, 11):
        QTest.mouseMove(v, QPoint(a.x() + (b.x() - a.x()) * k // 10, a.y() + (b.y() - a.y()) * k // 10))
    QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, b)
c = obj.position.copy()
L = gizmos.world_size(v.camera, c, gizmos.ARROW_PX, v.height())
drag(at(c + [L * 0.6, 0, 0]), at(c + [L * 0.6 + 12, 0, 0]))
moved = obj.position - c
R = gizmos.world_size(v.camera, obj.position, gizmos.RING_PX, v.height())
ring = gizmos.ring_points(obj.position, 2, R)
drag(at(ring[6]), at(ring[15]))                       # 30 -> 75 degrees around the Z ring
turned = obj.rotation.copy()
v.set_tools(move=False, rotate=False, lay_face=True)
laid = []
v.lay_face_requested.connect(lambda o, n: laid.append(n))
i, n = next((i, n) for i, n in v.lay_candidates(obj) if n[2] > 0.99)   # the top of the tall leg
f = v._flat_faces_of(obj)[i]
QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, at(obj.transformed().triangles[f.faces].mean(axis=(0, 1))))
print(round(moved[0]), round(abs(moved[1]) + abs(moved[2]), 6), round(turned[2]), len(laid), np.round(laid[0], 6).tolist() if laid else None)
'''
    r = _run_python(code)
    assert r.returncode == 0, r.stderr
    x, off_axis, turned, n_laid, normal = r.stdout.strip().split(" ", 4)
    assert abs(int(x) - 12) <= 1 and float(off_axis) == 0.0      # along X only
    assert abs(int(turned) - 45) <= 2                             # about Z, as far as the mouse went
    assert n_laid == "1" and normal == "[0.0, 0.0, 1.0]"


def test_lay_flat_automatically_picks_the_best_face():
    from openvat.core.orient import best_face, laid_on
    def best_after(mesh, rotation):
        obj = MeshObject("m", mesh)
        obj.rotation[:] = rotation
        obj.drop_to_plate()
        choice = best_face(obj)
        obj.rotation[:] = laid_on(obj.rotation, choice.normal)
        obj.drop_to_plate()
        return obj, choice
    # the bracket stood on its tall leg goes back onto its long base: most on the plate, nothing to support
    obj, c = best_after(_bracket(), [0, -90, 0])
    assert (round(c.contact), round(c.overhang), round(c.height)) == (360, 0, 24)
    assert obj.size() == pytest.approx([30, 12, 24])
    # a mushroom prints upside down, on its cap: then nothing overhangs
    mushroom = trimesh.boolean.union([trimesh.creation.box(extents=[4, 4, 10]).apply_translation([0, 0, 5]),
                                      trimesh.creation.box(extents=[20, 20, 3]).apply_translation([0, 0, 11.5])])
    obj, c = best_after(mushroom, [0, 0, 0])
    assert c.overhang == pytest.approx(0) and c.contact == pytest.approx(400)
    world = obj.transformed()
    assert world.area_faces[(world.face_normals[:, 2] < -0.99) & (world.triangles_center[:, 2] < 0.01)].sum() \
        == pytest.approx(400)
    # a slab on its edge lies down flat (all faces are free of overhangs: the lowest print wins)
    obj, c = best_after(trimesh.creation.box(extents=[40, 20, 4]), [90, 0, 0])
    assert obj.size()[2] == pytest.approx(4)
    # a cube stays as it is; a ball has nothing to lie on
    cube = MeshObject("cube", trimesh.creation.box(extents=[20, 20, 20]))
    assert best_face(cube).normal == pytest.approx([0, 0, -1])
    assert best_face(MeshObject("ball", trimesh.creation.icosphere(3, 10))) is None


def test_clone_many_with_supports_and_arrange():
    scene = Scene()
    slab = MeshObject("slab (3)", trimesh.creation.box(extents=[30, 20, 3]))
    slab.drop_to_plate()
    slab.position[2] += 20                                        # high enough for pillars to split
    scene.add(slab)
    scene.auto_supports(slab)
    n = len(scene.supports)
    assert n > 3 and any(s.trunk for s in scene.supports)          # some share a pillar
    before = [s.state() for s in scene.supports]
    made = scene.clone_many(slab, 5, arrange_on=PrinterProfile())
    assert [o.name for o in scene.objects] == ["slab (3)", "slab", "slab (2)", "slab (4)", "slab (5)", "slab (6)"]
    assert all(o.mesh is slab.mesh for o in made)                  # clones share the mesh
    assert scene.selected is made[-1]
    assert all(scene.fits(PrinterProfile()).values())
    for o in scene.objects:                                       # each has its own copy of the supports...
        mine = [s for s in scene.supports if s.owner_id == o.id]
        assert len(mine) == n
        lo, hi = o.world_bounds()                                 # ... under it, wherever it went
        assert all(lo[0] <= s.tip[0] <= hi[0] and lo[1] <= s.tip[1] <= hi[1] for s in mine)
        assert all(lo[0] - 6 <= s.trunk[0] <= hi[0] + 6 and lo[1] - 6 <= s.trunk[1] <= hi[1] + 6
                   for s in mine if s.trunk)                     # and so are their shared pillars
    assert len({s.id for s in scene.supports}) == len(scene.supports)
    # moving one copy leaves the others' shared pillars alone
    other = [s.trunk for s in scene.supports if s.owner_id == made[1].id]
    scene.move_object(made[0], made[0].position + [3.0, 0.0, 0.0])
    assert [s.trunk for s in scene.supports if s.owner_id == made[1].id] == other
    # one undo takes back all of it
    assert scene.undo_stack.undo()
    assert scene.objects == [slab] and len(scene.supports) == n
    assert all(np.allclose(s.state()["tip"], b["tip"]) for s, b in zip(scene.supports, before))
    # without arranging they line up beside the model, off it
    made = scene.clone_many(slab, 3)
    boxes = [o.world_bounds() for o in [slab, *made]]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[1][0] <= b[0][0] or b[1][0] <= a[0][0] or a[1][1] <= b[0][1] or b[1][1] <= a[0][1]


def test_tool_tabs_stay_on_top():
    """The tools column in two tabs; the tabs stay put while its cards scroll."""
    code = '''
from PySide6.QtWidgets import QApplication, QLabel
app = QApplication([])
from openvat.ui.collapsible import SectionColumn, CARD_GAP
col = SectionColumn(240)
for k in "abcd":
    lab = QLabel("x"); lab.setFixedHeight(150)
    col.add(k, k.upper(), lab)
col.set_pages([("One", ["a", "b", "c"]), ("Two", ["d"])])
col.finish(); col.resize(240, 300); col.show(); app.processEvents()
shown = lambda: "".join(k for k, c in col.sections.items() if not c.isHidden())
top = col._tabs.geometry().bottom() + 1 + CARD_GAP
first = col.sections["a"].geometry().y()
col.scroll_to(10_000); app.processEvents()
print(shown(), first == top, col.offset > 0, col._tabs.geometry().y(), col.sections["a"].geometry().y() < top,
      not col.mask().contains(col._tabs.geometry().bottomLeft() + __import__("PySide6.QtCore").QtCore.QPoint(20, 3)))
scrolled = col.offset
col.show_page(1); app.processEvents()
print(shown(), col.offset, col.current_page())
col.show_page(0); app.processEvents()
print(shown(), col.offset == scrolled)
'''
    r = _run_python(code, {"QT_QPA_PLATFORM": "offscreen"})
    assert r.returncode == 0, r.stderr
    lines = r.stdout.split("\n")
    assert lines[0] == "abc True True 0 True True"      # cards under the tabs are clipped, the tabs stay at 0
    assert lines[1] == "d 0 1"
    assert lines[2] == "abc True"                      # each tab keeps its scroll position


def test_support_meshes_rebuild_only_what_changed():
    """A new selection rebuilds nothing; removing or moving one support
    rebuilds just that, and the result is what a fresh build gives."""
    scene = Scene()
    slab = MeshObject("slab", trimesh.creation.box(extents=[30, 20, 3]))
    slab.drop_to_plate()
    slab.position[2] += 20
    scene.add(slab)
    scene.auto_supports(slab)
    def build(sc):
        m = trimesh.util.concatenate(sc.support_meshes())
        return np.round(m.vertices, 9), m.faces
    def fresh(sc):
        other = Scene()
        other.objects = sc.objects
        other.support_settings = sc.support_settings
        other.supports = [s.copy_for(s.owner_id) for s in sc.supports]
        return build(other)
    build(scene)
    version = scene.extras_version
    scene.selected_support = scene.supports[3]
    scene.changed()
    build(scene)
    assert scene.extras_version == version                       # selecting is free
    kept = {s.id: scene.support_mesh(s) for s in scene.supports}
    scene.remove_support(scene.supports[2])
    v, f = build(scene)
    assert scene.extras_version == version + 1
    assert all(scene.support_mesh(s) is kept[s.id] for s in scene.supports)   # the rest reused
    fv, ff = fresh(scene)
    assert np.array_equal(v, fv) and np.array_equal(f, ff)
    scene.move_object(slab, slab.position + [4.0, 0.0, 0.0])
    scene.changed()
    v, f = build(scene)
    fv, ff = fresh(scene)
    assert np.array_equal(v, fv) and np.array_equal(f, ff)
    scene.support_settings.pillar_diameter = 2.0                 # new settings: everything again
    scene.changed()
    v, f = build(scene)
    fv, ff = fresh(scene)
    assert np.array_equal(v, fv) and np.array_equal(f, ff)


def test_slicing_in_threads_gives_the_same_layers():
    import openvat.core.slicer as slicer
    scene = Scene()
    slab = MeshObject("slab", trimesh.creation.box(extents=[24, 16, 3]))
    slab.drop_to_plate()
    slab.position[2] += 6
    scene.add(slab)
    scene.auto_supports(slab)
    resin = ResinProfile()
    resin.normal.thickness = resin.bottom.thickness = 0.1
    saved = slicer.SLICE_THREADS
    try:
        slicer.SLICE_THREADS = 1
        one = slice_scene(scene, PrinterProfile(), resin)
        slicer.SLICE_THREADS = 4
        many = slice_scene(scene, PrinterProfile(), resin)
    finally:
        slicer.SLICE_THREADS = saved
    assert len(one.layers) == len(many.layers) > 50
    assert all(a.geometry.equals_exact(b.geometry, 0) for a, b in zip(one.layers, many.layers))
    assert [(i.first, i.last) for i in one.islands] == [(i.first, i.last) for i in many.islands]


def test_voxel_raster_per_polygon_matches_one_big_image():
    """Each polygon drawn in its own small window gives the same voxels as
    the whole layer drawn on one image (a ring with an island in its hole)."""
    from PIL import Image, ImageDraw
    from shapely.geometry import MultiPolygon, Point
    from openvat.core import voxels
    from openvat.core.slicer import Layer
    ring = Point(3.1, 2.2).buffer(2.0, quad_segs=24).difference(Point(3.1, 2.2).buffer(1.2, quad_segs=24))
    island = Point(3.1, 2.2).buffer(0.55, quad_segs=24)
    far = Point(9.3, 4.4).buffer(0.8, quad_segs=24)
    geom = MultiPolygon([ring, island, far])
    layer = Layer(0, 0.0, 0.05, 1, 1, 0, 0, 1, 1, 1, geom)
    vx, vy, origin, nx, ny = 0.05, 0.035, np.array([0.0, 0.0]), 220, 180
    got = voxels._raster(layer, origin, vx, vy, nx, ny)
    s = voxels.SUPERSAMPLE                                   # the reference: one image of the whole grid
    img = Image.new("1", (nx * s, ny * s), 0)
    draw = ImageDraw.Draw(img)
    shrunk = geom.buffer(-min(vx, vy) / s / 2, join_style="mitre")
    for poly in sorted(shrunk.geoms, key=lambda p: -p.area):
        draw.polygon(voxels._to_px(poly.exterior.coords, origin, vx / s, vy / s), fill=1)
        for hole in poly.interiors:
            draw.polygon(voxels._to_px(hole.coords, origin, vx / s, vy / s), fill=0)
    want = np.asarray(img, dtype=np.uint8).reshape(ny, s, nx, s).sum(axis=(1, 3)) >= voxels.MIN_COVERAGE
    assert got.sum() > 1000 and np.array_equal(got, want)


def test_picking_supports_and_turned_models():
    """Picking works on the scene's kept support meshes and on models in
    their own coordinates: the same hits as on a transformed copy."""
    code = '''
import numpy as np, trimesh
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication
app = QApplication([])
from openvat.core.scene import Scene
from openvat.core.mesh import MeshObject
from openvat.core.raycast import ray_mesh_closest
from openvat.ui.viewport3d import Viewport3D
from openvat.ui import gizmos
scene = Scene()
obj = MeshObject("slab", trimesh.creation.box(extents=[30, 20, 3]))
obj.rotation[:] = [0, 0, 30]; obj.scale[:] = [1.5, 0.8, 1.0]
obj.drop_to_plate(); obj.position[2] += 15
scene.add(obj); scene.auto_supports(obj)
v = Viewport3D(scene); v.resize(800, 600)
v.camera.aspect = 800 / 600
v.camera.set_view("iso"); v.camera.fit(scene.bounds(), (100, 100))
vp = v._view_projection()
def px(p):
    xy, _ = gizmos.project(vp, [p], 800, 600)
    return QPoint(int(round(xy[0][0])), int(round(xy[0][1])))
built = [(s, s.to_mesh(scene.support_settings)) for s in scene.supports]
world = obj.transformed()
def reference(pos):                                  # the old way: every mesh tested in world space
    o, d = v.camera.ray(pos.x(), pos.y(), 800, 600)
    best, best_t = None, np.inf
    for s, m in built:
        h = ray_mesh_closest(m, o, d)
        if h is not None and h[0] < best_t:
            best, best_t = s, h[0]
    h = ray_mesh_closest(world, o, d)
    if h is not None and h[0] < best_t:
        best, best_t = obj, h[0]
    return best, (o + d * best_t if best is not None else None)
v.camera.set_view("front"); v.camera.fit(scene.bounds(), (100, 100)); vp = v._view_projection()
same = supports = models = 0
for x in range(250, 560, 14):
    for y in range(150, 460, 14):
        got, ref = v._pick(QPoint(x, y)), reference(QPoint(x, y))
        same += got[0] is ref[0] and (got[1] is None or bool(np.allclose(got[1], ref[1], atol=1e-6)))
        supports += isinstance(got[0], type(scene.supports[0]))
        models += got[0] is obj
print(same == 23 * 23, supports > 10, models > 10, v._pick(QPoint(2, 2))[0] is None)
'''
    r = _run_python(code, {"QT_QPA_PLATFORM": "offscreen"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["True", "True", "True", "True"]


# --------------------------------------------------------------------------
# release force, support paint, section view, USB drives

def _wedge(lean: float = 30.0) -> MeshObject:
    """A 10 mm block on the plate whose +X side leans out ``lean`` degrees
    from vertical (too little to need supports at the default 45), in
    small triangles so paint can pick parts of it."""
    import math
    m = trimesh.creation.box(extents=[10, 10, 10])
    m.apply_translation([0, 0, 5])
    m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=1.5))
    v = m.vertices.copy()
    v[:, 0] += v[:, 2] * math.tan(math.radians(lean))
    obj = MeshObject("wedge", trimesh.Trimesh(v, m.faces))
    obj.drop_to_plate()
    return obj


def test_release_force_adds_supports_until_every_layer_holds():
    """Every layer pulls on the film with release_force x its area; the
    tips below it must hold that.  A stronger pull gets more supports."""
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports, release_check, model_space
    obj = MeshObject("slab", trimesh.creation.box(extents=[40, 30, 4]))
    obj.elevation = 5.0
    obj.drop_to_plate()
    st = SupportSettings(release_force=4.0)
    plain = generate_supports(obj, SupportSettings(release_force=0.0))
    before = release_check(model_space(obj, st).mesh, plain, st)
    assert not before.ok and before.area == pytest.approx(1200, rel=0.01)
    assert before.demand == pytest.approx(1200 * 4.0 / 100, rel=0.01)       # N/cm^2 x cm^2
    assert before.capacity == pytest.approx(sum(st.tip_force(s.tip[2]) for s in plain), rel=1e-6)
    report = {}
    held = generate_supports(obj, st, report)
    assert report["release_added"] > 0 and len(held) > len(plain)
    after = report["release"][0]
    assert after.ok and after.capacity >= after.demand and "holds" not in after.text()
    on_plate = release_check(model_space(_wedge(), st).mesh, [], st)       # standing on the plate
    assert on_plate.ok and on_plate.on_plate


def test_support_paint_brush_and_overhang_selection():
    from openvat.core.paint import (ENFORCE, BLOCK, NONE, brushed, overhang_faces, painted_with,
                                    paint_of, world_normals)
    obj = _wedge()
    n = len(obj.mesh.faces)
    assert paint_of(obj) is None
    centers = obj.transformed().triangles_center
    lean = world_normals(obj)[:, 0] > 0.4
    point = centers[lean].mean(axis=0)                                       # middle of the leaning side
    p = brushed(obj, point, 2.0, ENFORCE)
    hit = np.flatnonzero(p == ENFORCE)
    assert len(hit) > 3 and np.linalg.norm(centers[hit] - point, axis=1).max() <= 2.0 + 1e-9
    obj.paint = p
    assert brushed(obj, point, 2.0, ENFORCE) is None                         # nothing new: no change
    assert obj.paint is p and (brushed(obj, point, 2.0, NONE) == NONE).all() # erasing makes a new array
    # overhangs: the leaning side (30 deg from vertical) but never the bottom on the plate
    assert np.array_equal(overhang_faces(obj, 20), lean)
    assert not overhang_faces(obj, 40).any()
    obj.rotation[:] = [0, 180, 0]; obj.drop_to_plate()                        # upside down: the top faces down
    assert overhang_faces(obj, 40).sum() == 0                                  # ... but lies on the plate
    obj.elevation = 3.0; obj.drop_to_plate()
    assert overhang_faces(obj, 40).sum() == (np.asarray(obj.mesh.face_normals)[:, 2] > 0.99).sum()
    q = painted_with(obj, [0, 1, 2], BLOCK)
    assert q is not obj.paint and list(q[:3]) == [BLOCK] * 3 and len(q) == n


def test_painted_supports_go_where_painted_and_not_where_blocked():
    from openvat.core.paint import ENFORCE, BLOCK, painted_with, world_normals
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports
    st = SupportSettings()
    obj = _wedge()
    assert generate_supports(obj, st) == []                                  # 30 deg: no supports needed
    lean = world_normals(obj)[:, 0] > 0.4
    obj.paint = painted_with(obj, lean, ENFORCE)
    report = {}
    sups = generate_supports(obj, st, report)
    assert len(sups) >= 2 and report["straight"] == len(sups)
    centers = obj.transformed().triangles_center
    for s in sups:                                                           # on the painted side
        assert lean[np.argmin(np.linalg.norm(centers - s.tip, axis=1))]
    # blocking: a lifted slab with its +X half painted "no supports"
    slab = MeshObject("slab", trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(
        *(lambda b: (b.vertices, b.faces))(trimesh.creation.box(extents=[20, 14, 3])), max_edge=1.5)))
    slab.elevation = 4.0
    slab.drop_to_plate()
    m = slab.matrix()
    cx = (slab.mesh.triangles_center @ m[:3, :3].T + m[:3, 3])[:, 0]
    no_pull = SupportSettings(release_force=0.0)          # (else the open half gets more for the pull)
    full = generate_supports(slab, no_pull)
    slab.paint = painted_with(slab, cx > 0, BLOCK)
    report = {}
    half = generate_supports(slab, no_pull, report)
    assert report["blocked"] > 0 and 0 < len(half) < len(full)
    assert all(s.tip[0] <= 0.75 for s in half)
    report = {}                                           # with the pull: more on the open half, none blocked
    pulled = generate_supports(slab, st, report)
    assert len(pulled) > len(half) and all(s.tip[0] <= 0.75 for s in pulled) and report["release"][0].ok


def test_paint_section_and_usb_button_in_the_ui():
    """A paint stroke with the mouse (and undo), the section view's cut and
    picking through it, its controls, and the Export to USB button."""
    import os
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X server (run under xvfb-run)")
    code = '''
import numpy as np, trimesh
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
app = QApplication([])
from openvat.core.scene import Scene
from openvat.core.mesh import MeshObject
from openvat.core.paint import ENFORCE
from openvat.core.usb import Drive
from openvat.ui.viewport3d import Viewport3D
from openvat.ui.side_panel import SectionControls
from openvat.ui.usb_export import UsbButton, UsbWatcher
scene = Scene()
box = trimesh.creation.box(extents=[20, 20, 20])
obj = MeshObject("box", trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(box.vertices, box.faces, max_edge=2.0)))
obj.drop_to_plate(); scene.add(obj)
v = Viewport3D(scene); v.resize(600, 600); v.show()
v.camera.set_view("front"); v.camera.fit(scene.bounds(), (100, 100))
for _ in range(10): app.processEvents(); QTest.qWait(10)
out = []
v.set_paint(on=True, value=ENFORCE, radius=2.5)
c = QPoint(300, 300)
QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, c)
for k in range(1, 8):
    QTest.mouseMove(v, QPoint(300 + 8 * k, 300))
QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, QPoint(356, 300))
painted = np.flatnonzero(obj.paint == ENFORCE) if obj.paint is not None else []
front = np.asarray(obj.mesh.face_normals)[painted][:, 1] < -0.99 if len(painted) else [False]
out.append(len(painted) > 10 and bool(np.all(front)) and scene.selected is obj)
out.append(scene.undo_stack.undo() and obj.paint is None)
v.set_paint(on=False)
# the section view: cut at y = 0 showing the far half; a click passes the cut-away front
v.set_section(1, 0.0, True)
item, pt = v._pick(c)
shows = v.section_shows(np.array([0, -5.0, 10])), v.section_shows(np.array([0, 5.0, 10]))
out.append(item is obj and abs(pt[1] - 10) < 1e-6 and shows == (False, True))
v.set_section(None)
item, pt = v._pick(c)
out.append(item is obj and abs(pt[1] + 10) < 1e-6)
sec = SectionControls(scene)
got = []
sec.changed.connect(got.append)
sec.enable.setChecked(True)
sec.flip.setChecked(True)
sec.axes.button(0).setChecked(True)
sec.enable.setChecked(False)
out.append([g if g is None else (g[0], round(g[1], 1), g[2]) for g in got])
btn = UsbButton(); picked = []
btn.export_to.connect(picked.append)
one = [Drive("STICK", "/media/u/STICK", "/dev/sdb1")]
btn.set_drives(one); btn.click()
out.append((btn.isHidden(), btn.text(), picked == one))
btn.set_drives(one + [Drive("CARD", "", "/dev/mmcblk0p1")])
out.append((btn.text(), len(btn.menu().actions())))
btn.set_drives([])
out.append(btn.isHidden())
seen = []
w = UsbWatcher(probe=lambda: list(one)); w.changed.connect(seen.append)
w.refresh(); w.refresh()
out.append(len(seen))
print(repr(out))
'''
    r = _run_python(code)
    assert r.returncode == 0, r.stderr
    out = eval(r.stdout.strip().splitlines()[-1])
    assert out[:4] == [True, True, True, True]
    assert out[4] == [(2, 10.0, False), (2, 10.0, True), (0, 0.0, True), None]
    assert out[5] == (False, "Export to USB (STICK)", True)
    assert out[6] == ("Export to USB", 2)
    assert out[7] is True and out[8] == 1


def test_usb_drives_from_mountinfo_and_sysfs(tmp_path):
    """USB and SD card partitions are found, mounted or not (KDE leaves new
    sticks unmounted); internal disks, swap and read-only mounts are not."""
    import os
    from openvat.core import usb
    sysd = tmp_path / "sys" / "devices"
    blk = tmp_path / "sys" / "class" / "block"
    blk.mkdir(parents=True)
    dev = tmp_path / "dev"
    (dev / "disk" / "by-uuid").mkdir(parents=True)
    (dev / "disk" / "by-label").mkdir(parents=True)

    def disk(path, name, removable, parts):
        d = sysd / path / "block" / name
        d.mkdir(parents=True)
        (d / "removable").write_text(f"{removable}\n")
        os.symlink(d, blk / name)
        for p in parts:
            (d / p).mkdir()
            (d / p / "partition").write_text("1\n")
            os.symlink(d / p, blk / p)
            (dev / p).write_text("")
            os.symlink(dev / p, dev / "disk" / "by-uuid" / f"uuid-{p}")

    disk("pci0000:00/ata1/host0", "sda", 0, ["sda1", "sda2"])               # the internal disk
    disk("pci0000:00/usb2/2-1/host6", "sdb", 0, ["sdb1"])                    # USB stick (not flagged)
    disk("pci0000:00/usb2/2-2/host7", "sdc", 1, ["sdc1", "sdc2"])            # another, not mounted
    disk("platform/mmc0", "mmcblk0", 0, ["mmcblk0p1"])                      # an SD card
    os.symlink(dev / "sdc1", dev / "disk" / "by-label" / "MY\\x20STICK")
    mounts = tmp_path / "mountinfo"
    mounts.write_text(
        "22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw\n"
        "23 22 8:1 / /boot/efi rw,relatime shared:2 - vfat /dev/sda1 rw\n"
        "90 22 8:17 / /run/media/tyler/KING\\040STON rw,nosuid,nodev shared:9 - vfat /dev/sdb1 rw,fmask=0022\n"
        "91 22 179:1 / /run/media/tyler/SDCARD ro,nosuid shared:10 - exfat /dev/mmcblk0p1 ro\n"
        "92 22 0:40 / /tmp rw - tmpfs tmpfs rw\n")
    got = usb.linux_drives(str(mounts), str(blk), str(dev / "disk"))
    by_dev = {d.device: d for d in got}
    assert set(by_dev) == {"/dev/sdb1", "/dev/sdc1", "/dev/sdc2"}           # sdc2: a file system, no label
    assert by_dev["/dev/sdb1"].path == "/run/media/tyler/KING STON" and by_dev["/dev/sdb1"].label == "KING STON"
    assert not by_dev["/dev/sdc1"].mounted and by_dev["/dev/sdc1"].label == "MY STICK"
    assert "not mounted" in by_dev["/dev/sdc1"].describe()
    assert usb._parent_disk("/dev/sdc2", str(blk)) == "/dev/sdc"


# --------------------------------------------------------------------------
# faster support generation, lighting, the View card

def test_contact_search_keeps_its_slices():
    """The contact search cuts each layer once: the release check and the
    next run (release-force rounds, Automatic again) reuse the slices and
    find the same contacts."""
    from openvat.core import slicer
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import find_contact_points, release_check
    obj = _mushroom()
    obj.elevation = 4.0
    obj.drop_to_plate()
    mesh = obj.transformed()
    st = SupportSettings()
    cache = {}
    first = find_contact_points(mesh, st, 0.05, slices=cache)
    assert len(cache) == int(np.ceil((mesh.bounds[1, 2] - mesh.bounds[0, 2]) / 0.05))
    cuts = []
    at = slicer.MeshSlicer.at
    slicer.MeshSlicer.at = lambda self, z: cuts.append(z) or at(self, z)
    try:
        again = find_contact_points(mesh, st, 0.05, slices=cache)
        fresh = find_contact_points(mesh, st, 0.05)
        cached = release_check(mesh, [], st, slices=cache)
        n_cut = len(cuts)
        plain = release_check(mesh, [], st)
    finally:
        slicer.MeshSlicer.at = at
    assert again == first == fresh and n_cut == len(cache)      # the second run cut nothing
    assert cached.area == pytest.approx(plain.area, rel=0.02) and cached.area == pytest.approx(np.pi * 15 ** 2, rel=0.02)


def test_split_pillars_scales_to_many_supports():
    """Grouping pillars onto shared trunks looks up neighbours in a tree,
    so thousands of supports group in moments, within the branch limits."""
    import time
    from openvat.core.supports import SupportSettings
    from openvat.core.placement import generate_supports
    obj = MeshObject("plaque", trimesh.creation.box(extents=[120, 80, 3]))
    obj.elevation = 12.0
    obj.drop_to_plate()
    st = SupportSettings(release_force=0.0, density="high")
    t = time.time()
    sups = generate_supports(obj, st)
    took = time.time() - t
    assert len(sups) > 1000 and took < 60
    trunks: dict = {}
    for s in sups:
        if s.trunk is not None:
            trunks.setdefault(s.trunk, []).append(s)
    assert trunks and max(len(g) for g in trunks.values()) <= st.branch_max_count
    for trunk, group in trunks.items():
        spread = max(np.linalg.norm(a.pivot(st)[:2] - b.pivot(st)[:2]) for a in group for b in group)
        assert spread <= 2 * st.branch_max_distance + 1e-6


def test_lighting_settings():
    from openvat.ui.lighting import Lighting
    old_key, old_fill = np.array([0.4, -0.6, 1.0]), np.array([-0.6, 0.5, 0.3])
    key, fill = Lighting().directions()
    assert np.degrees(np.arccos(key @ old_key / np.linalg.norm(old_key))) < 1.0    # the lights as they were
    assert np.degrees(np.arccos(fill @ old_fill / np.linalg.norm(old_fill))) < 1.0
    light = Lighting.from_dict({"ambient": "0.5", "key": 9, "gloss": -3, "key_azimuth": 450, "junk": 1,
                                "follow_view": 1, "fill": "bad"})
    assert (light.ambient, light.key, light.gloss, light.key_azimuth, light.fill) == (0.5, 2.0, 1.0, 90.0, 0.25)
    assert light.follow_view is True and Lighting.from_dict(light.to_dict()) == light
    assert Lighting.from_dict(None) == Lighting()
    turned, _ = Lighting(key_azimuth=90.0, key_elevation=0.0, follow_view=True).directions(camera_yaw=90.0)
    assert np.allclose(turned, [-1, 0, 0], atol=1e-6)                 # follows the view: turns with it
    fixed, _ = Lighting(key_azimuth=90.0, key_elevation=0.0).directions(camera_yaw=90.0)
    assert np.allclose(fixed, [0, 1, 0], atol=1e-6)


def test_view_card_lighting_and_no_double_click_fit(tmp_path):
    """The View card's check boxes and the View menu stay in step; Edit ->
    Preferences -> Lighting previews in the 3D view, Cancel puts it back, OK
    keeps it for the next start; a double-click no longer fits the view."""
    import os
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X server (run under xvfb-run)")
    code = f'''
import time
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
app = QApplication([])
from openvat.core.profiles import ProfileStore
from openvat.ui.main_window import MainWindow
from openvat.ui.app_settings_dialog import PreferencesDialog
st = ProfileStore({str(tmp_path)!r}); st.printers.append(st.printer_presets[0])
w = MainWindow(st); w.resize(1100, 700); w.show()
for _ in range(10): app.processEvents(); time.sleep(0.01)
out = []
view = w.side.view
view.see_through.setChecked(True); w.plate_action.setChecked(False)
out.append((w.see_through_action.isChecked(), w.viewport.see_through, view.plate.isChecked(), w.viewport.show_plate))
out.append(w.side.sections["view"].summary.text())
d = w.viewport.camera.distance = 777.0
QTest.mouseDClick(w.viewport, Qt.LeftButton, Qt.NoModifier, QPoint(15, 15))
out.append(w.viewport.camera.distance == d)
def answer(ok):
    dlg = app.activeModalWidget()
    if not isinstance(dlg, PreferencesDialog):
        QTimer.singleShot(50, lambda: answer(ok)); return
    dlg.light_rows["ambient"].setValue(80)
    dlg.light_rows["key_azimuth"].setValue(10)
    out.append((round(w.viewport.lighting.ambient, 2), round(w.voxel_page.viewport.lighting.key_azimuth)))
    dlg.accept() if ok else dlg.reject()
QTimer.singleShot(50, lambda: answer(False)); w.open_preferences()
out.append(round(w.viewport.lighting.ambient, 2))
QTimer.singleShot(50, lambda: answer(True)); w.open_preferences()
again = MainWindow(st)
out.append((round(again.viewport.lighting.ambient, 2), round(again.viewport.lighting.key_azimuth)))
print(repr(out))
'''
    r = _run_python(code)
    assert r.returncode == 0, r.stderr
    out = eval(r.stdout.strip().splitlines()[-1])
    assert out[0] == (True, True, False, False)
    assert out[1] == "no plate · see-through"
    assert out[2] is True
    assert out[3] == (0.8, 10) and out[4] == 0.3 and out[5] == (0.8, 10) and out[6] == (0.8, 10)
