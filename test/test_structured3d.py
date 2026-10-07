from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from scenezoo.dataset.base import DataFormatError, UnsupportedOperationError
from scenezoo.dataset.scene import Structured3D
from scenezoo.dataset.scene._structured3d_store import Structured3DStore


def _save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_rendering(base, *, panorama=False):
    height, width = 4, 6
    rgba = np.full((height, width, 4), (10, 20, 30, 255), np.uint8)
    depth = np.full((height, width), 2000, np.uint16)
    semantic = np.full((height, width), 5, np.uint8)
    instance = np.full((height, width), 5, np.uint16)
    instance[0, 0] = 65535
    albedo = np.full((height, width, 4), (40, 50, 60, 255), np.uint8)
    normal = np.full((height, width, 3), (128, 128, 255), np.uint8)
    for name, value in (
        ("depth.png", depth),
        ("semantic.png", semantic),
        ("instance.png", instance),
        ("albedo.png", albedo),
        ("normal.png", normal),
    ):
        _save(base / name, value)
    if panorama:
        for lighting in ("raw", "cold", "warm"):
            _save(base / f"rgb_{lighting}light.png", rgba)
    else:
        _save(base / "rgb_rawlight.png", rgba)
        (base / "camera_pose.txt").write_text(
            "0 0 1000 0 1 0 0 0 1 0.7853981634 0.7853981634 1"
        )
        (base / "layout.json").write_text(json.dumps({"junctions": [], "planes": []}))


def _write_scene(root):
    scene = root / "data/scene_00000"
    room = scene / "2D_rendering/10"
    junctions = [
        [0, 0, 0],
        [2000, 0, 0],
        [2000, 3000, 0],
        [0, 3000, 0],
        [0, 0, 2500],
        [2000, 0, 2500],
        [2000, 3000, 2500],
        [0, 3000, 2500],
    ]
    pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    line_junction = np.zeros((12, 8), dtype=int)
    for line_id, pair in enumerate(pairs):
        line_junction[line_id, list(pair)] = 1
    plane_line = np.zeros((6, 12), dtype=int)
    plane_line[0, :4] = 1
    plane_line[1, 4:8] = 1
    for plane_id, line_ids in enumerate(
        ((0, 9, 4, 8), (1, 10, 5, 9), (2, 11, 6, 10), (3, 8, 7, 11)),
        start=2,
    ):
        plane_line[plane_id, list(line_ids)] = 1
    annotation = {
        "junctions": [
            {"ID": index, "coordinate": value} for index, value in enumerate(junctions)
        ],
        "lines": [
            {"ID": index, "point": [0, 0, 0], "direction": [1, 0, 0]}
            for index in range(12)
        ],
        "planes": [
            {"ID": 0, "type": "floor", "normal": [0, 0, 1], "offset": 0},
            {"ID": 1, "type": "ceiling", "normal": [0, 0, -1], "offset": 2500},
            *[
                {"ID": index, "type": "wall", "normal": [1, 0, 0], "offset": 0}
                for index in range(2, 6)
            ],
        ],
        "semantics": [{"ID": 10, "type": "office", "planeID": list(range(6))}],
        "planeLineMatrix": plane_line.tolist(),
        "lineJunctionMatrix": line_junction.tolist(),
        "cuboids": [],
        "manhattan": [],
    }
    scene.mkdir(parents=True)
    (scene / "annotation_3d.json").write_text(json.dumps(annotation))
    (scene / "bbox_3d.json").write_text(
        json.dumps(
            [
                {
                    "ID": 5,
                    "basis": np.eye(3).tolist(),
                    "coeffs": [100, 200, 300],
                    "centroid": [1000, 1500, 1000],
                }
            ]
        )
    )
    for position in (0, 2):
        _write_rendering(room / f"perspective/full/{position}")
    _write_rendering(room / "panorama/full", panorama=True)
    (room / "panorama/camera_xyz.txt").write_text("1000 1500 1000")
    (room / "panorama/layout.txt").write_text("0 0\n0 3\n2 0\n2 3\n")
    return scene


def test_splits_metadata_rooms_annotations_and_errata(tmp_path):
    _write_scene(tmp_path)
    dataset = Structured3D(tmp_path, offline=True)
    assert dataset.get_ids("train")[0] == "scene_00000"
    # Official ranges by default; the errata's invalid scenes can be dropped.
    assert len(dataset.get_ids("train")) == 3000
    assert len(dataset.get_ids("test")) == 250
    filtered = Structured3D(tmp_path, offline=True, include_invalid=False)
    assert len(filtered.get_ids("train")) == 2995
    assert len(filtered.get_ids("test")) == 249
    assert dataset.metadata["label_names"][5] == "chair"
    assert "scene_01155" in dataset.metadata["errata"]["invalid_scenes"]
    assert dataset.get_room_ids("scene_00000") == ["10"]
    assert dataset.get_frame_sources("scene_00000", "10") == (
        "perspective",
        "panorama",
    )
    assert len(dataset.get_structure_annotations("scene_00000")["planes"]) == 6
    assert dataset.get_layout("scene_00000", "10").shape == (4, 2)
    assert dataset.get_layout("scene_00000", "10", source="perspective", index=0) == {
        "junctions": [],
        "planes": [],
    }
    with pytest.raises(ValueError, match="scene_NNNNN"):
        dataset.get_room_ids("00000")
    with pytest.raises(DataFormatError, match="invalid scene"):
        filtered.get_room_ids("scene_01155")


def test_perspective_frames_preserve_ids_camera_and_modalities(tmp_path):
    _write_scene(tmp_path)
    dataset = Structured3D(tmp_path, offline=True)
    batch = dataset.get_frames(
        "scene_00000",
        room_id="10",
        source="perspective",
        indices=[2, 0],
        output_size=(3, 2),
        items=(
            "rgb",
            "depth",
            "semantic_maps",
            "instance_maps",
            "albedo",
            "normal_maps",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
    )
    assert batch.indices.tolist() == [2, 0]
    assert batch.source_frame_count == 3
    assert batch.frame_keys == (
        "10/perspective/full/2",
        "10/perspective/full/0",
    )
    assert batch.rgb.shape == batch.albedo.shape == (2, 2, 3, 3)
    assert batch.depth.shape == batch.semantic_maps.shape == (2, 2, 3)
    assert batch.depth.dtype == np.float32
    assert np.all(batch.depth == 2)
    assert batch.instance_maps.dtype == np.int32
    assert batch.normal_maps.dtype == np.float32
    assert batch.rgb_intrinsics.shape == batch.depth_intrinsics.shape == (2, 3, 3)
    camera_center = np.array([0, 0, 1, 1], dtype=np.float32)
    assert np.allclose(batch.world_to_camera[0] @ camera_center, (0, 0, 0, 1))

    cropped = dataset.get_frames(
        "scene_00000",
        room_id="10",
        source="perspective",
        indices=[0],
        output_size=(2, 2),
        center_crop=True,
        items=("rgb", "rgb_intrinsics"),
    )
    assert cropped.rgb.shape == (1, 2, 2, 3)
    # The 6x4 image centre (2.5, 1.5) maps to (1.0, 0.5) after the pixel-centred
    # 3x2 resize; the one-column crop removes the right edge only.
    assert np.allclose(cropped.rgb_intrinsics[0, :2, 2], (1.0, 0.5))


def test_panorama_frames_boxes_mesh_and_point_data(tmp_path):
    _write_scene(tmp_path)
    dataset = Structured3D(tmp_path, offline=True)
    batch = dataset.get_frames(
        "scene_00000",
        room_id="10",
        source="panorama",
        lighting="warm",
        items=("rgb", "depth", "instance_maps", "world_to_camera"),
    )
    assert batch.camera_model == "EQUIRECTANGULAR"
    assert batch.depth_mode == "ray_distance"
    assert batch.rgb.shape == (1, 4, 6, 3)
    with pytest.raises(UnsupportedOperationError, match="rgb_intrinsics"):
        dataset.get_frames(
            "scene_00000",
            room_id="10",
            source="panorama",
            items=("rgb_intrinsics",),
        )

    boxes, names = dataset.get_boxes("scene_00000")
    assert names == {5: "chair"}
    assert np.allclose(boxes[5].center, (1, 1.5, 1))
    assert np.allclose(boxes[5].extent, (0.2, 0.4, 0.6))
    mesh = dataset.get_mesh("scene_00000", room_id="10")
    assert len(mesh.vertices) == 8
    assert len(mesh.triangles) == 12
    assert np.allclose(mesh.get_axis_aligned_bounding_box().get_extent(), (2, 3, 2.5))

    points = dataset.get_point_data(
        "scene_00000",
        "10",
        semantic_label_space="pointcept25",
    )
    assert len(points) == 24
    assert points.rgb.shape == points.normals.shape == (24, 3)
    assert set(points.semantic_labels) == {4}
    assert (
        dataset.get_segmentation(
            "scene_00000",
            room_id="10",
            semantic_label_space="pointcept25",
        ).domain
        == "point"
    )


def test_official_zip_shards_require_extraction(tmp_path):
    source_root = tmp_path / "source"
    scene = _write_scene(source_root)
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "Structured3D_0.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in scene.rglob("*"):
            if path.is_file():
                archive.write(
                    path, f"Structured3D/{path.relative_to(source_root / 'data')}"
                )
    dataset = Structured3D(archive_root, offline=True)
    report = dataset.check(sample_ids=("scene_00000",))
    assert not report.ok
    assert any(issue.code == "scene-not-extracted" for issue in report.errors)
    with pytest.raises(FileNotFoundError, match="not fully extracted"):
        dataset.get_room_ids("scene_00000")


def test_store_uses_extracted_scene_only(tmp_path):
    archive_path = tmp_path / "Structured3D_0.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("scene_00000/annotation_3d.json", "{}")
    store = Structured3DStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="not fully extracted"):
        store.read_bytes("scene_00000/annotation_3d.json")

    annotation = tmp_path / "scene_00000" / "annotation_3d.json"
    annotation.parent.mkdir()
    annotation.write_text("{}")
    assert store.read_text("scene_00000/annotation_3d.json") == "{}"
