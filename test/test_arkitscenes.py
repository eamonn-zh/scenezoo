from __future__ import annotations

import csv
import json

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from scenezoo.dataset.base import DataFormatError, UnsupportedOperationError
from scenezoo.dataset.scene import ARKitScenes
import scenezoo.dataset.scene.arkitscenes as arkitscenes_module


SCENE_ID = "40776204"


def _write_png(path, value, *, shape, rgb=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if rgb else np.uint16
    array = (
        np.full((*shape, 3), value, dtype=dtype)
        if rgb
        else np.full(shape, value, dtype=dtype)
    )
    Image.fromarray(array).save(path)


def _make_capture(root):
    raw = root / "raw"
    scene = raw / "Training" / SCENE_ID
    scene.mkdir(parents=True)
    with open(raw / "metadata.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "video_id",
                "visit_id",
                "sky_direction",
                "fold",
                "has_laser_scanner_point_clouds",
                "is_in_upsampling",
                "is_in_threedod",
            )
        )
        writer.writerow((SCENE_ID, 1234, "Up", "Training", "True", "True", "True"))
    (scene / f"{SCENE_ID}.mov").touch()
    (scene / "lowres_wide.traj").write_text("0.000 0 0 0 0 0 0\n1.000 0 0 0 1 0 0\n")

    source_shapes = {
        "lowres_wide": (3, 4),
        "wide": (6, 8),
        "ultrawide": (4, 6),
        "vga_wide": (4, 6),
    }
    for source, shape in source_shapes.items():
        for index, timestamp in enumerate(("0.000", "1.000")):
            _write_png(
                scene / source / f"{SCENE_ID}_{timestamp}.png",
                20 + index,
                shape=shape,
                rgb=True,
            )
            width, height = shape[1], shape[0]
            intrinsic = (
                scene / f"{source}_intrinsics" / f"{SCENE_ID}_{timestamp}.pincam"
            )
            intrinsic.parent.mkdir(parents=True, exist_ok=True)
            intrinsic.write_text(f"{width} {height} 2 2 1.5 1.0 0 0")
    for index, timestamp in enumerate(("0.000", "1.000")):
        _write_png(
            scene / "lowres_depth" / f"{SCENE_ID}_{timestamp}.png",
            (index + 1) * 1000,
            shape=(3, 4),
        )
        _write_png(
            scene / "highres_depth" / f"{SCENE_ID}_{timestamp}.png",
            (index + 3) * 1000,
            shape=(6, 8),
        )
        _write_png(
            scene / "confidence" / f"{SCENE_ID}_{timestamp}.png",
            index + 1,
            shape=(3, 4),
        )
    return scene


def test_metadata_merges_human_rotation_corrections_and_builds_local_splits(tmp_path):
    _make_capture(tmp_path)
    dataset = ARKitScenes(tmp_path, offline=True)

    metadata = dataset.metadata[SCENE_ID]
    assert metadata == {
        "video_id": SCENE_ID,
        "visit_id": 1234,
        "fold": "Training",
        "official_sky_direction": "Up",
        "sky_direction": "Left",
        "has_direction_correction": True,
        "has_laser_scanner_point_clouds": True,
        "is_in_upsampling": True,
        "is_in_threedod": True,
    }
    assert dataset.get_ids("train_raw") == [SCENE_ID]
    assert dataset.get_ids("train_threedod") == [SCENE_ID]
    assert dataset.get_ids("train_depth_upsampling") == [SCENE_ID]


def test_mov_camera_only_does_not_open_video(tmp_path, monkeypatch):
    _make_capture(tmp_path)
    dataset = ARKitScenes(tmp_path, offline=True)
    for name in ("read_video_frames", "read_video_frame_size"):
        monkeypatch.setattr(
            arkitscenes_module,
            name,
            lambda *args, **kwargs: pytest.fail("camera-only access must not probe"),
        )
    native = dataset.get_frames(
        SCENE_ID,
        indices=[1, 0],
        items=("timestamps", "world_to_camera"),
        rotate_to_up=False,
    )
    assert native.indices.tolist() == [1, 0]
    assert native.timestamps.dtype == np.float64
    assert native.timestamps.tolist() == [1.0, 0.0]
    assert native.world_to_camera[:, 0, 3].tolist() == [1.0, 0.0]

    # The capture's sky direction is "Left": upright poses roll the camera by
    # the same 270-degree counter-clockwise rotation applied to its images.
    upright = dataset.get_frames(SCENE_ID, indices=[1, 0], items=("world_to_camera",))
    roll = np.eye(4, dtype=np.float32)
    roll[:2, :2] = [[0, -1], [1, 0]]
    assert np.allclose(upright.world_to_camera, roll @ native.world_to_camera)


def test_all_raw_rgb_sources_and_direction_rotation(tmp_path):
    _make_capture(tmp_path)
    dataset = ARKitScenes(tmp_path, offline=True)
    assert dataset.get_frame_sources(SCENE_ID) == (
        "mov",
        "lowres_wide",
        "wide",
        "ultrawide",
        "vga_wide",
    )
    lowres = dataset.get_frames(
        SCENE_ID,
        source="lowres_wide",
        indices=[0],
        items=("rgb", "rgb_intrinsics"),
    )
    # The packaged correction is Left, which is 270 degrees counter-clockwise.
    assert lowres.rgb.shape == (1, 4, 3, 3)
    assert lowres.rgb_intrinsics.shape == (1, 3, 3)

    for source in ("wide", "ultrawide", "vga_wide"):
        batch = dataset.get_frames(
            SCENE_ID,
            source=source,
            indices=[1],
            items=("rgb", "rgb_intrinsics", "world_to_camera"),
            output_size=(4, 3),
            rotate_to_up=False,
        )
        assert batch.rgb.shape == (1, 3, 4, 3)
        assert batch.rgb_intrinsics.shape == (1, 3, 3)


def test_lowres_and_highres_depth_confidence_are_metric_and_nearest(tmp_path):
    _make_capture(tmp_path)
    dataset = ARKitScenes(tmp_path, offline=True)
    lowres = dataset.get_frames(
        SCENE_ID,
        source="lowres_wide",
        indices=[1, 0],
        items=("depth", "depth_intrinsics", "confidence_maps"),
        output_size=(2, 2),
        rotate_to_up=False,
    )
    assert lowres.depth.dtype == np.float32
    assert lowres.depth[:, 0, 0].tolist() == [2.0, 1.0]
    assert lowres.confidence_maps.dtype == np.uint8
    assert lowres.confidence_maps[:, 0, 0].tolist() == [2, 1]
    assert lowres.depth_intrinsics.shape == (2, 3, 3)

    highres = dataset.get_frames(
        SCENE_ID,
        source="wide",
        indices=[0],
        items=("rgb", "depth", "depth_intrinsics", "confidence_maps"),
        output_size=(4, 3),
        rotate_to_up=False,
    )
    assert highres.rgb.shape == (1, 3, 4, 3)
    assert highres.depth.shape == highres.confidence_maps.shape == (1, 3, 4)
    assert np.all(highres.depth == 3.0)

    paired_lowres = dataset.get_frames(
        SCENE_ID,
        source="wide",
        depth_type="lowres",
        indices=[0],
        items=("depth", "depth_intrinsics"),
        rotate_to_up=False,
    )
    assert paired_lowres.depth.shape == (1, 3, 4)
    assert np.all(paired_lowres.depth == 1.0)

    mov_highres = dataset.get_frames(
        SCENE_ID,
        source="mov",
        depth_type="highres",
        indices=[0],
        items=("depth", "depth_intrinsics"),
        rotate_to_up=False,
    )
    assert mov_highres.depth.shape == (1, 6, 8)
    assert np.all(mov_highres.depth == 3.0)
    assert mov_highres.depth_intrinsics.shape == (1, 3, 3)

    with pytest.raises(UnsupportedOperationError):
        dataset.get_frames(SCENE_ID, source="ultrawide", items=("depth",))


def test_standalone_depth_upsampling_layout_needs_no_trajectory(tmp_path):
    root = tmp_path / "depth_upsampling"
    scene = root / "Validation" / "123"
    scene.mkdir(parents=True)
    (root / "metadata.csv").write_text(
        "video_id,visit_id,sky_direction,fold\n123,NA,Up,Validation\n"
    )
    for directory, value, shape, rgb in (
        ("color", 20, (6, 8), True),
        ("highres_depth", 3000, (6, 8), False),
        ("lowres_depth", 1000, (3, 4), False),
        ("confidence", 2, (3, 4), False),
    ):
        _write_png(
            scene / directory / "123_1.000.png",
            value,
            shape=shape,
            rgb=rgb,
        )
    dataset = ARKitScenes(root, offline=True)
    assert dataset.metadata["123"]["is_in_upsampling"] is True
    batch = dataset.get_frames(
        "123",
        source="wide",
        items=("rgb", "depth", "confidence_maps", "timestamps"),
        rotate_to_up=False,
    )
    assert batch.indices.tolist() == [0]
    assert batch.timestamps.tolist() == [1.0]
    assert batch.depth.shape == (1, 6, 8)
    assert np.all(batch.depth == 3.0)


def test_prepared_threedod_frame_layout(tmp_path):
    root = tmp_path / "threedod"
    scene = root / "Training" / "123"
    frames = scene / "123_frames"
    frames.mkdir(parents=True)
    (root / "metadata.csv").write_text(
        "video_id,visit_id,sky_direction,fold\n123,NA,Up,Training\n"
    )
    (frames / "color.traj").write_text("0.000 0 0 0 0 0 0\n1.000 0 0 0 0 0 0\n")
    for timestamp in ("0.000", "1.000"):
        intrinsic = frames / "color_intrinsics" / f"{timestamp}.pincam"
        intrinsic.parent.mkdir(exist_ok=True)
        intrinsic.write_text("4 3 2 2 1.5 1.0 0 0")
        _write_png(
            frames / "wide" / f"{timestamp}.png",
            20,
            shape=(3, 4),
            rgb=True,
        )
    _write_png(
        frames / "depth_densified" / "1.000.png",
        2500,
        shape=(3, 4),
    )
    dataset = ARKitScenes(root, offline=True)
    assert dataset.metadata["123"]["is_in_threedod"] is True
    batch = dataset.get_frames(
        "123",
        source="threedod",
        items=("rgb", "depth", "rgb_intrinsics", "world_to_camera"),
        rotate_to_up=False,
    )
    # The first released color frame has no densified depth and is filtered
    # before default selection rather than being paired to the wrong PNG.
    assert batch.indices.tolist() == [1]
    assert batch.rgb.shape == (1, 3, 4, 3)
    assert batch.depth.dtype == np.float32
    assert np.all(batch.depth == 2.5)
    assert batch.rgb_intrinsics.shape == (1, 3, 3)
    with pytest.raises(DataFormatError):
        dataset.get_frames(
            "123",
            source="threedod",
            indices=[0],
            items=("depth",),
            rotate_to_up=False,
        )


def test_faro_mapping_poses_and_point_clouds(tmp_path):
    _make_capture(tmp_path)
    with open(
        tmp_path / "laser_scanner_point_clouds_mapping.csv", "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("laser_scanner_point_clouds_id", "visit_id"))
        writer.writerow(("scan-a", 1234))
    directory = tmp_path / "laser_scanner_point_clouds" / "1234"
    directory.mkdir(parents=True)
    source_pose = np.eye(4)
    source_pose[3, :3] = (1, 2, 3)
    np.savetxt(directory / "scan-a_pose.txt", source_pose, delimiter=",")
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray([[1.0, 2.0, 3.0]]))
    )
    assert o3d.io.write_point_cloud(str(directory / "scan-a.ply"), cloud)

    dataset = ARKitScenes(tmp_path, offline=True)
    assert dataset.get_laser_scanner_ids(SCENE_ID) == ("scan-a",)
    pose = dataset.get_laser_scanner_poses(SCENE_ID, scan_ids=["scan-a"])["scan-a"]
    assert pose[:3, 3].tolist() == [1.0, 2.0, 3.0]
    clouds = dataset.get_laser_scanner_point_clouds(SCENE_ID, scan_ids=["scan-a"])
    assert len(clouds["scan-a"].points) == 1
    with pytest.raises(ValueError):
        dataset.get_laser_scanner_point_clouds(SCENE_ID, scan_ids=["unknown"])


def test_box_segmentation_uses_smaller_instance_for_overlaps(tmp_path):
    scene = _make_capture(tmp_path)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        ),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]])),
    )
    assert o3d.io.write_triangle_mesh(str(scene / f"{SCENE_ID}_3dod_mesh.ply"), mesh)

    def annotation(label, extent, uid):
        return {
            "label": label,
            "uid": uid,
            "segments": {
                "obbAligned": {
                    "centroid": [0, 0, 0],
                    "normalizedAxes": np.eye(3).reshape(-1).tolist(),
                    "axesLengths": [extent, extent, extent],
                }
            },
        }

    payload = {
        "data": [annotation("small", 1, "keep-me"), annotation("large", 4, "other")],
        "label_version": "synthetic",
    }
    (scene / f"{SCENE_ID}_3dod_annotation.json").write_text(json.dumps(payload))
    dataset = ARKitScenes(tmp_path, offline=True)
    assert dataset.get_annotations(SCENE_ID)["data"][0]["uid"] == "keep-me"
    segmentation = dataset.get_segmentation(SCENE_ID)
    assert segmentation.labels.tolist() == [0, 0, 0]
