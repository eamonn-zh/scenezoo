from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scenezoo.dataset.scene import ScanNetPP
from scenezoo.io.camera import read_colmap_cameras, read_colmap_images


def _write_colmap(directory, names, *, model="OPENCV_FISHEYE"):
    directory.mkdir(parents=True)
    params = "4 4 2 1.5 0.1 0.2 0.3 0.4"
    (directory / "cameras.txt").write_text(f"1 {model} 4 3 {params}\n")
    lines = []
    for index, name in enumerate(names):
        lines.extend([f"{index} 1 0 0 0 {index} 0 0 1 {name}", ""])
    (directory / "images.txt").write_text("\n".join(lines))
    (directory / "points3D.txt").write_text("7 1 2 3 10 20 30 0.1\n")


def _write_dslr(tmp_path):
    base = tmp_path / "data/scene/dslr"
    names = ["A.JPG", "B.JPG"]
    _write_colmap(base / "colmap", names)
    records = [
        {
            "file_path": name,
            "mask_path": name.replace("JPG", "png"),
            "is_bad": index == 1,
        }
        for index, name in enumerate(names)
    ]
    metadata = {
        "fl_x": 2,
        "fl_y": 2,
        "cx": 2,
        "cy": 1.5,
        "w": 4,
        "h": 3,
        "camera_model": "PINHOLE",
        "frames": records[:1],
        "test_frames": records[1:],
    }
    directory = base / "nerfstudio"
    directory.mkdir()
    (directory / "transforms.json").write_text(json.dumps(metadata))
    (directory / "transforms_undistorted.json").write_text(json.dumps(metadata))
    for folder in (
        "resized_images",
        "resized_anon_masks",
        "resized_undistorted_images",
        "resized_undistorted_masks",
    ):
        (base / folder).mkdir()
    for index, name in enumerate(names):
        rgb = np.full((3, 4, 3), 50 + index, dtype=np.uint8)
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[:, index] = 255
        for folder in ("resized_images", "resized_undistorted_images"):
            Image.fromarray(rgb).save(base / folder / name)
        for folder in ("resized_anon_masks", "resized_undistorted_masks"):
            Image.fromarray(mask).save(base / folder / name.replace("JPG", "png"))


def test_colmap_reader_preserves_opencv_models(tmp_path):
    directory = tmp_path / "model"
    _write_colmap(directory, ["A.JPG", "B.JPG"])
    camera = read_colmap_cameras(directory / "cameras.txt")[1]
    images = read_colmap_images(directory / "images.txt")
    assert camera.model == "OPENCV_FISHEYE"
    assert camera.distortion.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert images["B.JPG"].world_to_camera[0, 3] == 1


def test_laser_scan_uses_standard_point_batch(tmp_path, monkeypatch):
    class Cloud:
        points = np.asarray([[1.0, 2.0, 3.0]])
        colors = np.asarray([[0.0, 0.5, 1.0]])
        normals = np.asarray([[0.0, 0.0, 1.0]])

    dataset = ScanNetPP(tmp_path)
    monkeypatch.setattr(dataset, "get_point_cloud", lambda sample_id: Cloud())
    points = dataset.get_points("scene")
    assert points.xyz.dtype == np.float32
    assert points.xyz.tolist() == [[1.0, 2.0, 3.0]]
    assert points.rgb.tolist() == [[0, 128, 255]]
    assert points.normals.tolist() == [[0.0, 0.0, 1.0]]


def test_dslr_sources_camera_masks_keys_and_split(tmp_path):
    _write_dslr(tmp_path)
    dataset = ScanNetPP(tmp_path)
    assert dataset.get_frame_sources("scene") == ("dslr", "dslr_undistorted")
    raw = dataset.get_frames(
        "scene",
        source="dslr",
        frame_keys=["B.JPG", "A.JPG"],
        items=("rgb", "rgb_intrinsics", "world_to_camera", "frame_mask", "is_bad"),
        output_size=(2, 2),
    )
    assert raw.frame_keys == ("B.JPG", "A.JPG")
    assert raw.camera_model == "OPENCV_FISHEYE"
    assert raw.distortion.shape == (2, 4)
    assert raw.rgb.shape == (2, 2, 2, 3)
    assert raw.frame_mask.dtype == bool
    assert raw.world_to_camera[:, 0, 3].tolist() == [1, 0]
    assert raw.is_bad.tolist() == [True, False]

    test = dataset.get_frames(
        "scene",
        source="dslr_undistorted",
        view_split="test",
        indices=[0],
        items=("rgb_intrinsics", "world_to_camera"),
    )
    assert test.frame_keys == ("B.JPG",)
    assert test.camera_model == "PINHOLE"
    assert test.distortion.shape == (1, 0)


def test_panocam_maps_pose_and_backprojection(tmp_path):
    base = tmp_path / "data/scene"
    (base / "scans").mkdir(parents=True)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 4
    (base / "scans/scanner_poses.json").write_text(json.dumps(poses.tolist()))
    for folder in ("images", "depth", "anon_mask", "azim", "elev"):
        (base / "panocam" / folder).mkdir(parents=True)
    Image.fromarray(np.full((2, 4, 3), 128, np.uint8)).save(
        base / "panocam/images/1.jpg"
    )
    for folder, value in (
        ("depth", 2000),
        ("azim", 0),
        ("elev", 1571),
        ("anon_mask", 0),
    ):
        Image.fromarray(np.full((2, 4), value, np.uint16)).save(
            base / "panocam" / folder / "1.png"
        )
    dataset = ScanNetPP(tmp_path)
    batch = dataset.get_frames(
        "scene",
        source="panocam",
        indices=[1],
        items=("rgb", "depth", "azimuth", "elevation", "world_to_camera", "frame_mask"),
    )
    assert batch.camera_model == "EQUIRECTANGULAR"
    assert batch.depth_mode == "ray_distance"
    assert batch.frame_keys == ("1",)
    assert np.all(batch.depth == 2)
    assert np.allclose(batch.world_to_camera[0, 0, 3], -4)
    cloud = dataset.get_panorama_point_cloud("scene", 1, resized=False)
    assert len(cloud.points) == 8


def test_multilabel_instances_choose_smallest_for_single_label(tmp_path):
    scan = tmp_path / "data/scene/scans"
    scan.mkdir(parents=True)
    (scan / "segments.json").write_text(json.dumps({"segIndices": [0, 0, 1, 1, 1]}))
    (scan / "segments_anno.json").write_text(
        json.dumps(
            {
                "segGroups": [
                    {"objectId": 8, "label": "large", "segments": [0, 1]},
                    {"objectId": 3, "label": "small", "segments": [0]},
                ]
            }
        )
    )
    (scan / "mesh_aligned_0.05_mask.txt").write_text("")
    dataset = ScanNetPP(tmp_path)
    single = dataset.get_segmentation("scene")
    multi = dataset.get_segmentation("scene", multilabel=True)
    assert single.labels.tolist() == [3, 3, 8, 8, 8]
    assert multi.labels.tolist() == [[3, 8], [3, 8], [8, -1], [8, -1], [8, -1]]
    assert multi.is_multilabel
