from __future__ import annotations

import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from scenezoo import UnsupportedOperationError, get_dataset


def _write_common_scene(scene: Path, *, include_ground_truth: bool) -> None:
    (scene / "rgb").mkdir(parents=True)
    trajectory = (
        "graph_uid,tracking_timestamp_us,utc_timestamp_ns,tx_world_device,"
        "ty_world_device,tz_world_device,qx_world_device,qy_world_device,"
        "qz_world_device,qw_world_device\n"
        "simulated,0,-1,0,0,0,0,0,0,1\n"
        "simulated,100000,-1,1,0,0,0,0,0,1\n"
    )
    (scene / "trajectory.csv").write_text(trajectory)
    points = (
        "uid,graph_uid,px_world,py_world,pz_world,inv_dist_std,dist_std\n"
        "1,graph,1,2,3,0.004,0.009\n"
        "2,graph,4,5,6,0.006,0.020\n"
    )
    with gzip.open(scene / "semidense_points.csv.gz", "wt") as handle:
        handle.write(points)
    observations = (
        "uid,frame_tracking_timestamp_us,camera_serial,u,v\n"
        "1,0,16,10.5,20.5\n"
        "2,100000,16,30.5,40.5\n"
    )
    with gzip.open(scene / "semidense_observations.csv.gz", "wt") as handle:
        handle.write(observations)
    for index in range(2):
        rgb = np.full((704, 704, 3), (index + 1) * 20, dtype=np.uint8)
        Image.fromarray(rgb).save(scene / "rgb" / f"vignette{index:07d}.jpg")

    if not include_ground_truth:
        return
    (scene / "depth").mkdir()
    (scene / "instances").mkdir()
    for index in range(2):
        Image.fromarray(np.full((704, 704), (index + 1) * 1000, dtype=np.uint16)).save(
            scene / "depth" / f"depth{index:07d}.png"
        )
        Image.fromarray(np.full((704, 704), index + 2, dtype=np.uint16)).save(
            scene / "instances" / f"instance{index:07d}.png"
        )
    (scene / "object_instances_to_classes.json").write_text(
        json.dumps({"0": "empty_space", "2": "wall", "3": "door"})
    )
    (scene / "ase_scene_language.txt").write_text(
        "make_wall, id=0, a_x=0, a_y=0, a_z=0, b_x=2, b_y=0, "
        "b_z=0, height=3, thickness=0.1\n"
        "make_door, id=1000, wall0_id=0, wall1_id=-1, position_x=1, "
        "position_y=0, position_z=1, width=0.8, height=2\n"
    )


def _archive_scene(data_root: Path, split: str, scene_number: int, scene: Path) -> None:
    archive_path = data_root / f"{split}_chunk_{scene_number // 10:07d}.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in scene.rglob("*"):
            if path.is_file():
                archive.write(path, f"{scene_number}/{path.relative_to(scene)}")


def test_ase_extracted_scene_schema_and_camera(tmp_path):
    scene = tmp_path / "download" / "12"
    _write_common_scene(scene, include_ground_truth=True)
    dataset = get_dataset("ase", tmp_path, offline=True)

    report = dataset.check(sample_ids=("train/12",))
    assert report.ok, report.format()
    assert dataset.supports("points")
    assert dataset.supports("boxes")
    assert dataset.supports("frames")
    assert not dataset.supports("mesh")
    assert not dataset.supports("segmentation")

    frames = dataset.get_frames(
        "train/12",
        indices=(1, 0),
        items=(
            "rgb",
            "depth",
            "instance_maps",
            "frame_mask",
            "timestamps",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        rotate_to_up=False,
    )
    assert frames.indices.tolist() == [1, 0]
    assert frames.timestamps.tolist() == [0.1, 0.0]
    assert frames.rgb.shape == (2, 704, 704, 3)
    assert frames.depth.dtype == np.float32
    assert frames.depth[:, 0, 0].tolist() == [2.0, 1.0]
    assert frames.depth_mode == "ray_distance"
    assert frames.instance_maps.dtype == np.uint16
    assert frames.camera_model == "FISHEYE624"
    assert frames.distortion.shape == (2, 12)
    assert frames.rgb_intrinsics.shape == (2, 3, 3)
    assert frames.world_to_camera.shape == (2, 4, 4)
    assert frames.frame_mask.dtype == bool

    points = dataset.get_points(
        "train/12",
        max_inverse_distance_std=0.005,
        max_distance_std=0.01,
    )
    assert points.xyz.tolist() == [[1.0, 2.0, 3.0]]
    assert points.uids.tolist() == [1]
    observations = dataset.get_semidense_observations("train/12", frame_indices=(1,))
    assert observations["uids"].tolist() == [2]
    assert observations["frame_indices"].tolist() == [1]
    assert dataset.get_instance_classes("train/12")[2] == "wall"
    boxes, names = dataset.get_boxes("train/12")
    assert set(boxes) == {0, 1000}
    assert names == {0: "wall", 1000: "door"}


def test_ase_requires_train_scene_extraction(tmp_path):
    data_root = tmp_path / "download"
    scene = tmp_path / "staging" / "42"
    _write_common_scene(scene, include_ground_truth=True)
    data_root.mkdir()
    _archive_scene(data_root, "train", 42, scene)

    dataset = get_dataset("aria-ase", tmp_path)
    report = dataset.check(sample_ids=("train/42",))
    assert not report.ok
    assert any(issue.code == "scene-not-extracted" for issue in report.errors)
    assert dataset.get_ids("train") == []
    with pytest.raises(FileNotFoundError, match="is not extracted"):
        dataset.get_frames(
            "train/42",
            indices=(0,),
            items=("rgb", "world_to_camera"),
        )


def test_ase_test_release_has_no_withheld_ground_truth(tmp_path):
    scene = tmp_path / "download" / "test" / "2"
    _write_common_scene(scene, include_ground_truth=False)

    dataset = get_dataset("ase", tmp_path)
    report = dataset.check(sample_ids=("test/2",))
    assert report.ok, report.format()
    frames = dataset.get_frames(
        "test/2", indices=(0,), items=("rgb", "world_to_camera")
    )
    assert frames.rgb.shape == (1, 704, 704, 3)
    with pytest.raises(UnsupportedOperationError, match="withholds"):
        dataset.get_frames("test/2", indices=(0,), items=("depth",))
    with pytest.raises(UnsupportedOperationError, match="test scenes"):
        dataset.get_boxes("test/2")


def test_ase_manifest_splits_are_namespaced(tmp_path):
    (tmp_path / "download" / "train" / "0").mkdir(parents=True)
    (tmp_path / "download" / "test" / "0").mkdir(parents=True)
    manifest = [
        {"filename": "train_chunk_0000000.zip", "sha": "train"},
        {"filename": "test_chunk_0000000.zip", "sha": "test"},
    ]
    (tmp_path / "aria_synthetic_environments_dataset_download_urls.json").write_text(
        json.dumps(manifest)
    )
    dataset = get_dataset("ase", tmp_path)
    assert dataset.get_ids("train") == ["train/0"]
    assert dataset.get_ids("test") == ["test/0"]
    assert dataset.get_ids() == ["train/0", "test/0"]
    assert dataset.metadata["archive_counts"] == {"train": 1, "test": 1}


def _project_fisheye624(points, focal, center, distortion):
    """Reference Aria FisheyeRadTanThinPrism projection (projectaria_tools)."""

    k, p, s = distortion[:6], distortion[6:8], distortion[8:12]
    ab = points[:, :2] / points[:, 2:]
    radius = np.linalg.norm(ab, axis=1, keepdims=True)
    theta = np.arctan(radius)
    powers = theta ** (2 * np.arange(1, 7) + 1)
    uv = (theta + powers @ k[:, None]) / radius * ab
    r2 = np.sum(uv**2, axis=1, keepdims=True)
    uv = uv + 2 * (uv @ p[:, None]) * uv + r2 * p
    radial = np.hstack([r2, r2**2])
    uv = uv + np.stack([radial @ s[:2], radial @ s[2:]], axis=1)
    return uv * focal + center


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_fisheye_camera_projects_like_rotated_image(rotation):
    from scenezoo.dataset.scene.ase import AriaSyntheticEnvironments
    from scenezoo.ops.frame import build_pixel_transform, transform_intrinsics

    calibration = AriaSyntheticEnvironments.get_calibration()
    parameters = calibration["projection_parameters"]
    transform = build_pixel_transform(calibration["image_size"], rotation=rotation)
    points = np.random.default_rng(0).uniform([-1, -1, 1], [1, 1, 3], (50, 3))

    source = _project_fisheye624(points, parameters[0], parameters[1:3], parameters[3:])
    expected = (np.c_[source, np.ones(len(source))] @ transform.matrix.T)[:, :2]

    intrinsic = transform_intrinsics(calibration["intrinsic"], transform)
    rolled = AriaSyntheticEnvironments._rolled_distortion(parameters[3:], transform)
    actual = _project_fisheye624(
        points @ transform.camera_rotation.T,
        np.diag(intrinsic)[:2],
        intrinsic[:2, 2],
        rolled,
    )
    assert np.allclose(actual, expected, atol=1e-6)
