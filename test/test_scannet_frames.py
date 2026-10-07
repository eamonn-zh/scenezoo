from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from scenezoo import DataFormatError
from scenezoo.dataset.scene import ScanNet


def _jpeg(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_projection_archive(path, directory, frames):
    with zipfile.ZipFile(path, "w") as archive:
        for index, image in frames.items():
            archive.writestr(f"{directory}/{index}.png", _png(image))


def _write_label_mapping(path):
    path.write_text(
        "id\traw_category\tcount\tnyu40id\teigen13id\tnyuClass\t"
        "nyu40class\teigen13class\tmpcat40index\n"
        "1\twall\t1\t1\t12\twall\twall\tWall\t1\n"
        "2\tstack of chairs\t1\t5\t4\tchair\tchair\tChair\t3\n"
        "107\tpipe\t1\t40\t0\totherprop\totherprop\t\t40\n"
        "107\tpipes\t1\t38\t0\totherstructure\totherstructure\t\t38\n"
    )


def _write_sens(path):
    rgb_intrinsic = np.eye(4, dtype=np.float32)
    rgb_intrinsic[0, 0] = rgb_intrinsic[1, 1] = 4
    depth_intrinsic = np.eye(4, dtype=np.float32)
    depth_intrinsic[0, 0] = depth_intrinsic[1, 1] = 2
    frames = []
    for index in range(3):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index
        if index == 1:
            pose[0, 0] = np.nan
        rgb = np.full((3, 4, 3), index * 80, dtype=np.uint8)
        depth = np.full((2, 2), (index + 1) * 1000, dtype=np.uint16)
        frames.append((pose, _jpeg(rgb), zlib.compress(depth.tobytes())))
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", 4))
        handle.write(struct.pack("<Q", 4))
        handle.write(b"test")
        for matrix in (rgb_intrinsic, np.eye(4), depth_intrinsic, np.eye(4)):
            handle.write(np.asarray(matrix, dtype=np.float32).tobytes())
        handle.write(struct.pack("<ii", 2, 1))
        handle.write(struct.pack("<IIII", 4, 3, 2, 2))
        handle.write(struct.pack("<f", 1000.0))
        handle.write(struct.pack("<Q", len(frames)))
        for index, (pose, rgb, depth) in enumerate(frames):
            handle.write(pose.tobytes())
            timestamp = (index + 1) * 100
            handle.write(struct.pack("<QQ", timestamp, timestamp))
            handle.write(struct.pack("<QQ", len(rgb), len(depth)))
            handle.write(rgb)
            handle.write(depth)
        handle.write(struct.pack("<Q", 3))
        for index, timestamp in enumerate((90, 210, 290)):
            handle.write(struct.pack("<15dQ", *([float(index)] * 15), timestamp))


def test_scannet_filters_invalid_poses_and_preserves_explicit_order(tmp_path):
    scene = "scene0000_00"
    directory = tmp_path / "scans" / scene
    directory.mkdir(parents=True)
    _write_sens(directory / f"{scene}.sens")
    dataset = ScanNet(tmp_path)
    batch = dataset.get_frames(
        scene,
        indices=[2, 0],
        items=(
            "rgb",
            "depth",
            "timestamps",
            "imu",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        output_size=(2, 2),
    )
    assert batch.indices.tolist() == [2, 0]
    assert batch.source_frame_count == 3
    assert batch.rgb.shape == (2, 2, 2, 3)
    assert batch.depth.shape == (2, 2, 2)
    assert batch.depth.dtype == np.float32
    assert np.all(batch.depth[0] == 3.0)
    assert np.all(batch.depth[1] == 1.0)
    assert batch.world_to_camera[:, 0, 3].tolist() == pytest.approx([-2, 0])
    assert batch.rgb_intrinsics.shape == batch.depth_intrinsics.shape == (2, 3, 3)
    assert batch.timestamps.tolist() == [[300, 300], [100, 100]]
    assert batch.imu["valid"].tolist() == [True, True]
    assert batch.imu["timestamps"].tolist() == [290, 90]
    assert batch.imu["rotation_rate"][:, 0].tolist() == [2.0, 0.0]

    raw_imu = dataset.get_imu(scene)
    assert raw_imu["timestamps"].tolist() == [90, 210, 290]
    assert raw_imu["gravity"].shape == (3, 3)

    sampled = dataset.get_frames(scene, step=2, items=("world_to_camera",))
    assert sampled.indices.tolist() == [0]
    with pytest.raises(ValueError, match="not synchronized"):
        dataset.get_frames(scene, indices=[1], items=("world_to_camera",))

    empty = dataset.get_frames(scene, indices=[], items=("rgb", "depth"))
    assert empty.rgb.shape == (0, 3, 4, 3)
    assert empty.depth.shape == (0, 2, 2)


def test_scannet_projection_variants_normalize_ids_and_map_semantics(tmp_path):
    scene = "scene0000_00"
    directory = tmp_path / "scans" / scene
    directory.mkdir(parents=True)
    _write_sens(directory / f"{scene}.sens")
    (tmp_path / "tasks").mkdir()
    _write_label_mapping(tmp_path / "tasks/scannetv2-labels.combined.tsv")

    filtered_instance = {
        0: np.array([[0, 1], [2, 0]], dtype=np.uint8),
        2: np.array([[2, 0], [1, 2]], dtype=np.uint8),
    }
    filtered_semantic = {
        0: np.array([[0, 1], [2, 0]], dtype=np.uint16),
        2: np.array([[2, 0], [1, 2]], dtype=np.uint16),
    }
    raw_instance = {
        index: image + (image > 0) for index, image in filtered_instance.items()
    }
    raw_semantic = {index: image for index, image in filtered_semantic.items()}
    _write_projection_archive(
        directory / f"{scene}_2d-instance-filt.zip", "instance-filt", filtered_instance
    )
    _write_projection_archive(
        directory / f"{scene}_2d-label-filt.zip", "label-filt", filtered_semantic
    )
    _write_projection_archive(
        directory / f"{scene}_2d-instance.zip", "instance", raw_instance
    )
    _write_projection_archive(
        directory / f"{scene}_2d-label.zip", "label", raw_semantic
    )

    dataset = ScanNet(tmp_path)
    batch = dataset.get_frames(
        scene,
        indices=[2, 0],
        items=("instance_maps", "semantic_maps"),
        semantic_label_space="nyu40",
    )
    assert batch.instance_maps.dtype == np.int32
    assert batch.instance_maps[1].tolist() == [[-1, 0], [1, -1]]
    assert batch.semantic_maps[1].tolist() == [[0, 1], [5, 0]]
    duplicate_id = dataset._map_semantic_images_to_nyu40(
        np.array([[[107]]], dtype=np.uint16)
    )
    assert duplicate_id.item() == 38
    assert dataset.metadata["label_mapping"]["stack of chairs"]["nyu40id"] == 5
    assert dataset.metadata["label_mapping_source"].endswith(
        "tasks/scannetv2-labels.combined.tsv"
    )

    raw = dataset.get_frames(
        scene,
        indices=[0],
        items=("instance_maps", "semantic_maps"),
        annotation_variant="raw",
    )
    assert raw.instance_maps[0].tolist() == [[-1, 1], [2, -1]]
    assert raw.semantic_maps[0].dtype == np.uint16


def test_scannet_raw_and_decimated_semantic_segmentation(tmp_path):
    scene = "scene0000_00"
    directory = tmp_path / "scans" / scene
    directory.mkdir(parents=True)
    (tmp_path / "tasks").mkdir()
    _write_label_mapping(tmp_path / "tasks/scannetv2-labels.combined.tsv")
    groups = {
        "segGroups": [
            {"objectId": 0, "label": "wall", "segments": [0]},
            {"objectId": 1, "label": "stack of chairs", "segments": [1]},
        ]
    }
    (directory / f"{scene}.aggregation.json").write_text(json.dumps(groups))
    (directory / f"{scene}_vh_clean.aggregation.json").write_text(json.dumps(groups))
    (directory / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [0, 1, 1]})
    )
    (directory / f"{scene}_vh_clean.segs.json").write_text(
        json.dumps({"segIndices": [0, 0, 1, 1, 1]})
    )
    vertices = np.zeros(
        3,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "u2")],
    )
    vertices["label"] = [1, 5, 5]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(
        directory / f"{scene}_vh_clean_2.labels.ply"
    )

    dataset = ScanNet(tmp_path)
    decimated = dataset.get_segmentation(scene)
    raw = dataset.get_segmentation(scene, mesh_type="raw")
    semantic = dataset.get_segmentation(scene, segmentation_type="semantic")
    raw_semantic = dataset.get_segmentation(
        scene, segmentation_type="semantic", mesh_type="raw"
    )
    assert decimated.labels.tolist() == [0, 1, 1]
    assert raw.labels.tolist() == [0, 0, 1, 1, 1]
    assert semantic.labels.tolist() == [1, 5, 5]
    assert raw_semantic.labels.tolist() == [1, 1, 5, 5, 5]
    assert semantic.id_to_name[5] == "chair"


def test_scannet_scene_info_calibration_and_packaged_offline_splits(tmp_path):
    scene = "scene0000_00"
    directory = tmp_path / "scans" / scene
    directory.mkdir(parents=True)
    _write_sens(directory / f"{scene}.sens")
    identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (directory / f"{scene}.txt").write_text(
        f"axisAlignment = {identity}\n"
        f"colorToDepthExtrinsics = {identity}\n"
        "colorWidth = 4\n"
        "numIMUmeasurements = 3\n"
        "sceneType = Apartment\n"
    )

    dataset = ScanNet(tmp_path, offline=True)
    assert {key: len(value) for key, value in dataset.splits.items()} == {
        "train": 1201,
        "val": 312,
        "test": 100,
    }
    info = dataset.get_scene_info(scene)
    assert info["sceneType"] == "Apartment"
    assert info["colorWidth"] == 4
    assert info["axisAlignment"].shape == (4, 4)
    calibration = dataset.get_calibration(scene)
    assert calibration["sensor_name"] == "test"
    assert calibration["rgb_size"] == (4, 3)
    assert calibration["rgb_to_depth"].shape == (4, 4)


def test_scannet_rejects_truncated_sens_stream(tmp_path):
    scene = "scene0000_00"
    directory = tmp_path / "scans" / scene
    directory.mkdir(parents=True)
    path = directory / f"{scene}.sens"
    _write_sens(path)
    payload = path.read_bytes()
    path.write_bytes(payload[:-1])

    with pytest.raises(DataFormatError, match="Truncated ScanNet IMU stream"):
        ScanNet(tmp_path).get_frames(scene, indices=[0])


def test_boxes_group_vertices_by_annotated_segments(tmp_path):
    import open3d as o3d

    scene = "scene0000_00"
    directory = tmp_path / "scans" / scene
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    vertices = rng.uniform(-1, 1, (400, 3))
    segments = rng.integers(0, 40, len(vertices))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.arange(399).reshape(-1, 3)),
    )
    o3d.io.write_triangle_mesh(str(directory / f"{scene}_vh_clean_2.ply"), mesh)
    (directory / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": segments.tolist()})
    )
    groups = [
        {"objectId": 0, "label": "chair", "segments": [3, 7, 11, 7]},
        # Segment 99 has no vertices; the remaining segments still count.
        {"objectId": 1, "label": "table", "segments": [0, 39, 99]},
    ]
    (directory / f"{scene}.aggregation.json").write_text(
        json.dumps({"segGroups": groups})
    )
    boxes, names = ScanNet(tmp_path).get_boxes(scene, box_type="aabb")
    assert names == {0: "chair", 1: "table"}
    loaded = np.asarray(
        o3d.io.read_triangle_mesh(str(directory / f"{scene}_vh_clean_2.ply")).vertices
    )
    for object_id, group in enumerate(groups):
        expected = loaded[np.isin(segments, group["segments"])]
        np.testing.assert_allclose(boxes[object_id].min_bound, expected.min(axis=0))
        np.testing.assert_allclose(boxes[object_id].max_bound, expected.max(axis=0))
