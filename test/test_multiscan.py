from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from scenezoo.dataset.base import DataFormatError, UnsupportedOperationError
from scenezoo.dataset.scene import MultiScan
from scenezoo.io.compression import read_raw_deflate_frames


def _raw_deflate(array: np.ndarray) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(array.tobytes()) + compressor.flush()


def _write_multiscan_fixture(root, *, encoding="float16_zlib"):
    scene = "scene_fixture_00"
    directory = root / scene
    directory.mkdir()
    identity3 = np.eye(3, dtype=np.float32)
    identity4 = np.eye(4, dtype=np.float32)
    scene_info = {
        "camera_orientation_quaternion_format": "wxyz",
        "camera_orientation_euler_angles_format": "xyz",
        "depth_unit": "m",
        "streams": [
            {
                "type": "camera_info",
                "number_of_frames": 3,
                "encoding": "jsonl",
            },
            {
                "type": "confidence_map",
                "number_of_frames": 3,
                "encoding": "uint8_zlib",
            },
            {
                "type": "color_camera",
                "number_of_frames": 3,
                "resolution": [4, 6],
                "encoding": "h264",
            },
            {
                "type": "lidar_sensor",
                "number_of_frames": 3,
                "resolution": [2, 3],
                "encoding": encoding,
                "intrinsics": None,
            },
        ],
    }
    (directory / f"{scene}.json").write_text(json.dumps(scene_info))
    records = []
    for index in range(3):
        pose = identity4.copy()
        pose[0, 3] = index
        records.append(
            {
                "intrinsics": identity3.reshape(-1, order="F").tolist(),
                "transform": pose.reshape(-1, order="F").tolist(),
                "timestamp": 1000 + index,
                "exposure_duration": 20 + index,
                "quaternion": [1, 0, 0, index],
                "euler_angles": [0, 0, index],
            }
        )
    with open(directory / f"{scene}.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (directory / f"{scene}.align.json").write_text(
        json.dumps(
            {
                "coordinate_transform": identity4.reshape(-1, order="F").tolist(),
                "reference_scan_alignment": {
                    "target_id": "scene_reference_00",
                    "transformation": identity4.reshape(-1, order="F").tolist(),
                },
            }
        )
    )
    annotations = {
        "version": "test",
        "scanId": scene,
        "objects": [
            {
                "objectId": 1,
                "label": "cabinet.1",
                "partIds": [1, 2],
                "obb": {
                    "centroid": [0, 0, 0],
                    "normalizedAxes": identity3.reshape(-1, order="F").tolist(),
                    "axesLengths": [1, 2, 3],
                },
            },
            {"objectId": 2, "label": "remove.1", "partIds": [3]},
        ],
        "parts": [
            {
                "partId": 1,
                "label": "door.1",
                "parentId": 1,
                "articulations": [
                    {"type": "rotation", "axis": [0, 0, 1], "origin": [0, 0, 0]}
                ],
            },
            {"partId": 2, "label": "shelf.1", "parentId": 1, "articulations": []},
            {"partId": 3, "label": "door.2", "parentId": 2, "articulations": []},
        ],
        "scanBbox": {},
    }
    (directory / f"{scene}.annotations.json").write_text(json.dumps(annotations))
    ply = """ply
format ascii 1.0
element vertex 4
property double x
property double y
property double z
element face 2
property list uchar int vertex_indices
property ushort objectId
property ushort partId
end_header
0 0 0
1 0 0
0 1 0
0 0 1
3 0 1 2 1 1
3 0 2 3 2 3
"""
    (directory / f"{scene}.ply").write_text(ply)
    depth = np.stack([np.full((2, 3), value, dtype="<f2") for value in (1.0, 2.0, 3.0)])
    confidence = np.stack(
        [np.full((2, 3), value, dtype=np.uint8) for value in (0, 1, 2)]
    )
    (directory / f"{scene}.depth.zlib").write_bytes(_raw_deflate(depth))
    (directory / f"{scene}.confidence.zlib").write_bytes(_raw_deflate(confidence))
    return scene


def test_raw_deflate_reader_selects_and_reorders_frames(tmp_path):
    values = np.arange(5 * 2 * 3, dtype="<f2").reshape(5, 2, 3)
    path = tmp_path / "frames.zlib"
    path.write_bytes(_raw_deflate(values))
    result = read_raw_deflate_frames(
        path, [4, 1], frame_shape=(2, 3), dtype="<f2", frame_count=5, chunk_size=7
    )
    np.testing.assert_array_equal(result, values[[4, 1]])
    path.write_bytes(_raw_deflate(values)[:-2])
    with pytest.raises(DataFormatError, match="ended before frame"):
        read_raw_deflate_frames(
            path, [4], frame_shape=(2, 3), dtype="<f2", frame_count=5
        )


def test_multiscan_frames_depth_confidence_and_camera_metadata(tmp_path):
    scene = _write_multiscan_fixture(tmp_path)
    batch = MultiScan(tmp_path).get_frames(
        scene,
        indices=[2, 0],
        items=(
            "timestamps",
            "depth",
            "confidence_maps",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
            "exposure_durations",
        ),
        rotate_to_up=False,
    )
    assert batch.indices.tolist() == [2, 0]
    assert batch.source_frame_count == 3
    assert batch.camera_model == "PINHOLE"
    assert batch.timestamps.tolist() == [1002, 1000]
    assert batch.exposure_durations.tolist() == [22, 20]
    assert batch.depth.dtype == np.float32
    assert batch.depth.shape == (2, 2, 3)
    assert batch.depth[:, 0, 0].tolist() == [3.0, 1.0]
    assert batch.confidence_maps.dtype == np.uint8
    assert batch.confidence_maps[:, 0, 0].tolist() == [2, 0]
    np.testing.assert_allclose(batch.depth_intrinsics[:, 0, 0], 0.5)
    np.testing.assert_allclose(batch.depth_intrinsics[:, 1, 1], 0.5)
    assert batch.world_to_camera[:, 0, 3].tolist() == [-2, 0]

    metadata = MultiScan(tmp_path).get_camera_metadata(scene, indices=[2, 0])
    assert metadata["quaternion_format"] == "wxyz"
    assert metadata["euler_angles_format"] == "xyz"
    assert metadata["quaternions"][:, 3].tolist() == [2, 0]


def test_multiscan_rejects_android_depth_encoding(tmp_path):
    scene = _write_multiscan_fixture(tmp_path, encoding="uint16_zlib")
    with pytest.raises(UnsupportedOperationError, match="not Android"):
        MultiScan(tmp_path).get_frames(
            scene, indices=[0], items=("depth",), rotate_to_up=False
        )


def test_multiscan_all_segmentations_articulations_and_alignment(
    tmp_path, multiscan_metadata
):
    scene = _write_multiscan_fixture(tmp_path)
    dataset = MultiScan(tmp_path, **multiscan_metadata)
    assert dataset.get_segmentation(scene).labels.tolist() == [1, 0]
    assert dataset.get_segmentation(
        scene, segmentation_type="part_instance"
    ).labels.tolist() == [1, 0]
    assert dataset.get_segmentation(
        scene, segmentation_type="object_semantic"
    ).labels.tolist() == [7, 0]
    part_semantic = dataset.get_segmentation(scene, segmentation_type="part_semantic")
    assert part_semantic.labels.tolist() == [2, 0]
    assert part_semantic.id_to_name[1] == "static"
    articulation = dataset.get_articulations(scene)[0]
    assert articulation["object_id"] == articulation["part_id"] == 1
    assert articulation["object_label"] == "cabinet"
    assert articulation["part_label"] == "door"
    target, transform = dataset.get_reference_alignment(scene)
    assert target == "scene_reference_00"
    np.testing.assert_array_equal(transform, np.eye(4))
    boxes, names = dataset.get_boxes(scene, box_type="obb_gt")
    assert set(boxes) == set(names) == {1}

    custom_invalid = MultiScan(tmp_path, invalid_obj_id=-1, **multiscan_metadata)
    assert custom_invalid.get_segmentation(scene).labels.tolist() == [1, -1]
    assert custom_invalid.get_segmentation(
        scene, segmentation_type="object_semantic"
    ).labels.tolist() == [7, -1]


def test_multiscan_official_splits_and_alias_preserving_metadata(
    tmp_path, multiscan_metadata
):
    dataset = MultiScan(tmp_path, offline=True, **multiscan_metadata)
    assert sorted(dataset.splits) == ["test", "train", "val"]
    assert sum(len(ids) for ids in dataset.splits.values()) == 257
    assert len(dataset.get_ids()) == 273  # includes 16 scans with no split
    assert dataset.metadata["object_name_to_semantic_id"]["wall_cabinet"] == 7
    assert "cabinet_otherroom" in dataset.metadata["object_semantic_id_to_names"][7]
