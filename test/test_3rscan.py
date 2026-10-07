from __future__ import annotations

import json

import numpy as np
import pytest

from scenezoo import UnsupportedOperationError
from scenezoo.dataset.scene import ThreeRScan


def _write_metadata(root):
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 2.0
    environments = [
        {
            "reference": "train-ref",
            "type": "train",
            "ambiguity": [{"instance_source": 1}],
            "scans": [
                {
                    "reference": "train-rescan",
                    "transform": transform.reshape(-1, order="F").tolist(),
                    "rigid": [{"instance_reference": 1, "instance_rescan": 2}],
                    "nonrigid": [3],
                    "removed": [4],
                }
            ],
        },
        {
            "reference": "test-ref",
            "type": "test",
            "scans": [{"reference": "test-hidden", "rigid": [1]}],
        },
    ]
    (root / "3RScan.json").write_text(json.dumps(environments))


def _write_annotations(scan_dir):
    scan_dir.mkdir(parents=True)
    ply = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property ushort objectId
property ushort globalId
property uchar NYU40
property uchar Eigen13
property uchar RIO27
element face 2
property list uchar int vertex_indices
end_header
0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 4 5 4 5
1 1 0 0 0 0 1 4 5 4 5
0 1 0 0 0 0 1 4 5 4 5
3 0 1 2
3 0 2 3
"""
    (scan_dir / "labels.instances.annotated.v2.ply").write_text(ply)
    annotation = {
        "segGroups": [
            {
                "objectId": 1,
                "label": "armchair",
                "segments": [8],
                "obb": {
                    "centroid": [0.5, 0.5, 0.0],
                    "normalizedAxes": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "axesLengths": [1, 1, 0.1],
                },
            }
        ]
    }
    (scan_dir / "semseg.v2.json").write_text(json.dumps(annotation))
    (scan_dir / "mesh.refined.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [0, 8, 8, 8]})
    )


def test_3rscan_complete_split_scene_metadata_and_alignment(tmp_path):
    (tmp_path / "download").mkdir()
    _write_metadata(tmp_path)
    dataset = ThreeRScan(tmp_path, offline=True)

    assert dataset.get_ids("train") == ["train-ref", "train-rescan"]
    assert dataset.get_ids("test") == ["test-ref", "test-hidden"]
    assert sorted(dataset.splits) == ["test", "train", "val"]
    assert dataset.get_ids() == ["train-ref", "train-rescan", "test-ref", "test-hidden"]
    assert dataset.get_reference_id("train-rescan") == "train-ref"
    assert dataset.get_scene_info("test-hidden")["has_annotations"] is False
    assert dataset.get_changes("train-rescan")["removed"] == [4]
    np.testing.assert_allclose(
        dataset.get_scene_alignment("train-rescan")[:3, 3], [2, 0, 0]
    )
    with pytest.raises(UnsupportedOperationError, match="does not publish"):
        dataset.get_scene_alignment("test-hidden")


def test_3rscan_instance_and_semantic_spaces(tmp_path):
    scans = tmp_path / "download"
    scans.mkdir()
    _write_metadata(tmp_path)
    _write_annotations(scans / "train-ref")
    dataset = ThreeRScan(tmp_path, offline=True, invalid_obj_id=-1)

    instance = dataset.get_segmentation("train-ref")
    assert instance.labels.tolist() == [-1, 1, 1, 1]
    assert instance.id_to_name == {1: "armchair"}
    assert dataset.get_oversegmentation("train-ref").tolist() == [0, 8, 8, 8]

    expected = {
        "global": (4, "armchair"),
        "nyu40": (5, "chair"),
        "eigen13": (4, "Chair"),
        "rio27": (5, "chair"),
        "rio7": (1, "seating"),
    }
    for segmentation_type, (identifier, name) in expected.items():
        segmentation = dataset.get_segmentation(
            "train-ref", segmentation_type=segmentation_type
        )
        assert segmentation.labels.tolist() == [-1, identifier, identifier, identifier]
        assert segmentation.id_to_name == {identifier: name}

    boxes, names = dataset.get_boxes("train-ref", box_type="obb_gt")
    assert set(boxes) == {1}
    assert names == {1: "armchair"}
    np.testing.assert_allclose(boxes[1].center, [0.5, 0.5, 0])

    with pytest.raises(UnsupportedOperationError, match="not published"):
        dataset.get_segmentation("test-hidden")


def test_3rscan_mapping_is_packaged_for_offline_use(tmp_path):
    (tmp_path / "download").mkdir()
    _write_metadata(tmp_path)
    metadata = ThreeRScan(tmp_path, offline=True).metadata
    assert metadata["scan_count"] == 4
    assert metadata["reference_count"] == 2
    assert metadata["label_mapping"]["armchair"]["rio27_name"] == "chair"
