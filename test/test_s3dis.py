from __future__ import annotations

import numpy as np
import pytest

from scenezoo import DataFormatError, get_dataset, get_dataset_info
from scenezoo.dataset.scene import S3DIS, S3DISPointData


def _write_room(root, area, room, annotations, *, raw_points=None):
    room_path = root / "Stanford3dDataset_v1.2_Aligned_Version" / area / room
    annotation_path = room_path / "Annotations"
    annotation_path.mkdir(parents=True)
    for name, rows in annotations.items():
        (annotation_path / f"{name}.txt").write_text(rows)
    if raw_points is None:
        raw_points = "0 0 0 0 0 0\n"
    (room_path / f"{room}.txt").write_text(raw_points)
    return room_path


@pytest.fixture
def s3dis_root(tmp_path):
    _write_room(
        tmp_path,
        "Area_1",
        "office_10",
        {"wall_1": "0 0 0 10 20 30\n0 1 0 40 50 60\n"},
    )
    _write_room(
        tmp_path,
        "Area_1",
        "office_2",
        {"chair_1": "2 2 0 255 0 0\n2 3 1 0 255 0\n"},
    )
    hallway = _write_room(
        tmp_path,
        "Area_5",
        "hallway_6",
        {
            "ceiling_1": "10 20 30 1 2 3\n11 21 31 4 5 6\n",
            # PointNet-compatible behavior maps non-benchmark labels to clutter.
            "stairs_1": "12\x0022 32 7 8 9\n",
        },
        raw_points=("12 22 32 7 8 9\n11 21 31 4 5 6\n10 20 30 1 2 3\n"),
    )
    (hallway.parent / "Area_5_alignmentAngle.txt").write_text(
        "## Global alignment angle per disjoint space in Area_5 ##\nhallway_6 90\n"
    )
    return tmp_path


def test_registry_and_room_discovery_are_natural_and_lazy(s3dis_root):
    dataset = get_dataset("stanford3d", s3dis_root)
    assert isinstance(dataset, S3DIS)
    assert dataset.dataset_root.name == "Stanford3dDataset_v1.2_Aligned_Version"
    assert dataset.get_ids("area_1") == ["Area_1/office_2", "Area_1/office_10"]
    assert dataset.get_ids("area_5") == ["Area_5/hallway_6"]
    assert dataset.get_ids() == [
        "Area_1/office_2",
        "Area_1/office_10",
        "Area_5/hallway_6",
    ]
    assert dataset.metadata["Area_1/office_10"] == {
        "area": 1,
        "area_name": "Area_1",
        "room_name": "office_10",
        "room_type": "office",
        "room_number": 10,
        "annotation_count": 1,
        "alignment_angle": None,
    }
    assert get_dataset_info("stanford-3d").name == "s3dis"
    assert S3DIS.capabilities() == ("points", "segmentation", "boxes")


def test_cross_validation_supports_all_common_area_spellings(s3dis_root):
    dataset = S3DIS(s3dis_root)
    expected_train = ["Area_1/office_2", "Area_1/office_10"]
    expected_test = ["Area_5/hallway_6"]
    assert dataset.get_cross_validation_split(5) == (expected_train, expected_test)
    assert dataset.get_cross_validation_split("Area_5") == (
        expected_train,
        expected_test,
    )
    with pytest.raises(ValueError, match="1 through 6"):
        dataset.get_cross_validation_split(0)
    with pytest.raises(DataFormatError, match="does not contain Area_6"):
        dataset.get_cross_validation_split(6)


def test_point_labels_cloud_and_boxes_share_annotation_order(s3dis_root):
    dataset = S3DIS(s3dis_root)
    sample_id = "Area_5/hallway_6"
    data = dataset.get_point_data(sample_id)
    assert isinstance(data, S3DISPointData)
    assert data.xyz.dtype == np.float32
    assert data.rgb.dtype == np.uint8
    assert data.xyz.tolist() == [
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [12.0, 22.0, 32.0],
    ]
    assert data.semantic_labels.tolist() == [0, 0, 12]
    assert data.instance_labels.tolist() == [0, 0, 1]
    assert data.semantic_id_to_name[12] == "clutter"
    assert data.instance_id_to_name == {0: "ceiling", 1: "clutter"}
    assert dataset.get_points(sample_id).xyz.tolist() == data.xyz.tolist()

    instances = dataset.get_segmentation(sample_id)
    semantics = dataset.get_segmentation(sample_id, segmentation_type="semantic")
    assert instances.domain == semantics.domain == "point"
    assert instances.labels.tolist() == data.instance_labels.tolist()
    assert semantics.labels.tolist() == data.semantic_labels.tolist()
    assert semantics.id_to_name[12] == "clutter"

    cloud = data.to_point_cloud()
    assert np.asarray(cloud.points).tolist() == data.xyz.tolist()
    assert np.allclose(np.asarray(cloud.colors)[0], [1 / 255, 2 / 255, 3 / 255])

    boxes, names = dataset.get_boxes(sample_id)
    assert names == data.instance_id_to_name
    assert np.asarray(boxes[0].get_min_bound()).tolist() == [10.0, 20.0, 30.0]
    assert np.asarray(boxes[0].get_max_bound()).tolist() == [11.0, 21.0, 31.0]
    boxes, names = dataset.get_boxes(sample_id, classes="clutter")
    assert set(boxes) == set(names) == {1}
    with pytest.raises(ValueError, match="Unknown S3DIS box classes"):
        dataset.get_boxes(sample_id, classes=("plant",))


def test_origin_shift_and_raw_room_order_are_explicit(s3dis_root):
    dataset = S3DIS(s3dis_root)
    sample_id = "Area_5/hallway_6"
    shifted = dataset.get_point_data(sample_id, shift_to_origin=True)
    assert shifted.xyz.min(axis=0).tolist() == [0.0, 0.0, 0.0]
    raw = dataset.get_raw_room_points(sample_id)
    assert raw[:, :3].tolist() == [
        [12.0, 22.0, 32.0],
        [11.0, 21.0, 31.0],
        [10.0, 20.0, 30.0],
    ]
    assert dataset.get_alignment_angle(sample_id) == 90.0
    aligned = dataset.get_point_data(sample_id, align_to_axes=True)
    assert np.allclose(
        aligned.xyz,
        [[10.0, 22.0, 30.0], [11.0, 21.0, 31.0], [12.0, 20.0, 32.0]],
    )
    with pytest.raises(DataFormatError, match="No official alignment angle"):
        dataset.get_alignment_angle("Area_1/office_2")


def test_raw_room_file_tolerates_official_copyroom_typo(tmp_path):
    room = _write_room(
        tmp_path,
        "Area_6",
        "copyRoom_1",
        {"floor_1": "0 0 0 1 2 3\n"},
        raw_points="0 0 0 1 2 3\n",
    )
    (room / "copyRoom_1.txt").rename(room / "copy_Room_1.txt")
    points = S3DIS(tmp_path).get_raw_room_points("Area_6/copyRoom_1")
    assert points.tolist() == [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]


def test_s3dis_reports_malformed_xyzrgb(s3dis_root):
    path = (
        s3dis_root
        / "Stanford3dDataset_v1.2_Aligned_Version"
        / "Area_1"
        / "office_2"
        / "Annotations"
        / "chair_1.txt"
    )
    path.write_text("0 1 broken 2 3 4\n")
    with pytest.raises(DataFormatError, match="Cannot parse S3DIS XYZRGB"):
        S3DIS(s3dis_root).get_point_data("Area_1/office_2")
