import numpy as np
import pytest

from scenezoo.dataset import DataFormatError
from scenezoo.ops.annotation import parse_sstk_segmentation


@pytest.mark.parametrize("domain", ["vertex", "face"])
def test_sstk_segmentation_domain_remove_and_invalid(domain):
    result = parse_sstk_segmentation(
        {"segIndices": [0, 1, 2, 1]},
        {
            "segGroups": [
                {"objectId": 4, "label": "chair", "segments": [0, 2]},
                {"objectId": 7, "label": "remove", "segments": [1]},
            ]
        },
        invalid_id=-1,
        domain=domain,
        skip_remove_label=True,
    )
    assert result.labels.tolist() == [4, -1, 4, -1]
    assert result.id_to_name == {4: "chair"}
    assert result.domain == domain


def test_sstk_segmentation_empty_and_invalid_reference():
    empty = parse_sstk_segmentation({"segIndices": []}, {"segGroups": []}, invalid_id=0)
    assert empty.labels.shape == (0,)
    with pytest.raises(DataFormatError, match="unknown segments"):
        parse_sstk_segmentation(
            {"segIndices": [0]},
            {"segGroups": [{"objectId": 1, "label": "x", "segments": [3]}]},
        )


@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("faces", [[[0, 1, 2], [1, 2, 3]], [[0, 1, 2, 3], [1, 2, 3]]])
def test_read_ply_matches_plyfile_for_triangles_and_polygons(tmp_path, text, faces):
    from plyfile import PlyData, PlyElement

    from scenezoo.ops.annotation import read_ply

    vertex = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    face = np.empty(len(faces), dtype=[("vertex_indices", "O"), ("label", "i4")])
    face["vertex_indices"] = [np.asarray(value, dtype="i4") for value in faces]
    face["label"] = [7, 9]
    path = tmp_path / "mesh.ply"
    PlyData(
        [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")],
        text=text,
    ).write(path)
    expected = PlyData.read(path)["face"]
    actual = read_ply(path)["face"]
    assert actual["label"].tolist() == expected["label"].tolist() == [7, 9]
    assert [list(row) for row in actual["vertex_indices"]] == faces


def test_group_and_remap_labels():
    from scenezoo.ops.annotation import group_indices, remap_labels

    labels = np.array([5, -1, 3, 5, 3, 3])
    groups = group_indices(labels)
    assert list(groups) == [-1, 3, 5]
    assert {key: value.tolist() for key, value in groups.items()} == {
        -1: [1],
        3: [2, 4, 5],
        5: [0, 3],
    }
    multilabel = np.array([[5, -1], [3, 5]])
    remapped = remap_labels(multilabel, {5: 50, 3: 30}, default=-100)
    assert remapped.dtype == np.int32
    assert remapped.tolist() == [[50, -100], [30, 50]]


def test_sstk_objects_may_share_segments():
    from scenezoo.ops.annotation import sstk_object_indices

    vertex_segments = np.array([0, 1, 1, 2, 0, 2])
    objects = {
        4: {"segments": np.array([1, 0, 1])},  # Repeated segments are ignored.
        8: {"segments": np.array([1, 2, 99])},  # Unknown segments contribute nothing.
    }
    members = sstk_object_indices(vertex_segments, objects)
    assert members[4].tolist() == [0, 1, 2, 4]
    assert members[8].tolist() == [1, 2, 3, 5]
