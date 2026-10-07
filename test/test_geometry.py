import numpy as np
import pytest

from scenezoo.ops.geometry import construct_box, load_triangle_mesh


def test_box_errors_are_deterministic():
    with pytest.raises(ValueError, match="non-empty"):
        construct_box(np.empty((0, 3)))
    with pytest.raises(ValueError, match="Unknown box type"):
        construct_box(np.zeros((1, 3)), box_type="typo")
    with pytest.raises(ValueError, match="non-collinear"):
        construct_box(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), box_type="obb")


def test_aabb_accepts_a_single_point():
    box = construct_box(np.array([[1.0, 2.0, 3.0]]), box_type="aabb")
    assert np.asarray(box.get_min_bound()).tolist() == [1.0, 2.0, 3.0]


def test_mesh_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_triangle_mesh(tmp_path / "missing.ply")
