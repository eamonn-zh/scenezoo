from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scenezoo.dataset import UnsupportedOperationError
from scenezoo.ops.frame import (
    apply_image_transform,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    transform_world_to_camera,
    validate_frame_items,
)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotation_homography_matches_image(rotation):
    image = np.zeros((3, 4), dtype=np.uint8)
    image[1, 2] = 255
    transform = build_pixel_transform((4, 3), rotation=rotation)
    output = apply_image_transform(image, transform, resample=Image.Resampling.NEAREST)
    expected_xy = transform.matrix @ np.array([2, 1, 1], dtype=np.float64)
    y, x = np.argwhere(output == 255)[0]
    assert (x, y) == pytest.approx(expected_xy[:2])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("center_crop", [False, True])
def test_rotated_intrinsics_stay_pinhole_and_project_identically(rotation, center_crop):
    transform = build_pixel_transform(
        (640, 480), output_size=(300, 200), rotation=rotation, center_crop=center_crop
    )
    intrinsic = np.array([[500, 0, 310], [0, 490, 245], [0, 0, 1]], dtype=np.float64)
    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = [0.1, -0.2, 0.3]
    points = np.random.default_rng(0).uniform([-1, -1, 2], [1, 1, 4], (20, 3))

    rotated_k = transform_intrinsics(intrinsic, transform)
    rotated_pose = transform_world_to_camera(world_to_camera, transform)
    assert rotated_k[0, 1] == 0 and np.allclose(rotated_k[1:, 0], 0)
    assert rotated_k[0, 0] > 0 and rotated_k[1, 1] > 0

    def project(k, pose):
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        pixels = camera @ k.T
        return pixels[:, :2] / pixels[:, 2:]

    source = project(intrinsic, world_to_camera)
    expected = (np.c_[source, np.ones(len(source))] @ transform.matrix.T)[:, :2]
    assert np.allclose(project(rotated_k, rotated_pose), expected)
    assert np.allclose(
        transform_world_to_camera(world_to_camera, rotation), rotated_pose
    )


def test_resize_and_center_crop_share_one_homography():
    direct = build_pixel_transform((8, 4), output_size=(4, 4))
    assert np.allclose(direct.matrix, [[0.5, 0, -0.25], [0, 1, 0], [0, 0, 1]])
    cropped = build_pixel_transform((8, 4), output_size=(4, 4), center_crop=True)
    assert cropped.resized_size == (8, 4)
    assert cropped.crop_offset == (2, 0)
    assert np.allclose(cropped.matrix, [[1, 0, -2], [0, 1, 0], [0, 0, 1]])


@pytest.mark.parametrize("output_size", [(4, 3), (16, 12), (10, 3)])
def test_resize_homography_uses_pixel_centres(output_size):
    # A horizontal ramp stores each pixel centre's x coordinate. Resampling a
    # linear image is exact, so outputs reveal PIL's coordinate convention.
    ramp = np.tile(np.arange(8, dtype=np.float32), (6, 1))
    transform = build_pixel_transform((8, 6), output_size=output_size)
    output = apply_image_transform(ramp, transform, resample=Image.Resampling.BILINEAR)
    inverse = np.linalg.inv(transform.matrix)
    x = np.arange(output_size[0])
    expected = inverse[0, 0] * x + inverse[0, 2]
    # Exclude pixels whose (antialiasing) filter footprint is clipped by edges.
    support = max(1.0, inverse[0, 0])
    interior = (expected - support >= 0) & (expected + support <= 7)
    assert interior.sum() >= 2
    assert np.allclose(output[1, interior], expected[interior], atol=1e-4)


def test_nearest_neighbor_preserves_discrete_values():
    image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    transform = build_pixel_transform((2, 2), output_size=(7, 5))
    result = apply_image_transform(image, transform, resample=Image.Resampling.NEAREST)
    assert result.shape == (5, 7)
    assert set(np.unique(result)) == {0, 1, 2, 3}


def test_frame_selection_uses_raw_indices_and_preserves_order():
    selected, positions = select_frame_indices([0, 2, 5, 9], step=2)
    assert selected.tolist() == [0, 5]
    assert positions.tolist() == [0, 2]
    selected, positions = select_frame_indices([0, 2, 5, 9], indices=[9, 2])
    assert selected.tolist() == [9, 2]
    assert positions.tolist() == [3, 1]
    with pytest.raises(ValueError, match="step must be 1"):
        select_frame_indices([0, 1], indices=[0], step=2)
    with pytest.raises(ValueError, match="not synchronized"):
        select_frame_indices([0, 2], indices=[1])


def test_frame_items_distinguish_unknown_from_unsupported():
    with pytest.raises(ValueError, match="not one string"):
        validate_frame_items("rgb", {"rgb"})
    with pytest.raises(ValueError, match="Unknown"):
        validate_frame_items(("colour",), {"rgb"})
    with pytest.raises(UnsupportedOperationError):
        validate_frame_items(("depth",), {"rgb"})


def test_fractional_sizes_indices_and_steps_are_rejected():
    with pytest.raises(ValueError, match="integers"):
        build_pixel_transform((4, 3), output_size=(2.5, 2))
    with pytest.raises(ValueError, match="contain integers"):
        select_frame_indices([0, 1], indices=[0.5])
    with pytest.raises(ValueError, match="positive integer"):
        select_frame_indices([0, 1], step=1.5)
