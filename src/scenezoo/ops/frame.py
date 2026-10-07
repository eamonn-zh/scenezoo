"""Frame selection and coupled image/camera transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

FRAME_ITEMS = frozenset(
    {
        "timestamps",
        "rgb",
        "depth",
        "rgb_intrinsics",
        "depth_intrinsics",
        "world_to_camera",
        "instance_maps",
        "semantic_maps",
        "confidence_maps",
        "albedo",
        "normal_maps",
        "frame_mask",
        "exposure_durations",
        "imu",
        "is_bad",
        "azimuth",
        "elevation",
    }
)


def validate_frame_items(items: Sequence[str], supported: set[str]) -> tuple[str, ...]:
    """Validate and normalize requested :class:`FrameBatch` payload fields."""

    if isinstance(items, str):
        raise ValueError("items must be a sequence of field names, not one string.")
    normalized = tuple(items)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Frame items must not contain duplicates.")
    unknown = set(normalized) - FRAME_ITEMS
    if unknown:
        raise ValueError(f"Unknown frame items: {sorted(unknown)}")
    unavailable = set(normalized) - supported
    if unavailable:
        from ..dataset.base import UnsupportedOperationError

        raise UnsupportedOperationError(
            f"Frame items are not supported by this dataset: {sorted(unavailable)}"
        )
    return normalized


def valid_camera_mask(matrices: np.ndarray) -> np.ndarray:
    """Return rows containing finite, invertible 4x4 camera matrices."""

    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("Camera matrices must have shape (N, 4, 4).")
    valid = np.isfinite(matrices).all(axis=(1, 2))
    if valid.any():
        positions = np.flatnonzero(valid)
        valid[positions] &= np.abs(np.linalg.det(matrices[positions, :3, :3])) > 1e-8
    return valid


@dataclass(frozen=True, slots=True)
class PixelTransform:
    """A source-to-output pixel homography and its output image size.

    Pixel coordinates follow the OpenCV convention in which integer
    coordinates are pixel centres.
    """

    matrix: np.ndarray
    output_size: tuple[int, int]
    rotation: int
    resized_size: tuple[int, int]
    crop_offset: tuple[int, int]

    @property
    def camera_rotation(self) -> np.ndarray:
        """The 3x3 camera-frame rotation equivalent to the image rotation.

        Rotating an image by a right angle is the same as rolling the camera
        about its optical axis. Applying this rotation to camera coordinates
        keeps intrinsics in upper-triangular pinhole form.
        """

        return _CAMERA_ROLL[self.rotation].copy()


# Camera-frame rolls matching counter-clockwise image rotations. The 2x2 blocks
# equal the linear part of the corresponding pixel rotations.
_CAMERA_ROLL = {
    0: np.eye(3, dtype=np.float64),
    90: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),
    180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64),
    270: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
}


def build_pixel_transform(
    original_size: Sequence[int],
    *,
    output_size: Sequence[int] | None = None,
    rotation: int = 0,
    center_crop: bool = False,
) -> PixelTransform:
    """Build the exact transform applied by :func:`apply_image_transform`.

    Rotation is counter-clockwise and restricted to right angles. Rotation is
    applied before resize/crop.
    """

    def image_size(values, name):
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly (width, height).")
        converted = tuple(int(value) for value in values)
        if any(float(value) != converted[index] for index, value in enumerate(values)):
            raise ValueError(f"{name} values must be integers.")
        return converted

    source_width, source_height = image_size(original_size, "original_size")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if float(rotation) != int(rotation):
        raise ValueError("rotation must be an integer number of degrees.")
    rotation = int(rotation) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees.")

    if rotation == 0:
        rotation_matrix = np.eye(3, dtype=np.float64)
        rotated_size = (source_width, source_height)
    elif rotation == 90:
        rotation_matrix = np.array(
            [[0, 1, 0], [-1, 0, source_width - 1], [0, 0, 1]],
            dtype=np.float64,
        )
        rotated_size = (source_height, source_width)
    elif rotation == 180:
        rotation_matrix = np.array(
            [[-1, 0, source_width - 1], [0, -1, source_height - 1], [0, 0, 1]],
            dtype=np.float64,
        )
        rotated_size = (source_width, source_height)
    else:
        rotation_matrix = np.array(
            [[0, -1, source_height - 1], [1, 0, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        rotated_size = (source_height, source_width)

    if output_size is None:
        target_width, target_height = rotated_size
        resized_width, resized_height = rotated_size
        crop_left = crop_top = 0
    else:
        target_width, target_height = image_size(output_size, "output_size")
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Output image dimensions must be positive.")
        rotated_width, rotated_height = rotated_size
        if center_crop:
            scale = max(
                target_width / rotated_width,
                target_height / rotated_height,
            )
            resized_width = int(round(rotated_width * scale))
            resized_height = int(round(rotated_height * scale))
            crop_left = (resized_width - target_width) // 2
            crop_top = (resized_height - target_height) // 2
        else:
            resized_width, resized_height = target_width, target_height
            crop_left = crop_top = 0

    rotated_width, rotated_height = rotated_size
    scale_x = resized_width / rotated_width
    scale_y = resized_height / rotated_height
    # Resampling aligns pixel edges, so centred integer coordinates also shift
    # by half a pixel whenever the image is scaled.
    resize_crop_matrix = np.array(
        [
            [scale_x, 0, 0.5 * (scale_x - 1) - crop_left],
            [0, scale_y, 0.5 * (scale_y - 1) - crop_top],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return PixelTransform(
        matrix=resize_crop_matrix @ rotation_matrix,
        output_size=(target_width, target_height),
        rotation=rotation,
        resized_size=(resized_width, resized_height),
        crop_offset=(crop_left, crop_top),
    )


def apply_image_transform(
    image: np.ndarray,
    transform: PixelTransform,
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> np.ndarray:
    """Apply rotation, resize, and crop without changing channel semantics."""

    array = np.asarray(image)
    if transform.rotation:
        array = np.rot90(array, k=transform.rotation // 90)
    height, width = array.shape[:2]
    if (width, height) == transform.resized_size == transform.output_size:
        return np.ascontiguousarray(array)  # Nothing to resample or crop.
    pil_image = Image.fromarray(array)
    if pil_image.size != transform.resized_size:
        pil_image = pil_image.resize(transform.resized_size, resample=resample)
    left, top = transform.crop_offset
    if (left, top) != (0, 0) or pil_image.size != transform.output_size:
        width, height = transform.output_size
        pil_image = pil_image.crop((left, top, left + width, top + height))
    return np.asarray(pil_image)


def apply_transform_batch(
    images: np.ndarray,
    transform: PixelTransform,
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> np.ndarray:
    images = np.asarray(images)
    if len(images) == 0:
        width, height = transform.output_size
        trailing = images.shape[3:] if images.ndim > 3 else ()
        return np.empty((0, height, width, *trailing), dtype=images.dtype)
    return np.stack(
        [
            apply_image_transform(image, transform, resample=resample)
            for image in images
        ],
        axis=0,
    )


def transform_intrinsics(
    intrinsics: np.ndarray,
    transform: PixelTransform,
) -> np.ndarray:
    """Return pinhole intrinsics for images produced by ``transform``.

    Right-angle rotations are expressed as a camera roll, so the result stays
    upper triangular. Poses must be rotated with :func:`transform_world_to_camera`
    using the same transform.
    """

    matrices = np.asarray(intrinsics)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("Camera intrinsics must end with shape (3, 3).")
    pixel = transform.matrix.astype(matrices.dtype, copy=False)
    roll = transform.camera_rotation.astype(matrices.dtype, copy=False)
    return pixel @ matrices @ roll.T


def transform_world_to_camera(
    world_to_camera: np.ndarray,
    transform: PixelTransform | int,
) -> np.ndarray:
    """Apply the camera roll of a transform (or rotation angle) to poses."""

    matrices = np.asarray(world_to_camera)
    if matrices.shape[-2:] != (4, 4):
        raise ValueError("Camera poses must end with shape (4, 4).")
    rotation = (
        transform.rotation if isinstance(transform, PixelTransform) else transform
    )
    rotation = int(rotation) % 360
    if rotation not in _CAMERA_ROLL:
        raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees.")
    if rotation == 0:
        return matrices
    roll = np.eye(4, dtype=matrices.dtype)
    roll[:3, :3] = _CAMERA_ROLL[rotation]
    return roll @ matrices


def select_frame_indices(
    available_indices: Sequence[int] | np.ndarray,
    *,
    indices: Sequence[int] | np.ndarray | None = None,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve raw frame IDs and their positions in synchronized metadata."""

    def integer_array(values, name):
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence.")
        if raw.size == 0:
            return raw.astype(np.int64)
        if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
            raw.dtype, np.bool_
        ):
            raise ValueError(f"{name} must contain integers.")
        return raw.astype(np.int64, copy=False)

    available = integer_array(available_indices, "available_indices")
    if available.ndim != 1 or len(np.unique(available)) != len(available):
        raise ValueError("available_indices must be a unique one-dimensional sequence.")
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
        raise ValueError("step must be a positive integer.")
    if step <= 0:
        raise ValueError("step must be a positive integer.")
    if indices is None:
        positions = np.arange(0, len(available), step, dtype=np.int64)
        return available[positions], positions
    if step != 1:
        raise ValueError("step must be 1 when explicit indices are provided.")
    requested = integer_array(indices, "indices")
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise ValueError("indices must be a unique one-dimensional sequence.")
    position_by_id = {
        int(frame_id): position for position, frame_id in enumerate(available)
    }
    missing = [
        int(frame_id) for frame_id in requested if int(frame_id) not in position_by_id
    ]
    if missing:
        raise ValueError(
            "Frames are not synchronized or valid: " + ", ".join(map(str, missing))
        )
    positions = np.fromiter(
        (position_by_id[int(frame_id)] for frame_id in requested),
        dtype=np.int64,
        count=len(requested),
    )
    return requested, positions
