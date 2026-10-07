"""Small readers for the text form of COLMAP camera models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset.base import DataFormatError


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    intrinsic: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True, slots=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    world_to_camera: np.ndarray


def _quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def read_colmap_cameras(path: str | Path) -> dict[int, ColmapCamera]:
    """Read COLMAP ``cameras.txt`` without importing COLMAP."""

    result: dict[int, ColmapCamera] = {}
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            try:
                camera_id = int(fields[0])
                model = fields[1]
                width, height = int(fields[2]), int(fields[3])
                params = np.asarray(fields[4:], dtype=np.float64)
            except (IndexError, ValueError) as exc:
                raise DataFormatError(
                    f"Invalid COLMAP camera at {path}:{line_number}."
                ) from exc

            if model == "SIMPLE_PINHOLE" and len(params) == 3:
                fx = fy = params[0]
                cx, cy = params[1:3]
                distortion = np.empty((0,), dtype=np.float64)
            elif model == "PINHOLE" and len(params) == 4:
                fx, fy, cx, cy = params
                distortion = np.empty((0,), dtype=np.float64)
            elif model in {"OPENCV", "OPENCV_FISHEYE"} and len(params) == 8:
                fx, fy, cx, cy = params[:4]
                distortion = params[4:].copy()
            else:
                raise DataFormatError(
                    f"Unsupported COLMAP camera model {model!r} with {len(params)} parameters."
                )
            intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            result[camera_id] = ColmapCamera(
                camera_id, model, width, height, intrinsic, distortion
            )
    if not result:
        raise DataFormatError(f"No cameras found in {path}.")
    return result


def read_colmap_images(path: str | Path) -> dict[str, ColmapImage]:
    """Read only registered image poses from COLMAP ``images.txt``."""

    result: dict[str, ColmapImage] = {}
    with open(path) as handle:
        lines = iter(enumerate(handle, 1))
        for line_number, line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            try:
                image_id = int(fields[0])
                qvec = np.asarray(fields[1:5], dtype=np.float64)
                translation = np.asarray(fields[5:8], dtype=np.float64)
                camera_id = int(fields[8])
                name = " ".join(fields[9:])
                rotation = _quaternion_to_rotation(qvec)
            except (IndexError, ValueError) as exc:
                raise DataFormatError(
                    f"Invalid COLMAP image at {path}:{line_number}."
                ) from exc
            if not name or name in result:
                raise DataFormatError(
                    f"Missing or duplicate COLMAP image name at {path}:{line_number}."
                )
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
            result[name] = ColmapImage(image_id, camera_id, name, matrix)
            # Every registered-image record is followed by one POINTS2D line,
            # which may legitimately be empty.
            next(lines, None)
    if not result:
        raise DataFormatError(f"No registered images found in {path}.")
    return result


def read_colmap_sparse_points(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point IDs, XYZ coordinates, and uint8 RGB from ``points3D.txt``."""

    point_ids, points, colors = [], [], []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            try:
                point_ids.append(int(fields[0]))
                points.append([float(value) for value in fields[1:4]])
                colors.append([int(value) for value in fields[4:7]])
            except (IndexError, ValueError) as exc:
                raise DataFormatError(
                    f"Invalid COLMAP point at {path}:{line_number}."
                ) from exc
    return (
        np.asarray(point_ids, dtype=np.int64),
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
    )
