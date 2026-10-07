"""Small, copy-free return types shared by dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(slots=True)
class PointBatch:
    """Aligned point attributes with optional source-view provenance."""

    xyz: np.ndarray
    rgb: np.ndarray | None = None
    normals: np.ndarray | None = None
    semantic_labels: np.ndarray | None = None
    instance_labels: np.ndarray | None = None
    view_ids: np.ndarray | None = None
    view_keys: tuple[str, ...] = ()
    semantic_id_to_name: dict[int, str] = field(default_factory=dict)
    instance_id_to_name: dict[int, str] = field(default_factory=dict)
    invalid_id: int = -1

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1:] != (3,):
            raise ValueError("PointBatch.xyz must have shape (N, 3).")
        count = len(self.xyz)
        for name, dtype, trailing in (
            ("rgb", np.uint8, (3,)),
            ("normals", np.float32, (3,)),
            ("semantic_labels", np.int32, ()),
            ("instance_labels", np.int32, ()),
            ("view_ids", np.int32, ()),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=dtype)
            if array.shape != (count, *trailing):
                raise ValueError(
                    f"PointBatch.{name} must have shape {(count, *trailing)}."
                )
            setattr(self, name, array)
        self.view_keys = tuple(str(key) for key in self.view_keys)
        if self.view_ids is not None and len(self.view_ids):
            if np.any(self.view_ids < 0) or int(self.view_ids.max()) >= len(
                self.view_keys
            ):
                raise ValueError("PointBatch.view_ids contains an unknown view index.")

    def __len__(self) -> int:
        return len(self.xyz)

    def to_point_cloud(self):
        """Create an Open3D point cloud without importing it at package load."""

        import open3d as o3d

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz))
        if self.rgb is not None:
            cloud.colors = o3d.utility.Vector3dVector(
                self.rgb.astype(np.float64) / 255.0
            )
        if self.normals is not None:
            cloud.normals = o3d.utility.Vector3dVector(self.normals)
        return cloud


@dataclass(slots=True)
class Segmentation3D:
    """Labels defined on point-cloud points, mesh vertices, or mesh faces.

    ``labels`` is normally one-dimensional. A two-dimensional array represents
    padded multilabel annotations; entries equal to ``invalid_id`` are padding.
    """

    labels: np.ndarray
    id_to_name: dict[int, str]
    domain: Literal["point", "vertex", "face"]
    invalid_id: int

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)
        if self.labels.ndim not in {1, 2}:
            raise ValueError(
                "Segmentation labels must be a one- or two-dimensional array."
            )
        if self.domain not in {"point", "vertex", "face"}:
            raise ValueError(
                "Segmentation domain must be 'point', 'vertex', or 'face'."
            )

    @property
    def is_multilabel(self) -> bool:
        return self.labels.ndim == 2


@dataclass(slots=True)
class FrameBatch:
    """Synchronized frame data.

    All non-``None`` per-frame arrays have ``len(indices)`` as their first
    dimension. Depth is represented as float32 metres and camera poses are
    OpenCV-style world-to-camera matrices. ``camera_model`` and ``distortion``
    retain non-pinhole camera semantics rather than silently discarding them.
    ``depth_mode`` distinguishes conventional camera-Z depth from distance along
    a camera ray when a source dataset makes that distinction.
    ``instance_maps``, ``semantic_maps``, and ``confidence_maps`` contain
    discrete IDs and are always transformed with nearest-neighbor resampling by
    dataset adapters. ``normal_maps`` contains decoded unit-vector components,
    while ``albedo`` is an RGB image-like payload. ``exposure_durations``
    retains the source dataset's time unit rather than guessing or converting
    it.
    """

    indices: np.ndarray
    source_frame_count: int
    frame_keys: tuple[str, ...] | None = None
    camera_model: str | None = None
    depth_mode: str | None = None
    distortion: np.ndarray | None = None
    image_sizes: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    rgb_intrinsics: np.ndarray | None = None
    depth_intrinsics: np.ndarray | None = None
    world_to_camera: np.ndarray | None = None
    instance_maps: np.ndarray | None = None
    semantic_maps: np.ndarray | None = None
    confidence_maps: np.ndarray | None = None
    albedo: np.ndarray | None = None
    normal_maps: np.ndarray | None = None
    frame_mask: np.ndarray | None = None
    exposure_durations: np.ndarray | None = None
    imu: dict[str, np.ndarray] | None = None
    is_bad: np.ndarray | None = None
    azimuth: np.ndarray | None = None
    elevation: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError("Frame indices must be a one-dimensional array.")
        if self.source_frame_count < 0:
            raise ValueError("source_frame_count must be non-negative.")
        if np.any(self.indices < 0) or (
            len(self.indices) and int(self.indices.max()) >= self.source_frame_count
        ):
            raise ValueError("Frame indices are outside the source frame range.")
        if self.frame_keys is not None:
            self.frame_keys = tuple(str(key) for key in self.frame_keys)
            if len(self.frame_keys) != len(self.indices):
                raise ValueError("frame_keys must have one entry per frame.")
        if self.camera_model is not None:
            self.camera_model = str(self.camera_model).upper()
        if self.depth_mode is not None:
            self.depth_mode = str(self.depth_mode).lower()
        for name in (
            "distortion",
            "image_sizes",
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
            "is_bad",
            "azimuth",
            "elevation",
        ):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                setattr(self, name, value)
            if value is not None and len(value) != len(self.indices):
                raise ValueError(
                    f"FrameBatch.{name} has {len(value)} entries; "
                    f"expected {len(self.indices)}."
                )
        if self.distortion is not None and self.distortion.ndim != 2:
            raise ValueError("distortion must have shape (N, D).")
        if self.image_sizes is not None and self.image_sizes.shape != (len(self), 2):
            raise ValueError("image_sizes must have shape (N, 2).")
        if self.rgb_intrinsics is not None and self.rgb_intrinsics.shape[1:] != (3, 3):
            raise ValueError("rgb_intrinsics must have shape (N, 3, 3).")
        if self.depth_intrinsics is not None and self.depth_intrinsics.shape[1:] != (
            3,
            3,
        ):
            raise ValueError("depth_intrinsics must have shape (N, 3, 3).")
        if self.world_to_camera is not None and self.world_to_camera.shape[1:] != (
            4,
            4,
        ):
            raise ValueError("world_to_camera must have shape (N, 4, 4).")
        if self.depth is not None:
            self.depth = self.depth.astype(np.float32, copy=False)
            invalid = ~np.isfinite(self.depth) | (self.depth < 0)
            if invalid.any():
                self.depth = self.depth.copy()
                self.depth[invalid] = 0.0
        if self.imu is not None:
            converted = {}
            for name, value in self.imu.items():
                array = np.asarray(value)
                if len(array) != len(self):
                    raise ValueError(f"imu[{name!r}] must have one entry per frame.")
                converted[str(name)] = array
            self.imu = converted

    def __len__(self) -> int:
        return len(self.indices)
