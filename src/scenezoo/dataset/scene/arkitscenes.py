"""ARKitScenes scans, mobile RGB-D streams, and FARO captures."""

from __future__ import annotations

import csv
import json
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

from ...io.image import read_images
from ...io.video import (
    read_video_frames,
    read_video_frame_size,
)
from ...ops.frame import (
    apply_transform_batch,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    transform_world_to_camera,
    valid_camera_mask,
    validate_frame_items,
)
from ...ops.geometry import load_triangle_mesh
from ..base import DataFormatError, Dataset, UnsupportedOperationError
from ..registry import register_dataset
from ..types import FrameBatch, Segmentation3D


ROTATION_BY_SKY_DIRECTION = {"Up": 0, "Left": 270, "Right": 90, "Down": 180}
FRAME_SOURCES = (
    "mov",
    "lowres_wide",
    "wide",
    "ultrawide",
    "vga_wide",
    "threedod",
)
IMAGE_SOURCES = {
    "lowres_wide": ("lowres_wide",),
    "wide": ("wide", "color"),
    "ultrawide": ("ultrawide",),
    "vga_wide": ("vga_wide",),
}
INTRINSIC_SOURCE = {
    "mov": "lowres_wide_intrinsics",
    "lowres_wide": "lowres_wide_intrinsics",
    "wide": "wide_intrinsics",
    "ultrawide": "ultrawide_intrinsics",
    "vga_wide": "vga_wide_intrinsics",
}
DEPTH_DIRECTORY = {"lowres": "lowres_depth", "highres": "highres_depth"}
NATIVE_IMAGE_SIZES = {
    "lowres_wide": (256, 192),
    "wide": (1920, 1440),
    "ultrawide": (640, 480),
    "vga_wide": (640, 480),
    "lowres_depth": (256, 192),
    "highres_depth": (1920, 1440),
    "confidence": (256, 192),
}


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise DataFormatError(f"Invalid ARKitScenes boolean {field}={value!r}.")


def _timestamp_from_path(path: Path) -> float:
    try:
        return float(path.stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise DataFormatError(f"Invalid ARKitScenes frame filename: {path}") from exc


@register_dataset(
    "arkitscenes",
    description="ARKitScenes scans, mobile RGB-D streams, and FARO captures.",
)
class ARKitScenes(Dataset):
    """Adapter for the official raw, 3DOD, and depth-upsample assets.

    ``root_dir`` may point either at an ARKitScenes download root or directly at
    its ``raw`` directory. The packaged, human-reviewed direction corrections
    are applied on top of Apple's metadata and remain authoritative.
    """

    def __init__(
        self,
        root_dir,
        *,
        data_dir=None,
        metadata_file="metadata.csv",
        mesh_file="{scene_id}/{scene_id}_3dod_mesh.ply",
        rgb_video_file="{scene_id}/{scene_id}.mov",
        trajectory_file="{scene_id}/lowres_wide.traj",
        annotation_file="{scene_id}/{scene_id}_3dod_annotation.json",
        laser_mapping_file="laser_scanner_point_clouds_mapping.csv",
        laser_scanner_dir="laser_scanner_point_clouds/{visit_id}",
        depth_upsampling_attributes_file="val_attributes.csv",
        threedod_split_file="https://raw.githubusercontent.com/apple/ARKitScenes/refs/heads/main/threedod/3dod_train_val_splits.csv",
        raw_split_file="https://raw.githubusercontent.com/apple/ARKitScenes/refs/heads/main/raw/raw_train_val_splits.csv",
        depth_upsampling_split_file="https://raw.githubusercontent.com/apple/ARKitScenes/refs/heads/main/depth_upsampling/upsampling_train_val_splits.csv",
        timestamp_tolerance=0.005,
        invalid_obj_id=-1,
        cache_dir=None,
        offline=False,
    ):
        super().__init__(
            root_dir,
            cache_dir=cache_dir,
            offline=offline,
            invalid_obj_id=invalid_obj_id,
        )
        if timestamp_tolerance < 0:
            raise ValueError("timestamp_tolerance must be non-negative.")
        if data_dir is None:
            candidate = self.root_dir / "raw"
            self.data_root = (
                candidate if (candidate / metadata_file).is_file() else self.root_dir
            )
        else:
            self.data_root = self.root_dir / data_dir
        self.dataset_root = (
            self.data_root.parent if self.data_root.name == "raw" else self.root_dir
        )
        depth_candidate = self.dataset_root / "depth_upsampling"
        self.depth_upsampling_root = (
            depth_candidate if depth_candidate.is_dir() else self.data_root
        )
        self.metadata_file = metadata_file
        self.mesh_file = mesh_file
        self.rgb_video_file = rgb_video_file
        self.trajectory_file = trajectory_file
        self.annotation_file = annotation_file
        self.laser_mapping_file = laser_mapping_file
        self.laser_scanner_dir = laser_scanner_dir
        self.depth_upsampling_attributes_file = depth_upsampling_attributes_file
        self.timestamp_tolerance = float(timestamp_tolerance)
        self.split_files = {
            "threedod": threedod_split_file,
            "raw": raw_split_file,
            "depth_upsampling": depth_upsampling_split_file,
        }

    @property
    def _metadata_path(self) -> Path:
        return self.data_root / self.metadata_file

    def _load_splits(self):
        if self._metadata_path.is_file():
            result = {}
            groups = {
                "raw": lambda _: True,
                "threedod": lambda row: row["is_in_threedod"],
                "depth_upsampling": lambda row: row["is_in_upsampling"],
            }
            for part, include in groups.items():
                for fold, name in (("Training", "train"), ("Validation", "val")):
                    result[f"{name}_{part}"] = [
                        sample_id
                        for sample_id, row in self.metadata.items()
                        if row["fold"] == fold and include(row)
                    ]
            return result

        result = {}
        for dataset_part, path in self.split_files.items():
            with self.open_file(path, "r") as handle:
                rows = list(csv.DictReader(handle))
            for fold, name in (("Training", "train"), ("Validation", "val")):
                result[f"{name}_{dataset_part}"] = [
                    str(row["video_id"]) for row in rows if row["fold"] == fold
                ]
        return result

    def _load_metadata(self):
        if not self._metadata_path.is_file():
            return {}
        result = {}
        with open(self._metadata_path) as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row["video_id"])
                visit = row.get("visit_id", "").strip()
                official_direction = row["sky_direction"].strip()
                layout = self.data_root.name

                def subset_flag(field, subset):
                    value = row.get(field, "").strip()
                    return (
                        _parse_bool(value, field=field) if value else layout == subset
                    )

                result[sample_id] = {
                    "video_id": sample_id,
                    "visit_id": None if visit in {"", "NA"} else int(float(visit)),
                    "fold": row["fold"].strip(),
                    "official_sky_direction": official_direction,
                    "sky_direction": official_direction,
                    "has_direction_correction": False,
                    "has_laser_scanner_point_clouds": _parse_bool(
                        row["has_laser_scanner_point_clouds"],
                        field="has_laser_scanner_point_clouds",
                    )
                    if row.get("has_laser_scanner_point_clouds", "").strip()
                    else False,
                    "is_in_upsampling": subset_flag(
                        "is_in_upsampling", "depth_upsampling"
                    ),
                    "is_in_threedod": subset_flag("is_in_threedod", "threedod"),
                }
        correction_path = files("scenezoo.metadata") / "fixed_arkitscenes_direction.csv"
        with open(correction_path) as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row["video_id"])
                if sample_id in result:
                    result[sample_id]["sky_direction"] = row["sky_direction"].strip()
                    result[sample_id]["has_direction_correction"] = True
        return result

    def _check(self, *, sample_ids, require_complete):
        from ..check import (
            CheckBuilder,
            archive_candidates,
            check_scene_coverage,
            load_expected_ids,
        )

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The ARKitScenes root does not exist.",
            expected="<root>/raw/{metadata.csv,Training/,Validation/}",
            hint="Pass either the full ARKitScenes root or its raw directory.",
        ):
            return builder.finish()
        if not builder.require_directory(
            self.data_root,
            code="missing-raw-directory",
            message="The configured ARKitScenes raw-data directory is missing.",
            expected="<raw>/metadata.csv and <raw>/{Training,Validation}/<video_id>/",
            hint="Extract the official raw download; ZIP files cannot be read in place.",
        ):
            return builder.finish()

        if not self._metadata_path.is_file():
            builder.warning(
                "missing-local-metadata",
                "metadata.csv is absent, so local subset attributes cannot be verified.",
                path=self._metadata_path,
                expected="raw/metadata.csv from the official release.",
                hint="Place metadata.csv beside the Training and Validation directories.",
            )
        scene_paths = {
            path.name: path
            for fold in ("Training", "Validation")
            if (self.data_root / fold).is_dir()
            for path in (self.data_root / fold).iterdir()
            if path.is_dir()
        }
        expected = None if sample_ids is not None else load_expected_ids(self, builder)
        selected = check_scene_coverage(
            builder,
            present_ids=scene_paths,
            expected_ids=expected,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<raw>/<Training|Validation>/<video_id>/<video_id>.mov",
        )
        if not scene_paths:
            builder.error(
                "no-scenes",
                "No ARKitScenes video directories were found.",
                path=self.data_root,
                expected="Training/<video_id> and/or Validation/<video_id> directories.",
                hint="Extract the raw ZIP shards while preserving the split directories.",
            )
            archives = archive_candidates(self.data_root) or archive_candidates(
                self.root_dir
            )
            if archives:
                builder.info(
                    "archives-need-extraction",
                    "Archive files are present; ARKitScenes archives must be extracted first.",
                )
            return builder.finish()

        for sid in selected:
            base = scene_paths[sid]
            required = [base / f"{sid}.mov", base / "lowres_wide.traj"]
            builder.missing_paths(
                (path for path in required if not path.is_file()),
                code="incomplete-raw-scene",
                label="raw capture files",
                expected="<video_id>.mov and lowres_wide.traj in every raw scene.",
                hint="Re-download/re-extract the raw video shard for this video ID.",
            )

        metadata = self.metadata if self._metadata_path.is_file() else {}
        corrections = sum(
            bool(metadata.get(sid, {}).get("has_direction_correction"))
            for sid in selected
        )
        builder.stats["selected_direction_corrections"] = corrections
        builder.info(
            "direction-corrections-respected",
            "Packaged SceneZoo video-direction corrections will be applied in addition "
            "to Apple's metadata when frames are read.",
        )
        upsampling = self.dataset_root / "depth_upsampling"
        if upsampling.exists() and not upsampling.is_dir():
            builder.error(
                "invalid-depth-upsampling-path",
                "depth_upsampling exists but is not a directory.",
                path=upsampling,
                expected="<root>/depth_upsampling/{Training,Validation}/<video_id>/...",
            )
        return builder.finish()

    @cached_property
    def _split_by_id(self):
        if self.metadata:
            return {sample_id: row["fold"] for sample_id, row in self.metadata.items()}
        return {
            sample_id: directory
            for split_name, directory in (
                ("train_raw", "Training"),
                ("val_raw", "Validation"),
            )
            for sample_id in self.splits[split_name]
        }

    def _sample_directory(self, sample_id: str, *, upsampling=False) -> Path:
        sample_id = str(sample_id)
        try:
            directory = self._split_by_id[sample_id]
        except KeyError as exc:
            raise KeyError(
                f"ARKitScenes sample {sample_id!r} is not in a raw split."
            ) from exc
        root = self.depth_upsampling_root if upsampling else self.data_root
        return root / directory / sample_id

    def _sample_path(self, sample_id, template):
        return self._sample_directory(sample_id) / template.format(
            scene_id=sample_id
        ).removeprefix(f"{sample_id}/")

    def _asset_directory(self, sample_id: str, names: Sequence[str]) -> Path:
        raw = self._sample_directory(sample_id)
        for name in names:
            path = raw / name
            if path.is_dir():
                return path
        upsampling = self._sample_directory(sample_id, upsampling=True)
        for name in names:
            path = upsampling / name
            if path.is_dir():
                return path
        return raw / names[0]

    # 3DOD assets ------------------------------------------------------------
    def get_mesh(self, sample_id, *, mesh_type=None):
        if mesh_type not in {None, "raw"}:
            raise ValueError(f"Unknown ARKitScenes mesh type: {mesh_type!r}")
        path = self._sample_path(sample_id, self.mesh_file)
        if not path.is_file():
            raise UnsupportedOperationError(
                f"ARKitScenes sample {sample_id!r} has no downloaded 3DOD mesh: {path}"
            )
        return load_triangle_mesh(path)

    def get_annotations(self, sample_id) -> dict:
        """Return the complete official 3DOD annotation JSON."""

        path = self._sample_path(sample_id, self.annotation_file)
        if not path.is_file():
            raise UnsupportedOperationError(
                f"ARKitScenes sample {sample_id!r} has no downloaded 3DOD annotation: {path}"
            )
        try:
            with open(path) as handle:
                value = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                f"Invalid ARKitScenes annotation JSON: {path}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(value.get("data"), list):
            raise DataFormatError(f"Invalid ARKitScenes annotation schema: {path}")
        return value

    def get_boxes(self, sample_id, *, box_type="obb_gt"):
        if box_type != "obb_gt":
            raise ValueError("ARKitScenes provides only 'obb_gt' boxes.")
        annotations = self.get_annotations(sample_id)["data"]
        boxes, names = {}, {}
        for object_id, item in enumerate(annotations):
            try:
                obb = item["segments"]["obbAligned"]
                center = np.asarray(obb["centroid"], dtype=np.float32)
                rotation = (
                    np.asarray(obb["normalizedAxes"], dtype=np.float32).reshape(3, 3).T
                )
                extent = np.asarray(obb["axesLengths"], dtype=np.float32)
                name = str(item["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataFormatError(
                    f"Invalid ARKitScenes OBB for object {object_id}."
                ) from exc
            boxes[object_id] = o3d.geometry.OrientedBoundingBox(
                center, rotation, extent
            )
            names[object_id] = name
        return boxes, names

    def get_segmentation(self, sample_id):
        """Return box-derived vertex labels, resolving overlaps by smaller OBB."""

        boxes, names = self.get_boxes(sample_id)
        vertices = self.get_mesh(sample_id).vertices
        labels = np.full(len(vertices), self.invalid_obj_id, dtype=np.int32)
        # Larger boxes are written first so a smaller, more specific instance
        # deterministically owns vertices inside overlapping released OBBs.
        ordered = sorted(
            boxes.items(),
            key=lambda value: float(np.prod(value[1].extent)),
            reverse=True,
        )
        for object_id, box in ordered:
            labels[box.get_point_indices_within_bounding_box(vertices)] = object_id
        return Segmentation3D(labels, names, "vertex", self.invalid_obj_id)

    # Frame indexes ----------------------------------------------------------
    @staticmethod
    def _sorted_files(directory: Path, suffix=".png") -> list[Path]:
        paths = list(directory.glob(f"*{suffix}"))
        return sorted(paths, key=_timestamp_from_path)

    def _intrinsic_files(self, sample_id: str, source: str) -> list[Path]:
        if source == "threedod":
            directory = (
                self._sample_directory(sample_id)
                / f"{sample_id}_frames"
                / "color_intrinsics"
            )
            return self._sorted_files(directory, ".pincam")
        directory = self._asset_directory(sample_id, (INTRINSIC_SOURCE[source],))
        return self._sorted_files(directory, ".pincam")

    def _source_frames(self, sample_id: str, source: str):
        if source == "mov":
            paths = self._intrinsic_files(sample_id, source)
            return np.asarray([_timestamp_from_path(path) for path in paths]), paths
        if source == "threedod":
            directory = (
                self._sample_directory(sample_id) / f"{sample_id}_frames" / "wide"
            )
            paths = self._sorted_files(directory)
            return np.asarray([_timestamp_from_path(path) for path in paths]), paths
        directory = self._asset_directory(sample_id, IMAGE_SOURCES[source])
        paths = self._sorted_files(directory)
        return np.asarray([_timestamp_from_path(path) for path in paths]), paths

    def _trajectory_path(self, sample_id, source):
        if source == "threedod":
            return (
                self._sample_directory(sample_id) / f"{sample_id}_frames" / "color.traj"
            )
        return self._sample_path(sample_id, self.trajectory_file)

    def _camera_records(self, sample_id, *, source="mov"):
        path = self._trajectory_path(sample_id, source)
        if not path.is_file():
            raise UnsupportedOperationError(
                f"ARKitScenes sample {sample_id!r} has no camera trajectory: {path}"
            )
        # Each row: timestamp, axis-angle rotation (3), translation (3).
        text = path.read_text()
        if not text.strip():
            return np.empty(0, np.float64), np.empty((0, 4, 4), np.float32)
        try:
            rows = np.loadtxt(text.splitlines(), dtype=np.float64, ndmin=2)
        except ValueError as exc:
            raise DataFormatError(f"Invalid ARKitScenes trajectory in {path}.") from exc
        if rows.shape[1] != 7:
            raise DataFormatError(f"Invalid ARKitScenes trajectory row in {path}.")
        matrices = np.tile(np.eye(4, dtype=np.float32), (len(rows), 1, 1))
        matrices[:, :3, :3] = Rotation.from_rotvec(rows[:, 1:4]).as_matrix()
        matrices[:, :3, 3] = rows[:, 4:7]
        return rows[:, 0], matrices

    def _match_camera(self, frame_timestamps, camera_timestamps, matrices):
        if not len(frame_timestamps) or not len(camera_timestamps):
            return (
                np.empty(0, np.int64),
                np.empty(0, np.float64),
                np.empty((0, 4, 4), np.float32),
            )
        order = np.argsort(camera_timestamps)
        camera_timestamps = camera_timestamps[order]
        matrices = matrices[order]
        right = np.searchsorted(camera_timestamps, frame_timestamps)
        right = np.clip(right, 0, len(camera_timestamps) - 1)
        left = np.clip(right - 1, 0, len(camera_timestamps) - 1)
        use_right = np.abs(camera_timestamps[right] - frame_timestamps) < np.abs(
            camera_timestamps[left] - frame_timestamps
        )
        nearest = np.where(use_right, right, left)
        delta = np.abs(camera_timestamps[nearest] - frame_timestamps)
        keep = delta <= self.timestamp_tolerance + np.finfo(np.float64).eps
        indices = np.flatnonzero(keep).astype(np.int64)
        return indices, frame_timestamps[keep], matrices[nearest[keep]]

    @staticmethod
    def _nearest_paths(paths: Sequence[Path], timestamps, tolerance):
        if not paths and len(timestamps):
            raise FileNotFoundError(
                "The requested ARKitScenes image asset was not downloaded."
            )
        path_timestamps = np.asarray([_timestamp_from_path(path) for path in paths])
        result = []
        for timestamp in timestamps:
            position = int(np.searchsorted(path_timestamps, timestamp))
            candidates = [
                value for value in (position - 1, position) if 0 <= value < len(paths)
            ]
            nearest = min(
                candidates, key=lambda value: abs(path_timestamps[value] - timestamp)
            )
            if (
                abs(path_timestamps[nearest] - timestamp)
                > tolerance + np.finfo(float).eps
            ):
                raise DataFormatError(
                    f"No ARKitScenes asset matches timestamp {timestamp:.6f} "
                    f"within {tolerance:.6f} seconds."
                )
            result.append(paths[nearest])
        return result

    @staticmethod
    def _matching_timestamp_mask(paths: Sequence[Path], timestamps, tolerance):
        """Return timestamps that have an asset within the synchronization window."""

        if not paths:
            return np.zeros(len(timestamps), dtype=bool)
        candidates = np.asarray([_timestamp_from_path(path) for path in paths])
        right = np.searchsorted(candidates, timestamps)
        right = np.clip(right, 0, len(candidates) - 1)
        left = np.clip(right - 1, 0, len(candidates) - 1)
        delta = np.minimum(
            np.abs(candidates[right] - timestamps),
            np.abs(candidates[left] - timestamps),
        )
        return delta <= tolerance + np.finfo(np.float64).eps

    def _intrinsics(self, sample_id, source, timestamps):
        paths = self._nearest_paths(
            self._intrinsic_files(sample_id, source),
            timestamps,
            self.timestamp_tolerance,
        )
        matrices, sizes = [], []
        for path in paths:
            values = path.read_text().split()
            if len(values) < 6:
                raise DataFormatError(f"Invalid ARKitScenes intrinsic file: {path}")
            try:
                sizes.append((int(values[0]), int(values[1])))
                matrices.append(
                    np.array(
                        [
                            [values[2], 0, values[4]],
                            [0, values[3], values[5]],
                            [0, 0, 1],
                        ],
                        dtype=np.float32,
                    )
                )
            except ValueError as exc:
                raise DataFormatError(
                    f"Invalid ARKitScenes intrinsic file: {path}"
                ) from exc
        if not matrices:
            return np.empty((0, 3, 3), np.float32), sizes
        return np.stack(matrices), sizes

    def _rotation(self, sample_id, *, source):
        try:
            direction = self.metadata[str(sample_id)]["sky_direction"]
            sky_rotation = ROTATION_BY_SKY_DIRECTION[direction]
        except KeyError as exc:
            raise DataFormatError(
                f"Missing corrected ARKitScenes sky direction for {sample_id}."
            ) from exc
        # Video frames are decoded in their coded sensor orientation, the same
        # orientation as the corrected sky directions, intrinsics, and poses, so
        # MOV display-rotation metadata must not be added here.
        del source
        return sky_rotation % 360

    def get_frame_sources(self, sample_id) -> tuple[str, ...]:
        """Return frame sources that are present for one capture."""

        available = []
        video_path = self._sample_path(sample_id, self.rgb_video_file)
        if video_path.is_file() and self._intrinsic_files(sample_id, "mov"):
            available.append("mov")
        for source in FRAME_SOURCES[1:]:
            if source == "threedod":
                directory = (
                    self._sample_directory(sample_id) / f"{sample_id}_frames" / "wide"
                )
            else:
                directory = self._asset_directory(sample_id, IMAGE_SOURCES[source])
            if any(directory.glob("*.png")):
                available.append(source)
        return tuple(available)

    def _asset_files(self, sample_id, directory_name):
        if directory_name == "depth_densified":
            directory = (
                self._sample_directory(sample_id)
                / f"{sample_id}_frames"
                / directory_name
            )
            return self._sorted_files(directory)
        directory = self._asset_directory(sample_id, (directory_name,))
        return self._sorted_files(directory)

    def _asset_paths(self, sample_id, directory_name, timestamps):
        return self._nearest_paths(
            self._asset_files(sample_id, directory_name),
            timestamps,
            self.timestamp_tolerance,
        )

    @staticmethod
    def _image_size_from_batch(images, fallback):
        if len(images):
            return images.shape[2], images.shape[1]
        return fallback

    def _transform_intrinsics(
        self,
        intrinsics,
        intrinsic_sizes,
        native_size,
        *,
        output_size,
        rotation,
        center_crop,
    ):
        adjusted = []
        output_transform = build_pixel_transform(
            native_size,
            output_size=output_size,
            rotation=rotation,
            center_crop=center_crop,
        )
        for intrinsic, intrinsic_size in zip(intrinsics, intrinsic_sizes):
            to_native = build_pixel_transform(intrinsic_size, output_size=native_size)
            adjusted.append(
                transform_intrinsics(
                    transform_intrinsics(intrinsic, to_native), output_transform
                )
            )
        return (
            np.stack(adjusted).astype(np.float32)
            if adjusted
            else np.empty((0, 3, 3), np.float32)
        )

    def get_frames(
        self,
        sample_id,
        *,
        source="mov",
        depth_type=None,
        indices=None,
        step=1,
        items=("rgb", "rgb_intrinsics", "world_to_camera"),
        output_size=None,
        center_crop=False,
        rotate_to_up=True,
    ):
        """Return synchronized frames from an official ARKitScenes camera source.

        ``source`` is one of ``mov``, ``lowres_wide``, ``wide``, ``ultrawide``,
        ``vga_wide``, or ``threedod``. ``wide`` uses high-resolution depth by
        default; pass ``depth_type='lowres'`` to retrieve its Apple LiDAR input.
        """

        if source not in FRAME_SOURCES:
            raise ValueError(f"Unknown ARKitScenes frame source: {source!r}")
        if depth_type is None:
            depth_type = "highres" if source == "wide" else "lowres"
        if depth_type not in {"lowres", "highres"}:
            raise ValueError("depth_type must be 'lowres' or 'highres'.")
        if source not in {"mov", "wide"} and depth_type == "highres":
            raise ValueError(
                "High-resolution depth is synchronized with source='mov' or 'wide'."
            )
        supported = {"timestamps", "rgb", "rgb_intrinsics", "world_to_camera"}
        if source in {"mov", "lowres_wide", "wide", "threedod"}:
            supported |= {"depth", "depth_intrinsics", "confidence_maps"}
        if source == "threedod":
            supported.remove("confidence_maps")
        items = validate_frame_items(items, supported)
        depth_directory = (
            "depth_densified" if source == "threedod" else DEPTH_DIRECTORY[depth_type]
        )
        if source == "threedod":
            depth_source = "threedod"
        elif depth_type == "highres" and source == "wide":
            depth_source = "wide"
        else:
            # High-resolution depth used with the MOV is projected into the
            # same physical wide camera, so its low-resolution calibration is
            # scaled to the requested high-resolution raster.
            depth_source = "lowres_wide"

        frame_timestamps, source_paths = self._source_frames(str(sample_id), source)
        if not len(frame_timestamps):
            raise FileNotFoundError(
                f"ARKitScenes source {source!r} was not downloaded for {sample_id}."
            )
        trajectory_path = self._trajectory_path(sample_id, source)
        if trajectory_path.is_file():
            camera_timestamps, camera_matrices = self._camera_records(
                sample_id, source=source
            )
            available, synchronized_timestamps, world_to_camera = self._match_camera(
                frame_timestamps, camera_timestamps, camera_matrices
            )
            valid = valid_camera_mask(world_to_camera)
            available = available[valid]
            synchronized_timestamps = synchronized_timestamps[valid]
            world_to_camera = world_to_camera[valid]
        else:
            if "world_to_camera" in items:
                raise UnsupportedOperationError(
                    f"ARKitScenes source {source!r} has no camera trajectory for "
                    f"sample {sample_id!r}."
                )
            available = np.arange(len(frame_timestamps), dtype=np.int64)
            synchronized_timestamps = frame_timestamps
            world_to_camera = None

        if indices is None:
            requested_assets = []
            if "depth" in items:
                requested_assets.append(
                    ("depth", self._asset_files(sample_id, depth_directory))
                )
            if "confidence_maps" in items:
                requested_assets.append(
                    ("confidence", self._asset_files(sample_id, "confidence"))
                )
            if "rgb_intrinsics" in items:
                requested_assets.append(
                    ("RGB intrinsics", self._intrinsic_files(sample_id, source))
                )
            if "depth_intrinsics" in items:
                requested_assets.append(
                    (
                        "depth intrinsics",
                        self._intrinsic_files(sample_id, depth_source),
                    )
                )
            keep = np.ones(len(available), dtype=bool)
            for name, paths in requested_assets:
                if not paths:
                    raise FileNotFoundError(
                        f"ARKitScenes {name} was not downloaded for {sample_id}."
                    )
                keep &= self._matching_timestamp_mask(
                    paths, synchronized_timestamps, self.timestamp_tolerance
                )
            available = available[keep]
            synchronized_timestamps = synchronized_timestamps[keep]
            if world_to_camera is not None:
                world_to_camera = world_to_camera[keep]
        selected, positions = select_frame_indices(
            available, indices=indices, step=step
        )
        selected_timestamps = synchronized_timestamps[positions]

        values = {
            "indices": selected,
            "source_frame_count": len(frame_timestamps),
            "frame_keys": tuple(source_paths[index].name for index in selected),
            "camera_model": "PINHOLE",
        }
        if "timestamps" in items:
            values["timestamps"] = selected_timestamps.astype(np.float64, copy=False)
        oriented_items = {
            "rgb",
            "rgb_intrinsics",
            "depth",
            "depth_intrinsics",
            "confidence_maps",
            "world_to_camera",
        }
        rotation = (
            self._rotation(sample_id, source=source)
            if rotate_to_up and oriented_items.intersection(items)
            else 0
        )
        if "world_to_camera" in items:
            values["world_to_camera"] = transform_world_to_camera(
                world_to_camera[positions], rotation
            )

        if "rgb" in items:
            if source == "mov":
                video_path = self._sample_path(sample_id, self.rgb_video_file)
                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"ARKitScenes MOV was not downloaded: {video_path}"
                    )
                rgb = read_video_frames(video_path, frame_indices=selected)
                rgb_size = (
                    self._image_size_from_batch(rgb, NATIVE_IMAGE_SIZES["wide"])
                    if len(rgb)
                    else read_video_frame_size(video_path)
                )
            else:
                rgb = read_images([source_paths[index] for index in selected], rgb=True)
                fallback_size = NATIVE_IMAGE_SIZES.get(source, (1920, 1440))
                rgb_size = self._image_size_from_batch(rgb, fallback_size)
            rgb_transform = build_pixel_transform(
                rgb_size,
                output_size=output_size,
                rotation=rotation,
                center_crop=center_crop,
            )
            values["rgb"] = apply_transform_batch(rgb, rgb_transform)

        if "rgb_intrinsics" in items:
            intrinsics, sizes = self._intrinsics(sample_id, source, selected_timestamps)
            if source == "mov":
                video_path = self._sample_path(sample_id, self.rgb_video_file)
                rgb_size = (
                    read_video_frame_size(video_path)
                    if video_path.is_file()
                    else (sizes[0] if sizes else NATIVE_IMAGE_SIZES["lowres_wide"])
                )
            else:
                rgb_size = NATIVE_IMAGE_SIZES.get(source, (1920, 1440))
                if len(selected):
                    with Image.open(source_paths[int(selected[0])]) as image:
                        rgb_size = image.size
            values["rgb_intrinsics"] = self._transform_intrinsics(
                intrinsics,
                sizes,
                rgb_size,
                output_size=output_size,
                rotation=rotation,
                center_crop=center_crop,
            )

        if "depth" in items:
            depth_paths = self._asset_paths(
                sample_id, depth_directory, selected_timestamps
            )
            raw_depth = read_images(depth_paths, rgb=False)
            depth = raw_depth.astype(np.float32) / 1000.0
            depth_size = self._image_size_from_batch(
                depth, NATIVE_IMAGE_SIZES.get(depth_directory, (256, 192))
            )
            depth_transform = build_pixel_transform(
                depth_size,
                output_size=output_size,
                rotation=rotation,
                center_crop=center_crop,
            )
            values["depth"] = apply_transform_batch(
                depth, depth_transform, resample=Image.Resampling.NEAREST
            )

        if "depth_intrinsics" in items:
            intrinsics, sizes = self._intrinsics(
                sample_id, depth_source, selected_timestamps
            )
            depth_size = NATIVE_IMAGE_SIZES.get(depth_directory, (256, 192))
            if "depth" in items and len(values["depth"]):
                with Image.open(
                    self._asset_paths(
                        sample_id, depth_directory, selected_timestamps[:1]
                    )[0]
                ) as image:
                    depth_size = image.size
            values["depth_intrinsics"] = self._transform_intrinsics(
                intrinsics,
                sizes,
                depth_size,
                output_size=output_size,
                rotation=rotation,
                center_crop=center_crop,
            )

        if "confidence_maps" in items:
            confidence_paths = self._asset_paths(
                sample_id, "confidence", selected_timestamps
            )
            confidence = read_images(confidence_paths, rgb=False).astype(
                np.uint8, copy=False
            )
            confidence_size = self._image_size_from_batch(
                confidence, NATIVE_IMAGE_SIZES["confidence"]
            )
            confidence_transform = build_pixel_transform(
                confidence_size,
                output_size=output_size,
                rotation=rotation,
                center_crop=center_crop,
            )
            values["confidence_maps"] = apply_transform_batch(
                confidence, confidence_transform, resample=Image.Resampling.NEAREST
            )
        return FrameBatch(**values)

    # Depth-upsample and FARO metadata ---------------------------------------
    def get_depth_upsampling_attributes(self, sample_id=None):
        """Return official validation attributes, optionally for one video ID."""

        path = self.depth_upsampling_root / self.depth_upsampling_attributes_file
        if not path.is_file():
            raise FileNotFoundError(
                f"ARKitScenes depth-upsample attributes were not downloaded: {path}"
            )
        with open(path) as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if sample_id is None:
            return rows
        prefix = f"{sample_id}_"
        return [
            row
            for row in rows
            if str(row.get("video_id", "")) == str(sample_id)
            or any(str(value).startswith(prefix) for value in row.values())
        ]

    @cached_property
    def _laser_ids_by_visit(self):
        candidates = (
            self.dataset_root / self.laser_mapping_file,
            self.dataset_root / "raw" / self.laser_mapping_file,
        )
        path = next((value for value in candidates if value.is_file()), candidates[0])
        if not path.is_file():
            raise FileNotFoundError(
                f"ARKitScenes laser scanner mapping was not downloaded: {path}"
            )
        result = {}
        with open(path) as handle:
            for row in csv.DictReader(handle):
                visit_id = int(row["visit_id"])
                result.setdefault(visit_id, []).append(
                    str(row["laser_scanner_point_clouds_id"])
                )
        return result

    def get_laser_scanner_ids(self, sample_id) -> tuple[str, ...]:
        """Return FARO scan IDs associated with the video's venue visit."""

        try:
            visit_id = self.metadata[str(sample_id)]["visit_id"]
        except KeyError as exc:
            raise KeyError(f"Unknown ARKitScenes sample {sample_id!r}.") from exc
        if visit_id is None:
            return ()
        return tuple(self._laser_ids_by_visit.get(visit_id, ()))

    def _laser_directory(self, sample_id) -> Path:
        visit_id = self.metadata[str(sample_id)]["visit_id"]
        if visit_id is None:
            raise UnsupportedOperationError(
                f"ARKitScenes sample {sample_id!r} has no laser scanner visit."
            )
        return self.dataset_root / self.laser_scanner_dir.format(visit_id=visit_id)

    def _select_laser_ids(self, sample_id, scan_ids):
        available = self.get_laser_scanner_ids(sample_id)
        if scan_ids is None:
            return available
        if isinstance(scan_ids, str):
            raise ValueError("scan_ids must be a sequence, not one string.")
        selected = tuple(str(scan_id) for scan_id in scan_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("scan_ids must not contain duplicates.")
        missing = [scan_id for scan_id in selected if scan_id not in available]
        if missing:
            raise ValueError(f"Unknown ARKitScenes FARO scan IDs: {missing}")
        return selected

    def get_laser_scanner_poses(
        self, sample_id, *, scan_ids=None
    ) -> dict[str, np.ndarray]:
        """Return scanner-to-registered-coordinate transforms for FARO scans."""

        directory = self._laser_directory(sample_id)
        result = {}
        for scan_id in self._select_laser_ids(sample_id, scan_ids):
            path = directory / f"{scan_id}_pose.txt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"ARKitScenes FARO pose was not downloaded: {path}"
                )
            try:
                matrix = np.loadtxt(path, delimiter=",").astype(np.float32).T
            except ValueError as exc:
                raise DataFormatError(f"Invalid ARKitScenes FARO pose: {path}") from exc
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise DataFormatError(f"Invalid ARKitScenes FARO pose: {path}")
            result[scan_id] = matrix
        return result

    def get_laser_scanner_point_clouds(self, sample_id, *, scan_ids=None):
        """Return the released registered FARO RGB point clouds by scan ID."""

        directory = self._laser_directory(sample_id)
        result = {}
        for scan_id in self._select_laser_ids(sample_id, scan_ids):
            path = directory / f"{scan_id}.ply"
            if not path.is_file():
                raise FileNotFoundError(
                    f"ARKitScenes FARO point cloud was not downloaded: {path}"
                )
            cloud = o3d.io.read_point_cloud(str(path))
            if not cloud.has_points():
                raise DataFormatError(f"ARKitScenes FARO file has no points: {path}")
            result[scan_id] = cloud
        return result
