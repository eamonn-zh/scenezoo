"""Aria Synthetic Environments sequences, layouts, points, and RGB-D frames."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
import gzip
import io
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

from ...io.image import read_image
from ...ops.frame import (
    apply_transform_batch,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    transform_world_to_camera,
    validate_frame_items,
)
from ..base import DataFormatError, Dataset, UnsupportedOperationError
from ..registry import register_dataset
from ..types import FrameBatch, PointBatch
from ._ase_store import ASEStore


_MANIFEST_NAME = "aria_synthetic_environments_dataset_download_urls.json"
_ARCHIVE_PATTERN = re.compile(r"^(train|test)_chunk_([0-9]{7})\.zip$")
_IMAGE_PATTERNS = {
    "rgb": re.compile(r"^vignette([0-9]{7})\.jpg$"),
    "depth": re.compile(r"^depth([0-9]{7})\.png$"),
    "instances": re.compile(r"^instance([0-9]{7})\.png$"),
}
# Official calibration of the 704x704 renderings; 1408 is the 2x variant.
_IMAGE_SIZE = 704
_PROJECTION_704 = np.array(
    [
        297.6375381033778,
        357.6599197217746,
        349.1922497127481,
        0.3650890375644368,
        -0.1738082418112771,
        -0.7534945484033189,
        2.434788882752295,
        -2.57786220300886,
        0.8788483538598834,
        0.0008005198595407136,
        -0.000294237814554143,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
_T_DEVICE_CAMERA_QUATERNION_XYZW = np.array(
    [
        0.32640934382849085,
        0.029274992008313648,
        0.033361059956531547,
        0.9441858687689326,
    ],
    dtype=np.float64,
)
_T_DEVICE_CAMERA_TRANSLATION = np.array(
    [-0.007530096566173914, -0.01090854984158026, -0.003598063315542823],
    dtype=np.float64,
)
_REQUIRED_COMMON_ASSETS = (
    "rgb",
    "trajectory.csv",
    "semidense_points.csv.gz",
    "semidense_observations.csv.gz",
)
_REQUIRED_TRAIN_ASSETS = (
    "depth",
    "instances",
    "ase_scene_language.txt",
    "object_instances_to_classes.json",
)


@dataclass(slots=True)
class ASEPointData(PointBatch):
    """Semi-dense ASE points with the official uncertainty attributes."""

    uids: np.ndarray | None = None
    inverse_distance_std: np.ndarray | None = None
    distance_std: np.ndarray | None = None

    def __post_init__(self) -> None:
        PointBatch.__post_init__(self)
        for name, dtype in (
            ("uids", np.int64),
            ("inverse_distance_std", np.float32),
            ("distance_std", np.float32),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=dtype)
            if array.shape != (len(self),):
                raise ValueError(f"ASEPointData.{name} must have shape {(len(self),)}.")
            setattr(self, name, array)


@register_dataset(
    "ase",
    aliases=("aria-ase", "aria-synthetic-environments"),
    description="Aria Synthetic Environments egocentric synthetic scenes.",
)
class AriaSyntheticEnvironments(Dataset):
    """Read extracted ASE train/test sequences."""

    def __init__(
        self,
        root_dir,
        *,
        data_dir=None,
        manifest_file=None,
        extracted_split="train",
        invalid_obj_id=0,
        cache_dir=None,
        offline=False,
    ):
        super().__init__(
            root_dir,
            cache_dir=cache_dir,
            offline=offline,
            invalid_obj_id=invalid_obj_id,
        )
        self.data_dir = data_dir
        self.extracted_split = str(extracted_split)
        self._store = ASEStore(
            self.root_dir,
            data_dir,
            extracted_split=self.extracted_split,
        )
        self.data_root = self._store.data_root
        self.manifest_file = self._resolve_manifest(manifest_file)

    def _resolve_manifest(self, manifest_file) -> Path | None:
        if manifest_file is not None:
            path = Path(manifest_file).expanduser()
            return path if path.is_absolute() else self.root_dir / path
        candidates = (
            self.root_dir / _MANIFEST_NAME,
            self.data_root / _MANIFEST_NAME,
            self.data_root.parent / _MANIFEST_NAME,
        )
        return next((path for path in candidates if path.is_file()), None)

    @cached_property
    def _manifest_entries(self) -> tuple[tuple[str, str], ...]:
        if self.manifest_file is None:
            return ()
        try:
            payload = json.loads(self.manifest_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DataFormatError(
                f"Invalid ASE download manifest: {self.manifest_file}"
            ) from exc
        if not isinstance(payload, list):
            raise DataFormatError("ASE download manifest must contain a list.")
        result = []
        for row in payload:
            if not isinstance(row, dict) or not isinstance(row.get("filename"), str):
                raise DataFormatError("Invalid entry in ASE download manifest.")
            filename = row["filename"]
            if _ARCHIVE_PATTERN.fullmatch(filename) is None:
                raise DataFormatError(f"Invalid ASE archive filename: {filename!r}")
            result.append((filename, str(row.get("sha", ""))))
        if len({name for name, _ in result}) != len(result):
            raise DataFormatError("ASE download manifest contains duplicate archives.")
        return tuple(result)

    def _load_splits(self):
        result = {"train": [], "test": []}
        for sample_id in self._store.available_sample_ids:
            result[sample_id.split("/", 1)[0]].append(sample_id)
        return result

    @cached_property
    def _manifest_sample_ids(self) -> frozenset[str]:
        result = set()
        for filename, _ in self._manifest_entries:
            match = _ARCHIVE_PATTERN.fullmatch(filename)
            split, chunk = match.group(1), int(match.group(2))
            result.update(
                f"{split}/{number}" for number in range(chunk * 10, chunk * 10 + 10)
            )
        return frozenset(result)

    def _load_metadata(self):
        archive_counts = {"train": 0, "test": 0}
        for filename, _ in self._manifest_entries:
            archive_counts[_ARCHIVE_PATTERN.fullmatch(filename).group(1)] += 1
        return {
            "manifest_file": None
            if self.manifest_file is None
            else str(self.manifest_file),
            "archive_counts": archive_counts,
            "calibration": self.get_calibration(),
            "depth_mode": "ray_distance",
        }

    def get_available_ids(self, split=None) -> list[str]:
        """Return scenes currently available as extracted directories."""

        if split is not None and split not in {"train", "test"}:
            raise ValueError("split must be 'train', 'test', or None.")
        return [
            sample_id
            for sample_id in self._store.available_sample_ids
            if split is None or sample_id.startswith(f"{split}/")
        ]

    def _check(self, *, sample_ids, require_complete):
        from ..check import CheckBuilder, check_scene_coverage, normalize_sample_ids

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The ASE root does not exist.",
            expected="<root>/download/<scene_id>/ with extracted ASE sequences.",
            hint="Pass the directory containing the extracted download folder.",
        ):
            return builder.finish()
        if not builder.require_directory(
            self.data_root,
            code="missing-data-directory",
            message="The resolved ASE data directory is missing.",
            expected="Extracted numeric scene directories.",
            hint="Set data_dir correctly, then download and extract the ASE shards.",
        ):
            return builder.finish()
        if self.manifest_file is None:
            builder.warning(
                "missing-download-manifest",
                f"{_MANIFEST_NAME} was not found; official completeness cannot be verified.",
                expected=f"{_MANIFEST_NAME} beside the download directory.",
                hint="Keep the raw ASE download-URL manifest with the dataset root.",
            )

        requested = normalize_sample_ids(sample_ids)
        if requested is not None:
            try:
                normalized = tuple(
                    self._store.normalize_sample_id(value) for value in requested
                )
            except ValueError as exc:
                builder.error(
                    "invalid-scene-id",
                    str(exc),
                    expected="Globally unique IDs such as train/83788 or test/20.",
                    hint="Prefix test IDs because train and test numeric ranges overlap.",
                )
                return builder.finish()
            present = {
                sample_id
                for sample_id in normalized
                if self._store.scene_directory(sample_id) is not None
            }
            expected_ids = None
            coverage_sample_ids = normalized
        else:
            present = set(self._store.extracted_sample_ids)
            expected_ids = (
                set(self._manifest_sample_ids) if self.manifest_file else None
            )
            coverage_sample_ids = None

        archive_ids = set(self._store.archive_sample_ids)
        extracted_ids = set(self._store.extracted_sample_ids)
        expected_layout = (
            "<data>/<numeric_id>/ for extracted_split or "
            "<data>/<split>/<numeric_id>/; chunk ZIPs must be extracted first"
        )
        if requested is not None:
            builder.stats["scenes_found"] = len(present)
            builder.stats["scenes_requested"] = len(normalized)
            unavailable = sorted(set(normalized) - present - archive_ids)
            if unavailable:
                builder.error(
                    "missing-requested-scenes",
                    f"Requested scene IDs are missing: {', '.join(unavailable[:10])}",
                    expected=expected_layout,
                    hint="Download and extract these scenes, or correct sample_ids/root_dir.",
                )
            for sample_id in sorted(set(normalized) - present):
                if sample_id not in archive_ids:
                    continue
                archive = self._store.archive_path(sample_id)
                builder.error(
                    "scene-not-extracted",
                    f"{sample_id} is present only inside a downloaded chunk ZIP.",
                    path=archive,
                    expected=str(self._store.expected_scene_directory(sample_id)),
                    hint=(
                        f"Extract {archive.name} under {self.data_root}. SceneZoo does "
                        "not read ASE payloads directly from ZIP files."
                    ),
                )
            selected = tuple(
                sample_id for sample_id in normalized if sample_id in present
            )
        else:
            selected = check_scene_coverage(
                builder,
                present_ids=present,
                expected_ids=expected_ids,
                sample_ids=coverage_sample_ids,
                require_complete=require_complete,
                expected_layout=expected_layout,
            )

        if requested is None and self._manifest_entries:
            missing_shards = []
            for filename, _ in self._manifest_entries:
                match = _ARCHIVE_PATTERN.fullmatch(filename)
                split, chunk = match.group(1), int(match.group(2))
                chunk_ids = {
                    f"{split}/{number}" for number in range(chunk * 10, chunk * 10 + 10)
                }
                if (
                    not (self.data_root / filename).is_file()
                    and not chunk_ids <= present
                ):
                    missing_shards.append((filename, len(chunk_ids - present)))
            if missing_shards:
                examples = ", ".join(
                    f"{name} ({count}/10 scenes missing)"
                    for name, count in missing_shards[:5]
                )
                builder.add(
                    "error" if require_complete else "warning",
                    "missing-download-shards",
                    f"{len(missing_shards)} manifest shards are absent and not fully "
                    f"represented by extracted scenes. Examples: {examples}",
                    expected="Each manifest shard downloaded or all ten scenes extracted.",
                    hint="Download the named shard, then extract it before data access.",
                )

        unextracted_ids = archive_ids - extracted_ids
        unextracted_archives = {
            self._store.archive_path(sample_id) for sample_id in unextracted_ids
        }
        if requested is None and unextracted_archives:
            examples = ", ".join(path.name for path in sorted(unextracted_archives)[:5])
            builder.warning(
                "archives-need-extraction",
                f"{len(unextracted_archives)} downloaded ASE ZIPs still contain "
                f"unextracted scenes. Examples: {examples}",
                expected="One extracted directory per scene before using the dataset API.",
                hint=(
                    f"Extract the ZIPs under {self.data_root}; keep train and test in "
                    "separate directories when their numeric IDs overlap."
                ),
            )
        builder.stats["extracted_scenes"] = len(present)
        if requested is None:
            builder.stats["downloaded_archives"] = len(archive_ids) // 10
            builder.stats["archive_scenes_needing_extraction"] = len(unextracted_ids)
        if not present and requested is None:
            builder.error(
                "no-extracted-scenes",
                "No requested or discoverable extracted ASE scenes were found.",
                path=self.data_root,
                expected="Numeric scene directories containing the ASE payload files.",
                hint="Download and extract the required chunk ZIP before data access.",
            )
            return builder.finish()
        if requested is None:
            deep_selected = []
            for split in ("train", "test"):
                split_ids = [
                    value for value in selected if value.startswith(f"{split}/")
                ]
                deep_selected.extend(split_ids[:1])
            builder.info(
                "representative-deep-check",
                "The full check validates scene/shard coverage and deeply checks one "
                "extracted representative of each split. Pass sample_ids to deeply check "
                "particular sequences without scanning every payload tree.",
            )
        else:
            deep_selected = list(selected)
        builder.stats["scenes_deep_checked"] = len(deep_selected)
        for sample_id in deep_selected:
            split, _ = self._store.parts(sample_id)
            required_assets = _REQUIRED_COMMON_ASSETS + (
                _REQUIRED_TRAIN_ASSETS if split == "train" else ()
            )
            missing = [
                relative
                for relative in required_assets
                if not self._store.exists(sample_id, relative)
            ]
            if missing:
                builder.error(
                    "incomplete-scene",
                    f"{sample_id} is missing: {', '.join(missing)}.",
                    expected="All synchronized RGB/depth/instance and 3D annotation assets.",
                    hint="Re-download/re-extract the scene's complete chunk.",
                )
                continue
            folders = _IMAGE_PATTERNS if split == "train" else ("rgb",)
            frame_sets = {
                folder: set(self._frame_ids(sample_id, folder)) for folder in folders
            }
            try:
                trajectory_count = len(self.get_trajectory(sample_id)["timestamps"])
            except (DataFormatError, FileNotFoundError) as exc:
                builder.error(
                    "invalid-trajectory",
                    f"Cannot validate {sample_id} trajectory: {exc}",
                    expected="A non-empty, strictly timestamp-ordered trajectory.csv.",
                    hint="Restore trajectory.csv from the same official shard.",
                )
                continue
            synchronized = (
                len({frozenset(values) for values in frame_sets.values()}) == 1
            )
            image_count = len(next(iter(frame_sets.values())))
            if not synchronized or image_count != trajectory_count:
                split, number = self._store.parts(sample_id)
                bare_scene = self.data_root / str(number)
                possible_collision = (
                    split == self.extracted_split
                    and number < 1000
                    and bare_scene.is_dir()
                    and self._store.archive_path(f"test/{number}").is_file()
                )
                builder.error(
                    "possible-split-collision"
                    if possible_collision
                    else "unsynchronized-frame-files",
                    (
                        f"{sample_id} has different RGB/depth/instance frame IDs: "
                        + ", ".join(
                            f"{key}={len(value)}" for key, value in frame_sets.items()
                        )
                        + f", trajectory={trajectory_count}"
                        + (
                            ". Train and test reuse IDs 0-999; this bare directory "
                            "likely contains files extracted from both splits"
                            if possible_collision
                            else ""
                        )
                    ),
                    expected="Exactly matching frame IDs in rgb, depth, and instances.",
                    hint=(
                        "Remove the mixed directory and re-extract train/test into separate "
                        "namespaced roots."
                        if possible_collision
                        else "Restore the missing files from the same official chunk."
                    ),
                )
        builder.info(
            "extracted-directories-required",
            "ASE chunk ZIPs are download artifacts and must be extracted before data "
            f"access. Bare numeric folders are interpreted as {self.extracted_split!r}; "
            "use split-prefixed directories to avoid train/test numeric-ID collisions.",
        )
        return builder.finish()

    @staticmethod
    def _transform(translation, quaternion_xyzw) -> np.ndarray:
        quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise DataFormatError("ASE trajectory contains an invalid quaternion.")
        norm = np.linalg.norm(quaternion)
        if norm <= np.finfo(np.float64).eps:
            raise DataFormatError("ASE trajectory contains a zero quaternion.")
        translation = np.asarray(translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise DataFormatError("ASE trajectory contains an invalid translation.")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
        matrix[:3, 3] = translation
        return matrix

    @classmethod
    def get_calibration(cls, image_size=_IMAGE_SIZE) -> dict:
        """Return Meta's fixed ASE Fisheye624 calibration and RGB extrinsic."""

        if image_size not in {_IMAGE_SIZE, 2 * _IMAGE_SIZE}:
            raise ValueError("ASE calibration supports image_size 704 or 1408.")
        scale = image_size / _IMAGE_SIZE
        projection = _PROJECTION_704.copy()
        projection[:3] *= scale
        device_to_camera = cls._transform(
            _T_DEVICE_CAMERA_TRANSLATION,
            _T_DEVICE_CAMERA_QUATERNION_XYZW,
        )
        intrinsic = np.array(
            [
                [projection[0], 0.0, projection[1]],
                [0.0, projection[0], projection[2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return {
            "camera_model": "FISHEYE624",
            "image_size": (image_size, image_size),
            "projection_parameters": projection,
            "intrinsic": intrinsic,
            "device_to_camera": device_to_camera,
            "valid_radius": 1415.0 / 4.0 * scale,
            "max_solid_angle_radians": 1.0,
        }

    def get_trajectory(self, sample_id) -> dict[str, np.ndarray]:
        """Return ground-truth timestamps, device poses, and RGB camera poses."""

        required = {
            "tracking_timestamp_us",
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "qw_world_device",
        }
        timestamps = []
        world_to_device = []
        with self._store.open_binary(sample_id, "trajectory.csv") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            try:
                reader = csv.DictReader(text)
                if reader.fieldnames is None or not required <= set(reader.fieldnames):
                    raise DataFormatError("Invalid ASE trajectory header.")
                for line_number, row in enumerate(reader, start=2):
                    try:
                        timestamps.append(int(row["tracking_timestamp_us"]))
                        world_to_device.append(
                            self._transform(
                                [
                                    float(row["tx_world_device"]),
                                    float(row["ty_world_device"]),
                                    float(row["tz_world_device"]),
                                ],
                                [
                                    float(row["qx_world_device"]),
                                    float(row["qy_world_device"]),
                                    float(row["qz_world_device"]),
                                    float(row["qw_world_device"]),
                                ],
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise DataFormatError(
                            f"Invalid ASE trajectory row {line_number}."
                        ) from exc
            finally:
                text.detach()
        if not timestamps:
            raise DataFormatError("ASE trajectory contains no poses.")
        timestamp_array = np.asarray(timestamps, dtype=np.int64)
        if np.any(np.diff(timestamp_array) <= 0):
            raise DataFormatError(
                "ASE trajectory timestamps must be strictly increasing."
            )
        world_to_device_array = np.stack(world_to_device)
        device_to_camera = self.get_calibration()["device_to_camera"]
        camera_to_world = world_to_device_array @ device_to_camera
        return {
            "tracking_timestamp_us": timestamp_array,
            "timestamps": timestamp_array.astype(np.float64) / 1_000_000.0,
            "world_to_device": world_to_device_array,
            "world_to_camera": np.linalg.inv(camera_to_world),
        }

    def get_scene_commands(self, sample_id) -> list[tuple[str, dict[str, float]]]:
        """Parse the official ASE procedural floor-plan language."""

        self._require_train_ground_truth(sample_id, "scene-language layout")
        result = []
        for line_number, raw_line in enumerate(
            self._store.read_text(sample_id, "ase_scene_language.txt").splitlines(),
            start=1,
        ):
            fields = [field.strip() for field in raw_line.split(",")]
            if not fields or not fields[0]:
                continue
            params = {}
            try:
                for field in fields[1:]:
                    key, value = field.split("=", 1)
                    params[key.strip()] = float(value)
            except ValueError as exc:
                raise DataFormatError(
                    f"Invalid ASE scene-language command on line {line_number}."
                ) from exc
            result.append((fields[0], params))
        if not result:
            raise DataFormatError("ASE scene-language file contains no commands.")
        return result

    def get_instance_classes(self, sample_id) -> dict[int, str]:
        """Return the released mapping from image instance ID to class name."""

        self._require_train_ground_truth(sample_id, "instance-class mapping")
        try:
            payload = json.loads(
                self._store.read_text(sample_id, "object_instances_to_classes.json")
            )
            result = {int(key): str(value) for key, value in payload.items()}
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataFormatError("Invalid ASE instance-to-class mapping.") from exc
        if not result:
            raise DataFormatError("ASE instance-to-class mapping is empty.")
        return result

    def _require_train_ground_truth(self, sample_id, asset: str) -> None:
        split, _ = self._store.parts(sample_id)
        if split == "test":
            raise UnsupportedOperationError(
                f"ASE test scenes do not release {asset}; use a train scene."
            )

    @staticmethod
    def _z_rotation(angle: float) -> np.ndarray:
        sine, cosine = np.sin(angle), np.cos(angle)
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _rolled_distortion(distortion: np.ndarray, transform) -> np.ndarray:
        """Express Fisheye624 distortion in the rolled output camera frame.

        Radial terms are roll invariant. The tangential ``(p0, p1)`` term and
        the thin-prism ``(s0, s2)`` and ``(s1, s3)`` terms are planar vectors.
        """

        result = np.asarray(distortion, dtype=np.float64).copy()
        roll = transform.camera_rotation[:2, :2]
        result[6:8] = roll @ result[6:8]
        result[[8, 10]] = roll @ result[[8, 10]]
        result[[9, 11]] = roll @ result[[9, 11]]
        return result

    def get_boxes(self, sample_id, *, box_type="obb_gt"):
        """Interpret wall, door, and window commands as official layout OBBs."""

        if box_type != "obb_gt":
            raise ValueError("ASE provides only 'obb_gt' layout boxes.")
        commands = self.get_scene_commands(sample_id)
        boxes, names, walls = {}, {}, {}
        for command, params in commands:
            if command != "make_wall":
                continue
            try:
                identifier = int(params["id"])
                direction = np.array(
                    [
                        params["b_x"] - params["a_x"],
                        params["b_y"] - params["a_y"],
                    ]
                )
                walls[identifier] = (
                    float(np.arctan2(direction[1], direction[0])),
                    params["thickness"],
                )
            except KeyError as exc:
                raise DataFormatError(
                    f"ASE 'make_wall' command is missing parameter {exc.args[0]!r}."
                ) from exc
        for command, params in commands:
            try:
                identifier = int(params["id"])
                if command == "make_wall":
                    corner_a = np.array([params["a_x"], params["a_y"], params["a_z"]])
                    corner_b = np.array([params["b_x"], params["b_y"], params["b_z"]])
                    direction = corner_b - corner_a
                    angle = float(np.arctan2(direction[1], direction[0]))
                    thickness = params["thickness"]
                    center = (corner_a + corner_b) / 2
                    center[2] += params["height"] / 2
                    extent = np.array(
                        [np.linalg.norm(direction), thickness, params["height"]]
                    )
                    name = "wall"
                elif command in {"make_door", "make_window"}:
                    wall_ids = [
                        int(params[key])
                        for key in ("wall_id", "wall0_id", "wall1_id")
                        if key in params and params[key] >= 0
                    ]
                    wall = next(
                        (walls[value] for value in wall_ids if value in walls), None
                    )
                    if wall is None:
                        continue
                    angle, thickness = wall
                    center = np.array(
                        [
                            params["position_x"],
                            params["position_y"],
                            params["position_z"],
                        ]
                    )
                    extent = np.array([params["width"], thickness, params["height"]])
                    name = command.removeprefix("make_")
                else:
                    raise DataFormatError(f"Unknown ASE scene command: {command!r}")
            except KeyError as exc:
                raise DataFormatError(
                    f"ASE {command!r} command is missing parameter {exc.args[0]!r}."
                ) from exc
            if identifier in boxes:
                raise DataFormatError(f"Duplicate ASE layout ID: {identifier}")
            if not np.isfinite(center).all() or not np.isfinite(extent).all():
                raise DataFormatError(f"Non-finite ASE layout box {identifier}.")
            boxes[identifier] = o3d.geometry.OrientedBoundingBox(
                center.astype(np.float64),
                self._z_rotation(angle),
                extent.astype(np.float64),
            )
            names[identifier] = name
        return boxes, names

    def get_points(
        self,
        sample_id,
        *,
        max_inverse_distance_std=None,
        max_distance_std=None,
    ) -> ASEPointData:
        """Read semi-dense world points, optionally filtering by uncertainty."""

        for name, value in (
            ("max_inverse_distance_std", max_inverse_distance_std),
            ("max_distance_std", max_distance_std),
        ):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        try:
            with self._store.open_binary(sample_id, "semidense_points.csv.gz") as raw:
                with gzip.GzipFile(fileobj=raw) as compressed:
                    header = (
                        compressed.readline().decode("utf-8-sig").strip().split(",")
                    )
                    expected_header = [
                        "uid",
                        "graph_uid",
                        "px_world",
                        "py_world",
                        "pz_world",
                        "inv_dist_std",
                        "dist_std",
                    ]
                    if header != expected_header:
                        raise DataFormatError("Invalid ASE semi-dense point header.")
                    values = np.loadtxt(
                        compressed,
                        delimiter=",",
                        usecols=(0, 2, 3, 4, 5, 6),
                        ndmin=2,
                    )
        except (OSError, ValueError) as exc:
            raise DataFormatError("Invalid ASE semi-dense point CSV.") from exc
        if values.shape[1] != 6:
            raise DataFormatError("ASE semi-dense point CSV has an invalid schema.")
        keep = np.ones(len(values), dtype=bool)
        if max_inverse_distance_std is not None:
            keep &= values[:, 4] <= max_inverse_distance_std
        if max_distance_std is not None:
            keep &= values[:, 5] <= max_distance_std
        values = values[keep]
        return ASEPointData(
            xyz=values[:, 1:4],
            uids=values[:, 0].astype(np.int64),
            inverse_distance_std=values[:, 4],
            distance_std=values[:, 5],
            invalid_id=self.invalid_obj_id,
        )

    def get_semidense_observations(
        self,
        sample_id,
        *,
        frame_indices: Sequence[int] | None = None,
        point_ids: Sequence[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Stream and optionally filter the official semi-dense observations."""

        trajectory = self.get_trajectory(sample_id)
        timestamps = trajectory["tracking_timestamp_us"]
        timestamp_to_frame = {
            int(value): index for index, value in enumerate(timestamps)
        }
        requested_frames = None
        if frame_indices is not None:
            raw_frames = np.asarray(frame_indices)
            if raw_frames.ndim != 1 or not np.issubdtype(raw_frames.dtype, np.integer):
                raise ValueError(
                    "frame_indices must be a one-dimensional integer sequence."
                )
            if np.any(raw_frames < 0) or np.any(raw_frames >= len(timestamps)):
                raise ValueError("frame_indices are outside the trajectory range.")
            requested_frames = {int(value) for value in raw_frames}
        requested_points = (
            None if point_ids is None else {int(value) for value in point_ids}
        )
        uids, observation_timestamps, frames, serials, uv = [], [], [], [], []
        try:
            with self._store.open_binary(
                sample_id, "semidense_observations.csv.gz"
            ) as raw:
                with gzip.GzipFile(fileobj=raw) as compressed:
                    text = io.TextIOWrapper(
                        compressed, encoding="utf-8-sig", newline=""
                    )
                    reader = csv.DictReader(text)
                    required = {
                        "uid",
                        "frame_tracking_timestamp_us",
                        "camera_serial",
                        "u",
                        "v",
                    }
                    if reader.fieldnames is None or not required <= set(
                        reader.fieldnames
                    ):
                        raise DataFormatError(
                            "Invalid ASE semi-dense observation header."
                        )
                    for row in reader:
                        uid = int(row["uid"])
                        timestamp = int(row["frame_tracking_timestamp_us"])
                        frame = timestamp_to_frame.get(timestamp)
                        if frame is None:
                            continue
                        if (
                            requested_frames is not None
                            and frame not in requested_frames
                        ):
                            continue
                        if requested_points is not None and uid not in requested_points:
                            continue
                        uids.append(uid)
                        observation_timestamps.append(timestamp)
                        frames.append(frame)
                        serials.append(row["camera_serial"])
                        uv.append((float(row["u"]), float(row["v"])))
        except (OSError, TypeError, ValueError) as exc:
            raise DataFormatError("Invalid ASE semi-dense observation CSV.") from exc
        return {
            "uids": np.asarray(uids, dtype=np.int64),
            "tracking_timestamp_us": np.asarray(observation_timestamps, dtype=np.int64),
            "frame_indices": np.asarray(frames, dtype=np.int64),
            "camera_serials": np.asarray(serials, dtype=str),
            "uv": np.asarray(uv, dtype=np.float32).reshape(-1, 2),
        }

    def _frame_ids(self, sample_id, folder: str) -> tuple[int, ...]:
        pattern = _IMAGE_PATTERNS[folder]
        return tuple(
            sorted(
                int(match.group(1))
                for name in self._store.listdir(sample_id, folder)
                if (match := pattern.fullmatch(name)) is not None
            )
        )

    def get_frames(
        self,
        sample_id,
        *,
        indices=None,
        step=1,
        items=("rgb", "rgb_intrinsics", "world_to_camera"),
        output_size=None,
        center_crop=False,
        rotate_to_up=True,
    ):
        """Read synchronized fisheye RGB, ray-depth, instances, and camera poses."""

        supported = {
            "rgb",
            "depth",
            "instance_maps",
            "timestamps",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
            "frame_mask",
        }
        items = validate_frame_items(items, supported)
        trajectory = self.get_trajectory(sample_id)
        source_frame_count = len(trajectory["timestamps"])
        available = set(range(source_frame_count))
        folders_needed = {
            "rgb" for item in items if item in {"rgb", "rgb_intrinsics", "frame_mask"}
        }
        if "depth" in items or "depth_intrinsics" in items:
            folders_needed.add("depth")
        if "instance_maps" in items:
            folders_needed.add("instances")
        if not folders_needed:
            folders_needed.add("rgb")
        for folder in folders_needed:
            folder_ids = set(self._frame_ids(sample_id, folder))
            if not folder_ids:
                split, _ = self._store.parts(sample_id)
                detail = (
                    "The ASE test release withholds depth and instance ground truth."
                    if split == "test" and folder in {"depth", "instances"}
                    else f"The {folder!r} frame asset was not downloaded."
                )
                raise UnsupportedOperationError(detail)
            available &= folder_ids
        selected, _ = select_frame_indices(
            sorted(available), indices=indices, step=step
        )
        calibration = self.get_calibration()
        rotation = 270 if rotate_to_up else 0
        transform = build_pixel_transform(
            calibration["image_size"],
            output_size=output_size,
            rotation=rotation,
            center_crop=center_crop,
        )
        values = {
            "indices": selected,
            "source_frame_count": source_frame_count,
            "frame_keys": tuple(f"vignette{index:07d}.jpg" for index in selected),
            "camera_model": "FISHEYE624",
        }
        if "timestamps" in items:
            values["timestamps"] = trajectory["timestamps"][selected]
        if "world_to_camera" in items:
            values["world_to_camera"] = transform_world_to_camera(
                trajectory["world_to_camera"][selected], transform
            ).astype(np.float32)
        camera_items = {
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
            "frame_mask",
        }
        if camera_items & set(items):
            intrinsic = np.repeat(calibration["intrinsic"][None], len(selected), axis=0)
            transformed = transform_intrinsics(intrinsic, transform).astype(np.float32)
            if "rgb_intrinsics" in items:
                values["rgb_intrinsics"] = transformed
            if "depth_intrinsics" in items:
                values["depth_intrinsics"] = transformed.copy()
            values["distortion"] = np.repeat(
                self._rolled_distortion(
                    calibration["projection_parameters"][3:], transform
                )[None],
                len(selected),
                axis=0,
            ).astype(np.float32)
            values["image_sizes"] = np.repeat(
                np.asarray(transform.output_size, dtype=np.int32)[None],
                len(selected),
                axis=0,
            )
        if "rgb" in items:
            images = (
                np.stack(
                    [
                        read_image(payload, rgb=True)
                        for payload in self._store.read_many(
                            sample_id,
                            [f"rgb/vignette{index:07d}.jpg" for index in selected],
                        )
                    ]
                )
                if len(selected)
                else np.empty((0, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
            )
            values["rgb"] = apply_transform_batch(images, transform)
        if "depth" in items:
            images = (
                np.stack(
                    [
                        read_image(payload, rgb=False)
                        for payload in self._store.read_many(
                            sample_id,
                            [f"depth/depth{index:07d}.png" for index in selected],
                        )
                    ]
                )
                if len(selected)
                else np.empty((0, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint16)
            )
            values["depth"] = (
                apply_transform_batch(
                    images, transform, resample=Image.Resampling.NEAREST
                ).astype(np.float32)
                / 1000.0
            )
            values["depth_mode"] = "ray_distance"
        if "instance_maps" in items:
            images = (
                np.stack(
                    [
                        read_image(payload, rgb=False)
                        for payload in self._store.read_many(
                            sample_id,
                            [
                                f"instances/instance{index:07d}.png"
                                for index in selected
                            ],
                        )
                    ]
                )
                if len(selected)
                else np.empty((0, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint16)
            )
            values["instance_maps"] = apply_transform_batch(
                images, transform, resample=Image.Resampling.NEAREST
            )
        if "frame_mask" in items:
            width, height = calibration["image_size"]
            yy, xx = np.ogrid[:height, :width]
            radius = calibration["valid_radius"]
            center = calibration["projection_parameters"][1:3]
            mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
            masks = np.repeat(mask[None], len(selected), axis=0)
            values["frame_mask"] = apply_transform_batch(
                masks, transform, resample=Image.Resampling.NEAREST
            ).astype(bool)
        return FrameBatch(**values)
