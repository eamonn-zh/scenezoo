"""Stanford Large-Scale 3D Indoor Spaces (S3DIS) adapter."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from ...ops.annotation import group_indices
from ...ops.geometry import construct_box
from ..base import DataFormatError, Dataset
from ..registry import register_dataset
from ..types import PointBatch, Segmentation3D


S3DIS_CLASS_NAMES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)

# The five foreground categories used by the common S3DIS detection protocol.
S3DIS_DETECTION_CLASS_NAMES = ("table", "chair", "sofa", "bookcase", "board")

_AREA_PATTERN = re.compile(r"^Area_([1-6])$")
_ROOM_PATTERN = re.compile(r"^(.*)_([0-9]+)$")
_DATASET_DIRECTORY_NAMES = (
    "Stanford3dDataset_v1.2_Aligned_Version",
    "Stanford3dDataset_v1.2",
)
_UNICODE_WHITESPACE = str.maketrans(
    {
        "\ufeff": "",
        "\x00": " ",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
    }
)


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", value)
    )


class S3DISPointData(PointBatch):
    """Room points with semantic and instance labels in matching row order."""

    __slots__ = ()


@register_dataset(
    "s3dis",
    aliases=("stanford3d", "stanford-3d"),
    description=(
        "S3DIS room point clouds, point-domain semantic/instance labels, and boxes."
    ),
)
class S3DIS(Dataset):
    """Adapter for the raw S3DIS v1.2 and v1.2 aligned releases.

    Point and label order is derived from a deterministic concatenation of the
    files in each room's ``Annotations`` directory. This is the same approach
    used by PointNet and MMDetection3D and avoids assuming that the standalone
    room text file has an identical row order.
    """

    class_names = S3DIS_CLASS_NAMES
    detection_class_names = S3DIS_DETECTION_CLASS_NAMES

    def __init__(
        self,
        root_dir,
        *,
        dataset_dir=None,
        annotation_dir="Annotations",
        room_file="{room_name}.txt",
        unknown_class="clutter",
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
        if unknown_class not in self.class_names:
            raise ValueError(
                f"unknown_class must be one of {', '.join(self.class_names)}."
            )
        self.dataset_root = self._resolve_dataset_root(dataset_dir)
        self.annotation_dir = str(annotation_dir)
        self.room_file = str(room_file)
        self.unknown_class = unknown_class

    def _resolve_dataset_root(self, dataset_dir) -> Path:
        if dataset_dir is not None:
            path = Path(dataset_dir).expanduser()
            return path if path.is_absolute() else self.root_dir / path
        if any((self.root_dir / f"Area_{area}").is_dir() for area in range(1, 7)):
            return self.root_dir
        for name in _DATASET_DIRECTORY_NAMES:
            candidate = self.root_dir / name
            if candidate.is_dir():
                return candidate
        return self.root_dir

    def _discover_rooms(self) -> dict[str, list[str]]:
        areas: dict[str, list[str]] = {}
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"S3DIS dataset directory does not exist: {self.dataset_root}"
            )
        area_paths = sorted(
            (
                path
                for path in self.dataset_root.iterdir()
                if path.is_dir() and _AREA_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: int(_AREA_PATTERN.fullmatch(path.name).group(1)),
        )
        if not area_paths:
            raise DataFormatError(
                f"No Area_1 ... Area_6 directories found under {self.dataset_root}."
            )
        for area_path in area_paths:
            area_key = area_path.name.lower()
            room_ids = [
                f"{area_path.name}/{path.name}"
                for path in area_path.iterdir()
                if path.is_dir() and (path / self.annotation_dir).is_dir()
            ]
            areas[area_key] = sorted(room_ids, key=_natural_key)
        return areas

    def _load_splits(self):
        return self._discover_rooms()

    def _check(self, *, sample_ids, require_complete):
        from ..check import CheckBuilder, archive_candidates, check_scene_coverage

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The S3DIS root does not exist.",
            expected="<root>/Area_1/<room>/Annotations/*.txt through Area_6",
            hint="Pass the extracted Stanford3dDataset_v1.2 root.",
        ):
            return builder.finish()
        if not builder.require_directory(
            self.dataset_root,
            code="missing-dataset-directory",
            message="The resolved S3DIS dataset directory is missing.",
            expected="Area_1 ... Area_6 directories.",
            hint="Extract Stanford3dDataset_v1.2.zip/tar before using the adapter.",
        ):
            return builder.finish()
        areas = {
            path.name: path
            for path in self.dataset_root.iterdir()
            if path.is_dir() and _AREA_PATTERN.fullmatch(path.name)
        }
        builder.stats["areas_found"] = len(areas)
        missing_areas = [
            f"Area_{index}" for index in range(1, 7) if f"Area_{index}" not in areas
        ]
        if missing_areas:
            builder.add(
                "error" if require_complete else "warning",
                "missing-areas",
                "Missing official areas: " + ", ".join(missing_areas),
                expected="Area_1, Area_2, Area_3, Area_4, Area_5, and Area_6.",
                hint="Extract the missing area archives, or use a partial dataset intentionally.",
            )
        room_paths = {
            f"{area_name}/{room.name}": room
            for area_name, area in areas.items()
            for room in area.iterdir()
            if room.is_dir()
        }
        selected = check_scene_coverage(
            builder,
            present_ids=room_paths,
            expected_ids=None,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<root>/Area_N/<room>/Annotations/<class>_<instance>.txt",
        )
        if not room_paths:
            builder.error(
                "no-rooms",
                "No S3DIS room directories were found.",
                path=self.dataset_root,
                expected="Area_N/<room_name>/Annotations/*.txt",
                hint="Extract the official S3DIS v1.2 archive, preserving area folders.",
            )
            if archive_candidates(self.root_dir):
                builder.info(
                    "archives-need-extraction",
                    "S3DIS archives must be extracted before use.",
                )
            return builder.finish()
        missing_annotations = [
            room_paths[sid] / self.annotation_dir
            for sid in selected
            if not (room_paths[sid] / self.annotation_dir).is_dir()
        ]
        builder.missing_paths(
            missing_annotations,
            code="missing-annotation-directories",
            label="room annotation directories",
            expected="Every room has an Annotations directory.",
            hint="Use the raw S3DIS release, not only a preprocessed point-cloud export.",
        )
        empty_annotations = [
            room_paths[sid] / self.annotation_dir
            for sid in selected
            if (room_paths[sid] / self.annotation_dir).is_dir()
            and not any((room_paths[sid] / self.annotation_dir).glob("*.txt"))
        ]
        builder.missing_paths(
            empty_annotations,
            code="empty-annotation-directories",
            label="annotation directories with object text files",
            expected="At least one <class>_<instance>.txt file per room.",
            hint="Re-extract or restore the room's Annotations contents.",
        )
        return builder.finish()

    def _load_metadata(self):
        metadata = {}
        for sample_id in self.get_ids():
            area_name, room_name = sample_id.split("/", 1)
            room_match = _ROOM_PATTERN.fullmatch(room_name)
            annotation_path = self.dataset_root / sample_id / self.annotation_dir
            metadata[sample_id] = {
                "area": int(_AREA_PATTERN.fullmatch(area_name).group(1)),
                "area_name": area_name,
                "room_name": room_name,
                "room_type": room_match.group(1) if room_match else room_name,
                "room_number": int(room_match.group(2)) if room_match else None,
                "annotation_count": sum(1 for _ in annotation_path.glob("*.txt")),
                "alignment_angle": self._alignment_angles.get(sample_id),
            }
        return metadata

    @cached_property
    def _alignment_angles(self) -> dict[str, float]:
        angles = {}
        for area_key in (key for key in self.splits if key.startswith("area_")):
            area_name = f"Area_{area_key.removeprefix('area_')}"
            path = self.dataset_root / area_name / f"{area_name}_alignmentAngle.txt"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split()
                    if len(fields) != 2:
                        raise DataFormatError(
                            f"Invalid S3DIS alignment row at {path}:{line_number}."
                        )
                    room_name, angle_text = fields
                    try:
                        angle = float(angle_text)
                    except ValueError as exc:
                        raise DataFormatError(
                            f"Invalid S3DIS alignment angle at {path}:{line_number}."
                        ) from exc
                    if not np.isfinite(angle):
                        raise DataFormatError(
                            f"Non-finite S3DIS alignment angle at {path}:{line_number}."
                        )
                    angles[f"{area_name}/{room_name}"] = angle
        return angles

    def _room_path(self, sample_id: str) -> Path:
        if sample_id not in self._sample_ids:
            raise KeyError(f"Unknown S3DIS room ID: {sample_id!r}")
        return self.dataset_root / sample_id

    @cached_property
    def _sample_ids(self) -> frozenset[str]:
        return frozenset(self.get_ids())

    @staticmethod
    def _load_xyzrgb(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"S3DIS point file does not exist: {path}")
        if path.stat().st_size == 0:
            raise DataFormatError(f"S3DIS point file is empty: {path}")
        try:
            points = np.loadtxt(path, dtype=np.float32)
        except (UnicodeDecodeError, ValueError):
            # The official v1.2 aligned archive has a known non-ASCII whitespace
            # character in Area_5/hallway_6/Annotations/ceiling_1.txt.
            try:
                with path.open("r", encoding="utf-8-sig") as handle:
                    lines = (line.translate(_UNICODE_WHITESPACE) for line in handle)
                    points = np.loadtxt(lines, dtype=np.float32)
            except (UnicodeDecodeError, ValueError) as exc:
                raise DataFormatError(
                    f"Cannot parse S3DIS XYZRGB file: {path}"
                ) from exc
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 6:
            raise DataFormatError(
                f"S3DIS XYZRGB file must contain exactly six columns: {path}"
            )
        if not np.isfinite(points).all():
            raise DataFormatError(
                f"S3DIS XYZRGB file contains non-finite values: {path}"
            )
        colors = points[:, 3:]
        if np.any(colors < 0) or np.any(colors > 255):
            raise DataFormatError(f"S3DIS RGB values must be in [0, 255]: {path}")
        return points

    def _annotation_files(self, sample_id: str) -> list[Path]:
        path = self._room_path(sample_id) / self.annotation_dir
        files = sorted(path.glob("*.txt"), key=lambda value: _natural_key(value.stem))
        if not files:
            raise DataFormatError(f"No S3DIS instance annotations found in {path}.")
        return files

    def _class_name(self, annotation_path: Path) -> str:
        class_name, separator, _ = annotation_path.stem.rpartition("_")
        if not separator:
            class_name = annotation_path.stem
        return class_name if class_name in self.class_names else self.unknown_class

    def get_alignment_angle(self, sample_id) -> float:
        """Return the official Z-axis alignment angle in degrees."""

        self._room_path(sample_id)
        try:
            return self._alignment_angles[sample_id]
        except KeyError as exc:
            raise DataFormatError(
                f"No official alignment angle is available for {sample_id!r}."
            ) from exc

    def _align_xyz(self, sample_id: str, xyz: np.ndarray) -> np.ndarray:
        angle = -np.deg2rad(self.get_alignment_angle(sample_id))
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        return (xyz - center) @ rotation.T + center

    def get_cross_validation_split(self, test_area=5) -> tuple[list[str], list[str]]:
        """Return ``(train_ids, test_ids)`` for one held-out S3DIS area."""

        if isinstance(test_area, str):
            match = re.fullmatch(r"(?:area_)?([1-6])", test_area.strip().lower())
            if match is None:
                raise ValueError("test_area must identify an area from 1 through 6.")
            area_number = int(match.group(1))
        elif isinstance(test_area, (int, np.integer)) and 1 <= int(test_area) <= 6:
            area_number = int(test_area)
        else:
            raise ValueError("test_area must identify an area from 1 through 6.")
        key = f"area_{area_number}"
        if key not in self.splits:
            raise DataFormatError(
                f"The local S3DIS release does not contain Area_{area_number}."
            )
        test_ids = list(self.splits[key])
        train_ids = [
            sample_id
            for area_key, room_ids in self.splits.items()
            if area_key.startswith("area_") and area_key != key
            for sample_id in room_ids
        ]
        return train_ids, test_ids

    def get_points(
        self, sample_id, *, align_to_axes=False, shift_to_origin=False
    ) -> S3DISPointData:
        """Return annotation-aligned XYZ, RGB, semantic, and instance arrays."""

        point_parts = []
        semantic_parts = []
        instance_parts = []
        instance_id_to_name = {}
        class_to_id = {name: index for index, name in enumerate(self.class_names)}
        for instance_id, annotation_path in enumerate(
            self._annotation_files(sample_id)
        ):
            values = self._load_xyzrgb(annotation_path)
            class_name = self._class_name(annotation_path)
            point_parts.append(values)
            semantic_parts.append(
                np.full(len(values), class_to_id[class_name], dtype=np.int32)
            )
            instance_parts.append(np.full(len(values), instance_id, dtype=np.int32))
            instance_id_to_name[instance_id] = class_name
        points = np.concatenate(point_parts, axis=0)
        xyz = points[:, :3].astype(np.float32, copy=True)
        if align_to_axes:
            xyz = self._align_xyz(sample_id, xyz)
        if shift_to_origin:
            xyz -= xyz.min(axis=0)
        return S3DISPointData(
            xyz=xyz,
            rgb=np.rint(points[:, 3:]).astype(np.uint8),
            semantic_labels=np.concatenate(semantic_parts),
            instance_labels=np.concatenate(instance_parts),
            semantic_id_to_name=dict(enumerate(self.class_names)),
            instance_id_to_name=instance_id_to_name,
            invalid_id=self.invalid_obj_id,
        )

    def get_point_data(
        self, sample_id, *, align_to_axes=False, shift_to_origin=False
    ) -> S3DISPointData:
        """Dataset-specific alias for :meth:`get_points`."""

        return self.get_points(
            sample_id,
            align_to_axes=align_to_axes,
            shift_to_origin=shift_to_origin,
        )

    def get_raw_room_points(
        self, sample_id, *, align_to_axes=False, shift_to_origin=False
    ) -> np.ndarray:
        """Return the standalone room XYZRGB file without reordering its rows."""

        room_path = self._room_path(sample_id)
        path = room_path / self.room_file.format(room_name=room_path.name)
        if not path.is_file():
            # The official Area_6/copyRoom_1 room file is misspelled as
            # copy_Room_1.txt in some v1.2 aligned archives.
            candidates = sorted(room_path.glob("*.txt"))
            if len(candidates) == 1:
                path = candidates[0]
        values = self._load_xyzrgb(path).copy()
        if align_to_axes:
            values[:, :3] = self._align_xyz(sample_id, values[:, :3])
        if shift_to_origin:
            values[:, :3] -= values[:, :3].min(axis=0)
        return values

    def get_point_cloud(self, sample_id, *, align_to_axes=False, shift_to_origin=False):
        """Return an Open3D point cloud ordered like :meth:`get_segmentation`."""

        data = self.get_points(
            sample_id,
            align_to_axes=align_to_axes,
            shift_to_origin=shift_to_origin,
        )
        return data.to_point_cloud()

    def get_segmentation(self, sample_id, *, segmentation_type="instance"):
        data = self.get_points(sample_id)
        if segmentation_type == "instance":
            return Segmentation3D(
                data.instance_labels,
                data.instance_id_to_name,
                "point",
                self.invalid_obj_id,
            )
        if segmentation_type == "semantic":
            return Segmentation3D(
                data.semantic_labels,
                dict(enumerate(self.class_names)),
                "point",
                self.invalid_obj_id,
            )
        raise ValueError(
            "Unknown S3DIS segmentation type. Expected 'instance' or 'semantic'."
        )

    def get_boxes(
        self,
        sample_id,
        *,
        box_type="aabb",
        classes: Iterable[str] | None = None,
        align_to_axes=False,
    ):
        data = self.get_points(sample_id, align_to_axes=align_to_axes)
        selected = (
            None
            if classes is None
            else ({classes} if isinstance(classes, str) else set(classes))
        )
        if selected is not None:
            unknown = selected - set(self.class_names)
            if unknown:
                raise ValueError(
                    "Unknown S3DIS box classes: " + ", ".join(sorted(unknown))
                )
        boxes = {}
        names = {}
        members = group_indices(data.instance_labels)
        for instance_id, class_name in data.instance_id_to_name.items():
            if selected is not None and class_name not in selected:
                continue
            boxes[instance_id] = construct_box(
                data.xyz[members[instance_id]], box_type=box_type
            )
            names[instance_id] = class_name
        return boxes, names
