"""MultiScan adapter."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image

from ...io.compression import read_raw_deflate_frames
from ...io.video import read_video_frames
from ...ops.frame import (
    apply_transform_batch,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    transform_world_to_camera,
    valid_camera_mask,
    validate_frame_items,
)
from ...ops.annotation import group_indices, read_ply
from ...ops.geometry import construct_box, load_triangle_mesh
from ..base import DataFormatError, Dataset, UnsupportedOperationError
from ..registry import register_dataset
from ..types import FrameBatch, Segmentation3D


ROTATION_BY_SKY_DIRECTION = {"Up": 0, "Left": 270, "Right": 90, "Down": 180}
_SEGMENTATION_TYPES = {
    "object_instance",
    "part_instance",
    "object_semantic",
    "part_semantic",
}


def _base_label(label: str) -> str:
    return re.sub(r"\.\d+$", "", str(label))


def _matrix(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size != size * size:
        raise DataFormatError(f"MultiScan {name} must contain {size * size} values.")
    return array.reshape(size, size, order="F")


@register_dataset("multiscan", description="MultiScan indoor scenes.")
class MultiScan(Dataset):
    """Read the released Apple-device MultiScan captures and annotations."""

    def __init__(
        self,
        root_dir,
        *,
        ply_mesh_file="{scene_id}/{scene_id}.ply",
        obj_mesh_file="{scene_id}/textured_mesh/{scene_id}.obj",
        annotation_file="{scene_id}/{scene_id}.annotations.json",
        rgb_video_file="{scene_id}/{scene_id}.mp4",
        depth_file="{scene_id}/{scene_id}.depth.zlib",
        confidence_file="{scene_id}/{scene_id}.confidence.zlib",
        alignment_file="{scene_id}/{scene_id}.align.json",
        camera_metadata_file="{scene_id}/{scene_id}.jsonl",
        camera_file="{scene_id}/{scene_id}.json",
        split_file=None,
        object_label_file=None,
        part_label_file=None,
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
        metadata = files("scenezoo.metadata")
        self.mesh_files = {"ply": ply_mesh_file, "textured": obj_mesh_file}
        self.annotation_file = annotation_file
        self.rgb_video_file = rgb_video_file
        self.depth_file = depth_file
        self.confidence_file = confidence_file
        self.alignment_file = alignment_file
        self.camera_metadata_file = camera_metadata_file
        self.camera_file = camera_file
        self.split_file = split_file or metadata / "multiscan_scans_split.csv"
        self.object_label_file = (
            object_label_file or metadata / "multiscan_object_semantic_label_map.csv"
        )
        self.part_label_file = (
            part_label_file or metadata / "multiscan_part_semantic_label_map.csv"
        )

    @cached_property
    def _split_table(self):
        with self.open_file(self.split_file, "rb") as handle:
            table = pd.read_csv(handle, keep_default_na=False)
        if not {"scanId", "split"} <= set(table.columns):
            raise DataFormatError(
                "MultiScan split CSV must contain scanId and split columns."
            )
        return table.astype({"scanId": str, "split": str})

    def _load_splits(self):
        table = self._split_table
        values = table["split"].str.strip()
        return {
            split: table.loc[values == split, "scanId"].tolist()
            for split in ("train", "val", "test")
        }

    def _load_sample_ids(self):
        # The official CSV also lists 16 released scans with no split.
        return self._split_table["scanId"].tolist()

    def _check(self, *, sample_ids, require_complete):
        from ..check import (
            CheckBuilder,
            archive_candidates,
            check_required_scene_paths,
            check_scene_coverage,
            load_expected_ids,
        )

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The MultiScan root does not exist.",
            expected="<root>/<scan_id>/<scan_id>.ply and related files",
            hint="Pass the directory containing extracted MultiScan scan folders.",
        ):
            return builder.finish()
        present = {
            path.name
            for path in self.root_dir.iterdir()
            if path.is_dir() and (path / f"{path.name}.ply").exists()
        }
        expected = None if sample_ids is not None else load_expected_ids(self, builder)
        selected = check_scene_coverage(
            builder,
            present_ids=present,
            expected_ids=expected,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<root>/<scan_id>/<scan_id>.{ply,annotations.json,mp4,jsonl,json}",
        )
        if not present:
            builder.error(
                "no-scans",
                "No extracted MultiScan scan directories were found.",
                path=self.root_dir,
                expected="One directory per scan with a same-named .ply file.",
                hint="Extract the released Apple-device scan archives first.",
            )
            if archive_candidates(self.root_dir):
                builder.info(
                    "archives-need-extraction",
                    "MultiScan archive files are present and must be extracted before use.",
                )
            return builder.finish()
        check_required_scene_paths(
            builder,
            selected,
            {
                "mesh": lambda sid: self.root_dir
                / self.mesh_files["ply"].format(scene_id=sid),
                "annotations": lambda sid: self.root_dir
                / self.annotation_file.format(scene_id=sid),
                "rgb_video": lambda sid: self.root_dir
                / self.rgb_video_file.format(scene_id=sid),
                "camera_metadata": lambda sid: self.root_dir
                / self.camera_metadata_file.format(scene_id=sid),
                "camera_calibration": lambda sid: self.root_dir
                / self.camera_file.format(scene_id=sid),
            },
            expected=(
                "A complete released scan contains .ply, .annotations.json, .mp4, "
                ".jsonl, and .json files named after the scan."
            ),
            hint="Re-download/re-extract the incomplete MultiScan scan archive.",
        )
        depth_pairs = [
            (
                self.root_dir / self.depth_file.format(scene_id=sid),
                self.root_dir / self.confidence_file.format(scene_id=sid),
            )
            for sid in selected
        ]
        partial = [pair for pair in depth_pairs if pair[0].exists() != pair[1].exists()]
        if partial:
            builder.error(
                "incomplete-depth-pairs",
                f"{len(partial)} scans have depth without confidence, or vice versa.",
                expected="Both <scan_id>.depth.zlib and <scan_id>.confidence.zlib.",
                hint="Restore the missing member of each released depth pair.",
            )
        return builder.finish()

    @staticmethod
    def _read_label_table(handle, prefix: str) -> dict:
        table = pd.read_csv(handle)
        name_col = f"{prefix}Name"
        semantic_name_col = f"{prefix}SemanticName"
        semantic_id_col = f"{prefix}SemanticId"
        required = {name_col, semantic_name_col, semantic_id_col}
        if not required <= set(table.columns):
            raise DataFormatError(
                f"MultiScan {prefix} label CSV is missing columns: {sorted(required - set(table.columns))}"
            )
        name_to_id = {
            str(row[name_col]): int(row[semantic_id_col]) for _, row in table.iterrows()
        }
        id_to_names: dict[int, list[str]] = defaultdict(list)
        id_to_name = {}
        for _, row in table.iterrows():
            semantic_id = int(row[semantic_id_col])
            id_to_names[semantic_id].append(str(row[name_col]))
            id_to_name[semantic_id] = str(row[semantic_name_col])
        return {
            "name_to_semantic_id": name_to_id,
            "semantic_id_to_name": id_to_name,
            "semantic_id_to_names": {
                key: tuple(value) for key, value in id_to_names.items()
            },
        }

    def _load_metadata(self):
        with self.open_file(self.object_label_file, "rb") as handle:
            object_mapping = self._read_label_table(handle, "object")
        with self.open_file(self.part_label_file, "rb") as handle:
            part_mapping = self._read_label_table(handle, "part")
        part_mapping["semantic_id_to_name"] = {
            1: "static",
            **part_mapping["semantic_id_to_name"],
        }
        return {
            **{f"object_{key}": value for key, value in object_mapping.items()},
            **{f"part_{key}": value for key, value in part_mapping.items()},
            "sky_direction": self._load_sky_directions(),
        }

    @staticmethod
    def _load_sky_directions() -> dict[str, str]:
        result = {}
        with open(files("scenezoo.metadata") / "multiscan_direction.csv") as handle:
            for row in csv.DictReader(handle):
                result[row["video_id"]] = row["sky_direction"]
        return result

    def _path(self, template: str, sample_id: str) -> Path:
        return self.root_dir / template.format(scene_id=sample_id)

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            with open(path) as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataFormatError(f"Could not read MultiScan JSON: {path}") from exc
        if not isinstance(value, dict):
            raise DataFormatError(f"MultiScan JSON root must be an object: {path}")
        return value

    def get_scene_info(self, sample_id: str) -> dict:
        """Return the acquisition metadata from ``<scan>.json``."""
        return self._load_json(self._path(self.camera_file, sample_id))

    def get_annotations(self, sample_id: str) -> dict:
        """Return objects, parts, scan bounding box, and annotation version."""
        value = self._load_json(self._path(self.annotation_file, sample_id))
        if not isinstance(value.get("objects"), list) or not isinstance(
            value.get("parts"), list
        ):
            raise DataFormatError(
                "MultiScan annotations must contain objects and parts arrays."
            )
        return value

    def _alignment_data(self, sample_id: str) -> dict:
        return self._load_json(self._path(self.alignment_file, sample_id))

    def get_alignment(self, sample_id: str) -> np.ndarray:
        """Return the raw-mesh to gravity-aligned-scene transform."""
        try:
            return _matrix(
                self._alignment_data(sample_id)["coordinate_transform"], 4, "alignment"
            )
        except KeyError as exc:
            raise DataFormatError(
                "MultiScan alignment has no coordinate_transform."
            ) from exc

    def get_reference_alignment(self, sample_id: str):
        """Return ``(reference_scan_id, transform)`` or ``None``."""
        reference = self._alignment_data(sample_id).get("reference_scan_alignment")
        if reference is None:
            return None
        try:
            return str(reference["target_id"]), _matrix(
                reference["transformation"], 4, "reference alignment"
            )
        except KeyError as exc:
            raise DataFormatError("Incomplete MultiScan reference alignment.") from exc

    def get_articulations(self, sample_id: str) -> list[dict]:
        """Return articulation records enriched with object and part identity."""
        annotations = self.get_annotations(sample_id)
        object_by_part = {
            int(part_id): item
            for item in annotations["objects"]
            for part_id in item.get("partIds", ())
        }
        result = []
        for part in annotations["parts"]:
            part_id = int(part["partId"])
            owner = object_by_part.get(part_id)
            if owner is None and part.get("articulations"):
                raise DataFormatError(f"MultiScan part {part_id} has no owning object.")
            for articulation in part.get("articulations", ()):
                result.append(
                    {
                        "object_id": int(owner["objectId"]),
                        "object_label": _base_label(owner["label"]),
                        "part_id": part_id,
                        "part_label": _base_label(part["label"]),
                        "parent_id": int(part["parentId"]),
                        **articulation,
                    }
                )
        return result

    def get_mesh(self, sample_id, *, mesh_type=None):
        mesh_type = mesh_type or "ply"
        try:
            template = self.mesh_files[mesh_type]
        except KeyError as exc:
            raise ValueError(f"Unknown MultiScan mesh type: {mesh_type!r}") from exc
        return load_triangle_mesh(
            self._path(template, sample_id),
            enable_post_processing=mesh_type == "textured",
        )

    def _mesh_arrays(self, sample_id: str):
        path = self._path(self.mesh_files["ply"], sample_id)
        try:
            ply = read_ply(path)
            vertices = np.column_stack(
                [np.asarray(ply["vertex"][axis]) for axis in ("x", "y", "z")]
            )
            faces = np.asarray(ply["face"].data["vertex_indices"])
            if faces.dtype == object:  # ASCII PLY: one array per face.
                faces = np.stack(faces)
            faces = faces.astype(np.int64)
            object_ids = np.asarray(ply["face"].data["objectId"], dtype=np.int32)
            part_ids = np.asarray(ply["face"].data["partId"], dtype=np.int32)
        except (OSError, KeyError, ValueError) as exc:
            raise DataFormatError(f"Invalid MultiScan PLY: {path}") from exc
        if len(faces) != len(object_ids) or len(faces) != len(part_ids):
            raise DataFormatError("MultiScan PLY face annotation lengths differ.")
        return vertices, faces, object_ids, part_ids

    def get_segmentation(self, sample_id, *, segmentation_type="object_instance"):
        if segmentation_type not in _SEGMENTATION_TYPES:
            raise ValueError(
                f"Unknown MultiScan segmentation type: {segmentation_type!r}. "
                f"Expected one of {sorted(_SEGMENTATION_TYPES)}."
            )
        annotations = self.get_annotations(sample_id)
        _, _, object_ids, part_ids = self._mesh_arrays(sample_id)
        objects = {int(item["objectId"]): item for item in annotations["objects"]}
        parts = {int(item["partId"]): item for item in annotations["parts"]}
        removed = {
            object_id
            for object_id, item in objects.items()
            if _base_label(item["label"]).lower() == "remove"
        }
        if segmentation_type == "object_instance":
            labels = object_ids.copy()
            names = {
                object_id: _base_label(item["label"])
                for object_id, item in objects.items()
                if object_id not in removed
            }
            labels[~np.isin(labels, list(names))] = self.invalid_obj_id
        elif segmentation_type == "part_instance":
            labels = part_ids.copy()
            removed_parts = {
                int(part_id)
                for object_id in removed
                for part_id in objects[object_id].get("partIds", ())
            }
            labels[np.isin(labels, list(removed_parts))] = self.invalid_obj_id
            names = {
                part_id: _base_label(item["label"])
                for part_id, item in parts.items()
                if part_id not in removed_parts
            }
            labels[~np.isin(labels, list(names))] = self.invalid_obj_id
        elif segmentation_type == "object_semantic":
            lookup = self.metadata["object_name_to_semantic_id"]
            id_map = np.full(
                max(objects, default=0) + 1,
                self.invalid_obj_id,
                dtype=np.int32,
            )
            for object_id, item in objects.items():
                if object_id not in removed:
                    id_map[object_id] = lookup.get(_base_label(item["label"]), 0)
            labels = np.full_like(object_ids, self.invalid_obj_id)
            valid = (object_ids >= 0) & (object_ids < len(id_map))
            labels[valid] = id_map[object_ids[valid]]
            names = dict(self.metadata["object_semantic_id_to_name"])
        else:
            object_label_by_part = {
                int(part_id): _base_label(item["label"])
                for item in annotations["objects"]
                if int(item["objectId"]) not in removed
                for part_id in item.get("partIds", ())
            }
            object_lookup = self.metadata["object_name_to_semantic_id"]
            part_lookup = self.metadata["part_name_to_semantic_id"]
            id_map = np.full(
                max(parts, default=0) + 1,
                self.invalid_obj_id,
                dtype=np.int32,
            )
            for part_id, item in parts.items():
                object_label = object_label_by_part.get(part_id)
                if object_label is not None:
                    part_label = _base_label(item["label"])
                    id_map[part_id] = (
                        part_lookup[part_label]
                        if object_label in object_lookup and part_label in part_lookup
                        else 1
                    )
            labels = np.full_like(part_ids, self.invalid_obj_id)
            valid = (part_ids >= 0) & (part_ids < len(id_map))
            labels[valid] = id_map[part_ids[valid]]
            names = dict(self.metadata["part_semantic_id_to_name"])
        return Segmentation3D(labels, names, "face", self.invalid_obj_id)

    def get_boxes(self, sample_id, *, box_type="mobb_gravity"):
        annotations = self.get_annotations(sample_id)["objects"]
        names = {
            int(item["objectId"]): _base_label(item["label"])
            for item in annotations
            if _base_label(item["label"]).lower() != "remove"
        }
        if box_type == "obb_gt":
            boxes = {}
            for item in annotations:
                object_id = int(item["objectId"])
                if object_id not in names or "obb" not in item:
                    continue
                obb = item["obb"]
                try:
                    boxes[object_id] = o3d.geometry.OrientedBoundingBox(
                        np.asarray(obb["centroid"], dtype=np.float64),
                        np.asarray(obb["normalizedAxes"], dtype=np.float64)
                        .reshape(3, 3)
                        .T,
                        np.asarray(obb["axesLengths"], dtype=np.float64),
                    )
                except (KeyError, ValueError) as exc:
                    raise DataFormatError(
                        f"Invalid MultiScan OBB for object {object_id}."
                    ) from exc
            return boxes, {object_id: names[object_id] for object_id in boxes}
        vertices, faces, face_labels, _ = self._mesh_arrays(sample_id)
        boxes = {}
        for object_id, face_indices in group_indices(face_labels).items():
            if object_id not in names:
                continue
            vertex_indices = np.unique(faces[face_indices].reshape(-1))
            boxes[object_id] = construct_box(
                vertices[vertex_indices], box_type=box_type
            )
        return boxes, {object_id: names[object_id] for object_id in boxes}

    @staticmethod
    def _find_stream(scene_info: dict, stream_type: str, *, required=True):
        matches = [
            stream
            for stream in scene_info.get("streams", ())
            if stream.get("type") == stream_type
        ]
        if len(matches) > 1:
            raise DataFormatError(f"MultiScan has multiple {stream_type!r} streams.")
        if not matches:
            if required:
                raise DataFormatError(f"MultiScan has no {stream_type!r} stream.")
            return None
        return matches[0]

    def _read_camera_records(
        self,
        sample_id: str,
        *,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        scene_info = self.get_scene_info(sample_id)
        color_stream = self._find_stream(scene_info, "color_camera")
        expected_frames = int(color_stream["number_of_frames"])
        requested = None if indices is None else np.asarray(indices, dtype=np.int64)
        requested_set = None if requested is None else set(map(int, requested))
        records = {}
        path = self._path(self.camera_metadata_file, sample_id)
        line_count = 0
        if requested is None or requested.size:
            try:
                with open(path) as handle:
                    for line_count, line in enumerate(handle, start=1):
                        frame_id = line_count - 1
                        if requested_set is not None and frame_id not in requested_set:
                            if frame_id >= int(requested.max()):
                                break
                            continue
                        records[frame_id] = json.loads(line)
                        if requested_set is not None and len(records) == len(
                            requested_set
                        ):
                            break
            except (OSError, json.JSONDecodeError) as exc:
                raise DataFormatError(
                    f"Invalid MultiScan camera JSONL: {path}"
                ) from exc
        if requested is None:
            if line_count != expected_frames:
                raise DataFormatError(
                    "MultiScan camera metadata and declared frame counts differ."
                )
            order = np.arange(expected_frames, dtype=np.int64)
        else:
            missing = [int(index) for index in requested if int(index) not in records]
            if missing:
                raise DataFormatError(
                    "MultiScan camera metadata is missing frames: "
                    + ", ".join(map(str, missing))
                )
            order = requested

        intrinsics = []
        arkit_camera_to_world = []
        timestamps = []
        exposures = []
        quaternions = []
        euler_angles = []
        for frame_id in order:
            record = records[int(frame_id)]
            try:
                intrinsics.append(_matrix(record["intrinsics"], 3, "intrinsics"))
                arkit_camera_to_world.append(
                    _matrix(record["transform"], 4, "camera transform")
                )
                timestamps.append(int(record["timestamp"]))
                exposures.append(int(record["exposure_duration"]))
                quaternions.append(np.asarray(record["quaternion"], dtype=np.float32))
                euler_angles.append(
                    np.asarray(record["euler_angles"], dtype=np.float32)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DataFormatError(
                    f"Invalid MultiScan camera record at frame {int(frame_id)}."
                ) from exc
        intrinsics_array = np.asarray(intrinsics, dtype=np.float32).reshape(-1, 3, 3)
        arkit_array = np.asarray(arkit_camera_to_world, dtype=np.float32).reshape(
            -1, 4, 4
        )
        denominators = arkit_array[:, 3, 3]
        usable_scale = np.isfinite(denominators) & (np.abs(denominators) >= 1e-8)
        arkit_array[usable_scale] /= denominators[usable_scale, None, None]
        arkit_array[~usable_scale] = np.nan
        camera_to_world = arkit_array @ np.diag([1, -1, -1, 1]).astype(np.float32)
        world_to_camera = np.full_like(camera_to_world, np.nan)
        valid = valid_camera_mask(camera_to_world)
        if valid.any():
            world_to_camera[valid] = np.linalg.inv(
                camera_to_world[valid]
            ) @ self.get_alignment(sample_id)
        return {
            "indices": order,
            "source_frame_count": expected_frames,
            "intrinsics": intrinsics_array,
            "arkit_camera_to_world": arkit_array,
            "camera_to_world": camera_to_world,
            "world_to_camera": world_to_camera.astype(np.float32),
            "timestamps": np.asarray(timestamps, dtype=np.int64),
            "exposure_durations": np.asarray(exposures, dtype=np.int64),
            "quaternions": np.asarray(quaternions, dtype=np.float32).reshape(-1, 4),
            "euler_angles": np.asarray(euler_angles, dtype=np.float32).reshape(-1, 3),
        }

    def get_camera_metadata(self, sample_id: str, *, indices=None) -> dict:
        """Return capture records, including source quaternion and Euler fields."""
        info = self.get_scene_info(sample_id)
        color = self._find_stream(info, "color_camera")
        frame_count = int(color["number_of_frames"])
        if indices is None:
            values = self._read_camera_records(sample_id)
        else:
            selected, _ = select_frame_indices(
                np.arange(frame_count, dtype=np.int64), indices=indices
            )
            values = self._read_camera_records(sample_id, indices=selected)
        values["quaternion_format"] = info.get("camera_orientation_quaternion_format")
        values["euler_angles_format"] = info.get(
            "camera_orientation_euler_angles_format"
        )
        return values

    @staticmethod
    def _stream_size(stream: dict, name: str) -> tuple[int, int]:
        try:
            height, width = map(int, stream["resolution"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataFormatError(f"MultiScan {name} resolution is invalid.") from exc
        if width <= 0 or height <= 0:
            raise DataFormatError(f"MultiScan {name} resolution is invalid.")
        return width, height

    def _read_depth_or_confidence(
        self,
        sample_id: str,
        indices: np.ndarray,
        *,
        stream: dict,
        template: str,
        dtype: str,
        expected_encoding: str,
        frame_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        encoding = stream.get("encoding")
        if encoding != expected_encoding:
            raise UnsupportedOperationError(
                f"MultiScan encoding {encoding!r} is not supported; this adapter supports "
                f"only the released Apple {expected_encoding!r} stream, not Android depth."
            )
        if frame_shape is None:
            width, height = self._stream_size(stream, stream.get("type", "depth"))
            frame_shape = (height, width)
        return read_raw_deflate_frames(
            self._path(template, sample_id),
            indices,
            frame_shape=frame_shape,
            dtype=dtype,
            frame_count=int(stream["number_of_frames"]),
        )

    @staticmethod
    def _valid_camera_rows(intrinsics: np.ndarray, world_to_camera: np.ndarray):
        valid = valid_camera_mask(world_to_camera)
        valid &= np.isfinite(intrinsics).all(axis=(1, 2))
        positions = np.flatnonzero(valid)
        if len(positions):
            valid[positions] &= np.abs(np.linalg.det(intrinsics[positions])) > 1e-8
        return valid

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
        supported = {
            "timestamps",
            "rgb",
            "depth",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
            "confidence_maps",
            "exposure_durations",
        }
        items = validate_frame_items(items, supported)
        scene_info = self.get_scene_info(sample_id)
        color_stream = self._find_stream(scene_info, "color_camera")
        source_frame_count = int(color_stream["number_of_frames"])
        available = np.arange(source_frame_count, dtype=np.int64)

        if indices is None:
            cameras = self._read_camera_records(sample_id)
            valid = self._valid_camera_rows(
                cameras["intrinsics"], cameras["world_to_camera"]
            )
            synchronized = available[valid]
            selected, positions = select_frame_indices(synchronized, step=step)
            cameras = {
                key: value[valid]
                if isinstance(value, np.ndarray) and len(value) == source_frame_count
                else value
                for key, value in cameras.items()
            }
        else:
            selected, _ = select_frame_indices(available, indices=indices, step=step)
            cameras = self._read_camera_records(sample_id, indices=selected)
            valid = self._valid_camera_rows(
                cameras["intrinsics"], cameras["world_to_camera"]
            )
            if not valid.all():
                raise ValueError(
                    "Frames are not synchronized or valid: "
                    + ", ".join(map(str, selected[~valid]))
                )
            positions = np.arange(len(selected), dtype=np.int64)

        if rotate_to_up:
            direction = self._load_sky_directions().get(sample_id, "Up")
            try:
                rotation = ROTATION_BY_SKY_DIRECTION[direction]
            except KeyError as exc:
                raise DataFormatError(
                    f"Unknown MultiScan sky direction for {sample_id}: {direction!r}"
                ) from exc
        else:
            rotation = 0

        rgb_size = self._stream_size(color_stream, "color camera")
        rgb_transform = build_pixel_transform(
            rgb_size,
            output_size=output_size,
            rotation=rotation,
            center_crop=center_crop,
        )
        selected_intrinsics = cameras["intrinsics"][positions]
        values = {
            "indices": selected,
            "source_frame_count": source_frame_count,
            "camera_model": "PINHOLE",
        }
        for item in ("timestamps", "exposure_durations"):
            if item in items:
                values[item] = cameras[item][positions]
        if "world_to_camera" in items:
            values["world_to_camera"] = transform_world_to_camera(
                cameras["world_to_camera"][positions], rgb_transform
            )
        if "rgb_intrinsics" in items:
            values["rgb_intrinsics"] = transform_intrinsics(
                selected_intrinsics, rgb_transform
            ).astype(np.float32)
        if "rgb" in items:
            rgb = read_video_frames(
                self._path(self.rgb_video_file, sample_id), frame_indices=selected
            )
            actual_size = (rgb.shape[2], rgb.shape[1]) if len(rgb) else rgb_size
            if actual_size != rgb_size:
                raise DataFormatError(
                    f"MultiScan video size {actual_size} differs from metadata {rgb_size}."
                )
            values["rgb"] = apply_transform_batch(rgb, rgb_transform)

        depth_items = {"depth", "depth_intrinsics", "confidence_maps"} & set(items)
        if depth_items:
            depth_stream = self._find_stream(scene_info, "lidar_sensor")
            if int(depth_stream["number_of_frames"]) != source_frame_count:
                raise DataFormatError("MultiScan color and depth frame counts differ.")
            unit = scene_info.get("depth_unit", "m")
            if unit != "m":
                raise UnsupportedOperationError(
                    f"MultiScan depth unit {unit!r} is not supported; expected metres."
                )
            depth_size = self._stream_size(depth_stream, "depth")
            depth_transform = build_pixel_transform(
                depth_size,
                output_size=output_size,
                rotation=rotation,
                center_crop=center_crop,
            )
            if "depth_intrinsics" in items:
                if depth_stream.get("intrinsics") is None:
                    depth_intrinsics = transform_intrinsics(
                        selected_intrinsics,
                        build_pixel_transform(rgb_size, output_size=depth_size),
                    )
                else:
                    fixed = _matrix(depth_stream["intrinsics"], 3, "depth intrinsics")
                    depth_intrinsics = np.broadcast_to(
                        fixed, (len(selected), 3, 3)
                    ).copy()
                values["depth_intrinsics"] = transform_intrinsics(
                    depth_intrinsics, depth_transform
                ).astype(np.float32)
            if "depth" in items:
                depth = self._read_depth_or_confidence(
                    sample_id,
                    selected,
                    stream=depth_stream,
                    template=self.depth_file,
                    dtype="<f2",
                    expected_encoding="float16_zlib",
                ).astype(np.float32)
                values["depth"] = apply_transform_batch(
                    depth, depth_transform, resample=Image.Resampling.NEAREST
                )
            if "confidence_maps" in items:
                confidence_stream = self._find_stream(scene_info, "confidence_map")
                if int(confidence_stream["number_of_frames"]) != source_frame_count:
                    raise DataFormatError(
                        "MultiScan color and confidence frame counts differ."
                    )
                confidence = self._read_depth_or_confidence(
                    sample_id,
                    selected,
                    stream=confidence_stream,
                    template=self.confidence_file,
                    dtype="u1",
                    expected_encoding="uint8_zlib",
                    frame_shape=(depth_size[1], depth_size[0]),
                )
                values["confidence_maps"] = apply_transform_batch(
                    confidence, depth_transform, resample=Image.Resampling.NEAREST
                )
        return FrameBatch(**values)
