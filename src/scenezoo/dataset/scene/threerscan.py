"""3RScan adapter."""

from __future__ import annotations

import copy
import io
import json
from functools import cached_property
from importlib.resources import files
from pathlib import Path

from cachetools import LRUCache, cachedmethod
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image

from ...io.image import read_image
from ...io.archive import read_zip_members
from ...io.rio import RIOSequenceInfo, parse_rio_sequence_info
from ...ops.annotation import (
    group_indices,
    parse_sstk_groups,
    parse_sstk_segments,
    read_ply,
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
from ...ops.geometry import construct_box, load_triangle_mesh
from ..base import DataFormatError, Dataset, UnsupportedOperationError
from ..registry import register_dataset
from ..types import FrameBatch, Segmentation3D


_DATASET_METADATA_URL = "https://campar.in.tum.de/public_datasets/3RScan/3RScan.json"
_SEMANTIC_ATTRIBUTES = {
    "global": "globalId",
    "nyu40": "NYU40",
    "eigen13": "Eigen13",
    "rio27": "RIO27",
    "rio7": "globalId",
}


def _matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size != 16:
        raise DataFormatError(f"3RScan {name} must contain 16 values.")
    matrix = array.reshape(4, 4, order="F")
    if not np.isfinite(matrix).all():
        raise DataFormatError(f"3RScan {name} contains non-finite values.")
    return matrix


@register_dataset(
    "3rscan",
    aliases=("threerscan",),
    description="3RScan indoor scene sequences.",
)
class ThreeRScan(Dataset):
    """Read 3RScan meshes, annotations, scene changes, and RGB-D sequences."""

    def __init__(
        self,
        root_dir,
        *,
        textured_mesh_file="{scene_id}/mesh.refined.v2.obj",
        instance_aligned_mesh_file="{scene_id}/labels.instances.align.annotated.v2.ply",
        instance_mesh_file="{scene_id}/labels.instances.annotated.v2.ply",
        aggregation_file="{scene_id}/semseg.v2.json",
        oversegmentation_file="{scene_id}/mesh.refined.0.010000.segs.json",
        sequence_archive="{scene_id}/sequence.zip",
        dataset_metadata_file=None,
        label_mapping_file=None,
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
        self.scans_dir = (
            self.root_dir / "download"
            if (self.root_dir / "download").is_dir()
            else self.root_dir
        )
        self.mesh_files = {
            "instance": instance_mesh_file,
            "instance_aligned": instance_aligned_mesh_file,
            "textured": textured_mesh_file,
        }
        self.aggregation_file = aggregation_file
        self.oversegmentation_file = oversegmentation_file
        self.sequence_archive = sequence_archive
        self.dataset_metadata_file = (
            dataset_metadata_file or self._find_dataset_metadata()
        )
        self.label_mapping_file = (
            label_mapping_file
            or files("scenezoo.metadata") / "3rscan_semantic_mapping.csv"
        )
        self._sequence_info_cache = LRUCache(maxsize=64)

    def _find_dataset_metadata(self):
        candidates = [
            self.root_dir / "3RScan.json",
            self.scans_dir.parent / "3RScan.json",
        ]
        return next(
            (path for path in candidates if path.is_file()), _DATASET_METADATA_URL
        )

    def _scan_path(self, template: str, sample_id: str) -> Path:
        return self.scans_dir / template.format(scene_id=sample_id)

    @cached_property
    def _dataset_index(self) -> dict[str, dict]:
        with self.open_file(self.dataset_metadata_file, "r") as handle:
            try:
                environments = json.load(handle)
            except json.JSONDecodeError as exc:
                raise DataFormatError("Invalid 3RScan.json metadata.") from exc
        if not isinstance(environments, list):
            raise DataFormatError("3RScan.json must contain a list of environments.")

        index: dict[str, dict] = {}
        for environment in environments:
            try:
                reference_id = str(environment["reference"])
                split = str(environment["type"])
                rescans = environment.get("scans", [])
            except (KeyError, TypeError) as exc:
                raise DataFormatError("Incomplete environment in 3RScan.json.") from exc
            split = "val" if split == "validation" else split
            if split not in {"train", "val", "test"} or not isinstance(rescans, list):
                raise DataFormatError(f"Invalid 3RScan environment {reference_id!r}.")
            records = [(reference_id, None)]
            for rescan in rescans:
                try:
                    records.append((str(rescan["reference"]), rescan))
                except (KeyError, TypeError) as exc:
                    raise DataFormatError(
                        f"Invalid rescan in 3RScan environment {reference_id!r}."
                    ) from exc
            for scan_id, changes in records:
                if scan_id in index:
                    raise DataFormatError(
                        f"Duplicate scan ID in 3RScan.json: {scan_id}"
                    )
                index[scan_id] = {
                    "scan_id": scan_id,
                    "split": split,
                    "reference_id": reference_id,
                    "is_reference": scan_id == reference_id,
                    "has_annotations": split != "test" or scan_id == reference_id,
                    "ambiguity": copy.deepcopy(environment.get("ambiguity", [])),
                    "changes": copy.deepcopy(changes),
                }
        return index

    def _load_splits(self):
        return {
            split: [
                scan_id
                for scan_id, value in self._dataset_index.items()
                if value["split"] == split
            ]
            for split in ("train", "val", "test")
        }

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
            message="The 3RScan root does not exist.",
            expected="<root>/3RScan.json and <root>/download/<scan_uuid>/...",
            hint="Pass either the release root or its download directory.",
        ):
            return builder.finish()
        if not self.scans_dir.is_dir():
            builder.error(
                "missing-download-directory",
                "The configured 3RScan scan directory is missing.",
                path=self.scans_dir,
                expected="<root>/download/<scan_uuid>/ or <root>/<scan_uuid>/",
                hint="Extract the per-scan downloads while preserving UUID directories.",
            )
            return builder.finish()
        local_metadata = next(
            (
                path
                for path in (
                    self.root_dir / "3RScan.json",
                    self.scans_dir.parent / "3RScan.json",
                )
                if path.is_file()
            ),
            None,
        )
        if local_metadata is None:
            severity = "error" if self.offline else "warning"
            builder.add(
                severity,
                "missing-local-index",
                "3RScan.json is not available locally.",
                expected="3RScan.json beside the download directory.",
                hint=(
                    "Download the official index. Online mode can cache the public copy, "
                    "but offline checks and reproducible layouts need a local file."
                ),
            )
        present = {
            path.name
            for path in self.scans_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        expected = None if sample_ids is not None else load_expected_ids(self, builder)
        selected = check_scene_coverage(
            builder,
            present_ids=present,
            expected_ids=expected,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<download>/<scan_uuid>/{mesh.refined.v2.obj,sequence.zip,...}",
        )
        if not present:
            builder.error(
                "no-scans",
                "No extracted 3RScan UUID directories were found.",
                path=self.scans_dir,
                expected="One directory per reference scan or rescan UUID.",
                hint="Extract/move downloaded scan contents into download/<scan_uuid>.",
            )
            if archive_candidates(self.scans_dir):
                builder.info(
                    "outer-archives-need-extraction",
                    "Outer download archives must be extracted; sequence.zip stays compressed.",
                )
            return builder.finish()
        check_required_scene_paths(
            builder,
            selected,
            {
                "textured_mesh": lambda sid: self._scan_path(
                    self.mesh_files["textured"], sid
                ),
                "sequence_archive": lambda sid: self._scan_path(
                    self.sequence_archive, sid
                ),
            },
            expected="mesh.refined.v2.obj plus sequence.zip in every scan directory.",
            hint="Re-download the incomplete 3RScan scan; keep sequence.zip compressed.",
        )
        if expected is not None:
            index = self._dataset_index
            annotated = tuple(
                sid for sid in selected if index.get(sid, {}).get("has_annotations")
            )
            check_required_scene_paths(
                builder,
                annotated,
                {
                    "instance_mesh": lambda sid: self._scan_path(
                        self.mesh_files["instance"], sid
                    ),
                    "semantic_groups": lambda sid: self._scan_path(
                        self.aggregation_file, sid
                    ),
                    "oversegmentation": lambda sid: self._scan_path(
                        self.oversegmentation_file, sid
                    ),
                },
                expected=(
                    "Annotated scans need labels.instances.annotated.v2.ply, "
                    "semseg.v2.json, and mesh.refined.0.010000.segs.json."
                ),
                hint="Download the 3RScan semantic annotation assets.",
            )
        builder.info(
            "sequence-archive-supported",
            "sequence.zip is read on demand and should not be extracted.",
        )
        return builder.finish()

    @staticmethod
    def _semantic_spaces(table: pd.DataFrame) -> dict[str, dict[int, str]]:
        columns = {
            "global": ("Global ID", "Label"),
            "nyu40": ("Unnamed: 2", "NYU40 Mapping"),
            "eigen13": ("Unnamed: 4", "Eigen Mapping"),
            "rio27": ("Unnamed: 6", "RIO27 Mapping"),
            "rio7": ("Unnamed: 8", "RIO7 Mapping"),
        }
        spaces: dict[str, dict[int, str]] = {}
        for space, (id_column, name_column) in columns.items():
            mapping: dict[int, str] = {}
            for identifier, name in zip(table[id_column], table[name_column]):
                if pd.isna(identifier) or pd.isna(name):
                    continue
                identifier = int(identifier)
                if identifier == 0:
                    continue
                name = str(name)
                previous = mapping.setdefault(identifier, name)
                if previous != name:
                    raise DataFormatError(
                        f"Conflicting 3RScan {space} label {identifier}: "
                        f"{previous!r} and {name!r}."
                    )
            spaces[space] = mapping
        return spaces

    def _load_metadata(self):
        with self.open_file(self.label_mapping_file, "rb") as handle:
            table = pd.read_csv(handle, skiprows=1)
        required = {
            "Global ID",
            "Label",
            "Unnamed: 2",
            "NYU40 Mapping",
            "Unnamed: 4",
            "Eigen Mapping",
            "Unnamed: 6",
            "RIO27 Mapping",
            "Unnamed: 8",
            "RIO7 Mapping",
        }
        if not required <= set(table.columns):
            raise DataFormatError("Invalid 3RScan semantic mapping table.")
        spaces = self._semantic_spaces(table)
        label_mapping = {}
        global_to_semantic = {"rio7": {}}
        for row in table.to_dict(orient="records"):
            global_id = int(row["Global ID"])
            values = {
                "global_id": global_id,
                "nyu40_id": int(row["Unnamed: 2"]),
                "nyu40_name": str(row["NYU40 Mapping"]),
                "eigen13_id": int(row["Unnamed: 4"]),
                "eigen13_name": str(row["Eigen Mapping"]),
                "rio27_id": int(row["Unnamed: 6"]),
                "rio27_name": str(row["RIO27 Mapping"]),
                "rio7_id": int(row["Unnamed: 8"]),
                "rio7_name": str(row["RIO7 Mapping"]),
            }
            label_mapping[str(row["Label"])] = values
            global_to_semantic["rio7"][global_id] = values["rio7_id"]
        return {
            "label_mapping": label_mapping,
            "semantic_spaces": spaces,
            "global_to_semantic": global_to_semantic,
            "scan_count": len(self._dataset_index),
            "reference_count": sum(
                v["is_reference"] for v in self._dataset_index.values()
            ),
        }

    def get_scene_info(self, sample_id: str) -> dict:
        """Return split, reference grouping, and annotation availability."""

        try:
            return copy.deepcopy(self._dataset_index[sample_id])
        except KeyError as exc:
            raise KeyError(f"Unknown 3RScan scan ID: {sample_id!r}") from exc

    def get_reference_id(self, sample_id: str) -> str:
        """Return the reference scan ID of a scan (itself for references)."""

        return self.get_scene_info(sample_id)["reference_id"]

    def get_scene_alignment(self, sample_id: str) -> np.ndarray:
        """Return the metre-valued transform from scan to reference coordinates."""

        info = self.get_scene_info(sample_id)
        if info["is_reference"]:
            return np.eye(4, dtype=np.float32)
        changes = info["changes"]
        if not isinstance(changes, dict) or "transform" not in changes:
            raise UnsupportedOperationError(
                f"3RScan does not publish the scan-to-reference transform for {sample_id!r}."
            )
        return _matrix(changes["transform"], "scene transform")

    def get_changes(self, sample_id: str) -> dict:
        """Return the official ambiguity and object-change annotations."""

        info = self.get_scene_info(sample_id)
        return {
            "ambiguity": info["ambiguity"],
            "rigid": []
            if info["changes"] is None
            else info["changes"].get("rigid", []),
            "nonrigid": []
            if info["changes"] is None
            else info["changes"].get("nonrigid", []),
            "removed": []
            if info["changes"] is None
            else info["changes"].get("removed", []),
        }

    def get_mesh(self, sample_id, *, mesh_type=None):
        if mesh_type is None:
            instance_path = self._scan_path(self.mesh_files["instance"], sample_id)
            mesh_type = "instance" if instance_path.is_file() else "textured"
        try:
            template = self.mesh_files[mesh_type]
        except KeyError as exc:
            raise ValueError(f"Unknown 3RScan mesh type: {mesh_type!r}") from exc
        path = self._scan_path(template, sample_id)
        if not path.is_file() and mesh_type.startswith("instance"):
            raise UnsupportedOperationError(
                f"3RScan instance mesh is not published for {sample_id!r}."
            )
        return load_triangle_mesh(path, enable_post_processing=mesh_type == "textured")

    def get_annotations(self, sample_id: str) -> dict[int, dict]:
        """Return the per-instance SSTK groups for an annotated scan."""

        path = self._scan_path(self.aggregation_file, sample_id)
        if not path.is_file():
            raise UnsupportedOperationError(
                f"3RScan semantic annotations are not published for {sample_id!r}."
            )
        try:
            with path.open() as handle:
                return parse_sstk_groups(json.load(handle))
        except json.JSONDecodeError as exc:
            raise DataFormatError(f"Invalid 3RScan annotation file: {path}") from exc

    def get_oversegmentation(self, sample_id: str) -> np.ndarray:
        """Return the vertex-domain segment IDs used by ``semseg.v2.json``."""

        path = self._scan_path(self.oversegmentation_file, sample_id)
        if not path.is_file():
            raise UnsupportedOperationError(
                f"3RScan oversegmentation is not published for {sample_id!r}."
            )
        try:
            with path.open() as handle:
                return parse_sstk_segments(json.load(handle))
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                f"Invalid 3RScan oversegmentation file: {path}"
            ) from exc

    def _vertex_fields(
        self, sample_id: str, names: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        path = self._scan_path(self.mesh_files["instance"], sample_id)
        if not path.is_file():
            raise UnsupportedOperationError(
                f"3RScan vertex annotations are not published for {sample_id!r}."
            )
        data = read_ply(path)
        try:
            return {name: np.asarray(data["vertex"][name]) for name in names}
        except (KeyError, ValueError) as exc:
            raise DataFormatError(f"Missing 3RScan vertex field in {path}.") from exc

    def get_segmentation(self, sample_id, *, segmentation_type="instance"):
        """Return instance, Global, NYU40, Eigen13, RIO27, or RIO7 labels."""

        if segmentation_type == "instance":
            attribute = "objectId"
            names = {
                object_id: str(value["label"])
                for object_id, value in self.get_annotations(sample_id).items()
                if str(value.get("label", "")).lower() != "remove"
            }
        else:
            try:
                attribute = _SEMANTIC_ATTRIBUTES[segmentation_type]
            except KeyError as exc:
                valid = ", ".join(("instance", *_SEMANTIC_ATTRIBUTES))
                raise ValueError(
                    f"Unknown 3RScan segmentation type {segmentation_type!r}; use {valid}."
                ) from exc
            names = self.metadata["semantic_spaces"][segmentation_type]
        labels = self._vertex_fields(sample_id, (attribute,))[attribute].astype(
            np.int32
        )
        if segmentation_type == "rio7":
            global_ids = labels
            labels = np.zeros_like(global_ids)
            mapping = self.metadata["global_to_semantic"]["rio7"]
            lookup = np.zeros(max(mapping) + 1, dtype=np.int32)
            lookup[np.fromiter(mapping, dtype=np.int32)] = np.fromiter(
                mapping.values(), dtype=np.int32
            )
            known = global_ids < len(lookup)
            labels[known] = lookup[global_ids[known]]
        if self.invalid_obj_id != 0:
            labels = labels.copy()
            labels[labels == 0] = self.invalid_obj_id
        present = {
            int(value) for value in np.unique(labels) if value != self.invalid_obj_id
        }
        return Segmentation3D(
            labels,
            {
                identifier: names[identifier]
                for identifier in present
                if identifier in names
            },
            "vertex",
            self.invalid_obj_id,
        )

    def get_boxes(self, sample_id, *, box_type="mobb_gravity"):
        objects = self.get_annotations(sample_id)
        names = {
            object_id: str(value["label"])
            for object_id, value in objects.items()
            if str(value.get("label", "")).lower() != "remove"
        }
        boxes = {}
        if box_type == "obb_gt":
            for object_id, value in objects.items():
                if object_id not in names:
                    continue
                try:
                    obb = value["obb"]
                    boxes[object_id] = o3d.geometry.OrientedBoundingBox(
                        np.asarray(obb["centroid"], dtype=np.float64),
                        np.asarray(obb["normalizedAxes"], dtype=np.float64)
                        .reshape(3, 3)
                        .T,
                        np.asarray(obb["axesLengths"], dtype=np.float64),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise DataFormatError(
                        f"3RScan object {object_id} has an invalid ground-truth OBB."
                    ) from exc
        else:
            fields = self._vertex_fields(sample_id, ("x", "y", "z", "objectId"))
            vertices = np.column_stack((fields["x"], fields["y"], fields["z"]))
            for object_id, indices in group_indices(fields["objectId"]).items():
                if object_id == 0 or object_id not in names:
                    continue
                boxes[object_id] = construct_box(vertices[indices], box_type=box_type)
        return boxes, {object_id: names[object_id] for object_id in boxes}

    @cachedmethod(lambda self: self._sequence_info_cache)
    def _sequence_info(self, sample_id: str) -> tuple[Path, RIOSequenceInfo]:
        archive = self._scan_path(self.sequence_archive, sample_id)
        try:
            payload = read_zip_members(archive, ["_info.txt"])["_info.txt"]
        except (KeyError, OSError) as exc:
            raise DataFormatError(
                f"Cannot read 3RScan sequence metadata in {archive}."
            ) from exc
        return archive, parse_rio_sequence_info(payload)

    @staticmethod
    def _pose_members(indices: np.ndarray) -> list[str]:
        return [f"frame-{int(index):06d}.pose.txt" for index in indices]

    def _read_poses(self, archive: Path, indices: np.ndarray) -> np.ndarray:
        members = self._pose_members(indices)
        if not members:
            return np.empty((0, 4, 4), dtype=np.float32)
        try:
            payloads = read_zip_members(archive, members)
            matrices = np.stack(
                [
                    np.loadtxt(io.BytesIO(payloads[member]), dtype=np.float32)
                    for member in members
                ]
            )
        except (KeyError, OSError, ValueError) as exc:
            raise DataFormatError(f"Invalid 3RScan poses in {archive}.") from exc
        if matrices.shape != (len(indices), 4, 4):
            raise DataFormatError(f"Invalid 3RScan camera matrix shape in {archive}.")
        return matrices

    def get_calibration(self, sample_id: str) -> dict:
        """Return the native RGB/depth calibration declared by the sequence."""

        _, info = self._sequence_info(sample_id)
        return {
            "sensor_name": info.sensor_name,
            "rgb_size": info.color_size,
            "depth_size": info.depth_size,
            "depth_shift": info.depth_shift,
            "rgb_intrinsic": info.color_intrinsic.copy(),
            "rgb_extrinsic": info.color_extrinsic.copy(),
            "depth_intrinsic": info.depth_intrinsic.copy(),
            "depth_extrinsic": info.depth_extrinsic.copy(),
        }

    @staticmethod
    def _read_images(
        archive: Path,
        members: list[str],
        source_size: tuple[int, int],
        *,
        rgb: bool,
    ) -> np.ndarray:
        if not members:
            width, height = source_size
            shape = (0, height, width, 3) if rgb else (0, height, width)
            return np.empty(shape, dtype=np.uint8)
        try:
            payloads = read_zip_members(archive, members)
        except (KeyError, OSError) as exc:
            raise DataFormatError(f"Missing 3RScan frame in {archive}.") from exc
        frames = [
            read_image(payloads[member], rgb=rgb, name=member) for member in members
        ]
        for member, frame in zip(members, frames):
            if (frame.shape[1], frame.shape[0]) != tuple(source_size):
                raise DataFormatError(
                    f"3RScan frame {member!r} has size {frame.shape[1::-1]}; "
                    f"expected {source_size}."
                )
        return np.stack(frames)

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
        coordinate_space="scan",
    ):
        """Read requested RGB-D frames directly from ``sequence.zip``.

        Poses are RGB camera-to-world matrices in the archive and are returned
        as OpenCV world-to-camera matrices. ``coordinate_space='reference'``
        applies the official rescan-to-reference alignment before inversion.
        """

        items = validate_frame_items(
            items,
            {"rgb", "depth", "rgb_intrinsics", "depth_intrinsics", "world_to_camera"},
        )
        if coordinate_space not in {"scan", "reference"}:
            raise ValueError("coordinate_space must be 'scan' or 'reference'.")
        archive, info = self._sequence_info(sample_id)
        all_indices = np.arange(info.frame_count, dtype=np.int64)
        if indices is None:
            all_poses = self._read_poses(archive, all_indices)
            valid = valid_camera_mask(all_poses)
            available = all_indices[valid]
            selected, positions = select_frame_indices(available, step=step)
            camera_to_world = all_poses[valid][positions]
        else:
            selected, _ = select_frame_indices(all_indices, indices=indices, step=step)
            camera_to_world = self._read_poses(archive, selected)
            invalid = selected[~valid_camera_mask(camera_to_world)]
            if len(invalid):
                raise ValueError(
                    "Frames are not synchronized or valid: "
                    + ", ".join(map(str, invalid.tolist()))
                )

        rotation = 270 if rotate_to_up else 0
        color_transform = build_pixel_transform(
            info.color_size,
            output_size=output_size,
            rotation=rotation,
            center_crop=center_crop,
        )
        depth_transform = build_pixel_transform(
            info.depth_size,
            output_size=output_size,
            rotation=rotation,
            center_crop=center_crop,
        )
        values = {
            "indices": selected,
            "source_frame_count": info.frame_count,
            "camera_model": "PINHOLE",
        }
        if "rgb_intrinsics" in items:
            intrinsic = np.repeat(
                info.color_intrinsic[None, :3, :3], len(selected), axis=0
            )
            values["rgb_intrinsics"] = transform_intrinsics(
                intrinsic, color_transform
            ).astype(np.float32)
        if "depth_intrinsics" in items:
            intrinsic = np.repeat(
                info.depth_intrinsic[None, :3, :3], len(selected), axis=0
            )
            values["depth_intrinsics"] = transform_intrinsics(
                intrinsic, depth_transform
            ).astype(np.float32)
        if "rgb" in items:
            members = [f"frame-{index:06d}.color.jpg" for index in selected]
            rgb = self._read_images(archive, members, info.color_size, rgb=True)
            values["rgb"] = apply_transform_batch(rgb, color_transform)
        if "depth" in items:
            members = [f"frame-{index:06d}.depth.pgm" for index in selected]
            raw_depth = self._read_images(archive, members, info.depth_size, rgb=False)
            depth = raw_depth.astype(np.float32) / np.float32(info.depth_shift)
            values["depth"] = apply_transform_batch(
                depth, depth_transform, resample=Image.Resampling.NEAREST
            ).astype(np.float32)
        if "world_to_camera" in items:
            if coordinate_space == "reference":
                camera_to_world = self.get_scene_alignment(sample_id) @ camera_to_world
            values["world_to_camera"] = transform_world_to_camera(
                np.linalg.inv(camera_to_world), color_transform
            ).astype(np.float32)
        return FrameBatch(**values)
