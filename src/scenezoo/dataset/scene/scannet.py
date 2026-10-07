"""ScanNet v2 scene adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cachetools import LRUCache, cachedmethod
import numpy as np
from PIL import Image

from ...io.archive import read_zip_members
from ...io.image import read_image
from ...io.sens import (
    INVALID_TIMESTAMP,
    SensIndex,
    read_sens_frames,
    read_sens_imu,
    read_sens_index,
)
from ...ops.annotation import (
    parse_sstk_groups,
    parse_sstk_segmentation,
    parse_sstk_segments,
    read_ply_attribute,
    sstk_object_indices,
)
from ...ops.frame import (
    apply_transform_batch,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    valid_camera_mask,
    validate_frame_items,
)
from ...ops.geometry import construct_box, load_triangle_mesh
from ..base import DataFormatError, Dataset
from ..registry import register_dataset
from ..types import FrameBatch, Segmentation3D


_DEFAULT_LABEL_MAP_URL = (
    "https://kaldir.vc.in.tum.de/scannet/v2/tasks/scannetv2-labels.combined.tsv"
)
_PACKAGED_SPLIT_DIR = Path(__file__).resolve().parents[2] / "metadata"
_INTEGER_INFO_FIELDS = {
    "colorHeight",
    "colorWidth",
    "depthHeight",
    "depthWidth",
    "numColorFrames",
    "numDepthFrames",
    "numIMUmeasurements",
}


@register_dataset(
    "scannetv2",
    aliases=("scannet",),
    description="ScanNet v2 indoor RGB-D scenes.",
)
class ScanNet(Dataset):
    """ScanNet v2 meshes, annotations, calibration, and sensor streams."""

    def __init__(
        self,
        root_dir,
        *,
        raw_mesh_file="scans/{scene_id}/{scene_id}_vh_clean.ply",
        decimated_mesh_file="scans/{scene_id}/{scene_id}_vh_clean_2.ply",
        semantic_mesh_file="scans/{scene_id}/{scene_id}_vh_clean_2.labels.ply",
        aggregation_file="scans/{scene_id}/{scene_id}.aggregation.json",
        segmentation_file="scans/{scene_id}/{scene_id}_vh_clean_2.0.010000.segs.json",
        raw_aggregation_file="scans/{scene_id}/{scene_id}_vh_clean.aggregation.json",
        raw_segmentation_file="scans/{scene_id}/{scene_id}_vh_clean.segs.json",
        alignment_file="scans/{scene_id}/{scene_id}.txt",
        sens_file="scans/{scene_id}/{scene_id}.sens",
        instance_map_archive="scans/{scene_id}/{scene_id}_2d-instance-filt.zip",
        raw_instance_map_archive="scans/{scene_id}/{scene_id}_2d-instance.zip",
        semantic_map_archive="scans/{scene_id}/{scene_id}_2d-label-filt.zip",
        raw_semantic_map_archive="scans/{scene_id}/{scene_id}_2d-label.zip",
        label_mapping_file=None,
        train_split_file=None,
        val_split_file=None,
        test_split_file=None,
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
        self.mesh_files = {
            "raw": raw_mesh_file,
            "decimated": decimated_mesh_file,
            "semantic": semantic_mesh_file,
        }
        self.aggregation_files = {
            "raw": raw_aggregation_file,
            "decimated": aggregation_file,
        }
        self.segmentation_files = {
            "raw": raw_segmentation_file,
            "decimated": segmentation_file,
        }
        self.alignment_file = alignment_file
        self.sens_file = sens_file
        self.projection_archives = {
            ("instance", "filtered"): instance_map_archive,
            ("instance", "raw"): raw_instance_map_archive,
            ("semantic", "filtered"): semantic_map_archive,
            ("semantic", "raw"): raw_semantic_map_archive,
        }
        self.label_mapping_file = label_mapping_file
        self.split_files = {
            "train": train_split_file,
            "val": val_split_file,
            "test": test_split_file,
        }
        self._sens_index_cache = LRUCache(maxsize=4)

    def _root_source(self, source: str | Path) -> str | Path:
        if isinstance(source, str) and "://" in source:
            return source
        path = Path(source)
        return path if path.is_absolute() else self.root_dir / path

    def _load_splits(self):
        splits = {}
        for split, source in self.split_files.items():
            if source is None:
                source = _PACKAGED_SPLIT_DIR / f"scannetv2_{split}.txt"
            with self.open_file(self._root_source(source), "r") as handle:
                splits[split] = [line for line in handle.read().splitlines() if line]
        return splits

    def _label_mapping_source(self) -> str | Path:
        if self.label_mapping_file is not None:
            return self._root_source(self.label_mapping_file)
        candidates = (
            self.root_dir / "scannetv2-labels.combined.tsv",
            self.root_dir / "tasks/scannetv2-labels.combined.tsv",
        )
        for path in candidates:
            if path.is_file():
                return path
        return _DEFAULT_LABEL_MAP_URL

    def _load_metadata(self):
        mapping = {}
        source = self._label_mapping_source()
        with self.open_file(source, "r") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    name = row.pop("raw_category")
                except KeyError as exc:
                    raise DataFormatError(
                        f"Invalid ScanNet label mapping header: {source}"
                    ) from exc
                for key in ("id", "count", "nyu40id", "eigen13id", "mpcat40index"):
                    if row.get(key):
                        row[key] = int(row[key])
                mapping[name] = row
        return {"label_mapping": mapping, "label_mapping_source": str(source)}

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
            message="The ScanNet root does not exist.",
            expected="<root>/scans/<scene_id>/...",
            hint="Pass the directory created by the official ScanNet download script.",
        ):
            return builder.finish()
        scans_dir = self.root_dir / "scans"
        if not builder.require_directory(
            scans_dir,
            code="missing-scans-directory",
            message="The required 'scans' directory is missing.",
            expected="<root>/scans/scene0000_00/scene0000_00.sens",
            hint=(
                "Extract/move downloaded scene folders under <root>/scans. "
                "SceneZoo reads .sens directly; the 25k frame export is not needed."
            ),
        ):
            archives = archive_candidates(self.root_dir)
            if archives:
                builder.info(
                    "archives-found",
                    "Archives are present but the adapter expects extracted scene folders: "
                    + ", ".join(path.name for path in archives[:5]),
                    hint="Extract the archives so that <root>/scans/<scene_id> exists.",
                )
            return builder.finish()

        present = {
            path.name
            for path in scans_dir.iterdir()
            if path.is_dir() and path.name.startswith("scene")
        }
        expected = None if sample_ids is not None else load_expected_ids(self, builder)
        selected = check_scene_coverage(
            builder,
            present_ids=present,
            expected_ids=expected,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<root>/scans/<scene_id>/<scene_id>.*",
        )
        if not present:
            builder.error(
                "no-scenes",
                "No ScanNet scene directories were found.",
                path=scans_dir,
                expected="Directories named like scene0000_00.",
                hint="Run the official ScanNet downloader and extract each requested scene.",
            )
            return builder.finish()

        check_required_scene_paths(
            builder,
            selected,
            {
                "decimated_mesh": lambda sid: self.root_dir
                / self.mesh_files["decimated"].format(scene_id=sid),
                "sensor_stream": lambda sid: self.root_dir
                / self.sens_file.format(scene_id=sid),
                "scene_metadata": lambda sid: self.root_dir
                / self.alignment_file.format(scene_id=sid),
            },
            expected=(
                "Each scene needs *_vh_clean_2.ply, <scene_id>.sens, and "
                "<scene_id>.txt for mesh/frame/camera access."
            ),
            hint="Download the corresponding official ScanNet scene file group.",
        )
        if expected is not None:
            test_ids = set(self.splits.get("test", ()))
            annotated = tuple(sid for sid in selected if sid not in test_ids)
            check_required_scene_paths(
                builder,
                annotated,
                {
                    "semantic_mesh": lambda sid: self.root_dir
                    / self.mesh_files["semantic"].format(scene_id=sid),
                    "aggregation": lambda sid: self.root_dir
                    / self.aggregation_files["decimated"].format(scene_id=sid),
                    "segmentation": lambda sid: self.root_dir
                    / self.segmentation_files["decimated"].format(scene_id=sid),
                },
                expected=(
                    "Annotated train/validation scenes need *.labels.ply, "
                    "*.aggregation.json, and *.segs.json."
                ),
                hint="Download the ScanNet v2 task/annotation assets for these scenes.",
            )
        return builder.finish()

    @staticmethod
    def _annotation_mesh_type(mesh_type: str) -> str:
        if mesh_type == "semantic":
            return "decimated"
        if mesh_type not in {"raw", "decimated"}:
            raise ValueError("mesh_type must be 'raw' or 'decimated'.")
        return mesh_type

    def get_mesh(self, sample_id, *, mesh_type=None):
        mesh_type = mesh_type or "decimated"
        try:
            template = self.mesh_files[mesh_type]
        except KeyError as exc:
            raise ValueError(f"Unknown ScanNet mesh type: {mesh_type!r}") from exc
        return load_triangle_mesh(self.root_dir / template.format(scene_id=sample_id))

    def _load_annotation_json(self, sample_id, *, mesh_type="decimated"):
        mesh_type = self._annotation_mesh_type(mesh_type)
        aggregation_path = self.root_dir / self.aggregation_files[mesh_type].format(
            scene_id=sample_id
        )
        segmentation_path = self.root_dir / self.segmentation_files[mesh_type].format(
            scene_id=sample_id
        )
        try:
            with aggregation_path.open() as handle:
                groups = json.load(handle)
            with segmentation_path.open() as handle:
                segments = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                f"Invalid ScanNet annotation JSON for {sample_id}."
            ) from exc
        return segments, groups

    def _nyu40_names(self) -> dict[int, str]:
        names = {0: "unlabeled"}
        for row in self.metadata["label_mapping"].values():
            label_id = row.get("nyu40id")
            label_name = row.get("nyu40class")
            if label_id is not None and label_name:
                names.setdefault(int(label_id), str(label_name))
        return names

    def _semantic_segmentation_from_groups(self, sample_id, *, mesh_type):
        segments, groups = self._load_annotation_json(sample_id, mesh_type=mesh_type)
        vertex_segments = parse_sstk_segments(segments)
        objects = parse_sstk_groups(groups)
        max_segment = int(vertex_segments.max()) if len(vertex_segments) else -1
        segment_labels = np.zeros(max_segment + 1, dtype=np.int32)
        mapping = self.metadata["label_mapping"]
        for object_id, value in objects.items():
            name = str(value.get("label", ""))
            row = mapping.get(name)
            if row is None or not row.get("nyu40id"):
                raise DataFormatError(
                    f"ScanNet object {object_id} has no NYU40 mapping for {name!r}."
                )
            object_segments = np.asarray(value.get("segments", ()), dtype=np.int64)
            if np.any(object_segments < 0) or np.any(object_segments > max_segment):
                raise DataFormatError(
                    f"ScanNet object {object_id} references an unknown segment."
                )
            segment_labels[object_segments] = int(row["nyu40id"])
        return Segmentation3D(
            labels=segment_labels[vertex_segments],
            id_to_name=self._nyu40_names(),
            domain="vertex",
            invalid_id=0,
        )

    def get_segmentation(
        self,
        sample_id,
        *,
        segmentation_type="instance",
        mesh_type="decimated",
    ):
        """Return instance or NYU40 semantic labels for one mesh resolution."""

        mesh_type = self._annotation_mesh_type(mesh_type)
        if segmentation_type == "instance":
            segments, groups = self._load_annotation_json(
                sample_id, mesh_type=mesh_type
            )
            return parse_sstk_segmentation(
                segments,
                groups,
                invalid_id=self.invalid_obj_id,
                domain="vertex",
            )
        if segmentation_type != "semantic":
            raise ValueError("segmentation_type must be 'instance' or 'semantic'.")
        if mesh_type == "raw":
            return self._semantic_segmentation_from_groups(
                sample_id, mesh_type=mesh_type
            )
        path = self.root_dir / self.mesh_files["semantic"].format(scene_id=sample_id)
        labels = read_ply_attribute(path, "label").astype(np.int32, copy=False)
        return Segmentation3D(labels, self._nyu40_names(), "vertex", 0)

    def get_boxes(
        self,
        sample_id,
        *,
        box_type="mobb_gravity",
        mesh_type="decimated",
    ):
        mesh_type = self._annotation_mesh_type(mesh_type)
        vertices = np.asarray(self.get_mesh(sample_id, mesh_type=mesh_type).vertices)
        segments, groups = self._load_annotation_json(sample_id, mesh_type=mesh_type)
        vertex_segments = parse_sstk_segments(segments)
        if len(vertex_segments) != len(vertices):
            raise DataFormatError(
                f"ScanNet {mesh_type} segmentation and mesh vertex counts differ."
            )
        objects = parse_sstk_groups(groups)
        members = sstk_object_indices(vertex_segments, objects)
        boxes = {}
        names = {}
        for object_id, value in objects.items():
            object_vertices = vertices[members[object_id]]
            if len(object_vertices) == 0:
                raise DataFormatError(
                    f"ScanNet object {object_id} has no mesh vertices."
                )
            boxes[object_id] = construct_box(object_vertices, box_type=box_type)
            names[object_id] = value["label"]
        return boxes, names

    def get_scene_info(self, sample_id) -> dict:
        """Parse all fields from the release's per-scene ``.txt`` file."""

        path = self.root_dir / self.alignment_file.format(scene_id=sample_id)
        result = {}
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if "=" not in line:
                    raise DataFormatError(
                        f"Invalid ScanNet scene info line {line_number} in {path}."
                    )
                key, raw_value = (part.strip() for part in line.split("=", 1))
                if key in {"axisAlignment", "colorToDepthExtrinsics"}:
                    values = np.fromstring(raw_value, sep=" ", dtype=np.float32)
                    if values.size != 16:
                        raise DataFormatError(f"Invalid {key} matrix in {path}.")
                    result[key] = values.reshape(4, 4)
                elif key in _INTEGER_INFO_FIELDS:
                    result[key] = int(raw_value)
                else:
                    try:
                        result[key] = float(raw_value)
                    except ValueError:
                        result[key] = raw_value
        return result

    def get_alignment(self, sample_id):
        """Return the scene's 4x4 axis-alignment matrix from ``<scene_id>.txt``."""

        try:
            return self.get_scene_info(sample_id)["axisAlignment"]
        except KeyError as exc:
            raise DataFormatError(
                f"No axis alignment matrix found for ScanNet scene {sample_id}."
            ) from exc

    @cachedmethod(lambda self: self._sens_index_cache)
    def _get_sens_index(self, sample_id) -> SensIndex:
        return read_sens_index(
            self.root_dir / self.sens_file.format(scene_id=sample_id)
        )

    def get_calibration(self, sample_id) -> dict:
        """Return sensor header calibration and RGB-to-depth calibration."""

        index = self._get_sens_index(sample_id)
        scene_info = self.get_scene_info(sample_id)
        color_to_depth = scene_info.get("colorToDepthExtrinsics")
        result = {
            "sensor_name": index.sensor_name,
            "rgb_intrinsic": index.color_intrinsic.copy(),
            "depth_intrinsic": index.depth_intrinsic.copy(),
            "rgb_extrinsic": index.color_extrinsic.copy(),
            "depth_extrinsic": index.depth_extrinsic.copy(),
            "rgb_size": index.color_size,
            "depth_size": index.depth_size,
            "depth_shift": index.depth_shift,
        }
        if color_to_depth is not None:
            result["rgb_to_depth"] = color_to_depth.copy()
            result["depth_to_rgb"] = np.linalg.inv(color_to_depth).astype(np.float32)
        return result

    def get_imu(self, sample_id) -> dict[str, np.ndarray]:
        """Return the complete, independently sampled raw IMU stream."""

        return read_sens_imu(self._get_sens_index(sample_id))

    def _read_projection_maps(self, sample_id, indices, *, kind, variant):
        template = self.projection_archives[(kind, variant)]
        archive = self.root_dir / template.format(scene_id=sample_id)
        directory = {
            ("instance", "filtered"): "instance-filt",
            ("instance", "raw"): "instance",
            ("semantic", "filtered"): "label-filt",
            ("semantic", "raw"): "label",
        }[(kind, variant)]
        members = [f"{directory}/{int(index)}.png" for index in indices]
        try:
            payloads = read_zip_members(archive, members)
        except KeyError as exc:
            raise DataFormatError(
                f"ScanNet {kind} projections are missing requested frames."
            ) from exc
        maps = [read_image(payloads[member], name=member) for member in members]
        if not maps:
            return np.empty((0,), dtype=np.int32)
        if len({image.shape for image in maps}) != 1:
            raise DataFormatError("ScanNet projection images have inconsistent sizes.")
        result = np.stack(maps)
        if kind == "instance":
            source = result.astype(np.int32, copy=False)
            result = np.full(source.shape, self.invalid_obj_id, dtype=np.int32)
            foreground = source > 0
            result[foreground] = source[foreground] - 1
        return result

    def _map_semantic_images_to_nyu40(self, images: np.ndarray) -> np.ndarray:
        source_to_target = {0: 0}
        for row in self.metadata["label_mapping"].values():
            source = row.get("id")
            target = row.get("nyu40id")
            if source is None or target is None:
                continue
            # ScanNet's official helper constructs this mapping from file order;
            # a few numeric IDs repeat, so the final row intentionally wins.
            source_to_target[int(source)] = int(target)
        maximum = max(source_to_target, default=0)
        lut = np.zeros(maximum + 1, dtype=np.int32)
        for source, target in source_to_target.items():
            lut[source] = target
        result = np.zeros(images.shape, dtype=np.int32)
        valid = images <= maximum
        result[valid] = lut[images[valid]]
        return result

    @staticmethod
    def _frame_timestamps(index: SensIndex, selected: np.ndarray) -> np.ndarray:
        return np.stack(
            (index.color_timestamps[selected], index.depth_timestamps[selected]), axis=1
        )

    def _aligned_imu(self, index: SensIndex, selected: np.ndarray):
        raw = read_sens_imu(index)
        frame_timestamps = index.color_timestamps[selected].copy()
        use_depth = (frame_timestamps == 0) | (frame_timestamps == INVALID_TIMESTAMP)
        frame_timestamps[use_depth] = index.depth_timestamps[selected][use_depth]
        max_int64 = np.iinfo(np.int64).max
        valid_frames = (
            (frame_timestamps != 0)
            & (frame_timestamps != INVALID_TIMESTAMP)
            & (frame_timestamps <= max_int64)
        )
        imu_timestamps = raw["timestamps"]
        valid_imu = (
            (imu_timestamps != 0)
            & (imu_timestamps != INVALID_TIMESTAMP)
            & (imu_timestamps <= max_int64)
        )

        result = {
            name: np.full((len(selected), 3), np.nan, dtype=np.float64)
            for name in (
                "rotation_rate",
                "acceleration",
                "magnetic_field",
                "attitude",
                "gravity",
            )
        }
        result["timestamps"] = np.zeros(len(selected), dtype=np.uint64)
        result["valid"] = np.zeros(len(selected), dtype=bool)
        if not valid_frames.any() or not valid_imu.any():
            return result

        valid_positions = np.flatnonzero(valid_imu)
        order = np.argsort(imu_timestamps[valid_positions])
        valid_positions = valid_positions[order]
        sorted_timestamps = imu_timestamps[valid_positions].astype(np.int64)
        output_positions = np.flatnonzero(valid_frames)
        targets = frame_timestamps[output_positions].astype(np.int64)
        right = np.searchsorted(sorted_timestamps, targets)
        left = np.clip(right - 1, 0, len(sorted_timestamps) - 1)
        right = np.clip(right, 0, len(sorted_timestamps) - 1)
        choose_right = np.abs(sorted_timestamps[right] - targets) <= np.abs(
            targets - sorted_timestamps[left]
        )
        nearest = np.where(choose_right, right, left)
        source_positions = valid_positions[nearest]
        for name in (
            "rotation_rate",
            "acceleration",
            "magnetic_field",
            "attitude",
            "gravity",
        ):
            result[name][output_positions] = raw[name][source_positions]
        result["timestamps"][output_positions] = imu_timestamps[source_positions]
        result["valid"][output_positions] = True
        return result

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
        annotation_variant="filtered",
        semantic_label_space="scannet",
    ):
        """Read synchronized frames directly from ``.sens`` and projection ZIPs.

        ScanNet is already +Z upright, so ``rotate_to_up`` does not alter pixels.
        Timestamps have two columns in color/depth order and retain the release's
        raw microsecond integers. Frame-aligned IMU values use nearest timestamps;
        ``imu['valid']`` is false when the sensor timestamps are unavailable.
        """

        del rotate_to_up
        if annotation_variant not in {"raw", "filtered"}:
            raise ValueError("annotation_variant must be 'raw' or 'filtered'.")
        if semantic_label_space not in {"scannet", "nyu40"}:
            raise ValueError("semantic_label_space must be 'scannet' or 'nyu40'.")
        items = validate_frame_items(
            items,
            {
                "timestamps",
                "rgb",
                "depth",
                "rgb_intrinsics",
                "depth_intrinsics",
                "world_to_camera",
                "instance_maps",
                "semantic_maps",
                "imu",
            },
        )

        index = self._get_sens_index(sample_id)
        valid = valid_camera_mask(index.camera_to_world)
        available = np.flatnonzero(valid).astype(np.int64)
        selected, positions = select_frame_indices(
            available, indices=indices, step=step
        )
        valid_poses = index.camera_to_world[valid][positions]
        data = read_sens_frames(
            index,
            selected,
            rgb="rgb" in items,
            depth="depth" in items,
        )

        values = {
            "indices": selected,
            "source_frame_count": index.frame_count,
            "camera_model": "PINHOLE",
        }
        if "timestamps" in items:
            values["timestamps"] = self._frame_timestamps(index, selected)
        if "imu" in items:
            values["imu"] = self._aligned_imu(index, selected)
        if "world_to_camera" in items:
            values["world_to_camera"] = np.linalg.inv(valid_poses).astype(np.float32)
        if "rgb" in items or "rgb_intrinsics" in items:
            transform = build_pixel_transform(
                index.color_size, output_size=output_size, center_crop=center_crop
            )
            if "rgb" in items:
                values["rgb"] = apply_transform_batch(data["rgb"], transform)
            if "rgb_intrinsics" in items:
                intrinsic = np.repeat(
                    index.color_intrinsic[:3, :3][None], len(selected), axis=0
                )
                values["rgb_intrinsics"] = transform_intrinsics(
                    intrinsic, transform
                ).astype(np.float32)
        if "depth" in items or "depth_intrinsics" in items:
            transform = build_pixel_transform(
                index.depth_size, output_size=output_size, center_crop=center_crop
            )
            if "depth" in items:
                depth = apply_transform_batch(
                    data["depth"], transform, resample=Image.Resampling.NEAREST
                )
                values["depth"] = depth.astype(np.float32) / index.depth_shift
            if "depth_intrinsics" in items:
                intrinsic = np.repeat(
                    index.depth_intrinsic[:3, :3][None], len(selected), axis=0
                )
                values["depth_intrinsics"] = transform_intrinsics(
                    intrinsic, transform
                ).astype(np.float32)
        for kind, item in (
            ("instance", "instance_maps"),
            ("semantic", "semantic_maps"),
        ):
            if item not in items:
                continue
            maps = self._read_projection_maps(
                sample_id,
                selected,
                kind=kind,
                variant=annotation_variant,
            )
            if kind == "semantic" and semantic_label_space == "nyu40":
                maps = self._map_semantic_images_to_nyu40(maps)
            if len(maps):
                source_size = (maps.shape[2], maps.shape[1])
            else:
                source_size = index.color_size
                width, height = source_size
                maps = np.empty((0, height, width), dtype=np.int32)
            transform = build_pixel_transform(
                source_size, output_size=output_size, center_crop=center_crop
            )
            values[item] = apply_transform_batch(
                maps, transform, resample=Image.Resampling.NEAREST
            )
        return FrameBatch(**values)
