"""Structured3D renderings, structure annotations, boxes, and point clouds."""

from __future__ import annotations

import io
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Sequence

from cachetools import LRUCache, cachedmethod
import numpy as np
import open3d as o3d
from PIL import Image

from ...io.image import read_image
from ...ops.annotation import remap_labels
from ...ops.frame import (
    apply_transform_batch,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    validate_frame_items,
)
from ..base import DataFormatError, Dataset
from ..registry import register_dataset
from ..types import FrameBatch, PointBatch, Segmentation3D
from ._structured3d_store import Structured3DStore


STRUCTURED3D_VALID_CLASS_IDS_25 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    22,
    24,
    25,
    32,
    34,
    35,
    38,
    39,
    40,
)
STRUCTURED3D_CLASS_NAMES_25 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "picture",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "ceiling",
    "refrigerator",
    "television",
    "nightstand",
    "sink",
    "lamp",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)

_SCENE_PATTERN = re.compile(r"^scene_([0-9]{5})$")
_ROOM_PATTERN = re.compile(r"^[0-9]+$")
# Instance renderings mark background pixels with the maximum uint16 value.
_BACKGROUND_INSTANCE = 65535
_NYU40_CLASS_COUNT = 40
_ERRATA_SECTIONS = {
    "invalid scene": "invalid_scenes",
    "a pair of junctions are not aligned along the x-axis": "misaligned_rooms",
    "self-intersection layout": "self_intersection_rooms",
}
_IMAGE_ITEMS = {
    "rgb",
    "depth",
    "semantic_maps",
    "instance_maps",
    "albedo",
    "normal_maps",
}


def _first_point_per_voxel(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Indices of the first point in each occupied voxel, in input order.

    Unlike averaging voxel filters, this keeps original colours and labels.
    """

    grid = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, first = np.unique(grid, axis=0, return_index=True)
    return np.sort(first)


class Structured3DPointData(PointBatch):
    """One room's fused RGB-D points and labels in matching row order."""

    __slots__ = ()


@register_dataset(
    "structured3d",
    aliases=("structure3d",),
    description=(
        "Structured3D renderings, cameras, layouts, structure annotations, "
        "oriented boxes, and fused room point clouds."
    ),
)
class Structured3D(Dataset):
    """Adapter for extracted Structured3D scenes."""

    valid_class_ids_25 = STRUCTURED3D_VALID_CLASS_IDS_25
    class_names_25 = STRUCTURED3D_CLASS_NAMES_25

    def __init__(
        self,
        root_dir,
        *,
        data_dir=None,
        room_type_file=None,
        label_name_file=None,
        errata_file=None,
        include_invalid=True,
        cache_dir=None,
        offline=False,
        invalid_obj_id=-1,
    ):
        super().__init__(
            root_dir,
            cache_dir=cache_dir,
            offline=offline,
            invalid_obj_id=invalid_obj_id,
        )
        metadata = files("scenezoo.metadata")
        self.dataset_root = self._resolve_dataset_root(data_dir)
        self.room_type_file = room_type_file or metadata / "structured3d_room_types.txt"
        self.label_name_file = label_name_file or metadata / "structured3d_labelids.txt"
        self.errata_file = errata_file or metadata / "structured3d_errata.txt"
        self.include_invalid = bool(include_invalid)
        self._store = Structured3DStore(self.root_dir, data_dir)
        self._box_label_cache = LRUCache(maxsize=16)

    def _resolve_dataset_root(self, data_dir) -> Path:
        if data_dir is not None:
            path = Path(data_dir).expanduser()
            return path if path.is_absolute() else self.root_dir / path
        for candidate in (
            self.root_dir / "data",
            self.root_dir / "Structured3D",
            self.root_dir,
        ):
            if candidate.is_dir() and next(candidate.glob("scene_*"), None):
                return candidate
        return self.root_dir

    @staticmethod
    def _parse_errata(lines: Iterable[str]) -> dict[str, tuple[str, ...]]:
        result = {name: [] for name in _ERRATA_SECTIONS.values()}
        section = None
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                heading = line.removeprefix("#").strip().lower()
                try:
                    section = _ERRATA_SECTIONS[heading]
                except KeyError as exc:
                    raise DataFormatError(
                        f"Unknown Structured3D errata section at line {line_number}: {line}"
                    ) from exc
                continue
            if section is None:
                raise DataFormatError(
                    f"Structured3D errata entry has no section at line {line_number}."
                )
            result[section].append(line)
        return {name: tuple(values) for name, values in result.items()}

    def _load_splits(self):
        invalid = set(self.metadata["errata"]["invalid_scenes"])

        def values(start, end):
            scene_ids = [f"scene_{index:05d}" for index in range(start, end)]
            return (
                scene_ids
                if self.include_invalid
                else [scene_id for scene_id in scene_ids if scene_id not in invalid]
            )

        train = values(0, 3000)
        val = values(3000, 3250)
        test = values(3250, 3500)
        return {"train": train, "val": val, "test": test}

    def _check(self, *, sample_ids, require_complete):
        from ..check import CheckBuilder, check_scene_coverage, normalize_sample_ids

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The Structured3D root does not exist.",
            expected="Extracted data/scene_00000 or Structured3D/scene_00000 directories.",
            hint="Pass the extracted release root or set data_dir explicitly.",
        ):
            return builder.finish()
        extracted = {
            path.name
            for path in self.dataset_root.glob("scene_*")
            if path.is_dir() and re.fullmatch(r"scene_[0-9]{5}", path.name)
        }
        archives = self._store.zip_paths
        requested = normalize_sample_ids(sample_ids)
        expected_layout = (
            "<data_root>/scene_00000/{annotation_3d.json,bbox_3d.json,2D_rendering/}"
        )
        if requested is not None:
            invalid_requested = [
                sid for sid in requested if re.fullmatch(r"scene_[0-9]{5}", sid) is None
            ]
            if invalid_requested:
                builder.error(
                    "invalid-scene-ids",
                    "Invalid Structured3D scene IDs: "
                    + ", ".join(invalid_requested[:10]),
                    expected="IDs formatted exactly as scene_00000 through scene_03499.",
                    hint="Correct sample_ids; do not pass room IDs or filesystem paths.",
                )
            valid_requested = tuple(
                sid
                for sid in requested
                if re.fullmatch(r"scene_[0-9]{5}", sid) is not None
            )
            selected = tuple(sid for sid in valid_requested if sid in extracted)
            builder.stats["scenes_found"] = len(selected)
            builder.stats["scenes_requested"] = len(requested)
            unavailable = []
            for sid in valid_requested:
                if sid in extracted:
                    continue
                candidates = self._store.archive_candidates(sid)
                if candidates:
                    names = ", ".join(path.name for path in candidates[:5])
                    builder.error(
                        "scene-not-extracted",
                        f"{sid} is not extracted; matching download shards were found.",
                        path=candidates[0],
                        expected=(
                            f"{self.root_dir}/data/{sid}, "
                            f"{self.root_dir}/Structured3D/{sid}, or "
                            f"<data_dir>/{sid}"
                        ),
                        hint=(
                            f"Extract all {sid} assets from the relevant shards, including "
                            f"{names}, while preserving the official scene tree."
                        ),
                    )
                else:
                    unavailable.append(sid)
            if unavailable:
                builder.error(
                    "missing-requested-scenes",
                    f"Requested scene IDs are missing: {', '.join(unavailable[:10])}",
                    expected=expected_layout,
                    hint="Download and extract the required Structured3D shards.",
                )
        else:
            selected = check_scene_coverage(
                builder,
                present_ids=extracted,
                expected_ids=self.get_ids(),
                sample_ids=None,
                require_complete=require_complete,
                expected_layout=expected_layout,
            )
        builder.stats["zip_shards_found"] = len(archives)
        builder.stats["extracted_scenes"] = len(extracted)
        if archives and len(extracted) < len(self.get_ids()):
            examples = ", ".join(path.name for path in archives[:5])
            builder.warning(
                "archives-need-extraction",
                f"{len(archives)} Structured3D ZIP shards are present but are not read "
                f"by SceneZoo. Examples: {examples}",
                expected="Fully extracted scene_NNNNN directories.",
                hint="Extract every required annotation and rendering shard before use.",
            )
        if not extracted and requested is None:
            builder.error(
                "no-extracted-scenes",
                "No extracted Structured3D scene directories were found.",
                path=self.root_dir,
                expected=expected_layout,
                hint="Download and extract the official shards before using the adapter.",
            )
            return builder.finish()
        missing_annotations = [
            sid
            for sid in selected
            if sid in extracted
            and not (self.dataset_root / sid / "annotation_3d.json").is_file()
        ]
        if missing_annotations:
            builder.error(
                "missing-structure-annotations",
                f"{len(missing_annotations)} extracted scenes lack annotation_3d.json. "
                f"Examples: {', '.join(missing_annotations[:10])}",
                expected="<scene_id>/annotation_3d.json",
                hint="Extract/download the Structured3D annotation asset shard.",
            )
        builder.info(
            "extracted-directories-required",
            "Structured3D ZIP shards are download artifacts. SceneZoo reads only fully "
            "extracted scene directories.",
        )
        return builder.finish()

    def _load_metadata(self):
        with self.open_file(self.room_type_file, "r") as handle:
            room_types = tuple(line.strip() for line in handle if line.strip())
        label_names = {}
        with self.open_file(self.label_name_file, "r") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.strip().split(maxsplit=1)
                if not fields:
                    continue
                if len(fields) != 2 or not fields[0].isdigit():
                    raise DataFormatError(
                        f"Invalid Structured3D label row at line {line_number}."
                    )
                label_names[int(fields[0])] = fields[1]
        with self.open_file(self.errata_file, "r") as handle:
            errata = self._parse_errata(handle)
        return {
            "room_types": room_types,
            "label_names": label_names,
            "label_spaces": {
                "nyu40": label_names,
                "pointcept25": dict(enumerate(self.class_names_25)),
            },
            "errata": errata,
            "official_scene_count": 3500,
        }

    def _validate_scene_id(self, sample_id) -> str:
        scene_id = str(sample_id)
        match = _SCENE_PATTERN.fullmatch(scene_id)
        if match is None or int(match.group(1)) >= 3500:
            raise ValueError(
                "Structured3D scene IDs must use the official scene_NNNNN form."
            )
        if (
            not self.include_invalid
            and scene_id in self.metadata["errata"]["invalid_scenes"]
        ):
            raise DataFormatError(f"Structured3D marks {scene_id} as an invalid scene.")
        return scene_id

    @staticmethod
    def _validate_room_id(room_id) -> str:
        value = str(room_id)
        if _ROOM_PATTERN.fullmatch(value) is None:
            raise ValueError("Structured3D room IDs must contain only decimal digits.")
        return value

    def _logical(self, sample_id, *parts) -> str:
        scene_id = self._validate_scene_id(sample_id)
        return "/".join((scene_id, *(str(part) for part in parts)))

    def _read_json(self, logical: str):
        try:
            return json.loads(self._store.read_text(logical))
        except json.JSONDecodeError as exc:
            raise DataFormatError(f"Invalid Structured3D JSON: {logical}") from exc

    def get_room_ids(self, sample_id) -> list[str]:
        """Return the numeric rendering-room IDs available in one scene."""

        logical = self._logical(sample_id, "2D_rendering")
        rooms = [
            name
            for name in self._store.listdir(logical)
            if _ROOM_PATTERN.fullmatch(name)
        ]
        if not rooms:
            raise FileNotFoundError(
                f"No Structured3D rendered rooms found under {logical}."
            )
        return rooms

    def get_frame_sources(self, sample_id, room_id) -> tuple[str, ...]:
        """Return ``perspective`` and/or ``panorama`` when present for a room."""

        room_id = self._validate_room_id(room_id)
        base = self._logical(sample_id, "2D_rendering", room_id)
        return tuple(
            source
            for source in ("perspective", "panorama")
            if self._store.exists(f"{base}/{source}")
        )

    def get_structure_annotations(self, sample_id) -> dict:
        """Return the official primitive-and-relationship 3D annotation."""

        value = self._read_json(self._logical(sample_id, "annotation_3d.json"))
        required = {
            "junctions",
            "lines",
            "planes",
            "semantics",
            "planeLineMatrix",
            "lineJunctionMatrix",
            "cuboids",
            "manhattan",
        }
        if not isinstance(value, dict) or not required <= set(value):
            raise DataFormatError(
                "Structured3D annotation_3d.json is missing required fields."
            )
        return value

    def get_layout(self, sample_id, room_id, *, source="panorama", index=None):
        """Return one room's official panorama or perspective 2D layout."""

        room_id = self._validate_room_id(room_id)
        base = self._logical(sample_id, "2D_rendering", room_id, source)
        if source == "panorama":
            if index not in {None, 0}:
                raise ValueError("Panorama layout has no perspective position index.")
            try:
                values = np.loadtxt(
                    io.StringIO(self._store.read_text(f"{base}/layout.txt")),
                    dtype=np.float32,
                )
            except ValueError as exc:
                raise DataFormatError(
                    "Invalid Structured3D panorama layout.txt."
                ) from exc
            values = np.atleast_2d(values)
            if values.ndim != 2 or values.shape[1] != 2:
                raise DataFormatError("Panorama layout must contain two columns.")
            return values
        if source == "perspective":
            if index is None or isinstance(index, (bool, np.bool_)):
                raise ValueError(
                    "Perspective layout requires an integer position index."
                )
            return self._read_json(f"{base}/full/{int(index)}/layout.json")
        raise ValueError(
            "Structured3D layout source must be 'panorama' or 'perspective'."
        )

    def _read_image(self, logical: str, *, rgb=False) -> np.ndarray:
        return read_image(self._store.read_bytes(logical), rgb=rgb, name=logical)

    @staticmethod
    def _camera_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if values.shape != (12,) or not np.isfinite(values).all():
            raise DataFormatError(
                "Structured3D perspective camera_pose.txt needs 12 finite values."
            )
        eye = values[:3].astype(np.float64) / 1000.0
        forward = values[3:6].astype(np.float64)
        up = values[6:9].astype(np.float64)
        forward_norm, up_norm = np.linalg.norm(forward), np.linalg.norm(up)
        if forward_norm < 1e-8 or up_norm < 1e-8:
            raise DataFormatError(
                "Structured3D camera direction and up vectors must be non-zero."
            )
        forward /= forward_norm
        up /= up_norm
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            raise DataFormatError(
                "Structured3D camera direction and up vectors are parallel."
            )
        right /= right_norm
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        rotation = np.stack((right, down, forward))
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ eye
        return world_to_camera, values[9:11]

    @staticmethod
    def _panorama_camera(values: np.ndarray) -> np.ndarray:
        if values.shape != (3,) or not np.isfinite(values).all():
            raise DataFormatError(
                "Structured3D camera_xyz.txt needs three finite values."
            )
        # The official panorama projection utilities map the image centre to
        # +Y in the scene coordinates (xyz_2_coorxy uses atan2(x, y)).
        forward = np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        eye = values.astype(np.float64) / 1000.0
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = -rotation @ eye
        return matrix

    @staticmethod
    def _intrinsic(size: tuple[int, int], half_fov: np.ndarray) -> np.ndarray:
        width, height = size
        if (
            half_fov.shape != (2,)
            or np.any(half_fov <= 0)
            or np.any(half_fov >= np.pi / 2)
        ):
            raise DataFormatError("Structured3D camera half-FOV values are invalid.")
        # The field of view spans the outer pixel edges; the principal point is
        # the image centre in integer-pixel-centre (OpenCV) coordinates.
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = width / 2.0 / np.tan(half_fov[0])
        intrinsic[1, 1] = height / 2.0 / np.tan(half_fov[1])
        intrinsic[0, 2] = (width - 1) / 2.0
        intrinsic[1, 2] = (height - 1) / 2.0
        return intrinsic

    def _frame_paths(
        self, sample_id, room_id, *, source, configuration, lighting, position
    ) -> tuple[str, dict[str, str]]:
        room_base = self._logical(sample_id, "2D_rendering", room_id)
        if source == "perspective":
            base = f"{room_base}/perspective/{configuration}/{position}"
            rgb_name = "rgb_rawlight.png"
            camera = f"{base}/camera_pose.txt"
        else:
            base = f"{room_base}/panorama/{configuration}"
            rgb_name = f"rgb_{lighting}light.png"
            camera = f"{room_base}/panorama/camera_xyz.txt"
        return camera, {
            "rgb": f"{base}/{rgb_name}",
            "depth": f"{base}/depth.png",
            "semantic_maps": f"{base}/semantic.png",
            "instance_maps": f"{base}/instance.png",
            "albedo": f"{base}/albedo.png",
            "normal_maps": f"{base}/normal.png",
        }

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
        room_id=None,
        source="perspective",
        configuration="full",
        lighting="raw",
    ) -> FrameBatch:
        """Read rendered views for one room without loading unrequested images."""

        del rotate_to_up
        if room_id is None:
            raise ValueError("Structured3D get_frames() requires room_id.")
        room_id = self._validate_room_id(room_id)
        if room_id not in self.get_room_ids(sample_id):
            raise KeyError(f"Unknown Structured3D room {room_id!r} in {sample_id}.")
        if source not in {"perspective", "panorama"}:
            raise ValueError("Structured3D source must be 'perspective' or 'panorama'.")
        valid_configurations = {
            "perspective": {"empty", "full"},
            "panorama": {"empty", "simple", "full"},
        }
        if configuration not in valid_configurations[source]:
            raise ValueError(
                f"Invalid {source} configuration {configuration!r}; expected "
                f"{sorted(valid_configurations[source])}."
            )
        if source == "perspective" and lighting != "raw":
            raise ValueError(
                "Structured3D perspective RGB is only available with raw lighting."
            )
        if source == "panorama" and lighting not in {"raw", "cold", "warm"}:
            raise ValueError(
                "Structured3D panorama lighting must be raw, cold, or warm."
            )
        if source == "panorama" and center_crop:
            raise ValueError(
                "center_crop is not defined for equirectangular panoramas."
            )
        supported = _IMAGE_ITEMS | {"world_to_camera"}
        if source == "perspective":
            supported |= {"rgb_intrinsics", "depth_intrinsics"}
        if configuration != "full":
            supported.remove("instance_maps")
        items = validate_frame_items(items, supported)

        if source == "perspective":
            directory = self._logical(
                sample_id,
                "2D_rendering",
                room_id,
                "perspective",
                configuration,
            )
            available = np.asarray(
                [
                    int(name)
                    for name in self._store.listdir(directory)
                    if name.isdigit()
                ],
                dtype=np.int64,
            )
            if len(available) == 0:
                raise FileNotFoundError(
                    f"Structured3D {configuration} perspective views were not "
                    f"downloaded for {sample_id}/{room_id}."
                )
            source_frame_count = int(available.max()) + 1
        else:
            camera, paths = self._frame_paths(
                sample_id,
                room_id,
                source=source,
                configuration=configuration,
                lighting=lighting,
                position=0,
            )
            if not self._store.exists(camera) or not any(
                self._store.exists(path) for path in paths.values()
            ):
                raise FileNotFoundError(
                    f"Structured3D {configuration} panorama was not downloaded "
                    f"for {sample_id}/{room_id}."
                )
            available = np.asarray([0], dtype=np.int64)
            source_frame_count = 1
        selected, _ = select_frame_indices(available, indices=indices, step=step)
        records = [
            self._frame_paths(
                sample_id,
                room_id,
                source=source,
                configuration=configuration,
                lighting=lighting,
                position=int(position),
            )
            for position in selected
        ]

        probe_records = records or [
            self._frame_paths(
                sample_id,
                room_id,
                source=source,
                configuration=configuration,
                lighting=lighting,
                position=int(available[0]),
            )
        ]
        probe = None
        for _, paths in probe_records:
            for name in ("rgb", "depth", "semantic_maps", "albedo"):
                if self._store.exists(paths[name]):
                    probe = self._read_image(paths[name], rgb=name in {"rgb", "albedo"})
                    break
            if probe is not None:
                break
        if probe is None:
            raise FileNotFoundError(
                f"No Structured3D raster found for {sample_id}/{room_id}."
            )
        source_size = (probe.shape[1], probe.shape[0])
        transform = build_pixel_transform(
            source_size,
            output_size=output_size,
            center_crop=center_crop,
        )
        values = {
            "indices": selected,
            "source_frame_count": source_frame_count,
            "frame_keys": tuple(
                f"{room_id}/{source}/{configuration}"
                + (f"/{int(position)}" if source == "perspective" else "")
                for position in selected
            ),
            "camera_model": (
                "PINHOLE" if source == "perspective" else "EQUIRECTANGULAR"
            ),
            # Panorama depth is measured along each pixel's viewing ray.
            "depth_mode": (
                "ray_distance" if source == "panorama" and "depth" in items else None
            ),
            "image_sizes": np.repeat(
                np.asarray(transform.output_size, dtype=np.int32)[None],
                len(selected),
                axis=0,
            ),
        }

        for item in set(items) & _IMAGE_ITEMS:
            images = []
            for _, paths in records:
                logical = paths[item]
                image = self._read_image(logical, rgb=item in {"rgb", "albedo"})
                if image.shape[:2] != (source_size[1], source_size[0]):
                    raise DataFormatError(
                        "Structured3D synchronized rasters have different sizes: "
                        f"{logical}"
                    )
                images.append(image)
            if images:
                batch = np.stack(images)
            else:
                channels = 3 if item in {"rgb", "albedo", "normal_maps"} else None
                shape = (0, source_size[1], source_size[0]) + (
                    () if channels is None else (channels,)
                )
                batch = np.empty(shape, dtype=np.uint8)
            resample = (
                Image.Resampling.LANCZOS
                if item in {"rgb", "albedo"}
                else Image.Resampling.NEAREST
            )
            batch = apply_transform_batch(batch, transform, resample=resample)
            if item == "depth":
                batch = batch.astype(np.float32) / 1000.0
            elif item in {"semantic_maps", "instance_maps"}:
                batch = batch.astype(np.int32)
            elif item == "normal_maps":
                batch = np.clip(batch.astype(np.float32) / 128.0 - 1.0, -1.0, 1.0)
            values[item] = batch

        cameras = []
        half_fovs = []
        for camera_path, _ in records:
            camera_values = np.fromstring(
                self._store.read_text(camera_path), sep=" ", dtype=np.float64
            )
            if source == "perspective":
                matrix, half_fov = self._camera_from_values(camera_values)
                cameras.append(matrix)
                half_fovs.append(half_fov)
            else:
                cameras.append(self._panorama_camera(camera_values))
        if "world_to_camera" in items:
            values["world_to_camera"] = (
                np.stack(cameras).astype(np.float32)
                if cameras
                else np.empty((0, 4, 4), dtype=np.float32)
            )
        intrinsic_items = {"rgb_intrinsics", "depth_intrinsics"}
        if source == "perspective" and set(items) & intrinsic_items:
            intrinsics = (
                np.stack(
                    [self._intrinsic(source_size, half_fov) for half_fov in half_fovs]
                )
                if half_fovs
                else np.empty((0, 3, 3), dtype=np.float32)
            )
            intrinsics = transform_intrinsics(intrinsics, transform).astype(np.float32)
            if "rgb_intrinsics" in items:
                values["rgb_intrinsics"] = intrinsics
            if "depth_intrinsics" in items:
                values["depth_intrinsics"] = intrinsics.copy()
        return FrameBatch(**values)

    @staticmethod
    def _box_from_annotation(record: dict) -> o3d.geometry.OrientedBoundingBox:
        try:
            basis = np.asarray(record["basis"], dtype=np.float64)
            coeffs = np.asarray(record["coeffs"], dtype=np.float64)
            center = np.asarray(record["centroid"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataFormatError("Invalid Structured3D bounding-box record.") from exc
        if basis.shape != (3, 3) or coeffs.shape != (3,) or center.shape != (3,):
            raise DataFormatError(
                "Structured3D box basis, coeffs, and centroid have invalid shapes."
            )
        if (
            not np.isfinite(basis).all()
            or not np.isfinite(coeffs).all()
            or not np.isfinite(center).all()
        ):
            raise DataFormatError("Structured3D bounding-box values must be finite.")
        if np.any(coeffs <= 0) or not np.allclose(
            basis @ basis.T, np.eye(3), atol=1e-4
        ):
            raise DataFormatError(
                "Structured3D bounding-box basis or radii are invalid."
            )
        rotation = basis.T.copy()
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        return o3d.geometry.OrientedBoundingBox(
            center / 1000.0,
            rotation,
            2.0 * np.abs(coeffs) / 1000.0,
        )

    @cachedmethod(lambda self: self._box_label_cache)
    def _infer_box_labels(self, sample_id) -> dict[int, int]:
        scene_id = self._validate_scene_id(sample_id)
        counts: dict[int, np.ndarray] = {}
        for room_id in self.get_room_ids(scene_id):
            _, paths = self._frame_paths(
                scene_id,
                room_id,
                source="panorama",
                configuration="full",
                lighting="raw",
                position=0,
            )
            if not self._store.exists(paths["instance_maps"]) or not self._store.exists(
                paths["semantic_maps"]
            ):
                continue
            instances = self._read_image(paths["instance_maps"]).astype(np.int64)
            semantics = self._read_image(paths["semantic_maps"]).astype(np.int64)
            if instances.shape != semantics.shape:
                continue
            valid = (
                (instances != _BACKGROUND_INSTANCE)
                & (semantics > 0)
                & (semantics <= _NYU40_CLASS_COUNT)
            )
            encoded, frequencies = np.unique(
                instances[valid] * 64 + semantics[valid], return_counts=True
            )
            for value, frequency in zip(encoded, frequencies):
                instance_id, semantic_id = divmod(int(value), 64)
                histogram = counts.setdefault(
                    instance_id, np.zeros(_NYU40_CLASS_COUNT + 1, dtype=np.int64)
                )
                histogram[semantic_id] += int(frequency)
        return {
            instance_id: int(np.argmax(histogram))
            for instance_id, histogram in counts.items()
            if histogram.sum()
        }

    def get_boxes(self, sample_id, *, box_type="obb_gt", infer_labels=True):
        """Return official OBBs in metres and labels inferred from visible pixels."""

        if box_type != "obb_gt":
            raise ValueError("Structured3D only supports box_type='obb_gt'.")
        records = self._read_json(self._logical(sample_id, "bbox_3d.json"))
        if not isinstance(records, list):
            raise DataFormatError("Structured3D bbox_3d.json must contain a list.")
        inferred = self._infer_box_labels(sample_id) if infer_labels else {}
        label_names = self.metadata["label_names"]
        boxes = {}
        names = {}
        for record in records:
            try:
                instance_id = int(record["ID"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataFormatError(
                    "Structured3D box has an invalid instance ID."
                ) from exc
            if instance_id in boxes:
                raise DataFormatError(f"Duplicate Structured3D box ID: {instance_id}")
            boxes[instance_id] = self._box_from_annotation(record)
            names[instance_id] = label_names.get(inferred.get(instance_id), "unknown")
        return boxes, names

    @staticmethod
    def _line_polygons(pairs: Sequence[Sequence[int]]) -> list[list[int]]:
        remaining = [list(map(int, pair)) for pair in pairs]
        polygons = []
        while remaining:
            edge = remaining.pop(0)
            if len(edge) != 2 or edge[0] == edge[1]:
                raise DataFormatError("Structured3D plane has an invalid line edge.")
            polygon = edge[:]
            while polygon[-1] != polygon[0]:
                position = next(
                    (
                        index
                        for index, pair in enumerate(remaining)
                        if polygon[-1] in pair
                    ),
                    None,
                )
                if position is None:
                    raise DataFormatError(
                        "Structured3D plane edges do not form a closed polygon."
                    )
                pair = remaining.pop(position)
                polygon.append(pair[1] if pair[0] == polygon[-1] else pair[0])
                if len(polygon) > len(pairs) + 1:
                    raise DataFormatError(
                        "Structured3D plane polygon traversal did not close."
                    )
            polygons.append(polygon[:-1])
        return polygons

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        return 0.5 * float(
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
        )

    @classmethod
    def _triangulate_polygon(cls, points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            raise DataFormatError(
                "Structured3D room polygon needs at least three corners."
            )
        orientation = 1.0 if cls._polygon_area(points) > 0 else -1.0
        remaining = list(range(len(points)))
        triangles = []

        def cross(a, b, c):
            ab = b - a
            ac = c - a
            return float(ab[0] * ac[1] - ab[1] * ac[0])

        def inside(point, a, b, c):
            values = (
                cross(a, b, point),
                cross(b, c, point),
                cross(c, a, point),
            )
            return all(value >= -1e-8 for value in values) or all(
                value <= 1e-8 for value in values
            )

        while len(remaining) > 3:
            ear = None
            for position, current in enumerate(remaining):
                previous = remaining[position - 1]
                following = remaining[(position + 1) % len(remaining)]
                if (
                    orientation
                    * cross(points[previous], points[current], points[following])
                    <= 1e-8
                ):
                    continue
                if any(
                    inside(
                        points[other],
                        points[previous],
                        points[current],
                        points[following],
                    )
                    for other in remaining
                    if other not in {previous, current, following}
                ):
                    continue
                ear = position
                triangles.append((previous, current, following))
                break
            if ear is None:
                raise DataFormatError(
                    "Structured3D room polygon is self-intersecting or degenerate."
                )
            remaining.pop(ear)
        triangles.append(tuple(remaining))
        return np.asarray(triangles, dtype=np.int32)

    def _room_floor_polygon(self, sample_id, room_id) -> tuple[np.ndarray, float]:
        annotations = self.get_structure_annotations(sample_id)
        try:
            room_semantic = next(
                value
                for value in annotations["semantics"]
                if int(value["ID"]) == int(room_id)
            )
        except (StopIteration, KeyError, TypeError, ValueError) as exc:
            raise DataFormatError(
                f"No Structured3D 3D semantic annotation matches room {room_id}."
            ) from exc
        junctions = np.asarray(
            [value["coordinate"] for value in annotations["junctions"]],
            dtype=np.float64,
        )
        plane_lines = np.asarray(annotations["planeLineMatrix"])
        line_junctions = np.asarray(annotations["lineJunctionMatrix"])
        if junctions.ndim != 2 or junctions.shape[1] != 3:
            raise DataFormatError(
                "Structured3D junction coordinates must have shape (N, 3)."
            )
        if plane_lines.ndim != 2 or line_junctions.ndim != 2:
            raise DataFormatError(
                "Structured3D relationship matrices must be two-dimensional."
            )
        hole_lines = set()
        for semantic in annotations["semantics"]:
            if semantic.get("type") in {"door", "window"}:
                for plane_id in semantic.get("planeID", ()):
                    hole_lines.update(
                        np.flatnonzero(plane_lines[int(plane_id)]).tolist()
                    )
        polygons_by_type = {}
        for plane_id in room_semantic.get("planeID", ()):
            plane_id = int(plane_id)
            plane_type = annotations["planes"][plane_id]["type"]
            if plane_type not in {"floor", "ceiling"}:
                continue
            line_ids = [
                int(value)
                for value in np.flatnonzero(plane_lines[plane_id])
                if int(value) not in hole_lines
            ]
            pairs = [
                np.flatnonzero(line_junctions[line_id]).tolist() for line_id in line_ids
            ]
            polygons = self._line_polygons(pairs)
            polygons_by_type[plane_type] = max(
                polygons,
                key=lambda polygon: abs(self._polygon_area(junctions[polygon, :2])),
            )
        if "floor" not in polygons_by_type or "ceiling" not in polygons_by_type:
            raise DataFormatError(
                f"Structured3D room {room_id} has no floor/ceiling polygon."
            )
        floor = junctions[polygons_by_type["floor"]]
        ceiling_z = float(np.mean(junctions[polygons_by_type["ceiling"], 2]))
        return floor / 1000.0, ceiling_z / 1000.0

    def _room_mesh(
        self, sample_id, room_id, *, include_ceiling
    ) -> o3d.geometry.TriangleMesh:
        floor, ceiling_z = self._room_floor_polygon(sample_id, room_id)
        floor_triangles = self._triangulate_polygon(floor[:, :2])
        count = len(floor)
        ceiling = floor.copy()
        ceiling[:, 2] = ceiling_z
        triangles = []
        orientation = 1 if self._polygon_area(floor[:, :2]) > 0 else -1
        triangles.extend(
            triangle[::-1] if orientation > 0 else triangle
            for triangle in floor_triangles
        )
        if include_ceiling:
            triangles.extend(
                triangle + count if orientation > 0 else triangle[::-1] + count
                for triangle in floor_triangles
            )
        for index in range(count):
            following = (index + 1) % count
            triangles.extend(
                (
                    (index, following, count + following),
                    (index, count + following, count + index),
                )
            )
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.concatenate((floor, ceiling))),
            o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
        )
        mesh.compute_vertex_normals()
        return mesh

    def get_mesh(
        self,
        sample_id,
        *,
        mesh_type="layout",
        room_id=None,
        include_ceiling=True,
        skip_invalid_rooms=True,
    ) -> o3d.geometry.TriangleMesh:
        """Reconstruct structural room layout; this is not a furniture mesh."""

        if mesh_type not in {None, "layout"}:
            raise ValueError("Structured3D only supports mesh_type='layout'.")
        scene_id = self._validate_scene_id(sample_id)
        bad_rooms = set(self.metadata["errata"]["self_intersection_rooms"])
        if room_id is not None:
            room_id = self._validate_room_id(room_id)
            if f"{scene_id}_room_{room_id}" in bad_rooms:
                raise DataFormatError(
                    f"Structured3D marks {scene_id} room {room_id} as "
                    "self-intersecting."
                )
            return self._room_mesh(scene_id, room_id, include_ceiling=include_ceiling)
        meshes = []
        for value in self.get_room_ids(scene_id):
            if f"{scene_id}_room_{value}" in bad_rooms and skip_invalid_rooms:
                continue
            meshes.append(
                self._room_mesh(scene_id, value, include_ceiling=include_ceiling)
            )
        if not meshes:
            raise DataFormatError(
                f"Structured3D scene {scene_id} has no valid room mesh."
            )
        result = meshes[0]
        for mesh in meshes[1:]:
            result += mesh
        return result

    @staticmethod
    def _organized_normals(points: np.ndarray) -> np.ndarray:
        horizontal = np.gradient(points, axis=1)
        vertical = np.gradient(points, axis=0)
        normals = np.cross(horizontal, vertical)
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        return np.divide(
            normals,
            lengths,
            out=np.zeros_like(normals),
            where=lengths > 1e-8,
        )

    @staticmethod
    def _perspective_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        yy, xx = np.indices((height, width), dtype=np.float32)
        x = (xx - intrinsic[0, 2]) / intrinsic[0, 0] * depth
        y = (yy - intrinsic[1, 2]) / intrinsic[1, 1] * depth
        return np.stack((x, y, depth), axis=-1)

    @staticmethod
    def _panorama_points(depth: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        longitude = (np.arange(width, dtype=np.float32) + 0.5) / width * (
            2 * np.pi
        ) - np.pi
        latitude = (
            np.pi / 2 - (np.arange(height, dtype=np.float32) + 0.5) / height * np.pi
        )
        longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
        cosine = np.cos(latitude_grid)
        directions = np.stack(
            (
                np.sin(longitude_grid) * cosine,
                -np.sin(latitude_grid),
                np.cos(longitude_grid) * cosine,
            ),
            axis=-1,
        )
        return directions * depth[..., None]

    def get_points(
        self,
        sample_id,
        room_id,
        *,
        sources=("panorama",),
        configuration="full",
        lighting="raw",
        indices=None,
        semantic_label_space="nyu40",
        voxel_size=None,
        min_view_cosine=None,
    ) -> Structured3DPointData:
        """Fuse selected RGB-D renderings into a standard world-space point batch."""

        if isinstance(sources, str):
            sources = (sources,)
        sources = tuple(sources)
        if not sources or set(sources) - {"perspective", "panorama"}:
            raise ValueError("sources must contain perspective and/or panorama.")
        if configuration != "full":
            raise ValueError(
                "Point fusion requires configuration='full' for instance labels."
            )
        if semantic_label_space not in {"nyu40", "pointcept25"}:
            raise ValueError("semantic_label_space must be 'nyu40' or 'pointcept25'.")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        if min_view_cosine is not None and not 0 <= min_view_cosine <= 1:
            raise ValueError("min_view_cosine must be between 0 and 1.")
        room_id = self._validate_room_id(room_id)
        xyz_parts = []
        rgb_parts = []
        normal_parts = []
        semantic_parts = []
        instance_parts = []
        view_parts = []
        view_keys = []
        for source in sources:
            requested = (
                "rgb",
                "depth",
                "semantic_maps",
                "instance_maps",
                "world_to_camera",
            )
            if source == "perspective":
                requested += ("depth_intrinsics",)
            batch = self.get_frames(
                sample_id,
                room_id=room_id,
                source=source,
                configuration=configuration,
                lighting=lighting,
                indices=indices,
                items=requested,
            )
            for frame_index in range(len(batch)):
                depth = batch.depth[frame_index]
                if source == "perspective":
                    camera_points = self._perspective_points(
                        depth, batch.depth_intrinsics[frame_index]
                    )
                else:
                    camera_points = self._panorama_points(depth)
                camera_normals = self._organized_normals(camera_points)
                valid = np.isfinite(camera_points).all(axis=-1) & (depth > 0)
                if min_view_cosine is not None:
                    distance = np.linalg.norm(camera_points, axis=-1)
                    cosine = np.divide(
                        np.abs(np.sum(camera_points * camera_normals, axis=-1)),
                        distance,
                        out=np.zeros_like(distance),
                        where=distance > 1e-8,
                    )
                    valid &= cosine >= min_view_cosine
                camera_to_world = np.linalg.inv(batch.world_to_camera[frame_index])
                world_points = (
                    camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
                )
                world_normals = camera_normals @ camera_to_world[:3, :3].T
                semantic = batch.semantic_maps[frame_index].astype(np.int32, copy=True)
                semantic[semantic == 0] = self.invalid_obj_id
                instance = batch.instance_maps[frame_index].astype(np.int32, copy=True)
                instance[instance == _BACKGROUND_INSTANCE] = self.invalid_obj_id
                frame_xyz = world_points[valid]
                if not len(frame_xyz):
                    continue
                frame_rgb = batch.rgb[frame_index][valid]
                frame_normals = world_normals[valid]
                frame_semantic = semantic[valid]
                frame_instance = instance[valid]
                if voxel_size is not None:
                    local_selected = _first_point_per_voxel(frame_xyz, voxel_size)
                    frame_xyz = frame_xyz[local_selected]
                    frame_rgb = frame_rgb[local_selected]
                    frame_normals = frame_normals[local_selected]
                    frame_semantic = frame_semantic[local_selected]
                    frame_instance = frame_instance[local_selected]
                view_id = len(view_keys)
                view_keys.append(batch.frame_keys[frame_index])
                xyz_parts.append(frame_xyz)
                rgb_parts.append(frame_rgb)
                normal_parts.append(frame_normals)
                semantic_parts.append(frame_semantic)
                instance_parts.append(frame_instance)
                view_parts.append(np.full(len(frame_xyz), view_id, dtype=np.int32))
        if not xyz_parts:
            raise DataFormatError(
                f"No valid Structured3D RGB-D views for {sample_id}/{room_id}."
            )
        xyz = np.concatenate(xyz_parts).astype(np.float32)
        rgb = np.concatenate(rgb_parts).astype(np.uint8)
        normals = np.concatenate(normal_parts).astype(np.float32)
        semantic = np.concatenate(semantic_parts).astype(np.int32)
        instance = np.concatenate(instance_parts).astype(np.int32)
        view_ids = np.concatenate(view_parts)
        if semantic_label_space == "pointcept25":
            semantic = remap_labels(
                semantic,
                {
                    source: target
                    for target, source in enumerate(self.valid_class_ids_25)
                },
                self.invalid_obj_id,
            )
            semantic_names = dict(enumerate(self.class_names_25))
        else:
            semantic_names = self.metadata["label_names"]
        if voxel_size is not None and len(xyz):
            selected = _first_point_per_voxel(xyz, voxel_size)
            xyz, rgb, normals = xyz[selected], rgb[selected], normals[selected]
            semantic = semantic[selected]
            instance = instance[selected]
            view_ids = view_ids[selected]
        box_labels = self._infer_box_labels(sample_id)
        instance_names = {
            instance_id: self.metadata["label_names"].get(semantic_id, "unknown")
            for instance_id, semantic_id in box_labels.items()
        }
        return Structured3DPointData(
            xyz=xyz,
            rgb=rgb,
            normals=normals,
            semantic_labels=semantic,
            instance_labels=instance,
            view_ids=view_ids,
            view_keys=tuple(view_keys),
            semantic_id_to_name=semantic_names,
            instance_id_to_name=instance_names,
            invalid_id=self.invalid_obj_id,
        )

    def get_point_cloud(
        self,
        sample_id,
        room_id,
        *,
        sources=("panorama",),
        configuration="full",
        lighting="raw",
        indices=None,
        semantic_label_space="nyu40",
        voxel_size=None,
        min_view_cosine=None,
    ) -> o3d.geometry.PointCloud:
        """Return the Open3D representation of :meth:`get_points`."""

        return self.get_points(
            sample_id,
            room_id,
            sources=sources,
            configuration=configuration,
            lighting=lighting,
            indices=indices,
            semantic_label_space=semantic_label_space,
            voxel_size=voxel_size,
            min_view_cosine=min_view_cosine,
        ).to_point_cloud()

    def get_point_data(
        self,
        sample_id,
        room_id,
        *,
        sources=("panorama",),
        configuration="full",
        lighting="raw",
        indices=None,
        semantic_label_space="nyu40",
        voxel_size=None,
        min_view_cosine=None,
    ) -> Structured3DPointData:
        """Dataset-specific alias for :meth:`get_points`."""

        return self.get_points(
            sample_id,
            room_id,
            sources=sources,
            configuration=configuration,
            lighting=lighting,
            indices=indices,
            semantic_label_space=semantic_label_space,
            voxel_size=voxel_size,
            min_view_cosine=min_view_cosine,
        )

    def get_segmentation(
        self,
        sample_id,
        *,
        room_id=None,
        segmentation_type="semantic",
        sources=("panorama",),
        configuration="full",
        lighting="raw",
        indices=None,
        semantic_label_space="nyu40",
        voxel_size=None,
        min_view_cosine=None,
    ) -> Segmentation3D:
        """Return point-domain labels aligned with one fused room point set."""

        if room_id is None:
            raise ValueError("Structured3D get_segmentation() requires room_id.")
        data = self.get_points(
            sample_id,
            room_id,
            sources=sources,
            configuration=configuration,
            lighting=lighting,
            indices=indices,
            semantic_label_space=semantic_label_space,
            voxel_size=voxel_size,
            min_view_cosine=min_view_cosine,
        )
        if segmentation_type == "semantic":
            return Segmentation3D(
                data.semantic_labels,
                data.semantic_id_to_name,
                "point",
                self.invalid_obj_id,
            )
        if segmentation_type == "instance":
            return Segmentation3D(
                data.instance_labels,
                data.instance_id_to_name,
                "point",
                self.invalid_obj_id,
            )
        raise ValueError("segmentation_type must be 'semantic' or 'instance'.")
