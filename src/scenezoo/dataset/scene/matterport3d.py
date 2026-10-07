"""Matterport3D houses, region annotations, boxes, and RGB-D views."""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path

from cachetools import LRUCache, cachedmethod
import numpy as np
import open3d as o3d
from PIL import Image

from ...io.image import read_image
from ...ops.frame import (
    apply_image_transform,
    build_pixel_transform,
    select_frame_indices,
    transform_intrinsics,
    valid_camera_mask,
    validate_frame_items,
)
from ...ops.annotation import read_ply, remap_labels
from ...ops.geometry import construct_box, load_triangle_mesh
from ..base import DataFormatError, Dataset
from ..registry import register_dataset
from ..types import FrameBatch, Segmentation3D

_CATEGORY_MAPPING_URL = (
    "https://raw.githubusercontent.com/niessner/Matterport/"
    "master/metadata/category_mapping.tsv"
)
_SPLIT_URL = (
    "https://raw.githubusercontent.com/niessner/Matterport/"
    "master/tasks/benchmark/scenes_{split}.txt"
)
_TRAIN_SPLIT_URL = _SPLIT_URL.format(split="train")
_VAL_SPLIT_URL = _SPLIT_URL.format(split="val")
_TEST_SPLIT_URL = _SPLIT_URL.format(split="test")
_REGION_PLY = re.compile(r"region(?P<id>\d+)\.ply", re.IGNORECASE)
_RAW_IMAGE = re.compile(
    r"^(?P<pano>.+)_(?P<kind>[id])(?P<camera>\d+)_(?P<yaw>\d+)\.(?:jpg|png)$",
    re.IGNORECASE,
)
_RAW_POSE = re.compile(
    r"^(?P<pano>.+)_pose_?(?P<camera>\d+)_(?P<yaw>\d+)\.(?:txt|pose)$",
    re.IGNORECASE,
)
_RAW_INTRINSIC = re.compile(
    r"^(?P<pano>.+)_intrinsics?_?(?P<camera>\d+)"
    r"(?:_(?P<yaw>\d+))?\.(?:txt|intrinsics?)$",
    re.IGNORECASE,
)
_INTEGER_COLUMNS = {
    "index",
    "count",
    "nyuId",
    "nyu40id",
    "eigen13id",
    "mpcat40index",
}
_LABEL_SPACES = {
    "raw": ("index", "raw_category"),
    "mpcat40": ("mpcat40index", "mpcat40"),
    "nyu40": ("nyu40id", "nyu40class"),
    "eigen13": ("eigen13id", "eigen13class"),
}
# The official category TSV has one anomalous ``floor mat`` row for NYU40 ID
# 40; NYU40's canonical class name for that ID is ``otherprop``.
_LABEL_NAME_OVERRIDES = {("nyu40", 40): "otherprop"}
_CV_FROM_UNDISTORTED = np.diag([1.0, -1.0, -1.0, 1.0])
# Depth PNGs store integer units of 0.25 mm.
_DEPTH_UNITS_PER_METRE = 4000.0


def _numbers(payload: bytes, *, expected: int | None, source: str) -> np.ndarray:
    try:
        values = np.fromstring(payload.decode("utf-8"), sep=" ", dtype=np.float64)
    except UnicodeDecodeError as exc:
        raise DataFormatError(f"Matterport3D {source} is not UTF-8 text.") from exc
    if expected is not None and values.size != expected:
        raise DataFormatError(
            f"Matterport3D {source} contains {values.size} values; expected {expected}."
        )
    if not np.isfinite(values).all():
        raise DataFormatError(f"Matterport3D {source} contains non-finite values.")
    return values


def _matrix(values, shape: tuple[int, int], source: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != shape[0] * shape[1] or not np.isfinite(array).all():
        raise DataFormatError(
            f"Matterport3D {source} must contain a finite {shape[0]}x{shape[1]} matrix."
        )
    return array.reshape(shape)


def _decode_name(value: str) -> str:
    return value.replace("#", " ")


def _parse_house(payload: bytes, source: str) -> dict:
    """Parse the useful, documented records from one official ``.house`` file."""

    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DataFormatError(
            f"Matterport3D house file is not UTF-8: {source}"
        ) from exc
    result = {
        "name": None,
        "label": None,
        "bbox": None,
        "levels": {},
        "regions": {},
        "categories": {},
        "objects": {},
        "images": [],
    }
    expected = {}
    line_number = 0
    try:
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            if not fields or fields[0].upper() == "ASCII":
                continue
            command = fields[0]
            if command == "H":
                if len(fields) < 25:
                    raise ValueError("incomplete H record")
                result["name"] = fields[1]
                result["label"] = _decode_name(fields[2])
                expected = {
                    "images": int(fields[3]),
                    "objects": int(fields[8]),
                    "categories": int(fields[9]),
                    "regions": int(fields[10]),
                    "levels": int(fields[12]),
                }
                result["bbox"] = np.asarray(fields[18:24], dtype=np.float64)
            elif command == "L":
                identifier = int(fields[1])
                if identifier in result["levels"]:
                    raise ValueError(f"duplicate level {identifier}")
                result["levels"][identifier] = {
                    "id": identifier,
                    "region_count": int(fields[2]),
                    "label": _decode_name(fields[3]),
                    "position": np.asarray(fields[4:7], dtype=np.float64),
                    "bbox": np.asarray(fields[7:13], dtype=np.float64),
                }
            elif command == "R":
                identifier = int(fields[1])
                if identifier in result["regions"]:
                    raise ValueError(f"duplicate region {identifier}")
                result["regions"][identifier] = {
                    "id": identifier,
                    "level_id": int(fields[2]),
                    "label": _decode_name(fields[5]),
                    "position": np.asarray(fields[6:9], dtype=np.float64),
                    "bbox": np.asarray(fields[9:15], dtype=np.float64),
                    "height": float(fields[15]),
                }
            elif command == "C":
                identifier = int(fields[1])
                if identifier in result["categories"]:
                    raise ValueError(f"duplicate category {identifier}")
                result["categories"][identifier] = {
                    "id": identifier,
                    "mapping_index": int(fields[2]),
                    "mapping_name": _decode_name(fields[3]),
                    "mpcat40_id": int(fields[4]),
                    "mpcat40_name": _decode_name(fields[5]),
                }
            elif command == "O":
                identifier = int(fields[1])
                if identifier in result["objects"]:
                    raise ValueError(f"duplicate object {identifier}")
                result["objects"][identifier] = {
                    "id": identifier,
                    "region_id": int(fields[2]),
                    "category_id": int(fields[3]),
                    "center": np.asarray(fields[4:7], dtype=np.float64),
                    "axis0": np.asarray(fields[7:10], dtype=np.float64),
                    "axis1": np.asarray(fields[10:13], dtype=np.float64),
                    "radii": np.asarray(fields[13:16], dtype=np.float64),
                }
            elif command == "I":
                result["images"].append(
                    {
                        "id": int(fields[1]),
                        "panorama_id": int(fields[2]),
                        "name": fields[3],
                        "camera_id": int(fields[4]),
                        "yaw_id": int(fields[5]),
                        "world_to_camera": _matrix(
                            fields[6:22], (4, 4), f"image matrix on line {line_number}"
                        ),
                        "intrinsics": _matrix(
                            fields[22:31],
                            (3, 3),
                            f"image intrinsics on line {line_number}",
                        ),
                        "image_size": (int(fields[31]), int(fields[32])),
                    }
                )
    except (IndexError, TypeError, ValueError) as exc:
        raise DataFormatError(
            f"Invalid Matterport3D house record at {source}:{line_number}."
        ) from exc
    if result["name"] is None:
        raise DataFormatError(f"Matterport3D house file has no H record: {source}")
    for name in ("levels", "regions", "categories", "objects", "images"):
        if name in expected and len(result[name]) != expected[name]:
            raise DataFormatError(
                f"Matterport3D house file declares {expected[name]} {name}, "
                f"but contains {len(result[name])}: {source}"
            )
    return result


@register_dataset(
    "matterport3d",
    description="Matterport3D houses with region annotations and RGB-D views.",
)
class Matterport3D(Dataset):
    """Read the official V1 Matterport3D house-oriented release."""

    def __init__(
        self,
        root_dir,
        *,
        mesh_dir="{scene_id}/matterport_mesh",
        house_segmentation_dir="{scene_id}/house_segmentations",
        region_segmentation_dir="{scene_id}/region_segmentations",
        poisson_dir="{scene_id}/poisson_meshes",
        raw_color_dir="{scene_id}/matterport_color_images",
        raw_depth_dir="{scene_id}/matterport_depth_images",
        raw_intrinsics_dir="{scene_id}/matterport_camera_intrinsics",
        raw_poses_dir="{scene_id}/matterport_camera_poses",
        undistorted_camera_dir="{scene_id}/undistorted_camera_parameters",
        undistorted_color_dir="{scene_id}/undistorted_color_images",
        undistorted_depth_dir="{scene_id}/undistorted_depth_images",
        label_mapping_file=_CATEGORY_MAPPING_URL,
        train_split_file=_TRAIN_SPLIT_URL,
        val_split_file=_VAL_SPLIT_URL,
        test_split_file=_TEST_SPLIT_URL,
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
        self.scans_dir = self._find_scans_dir()
        self.mesh_dir = mesh_dir
        self.house_segmentation_dir = house_segmentation_dir
        self.region_segmentation_dir = region_segmentation_dir
        self.poisson_dir = poisson_dir
        self.frame_dirs = {
            "raw": {
                "rgb": raw_color_dir,
                "depth": raw_depth_dir,
                "intrinsics": raw_intrinsics_dir,
                "poses": raw_poses_dir,
            },
            "undistorted": {
                "camera": undistorted_camera_dir,
                "rgb": undistorted_color_dir,
                "depth": undistorted_depth_dir,
            },
        }
        self.label_mapping_file = label_mapping_file
        self.split_files = {
            "train": train_split_file,
            "val": val_split_file,
            "test": test_split_file,
        }
        self._directory_indices: dict[Path, dict[str, str]] = {}
        self._house_cache = LRUCache(maxsize=8)
        self._raw_record_cache = LRUCache(maxsize=8)
        self._undistorted_record_cache = LRUCache(maxsize=8)

    def _find_scans_dir(self) -> Path:
        candidates = (self.root_dir / "v1" / "scans", self.root_dir / "scans")
        return next((path for path in candidates if path.is_dir()), self.root_dir)

    def _path(self, template: str, sample_id: str) -> Path:
        return self.scans_dir / template.format(scene_id=str(sample_id))

    def _archive_path(self, template: str, sample_id: str) -> Path:
        directory = self._path(template, sample_id)
        return directory.parent / f"{directory.name}.zip"

    def _load_splits(self):
        result = {}
        for split, path in self.split_files.items():
            with self.open_file(path, "r") as handle:
                values = [line.strip() for line in handle if line.strip()]
            if len(values) != len(set(values)):
                raise DataFormatError(f"Duplicate Matterport3D ID in {split} split.")
            result[split] = values
        return result

    def _check(self, *, sample_ids, require_complete):
        from ..check import (
            CheckBuilder,
            check_scene_coverage,
            load_expected_ids,
            normalize_sample_ids,
        )

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The Matterport3D root does not exist.",
            expected="<root>/v1/scans/<house_id>/<extracted_bundle>/",
            hint="Pass the Matterport3D download root, scans directory, or v1/scans parent.",
        ):
            return builder.finish()
        if not self.scans_dir.is_dir():
            builder.error(
                "missing-scans-directory",
                "The resolved Matterport3D scans directory is missing.",
                path=self.scans_dir,
                expected="v1/scans/<house_id>/ or scans/<house_id>/",
                hint="Keep the official per-house directory level around the bundles.",
            )
            return builder.finish()

        templates = (
            self.mesh_dir,
            self.house_segmentation_dir,
            self.region_segmentation_dir,
            self.poisson_dir,
            *(value for group in self.frame_dirs.values() for value in group.values()),
        )

        houses = tuple(path for path in self.scans_dir.iterdir() if path.is_dir())
        present = {
            house.name
            for house in houses
            if any(self._path(template, house.name).is_dir() for template in templates)
        }
        archive_houses = {
            house.name
            for house in houses
            if any(
                self._archive_path(template, house.name).is_file()
                for template in templates
            )
        }
        archives = tuple(
            self._archive_path(template, house.name)
            for house in houses
            for template in templates
            if self._archive_path(template, house.name).is_file()
        )
        expected_layout = "<scans>/<house_id>/<official_bundle_name>/"
        requested = normalize_sample_ids(sample_ids)
        if requested is None:
            selected = check_scene_coverage(
                builder,
                present_ids=present,
                expected_ids=load_expected_ids(self, builder),
                sample_ids=None,
                require_complete=require_complete,
                expected_layout=expected_layout,
            )
        else:
            builder.stats["scenes_found"] = len(present & set(requested))
            builder.stats["scenes_requested"] = len(requested)
            selected = tuple(sid for sid in requested if sid in present)
            unavailable = []
            for sid in requested:
                if sid in present:
                    continue
                if sid in archive_houses:
                    candidates = [
                        self._archive_path(template, sid)
                        for template in templates
                        if self._archive_path(template, sid).is_file()
                    ]
                    builder.error(
                        "house-not-extracted",
                        f"{sid} has downloaded Matterport3D bundles but none are extracted.",
                        path=candidates[0],
                        expected=str(candidates[0].with_suffix("")),
                        hint=(
                            f"Extract the downloaded ZIP bundles inside "
                            f"{self.scans_dir / sid} before using the dataset API."
                        ),
                    )
                else:
                    unavailable.append(sid)
            if unavailable:
                builder.error(
                    "missing-requested-scenes",
                    f"Requested house IDs are missing: {', '.join(unavailable[:10])}",
                    expected=expected_layout,
                    hint="Download and extract the requested Matterport3D houses.",
                )
        builder.stats["extracted_houses"] = len(present)
        builder.stats["zip_bundles_found"] = len(archives)
        if archives:
            examples = ", ".join(str(path) for path in archives[:5])
            builder.warning(
                "archives-need-extraction",
                f"{len(archives)} Matterport3D ZIP bundles are present but ignored by "
                f"SceneZoo. Examples: {examples}",
                expected="A same-named extracted directory for every required bundle.",
                hint="Extract the ZIP bundles in place before using their modalities.",
            )
        if not present and requested is None:
            builder.error(
                "no-extracted-houses",
                "No Matterport3D houses with extracted bundle directories were found.",
                path=self.scans_dir,
                expected=expected_layout,
                hint="Download and extract at least one official house bundle.",
            )
            return builder.finish()

        for sid in selected:
            for label, template in (
                ("house segmentation", self.house_segmentation_dir),
                ("region segmentation", self.region_segmentation_dir),
            ):
                directory = self._path(template, sid)
                archive = self._archive_path(template, sid)
                if not directory.is_dir():
                    builder.error(
                        "bundle-not-extracted"
                        if archive.is_file()
                        else "missing-annotation-bundle",
                        f"{sid} is missing its extracted {label} directory.",
                        path=archive if archive.is_file() else directory,
                        expected=str(directory),
                        hint=(
                            f"Extract {archive.name} in place."
                            if archive.is_file()
                            else "Download and extract the official annotation bundle."
                        ),
                    )
            for source, group in self.frame_dirs.items():
                extracted = {
                    key: self._path(template, sid).is_dir()
                    for key, template in group.items()
                }
                downloaded = {
                    key: value or self._archive_path(group[key], sid).is_file()
                    for key, value in extracted.items()
                }
                if any(downloaded.values()) and not all(extracted.values()):
                    missing = [key for key, value in extracted.items() if not value]
                    builder.error(
                        "incomplete-frame-bundle-group",
                        f"{sid} lacks extracted {source} frame directories: "
                        f"{', '.join(missing)}.",
                        expected=f"All {source} camera/RGB/depth directories together.",
                        hint="Download any missing bundles and extract every ZIP in place.",
                    )
        builder.info(
            "extracted-directories-required",
            "Matterport3D ZIP bundles are download artifacts. SceneZoo reads only "
            "same-named extracted directories.",
        )
        return builder.finish()

    def _load_metadata(self):
        rows = []
        with self.open_file(self.label_mapping_file, "r") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"index", "raw_category"} | {
                column for pair in _LABEL_SPACES.values() for column in pair
            }
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise DataFormatError(
                    "Invalid Matterport3D category_mapping.tsv header."
                )
            for raw in reader:
                row = dict(raw)
                try:
                    for key in _INTEGER_COLUMNS:
                        if key in row and row[key] not in {"", None}:
                            row[key] = int(row[key])
                except ValueError as exc:
                    raise DataFormatError(
                        f"Invalid Matterport3D category row: {raw.get('raw_category')!r}."
                    ) from exc
                rows.append(row)
        by_index = {}
        by_raw = {}
        for row in rows:
            identifier = row["index"]
            name = row["raw_category"]
            if identifier in by_index or name in by_raw:
                raise DataFormatError(
                    f"Duplicate Matterport3D category: {identifier}/{name!r}."
                )
            by_index[identifier] = row
            by_raw[name] = row
        spaces = {}
        for space, (id_key, name_key) in _LABEL_SPACES.items():
            mapping = {}
            for row in rows:
                identifier, name = row.get(id_key), row.get(name_key)
                if not isinstance(identifier, int) or not name:
                    continue
                name = _LABEL_NAME_OVERRIDES.get((space, identifier), str(name))
                previous = mapping.setdefault(identifier, name)
                if previous != name:
                    raise DataFormatError(
                        f"Conflicting Matterport3D {space} label {identifier}."
                    )
            spaces[space] = mapping
        return {
            "label_mapping": by_raw,
            "index_to_name": {
                key: row["raw_category"] for key, row in by_index.items()
            },
            "categories_by_index": by_index,
            "label_spaces": spaces,
        }

    def _directory_path(self, template: str, sample_id: str) -> Path:
        directory = self._path(template, sample_id)
        if directory.is_dir():
            return directory
        archive = self._archive_path(template, sample_id)
        if archive.is_file():
            raise FileNotFoundError(
                f"Matterport3D bundle for {sample_id!r} is not extracted. Extract "
                f"{archive} to {directory} before using the dataset API."
            )
        raise FileNotFoundError(
            f"Matterport3D directory was not found for {sample_id!r}: {directory}"
        )

    def _optional_directory_path(self, template: str, sample_id: str) -> Path | None:
        try:
            return self._directory_path(template, sample_id)
        except FileNotFoundError:
            return None

    def _directory_index(self, path: Path) -> dict[str, str]:
        if path in self._directory_indices:
            return self._directory_indices[path]
        matches = {}
        try:
            names = [
                str(file.relative_to(path))
                for file in path.rglob("*")
                if file.is_file()
            ]
        except OSError as exc:
            raise DataFormatError(
                f"Cannot index Matterport3D directory: {path}"
            ) from exc
        for name in names:
            basename = Path(name).name
            if basename in matches:
                raise DataFormatError(
                    f"Matterport3D bundle contains duplicate basename {basename!r}: {path}"
                )
            matches[basename] = name
        self._directory_indices[path] = matches
        return matches

    def _index(
        self, template: str, sample_id: str, *, optional=False
    ) -> tuple[Path | None, dict[str, str]]:
        path = (
            self._optional_directory_path(template, sample_id)
            if optional
            else self._directory_path(template, sample_id)
        )
        return (path, {}) if path is None else (path, self._directory_index(path))

    @staticmethod
    def _read_member(path: Path, member: str) -> bytes:
        try:
            return (path / member).read_bytes()
        except OSError as exc:
            raise DataFormatError(
                f"Cannot read Matterport3D file {member!r} from {path}."
            ) from exc

    @staticmethod
    def _read_members(path: Path, members: dict[str, str]) -> dict[str, bytes]:
        """Read several files from one extracted directory."""

        try:
            return {
                basename: (path / member).read_bytes()
                for basename, member in members.items()
            }
        except OSError as exc:
            raise DataFormatError(
                f"Cannot read Matterport3D files from {path}."
            ) from exc

    @staticmethod
    def _member(index: dict[str, str], basename: str, source: Path) -> str:
        try:
            return index[basename]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Matterport3D member {basename!r} was not found in {source}."
            ) from exc

    @staticmethod
    def _region_stem(region_id) -> str:
        value = str(region_id)
        if value.lower().startswith("region"):
            value = value[6:]
        if not value.isdigit():
            raise ValueError("Matterport3D region_id must be an integer or 'regionN'.")
        return f"region{int(value)}"

    def get_region_ids(self, sample_id) -> list[str]:
        """Return numeric region IDs whose annotated PLY is available."""

        path, index = self._index(self.region_segmentation_dir, str(sample_id))
        values = [
            int(match.group("id"))
            for name in index
            if (match := _REGION_PLY.fullmatch(name)) is not None
        ]
        if not values:
            raise DataFormatError(f"No region PLY files found in {path}.")
        return [str(value) for value in sorted(values)]

    @cachedmethod(lambda self: self._house_cache)
    def _house_data(self, sample_id: str) -> dict:
        path, index = self._index(self.house_segmentation_dir, sample_id)
        basename = f"{sample_id}.house"
        if basename not in index:
            candidates = [name for name in index if name.lower().endswith(".house")]
            if len(candidates) != 1:
                raise DataFormatError(f"Cannot identify one .house file in {path}.")
            basename = candidates[0]
        member = index[basename]
        return _parse_house(self._read_member(path, member), f"{path}:{member}")

    def get_house_info(self, sample_id) -> dict:
        """Return the official level, region, category, object, and image hierarchy."""

        return copy.deepcopy(self._house_data(str(sample_id)))

    def get_mesh(
        self,
        sample_id,
        *,
        mesh_type=None,
        region_id=None,
        poisson_level=None,
    ):
        """Load a textured house, annotated house/region, or Poisson mesh."""

        sample_id = str(sample_id)
        mesh_type = mesh_type or "house"
        if mesh_type == "region":
            if region_id is None:
                raise ValueError("mesh_type='region' requires region_id.")
            stem = self._region_stem(region_id)
            path, index = self._index(self.region_segmentation_dir, sample_id)
            basename = f"{stem}.ply"
            member = self._member(index, basename, path)
            return load_triangle_mesh(path / member)
        if region_id is not None:
            raise ValueError("region_id is only valid with mesh_type='region'.")
        if mesh_type == "house":
            path, index = self._index(self.house_segmentation_dir, sample_id)
            basename = f"{sample_id}.ply"
            if basename not in index:
                candidates = [name for name in index if name.lower().endswith(".ply")]
                if len(candidates) != 1:
                    raise DataFormatError(f"Cannot identify one house PLY in {path}.")
                basename = candidates[0]
            return load_triangle_mesh(path / index[basename])
        if mesh_type == "poisson":
            path, index = self._index(self.poisson_dir, sample_id)
            suffix = "" if poisson_level is None else f"_{int(poisson_level)}"
            basename = f"{sample_id}{suffix}.ply"
            member = self._member(index, basename, path)
            return load_triangle_mesh(path / member)
        if mesh_type != "raw":
            raise ValueError(
                f"Unknown Matterport3D mesh type: {mesh_type!r}; "
                "use 'raw', 'house', 'region', or 'poisson'."
            )
        if poisson_level is not None:
            raise ValueError("poisson_level is only valid with mesh_type='poisson'.")
        path, index = self._index(self.mesh_dir, sample_id)
        basename = f"{sample_id}.obj"
        candidates = [name for name in index if name.lower().endswith(".obj")]
        if basename in index:
            member = index[basename]
        elif len(candidates) == 1:
            member = index[candidates[0]]
        else:
            raise DataFormatError(f"Cannot identify one textured OBJ in {path}.")
        return load_triangle_mesh(path / member, enable_post_processing=True)

    def _annotation_source(self, sample_id: str, region_id):
        if region_id is None:
            path, index = self._index(self.house_segmentation_dir, sample_id)
            stem = sample_id
        else:
            path, index = self._index(self.region_segmentation_dir, sample_id)
            stem = self._region_stem(region_id)
        return path, index, stem

    def _face_annotations(self, sample_id: str, region_id):
        path, index, stem = self._annotation_source(sample_id, region_id)
        ply_member = self._member(index, f"{stem}.ply", path)
        try:
            ply = read_ply(path / ply_member)
            face = ply["face"].data
        except (OSError, KeyError, ValueError) as exc:
            raise DataFormatError(
                f"Invalid Matterport3D annotated PLY in {path}."
            ) from exc
        fields = set(face.dtype.names or ())

        def attribute(*names):
            name = next((name for name in names if name in fields), None)
            return None if name is None else np.asarray(face[name], dtype=np.int64)

        instance = attribute("segment_id", "face_segment")
        category = attribute("category_id", "face_category")
        if instance is not None and category is not None:
            return instance, category, len(face)

        semseg = self._member(index, f"{stem}.semseg.json", path)
        fsegs = self._member(index, f"{stem}.fsegs.json", path)
        try:
            groups = json.loads(self._read_member(path, semseg))
            segment_data = json.loads(self._read_member(path, fsegs))
            segment_ids = np.asarray(segment_data["segIndices"], dtype=np.int64)
            group_values = groups["segGroups"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise DataFormatError(
                f"Invalid Matterport3D JSON annotation in {path}."
            ) from exc
        if segment_ids.ndim != 1 or len(segment_ids) != len(face):
            raise DataFormatError(
                f"Matterport3D face/segment count mismatch in {path}: "
                f"{len(face)} faces and {len(segment_ids)} labels."
            )
        assigned_segments = set()
        known_segments = set(map(int, np.unique(segment_ids)))
        if np.any(segment_ids < 0):
            raise DataFormatError(
                f"Matterport3D face segments must be non-negative: {path}"
            )
        segment_count = int(segment_ids.max(initial=-1)) + 1
        segment_object = np.full(segment_count, self.invalid_obj_id, dtype=np.int64)
        segment_category = np.full(segment_count, self.invalid_obj_id, dtype=np.int64)
        for fallback_id, group in enumerate(group_values):
            try:
                object_id = int(group.get("id", fallback_id))
                raw_label = group.get("label")
                if raw_label is None and "label_index" in group:
                    row = self.metadata["categories_by_index"].get(
                        int(group["label_index"])
                    )
                    raw_label = None if row is None else row["raw_category"]
                if raw_label is None:
                    raise KeyError("label")
                row = self.metadata["label_mapping"].get(str(raw_label))
                category_id = self.invalid_obj_id if row is None else int(row["index"])
                members = np.asarray(group["segments"], dtype=np.int64)
            except (TypeError, ValueError, KeyError) as exc:
                raise DataFormatError(
                    f"Invalid Matterport3D object group {fallback_id} in {path}."
                ) from exc
            requested_segments = set(map(int, members))
            unknown = requested_segments - known_segments
            overlap = requested_segments & assigned_segments
            if unknown:
                raise DataFormatError(
                    f"Matterport3D object {object_id} references unknown segments "
                    f"{sorted(unknown)} in {path}."
                )
            if overlap:
                raise DataFormatError(
                    f"Matterport3D segments belong to multiple objects in {path}: "
                    f"{sorted(overlap)}"
                )
            assigned_segments.update(requested_segments)
            segment_object[members] = object_id
            segment_category[members] = category_id
        return segment_object[segment_ids], segment_category[segment_ids], len(face)

    def _category_value(self, category_id: int, label_space: str):
        try:
            id_key, name_key = _LABEL_SPACES[label_space]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Matterport3D label space {label_space!r}; "
                f"use {', '.join(_LABEL_SPACES)}."
            ) from exc
        row = self.metadata["categories_by_index"].get(int(category_id))
        if row is None or not isinstance(row.get(id_key), int) or not row.get(name_key):
            return None
        identifier = int(row[id_key])
        name = self.metadata["label_spaces"][label_space].get(
            identifier, str(row[name_key])
        )
        return identifier, name

    def get_segmentation(
        self,
        sample_id,
        *,
        region_id=None,
        segmentation_type="instance",
        label_space="raw",
    ):
        """Return face-domain instance or semantic labels for a house/region mesh."""

        if segmentation_type not in {"instance", "semantic"}:
            raise ValueError(
                "Matterport3D segmentation_type must be 'instance' or 'semantic'."
            )
        if label_space not in _LABEL_SPACES:
            raise ValueError(
                f"Unknown Matterport3D label space {label_space!r}; "
                f"use {', '.join(_LABEL_SPACES)}."
            )
        instance, category, _ = self._face_annotations(str(sample_id), region_id)
        invalid = self.invalid_obj_id
        if segmentation_type == "instance":
            labelled = (instance != invalid) & (category != invalid)
            pairs = np.unique(np.column_stack((instance, category))[labelled], axis=0)
            object_ids, counts = np.unique(pairs[:, 0], return_counts=True)
            if np.any(counts > 1):
                object_id = int(object_ids[counts > 1][0])
                conflicting = pairs[pairs[:, 0] == object_id, 1].tolist()
                raise DataFormatError(
                    f"Matterport3D object {object_id} has conflicting categories: "
                    f"{conflicting}"
                )
            names = {}
            for object_id, category_id in pairs.tolist():
                converted = self._category_value(category_id, label_space)
                if converted is not None:
                    names[object_id] = converted[1]
            labels = instance.astype(np.int32, copy=False)
        else:
            converted = {
                int(category_id): self._category_value(int(category_id), label_space)
                for category_id in np.unique(category)
                if category_id != invalid
            }
            converted = {key: value for key, value in converted.items() if value}
            mapping = {key: value[0] for key, value in converted.items()}
            labels = remap_labels(category, mapping, invalid)
            names = dict(converted.values())
        return Segmentation3D(labels, names, "face", invalid)

    @staticmethod
    def _official_obb(record: dict):
        center = record["center"]
        axis0 = record["axis0"]
        axis1 = record["axis1"]
        radii = record["radii"]
        norm0, norm1 = np.linalg.norm(axis0), np.linalg.norm(axis1)
        if (
            not np.isfinite(np.concatenate((center, axis0, axis1, radii))).all()
            or norm0 <= 1e-8
            or norm1 <= 1e-8
            or np.any(radii < 0)
        ):
            raise DataFormatError(
                f"Invalid Matterport3D OBB for object {record['id']}."
            )
        axis0 = axis0 / norm0
        axis1 = axis1 - axis0 * np.dot(axis0, axis1)
        norm1 = np.linalg.norm(axis1)
        if norm1 <= 1e-8:
            raise DataFormatError(
                f"Degenerate Matterport3D OBB for object {record['id']}."
            )
        axis1 /= norm1
        axis2 = np.cross(axis0, axis1)
        rotation = np.column_stack((axis0, axis1, axis2))
        return o3d.geometry.OrientedBoundingBox(center, rotation, 2.0 * radii)

    def get_boxes(
        self,
        sample_id,
        *,
        box_type="obb_gt",
        region_id=None,
        label_space="raw",
    ):
        """Return official object OBBs, optionally filtered to one region."""

        if label_space not in _LABEL_SPACES:
            raise ValueError(
                f"Unknown Matterport3D label space {label_space!r}; "
                f"use {', '.join(_LABEL_SPACES)}."
            )
        valid_box_types = {"obb_gt", "aabb", "obb", "mobb", "mobb_gravity"}
        if box_type not in valid_box_types:
            raise ValueError(
                f"Unknown Matterport3D box type {box_type!r}; "
                f"use {', '.join(sorted(valid_box_types))}."
            )
        house = self._house_data(str(sample_id))
        region = None if region_id is None else int(self._region_stem(region_id)[6:])
        boxes, names = {}, {}
        for object_id, record in house["objects"].items():
            if region is not None and record["region_id"] != region:
                continue
            if record["category_id"] < 0:
                # Official houses mark unlabeled objects with category index -1;
                # like unmapped categories, they have no box label.
                continue
            category = house["categories"].get(record["category_id"])
            if category is None:
                raise DataFormatError(
                    f"Matterport3D object {object_id} references an unknown category."
                )
            converted = self._category_value(category["mapping_index"], label_space)
            if converted is None:
                continue
            official = self._official_obb(record)
            if box_type == "obb_gt":
                box = official
            else:
                box = construct_box(
                    np.asarray(official.get_box_points()), box_type=box_type
                )
            boxes[object_id] = box
            names[object_id] = converted[1]
        return boxes, names

    @staticmethod
    def _parse_undistorted_config(payload: bytes, source: str) -> list[dict]:
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise DataFormatError(
                f"Invalid Matterport3D camera config: {source}"
            ) from exc
        records = []
        intrinsics = None
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                if fields[0] in {
                    "intrinsics_matrix",
                    "color_intrinsics_matrix",
                    "depth_intrinsics_matrix",
                }:
                    intrinsics = _matrix(
                        fields[1:],
                        (3, 3),
                        f"camera intrinsics at {source}:{line_number}",
                    )
                elif fields[0] == "scan":
                    if intrinsics is None:
                        raise ValueError("scan appears before intrinsics_matrix")
                    records.append(
                        {
                            "key": Path(fields[2]).name,
                            "depth": Path(fields[1]).name,
                            "rgb": Path(fields[2]).name,
                            "intrinsics": intrinsics.copy(),
                            "camera_to_world": _matrix(
                                fields[3:],
                                (4, 4),
                                f"camera pose at {source}:{line_number}",
                            ),
                            "distortion": np.empty(0, dtype=np.float64),
                            "image_size": None,
                        }
                    )
            except (IndexError, ValueError) as exc:
                raise DataFormatError(
                    f"Invalid Matterport3D camera config at {source}:{line_number}."
                ) from exc
        if not records:
            raise DataFormatError(f"No scan records in Matterport3D config: {source}")
        return records

    @cachedmethod(lambda self: self._undistorted_record_cache)
    def _undistorted_records(self, sample_id: str) -> tuple[dict, ...]:
        template = self.frame_dirs["undistorted"]["camera"]
        path, index = self._index(template, sample_id)
        basename = f"{sample_id}.conf"
        if basename not in index:
            candidates = [name for name in index if name.lower().endswith(".conf")]
            if len(candidates) != 1:
                raise DataFormatError(f"Cannot identify one camera config in {path}.")
            basename = candidates[0]
        records = self._parse_undistorted_config(
            self._read_member(path, index[basename]), f"{path}:{index[basename]}"
        )
        return tuple(records)

    @staticmethod
    def _raw_key(match: re.Match) -> tuple[str, int, int]:
        return (
            match.group("pano"),
            int(match.group("camera")),
            int(match.group("yaw")),
        )

    @cachedmethod(lambda self: self._raw_record_cache)
    def _raw_records(self, sample_id: str) -> tuple[dict, ...]:
        templates = self.frame_dirs["raw"]
        bundles = {
            name: self._index(template, sample_id, optional=True)
            for name, template in templates.items()
        }
        unextracted = [
            self._archive_path(template, sample_id)
            for name, template in templates.items()
            if bundles[name][0] is None
            and self._archive_path(template, sample_id).is_file()
        ]
        if unextracted:
            raise FileNotFoundError(
                f"Matterport3D raw frame bundles are not extracted for {sample_id!r}: "
                + ", ".join(str(path) for path in unextracted)
            )
        images = {}
        for item in ("rgb", "depth"):
            _, index = bundles[item]
            for basename in index:
                match = _RAW_IMAGE.fullmatch(basename)
                if match is not None:
                    images.setdefault(self._raw_key(match), {})[item] = basename
        poses = {}
        for basename in bundles["poses"][1]:
            match = _RAW_POSE.fullmatch(basename)
            if match is not None:
                poses[self._raw_key(match)] = basename
        intrinsics = {}
        for basename in bundles["intrinsics"][1]:
            match = _RAW_INTRINSIC.fullmatch(basename)
            if match is not None:
                yaw = match.group("yaw")
                intrinsics[
                    (
                        match.group("pano"),
                        int(match.group("camera")),
                        None if yaw is None else int(yaw),
                    )
                ] = basename
        keys = sorted(set(images) | set(poses))
        if not keys:
            raise FileNotFoundError(
                f"No raw Matterport3D frames were downloaded for {sample_id!r}."
            )
        pose_path = bundles["poses"][0]
        pose_payloads = (
            {}
            if pose_path is None
            else self._read_members(
                pose_path,
                {
                    basename: bundles["poses"][1][basename]
                    for basename in poses.values()
                },
            )
        )
        intrinsic_path = bundles["intrinsics"][0]
        intrinsic_payloads = (
            {}
            if intrinsic_path is None
            else self._read_members(
                intrinsic_path,
                {
                    basename: bundles["intrinsics"][1][basename]
                    for basename in intrinsics.values()
                },
            )
        )
        records = []
        for key in keys:
            pose = None
            if key in poses:
                basename = poses[key]
                pose = _numbers(
                    pose_payloads[basename],
                    expected=16,
                    source=f"pose {basename}",
                ).reshape(4, 4)
            intrinsic_key = key if key in intrinsics else (key[0], key[1], None)
            matrix = distortion = image_size = None
            if intrinsic_key in intrinsics:
                basename = intrinsics[intrinsic_key]
                values = _numbers(
                    intrinsic_payloads[basename],
                    expected=11,
                    source=f"intrinsics {basename}",
                )
                width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3 = values
                matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                distortion = np.asarray([k1, k2, p1, p2, k3])
                image_size = (int(width), int(height))
            value = images.get(key, {})
            records.append(
                {
                    "key": value.get("rgb")
                    or value.get("depth")
                    or "/".join(map(str, key)),
                    "rgb": value.get("rgb"),
                    "depth": value.get("depth"),
                    "intrinsics": matrix,
                    "camera_to_world": pose,
                    "distortion": distortion,
                    "image_size": image_size,
                }
            )
        return tuple(records)

    def get_frame_keys(self, sample_id, *, source="undistorted") -> tuple[str, ...]:
        """Return deterministic view keys corresponding to integer frame indices."""

        records = self._frame_records(str(sample_id), source)
        return tuple(str(record["key"]) for record in records)

    def get_frame_sources(self, sample_id) -> tuple[str, ...]:
        """Return downloaded frame sources with camera and RGB information."""

        sample_id = str(sample_id)
        result = []
        for source in ("raw", "undistorted"):
            try:
                records = self._frame_records(sample_id, source)
            except FileNotFoundError:
                continue
            rgb_path, _ = self._asset_index(sample_id, source, "rgb")
            if rgb_path is not None and any(
                self._camera_valid(record) for record in records
            ):
                result.append(source)
        return tuple(result)

    def _frame_records(self, sample_id: str, source: str) -> tuple[dict, ...]:
        if source == "undistorted":
            return self._undistorted_records(sample_id)
        if source == "raw":
            return self._raw_records(sample_id)
        raise ValueError("Matterport3D frame source must be 'raw' or 'undistorted'.")

    def _asset_index(self, sample_id: str, source: str, item: str):
        template = self.frame_dirs[source][item]
        return self._index(template, sample_id, optional=True)

    def _read_image(self, bundle, basename: str | None, *, item: str) -> np.ndarray:
        return self._read_images(bundle, [basename], item=item)[0]

    def _read_images(self, bundle, basenames, *, item: str) -> list[np.ndarray]:
        path, index = bundle
        missing = [
            basename
            for basename in basenames
            if basename is None or basename not in index
        ]
        if path is None or missing:
            raise FileNotFoundError(
                f"Matterport3D {item} frames were not downloaded: {missing}"
            )
        payloads = self._read_members(
            path, {basename: index[basename] for basename in basenames}
        )
        return [
            read_image(payloads[basename], rgb=item == "rgb", name=basename)
            for basename in basenames
        ]

    @staticmethod
    def _camera_valid(record: dict) -> bool:
        intrinsic = record["intrinsics"]
        pose = record["camera_to_world"]
        if intrinsic is None or pose is None:
            return False
        return bool(
            np.isfinite(intrinsic).all()
            and intrinsic.shape == (3, 3)
            and valid_camera_mask(np.asarray(pose)[None])[0]
        )

    def get_frames(
        self,
        sample_id,
        *,
        source="undistorted",
        indices=None,
        step=1,
        items=("rgb", "rgb_intrinsics", "world_to_camera"),
        output_size=None,
        center_crop=False,
        rotate_to_up=True,
    ):
        """Return raw or undistorted official RGB-D views.

        Frame indices are deterministic ordinals; ``frame_keys`` retains the
        official image name. Raw distortion is returned without being discarded.
        """

        del rotate_to_up  # Matterport images contain no display-rotation metadata.
        if source not in {"raw", "undistorted"}:
            raise ValueError(
                "Matterport3D frame source must be 'raw' or 'undistorted'."
            )
        supported = {
            "rgb",
            "depth",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        }
        items = validate_frame_items(items, supported)
        records = self._frame_records(str(sample_id), source)
        camera_requested = bool(
            {"rgb_intrinsics", "depth_intrinsics", "world_to_camera"} & set(items)
        )
        bundles = {
            item: self._asset_index(str(sample_id), source, item)
            for item in ("rgb", "depth")
            if item in items
        }
        for item, (path, _) in bundles.items():
            if path is None:
                template = self.frame_dirs[source][item]
                archive = self._archive_path(template, str(sample_id))
                if archive.is_file():
                    raise FileNotFoundError(
                        f"Matterport3D {item} bundle is not extracted for "
                        f"{sample_id!r}. Extract {archive} to "
                        f"{self._path(template, str(sample_id))}."
                    )
                raise FileNotFoundError(
                    f"Matterport3D {item} directory was not found for "
                    f"{sample_id!r}: {self._path(template, str(sample_id))}"
                )
        available = []
        for index, record in enumerate(records):
            if camera_requested and not self._camera_valid(record):
                continue
            if indices is None and any(
                record[item] is None
                or bundles[item][0] is None
                or record[item] not in bundles[item][1]
                for item in bundles
            ):
                continue
            available.append(index)
        if records and not available:
            reason = (
                "valid camera parameters" if camera_requested else "requested assets"
            )
            raise DataFormatError(
                f"Matterport3D has no frames with {reason} for {sample_id!r}."
            )
        selected, _ = select_frame_indices(available, indices=indices, step=step)
        chosen = [records[int(index)] for index in selected]

        values = {
            "indices": selected,
            "source_frame_count": len(records),
            "frame_keys": tuple(str(record["key"]) for record in chosen),
            "camera_model": "OPENCV" if source == "raw" else "PINHOLE",
        }
        camera_to_world = []
        base_intrinsics = []
        for record in chosen:
            if camera_requested:
                pose = np.asarray(record["camera_to_world"], dtype=np.float64)
                if source == "undistorted":
                    pose = pose @ _CV_FROM_UNDISTORTED
                camera_to_world.append(pose)
                base_intrinsics.append(
                    np.asarray(record["intrinsics"], dtype=np.float64).copy()
                )
        if "world_to_camera" in items:
            values["world_to_camera"] = np.asarray(
                [np.linalg.inv(pose) for pose in camera_to_world], dtype=np.float32
            ).reshape((-1, 4, 4))
        if source == "raw" and camera_requested:
            values["distortion"] = (
                np.asarray(
                    [record["distortion"] for record in chosen], dtype=np.float32
                ).reshape((len(chosen), 5))
                if chosen
                else np.empty((0, 5), dtype=np.float32)
            )

        loaded = {
            item: self._read_images(
                bundles[item], [record[item] for record in chosen], item=item
            )
            for item in bundles
        }

        def image_size(record) -> tuple[int, int]:
            # Undistorted records carry no size; read it from an image header.
            if record["image_size"] is not None:
                return record["image_size"]
            for probe_item in ("rgb", "depth"):
                directory, index = self._asset_index(str(sample_id), source, probe_item)
                name = record[probe_item]
                if directory is not None and name in index:
                    with Image.open(directory / index[name]) as image:
                        return image.size
            raise FileNotFoundError(
                "Matterport3D image data is required to determine camera size."
            )

        def transformed(item: str, resample):
            output, matrices, sizes = [], [], []
            for position, image in enumerate(loaded[item]):
                size = (image.shape[1], image.shape[0])
                transform = build_pixel_transform(
                    size, output_size=output_size, center_crop=center_crop
                )
                output.append(
                    apply_image_transform(image, transform, resample=resample)
                )
                sizes.append(transform.output_size)
                if camera_requested:
                    intrinsic = base_intrinsics[position]
                    matrices.append(transform_intrinsics(intrinsic, transform))
            if output:
                try:
                    array = np.stack(output)
                except ValueError as exc:
                    raise DataFormatError(
                        "Matterport3D frames have mixed sizes; pass output_size."
                    ) from exc
            else:
                width, height = (
                    (0, 0) if output_size is None else tuple(map(int, output_size))
                )
                trailing = (3,) if item == "rgb" else ()
                dtype = np.uint8 if item == "rgb" else np.float32
                array = np.empty((0, height, width, *trailing), dtype=dtype)
            return (
                array,
                np.asarray(matrices, dtype=np.float32).reshape((-1, 3, 3)),
                sizes,
            )

        rgb_matrices = depth_matrices = None
        image_sizes = []
        if "rgb" in loaded:
            rgb, rgb_matrices, image_sizes = transformed(
                "rgb", Image.Resampling.LANCZOS
            )
            values["rgb"] = rgb.astype(np.uint8, copy=False)
        if "depth" in loaded:
            depth, depth_matrices, depth_sizes = transformed(
                "depth", Image.Resampling.NEAREST
            )
            values["depth"] = depth.astype(np.float32) / _DEPTH_UNITS_PER_METRE
            if not image_sizes:
                image_sizes = depth_sizes

        def camera_matrices(item: str):
            nonlocal image_sizes
            matrices = rgb_matrices if item == "rgb" else depth_matrices
            if matrices is not None:
                return matrices
            result, sizes = [], []
            for record, intrinsic in zip(chosen, base_intrinsics):
                size = image_size(record)
                transform = build_pixel_transform(
                    size, output_size=output_size, center_crop=center_crop
                )
                result.append(transform_intrinsics(intrinsic, transform))
                sizes.append(transform.output_size)
            if not image_sizes:
                image_sizes = sizes
            return np.asarray(result, dtype=np.float32).reshape((-1, 3, 3))

        if "rgb_intrinsics" in items:
            values["rgb_intrinsics"] = camera_matrices("rgb")
        if "depth_intrinsics" in items:
            values["depth_intrinsics"] = camera_matrices("depth")
        if image_sizes:
            values["image_sizes"] = np.asarray(image_sizes, dtype=np.int32)
        return FrameBatch(**values)
