"""ScanNet++ scans and synchronized iPhone, DSLR, and panoramic captures."""

from __future__ import annotations

import csv
import io
import json
import zlib
from pathlib import Path

import lz4.block
import numpy as np
import open3d as o3d
from PIL import Image

from ...io.image import read_images
from ...io.camera import (
    read_colmap_cameras,
    read_colmap_images,
    read_colmap_sparse_points,
)
from ...io.video import (
    read_grayscale_video,
    read_video_frame_count,
    read_video_frames,
    read_video_frame_size,
)
from ...ops.annotation import (
    parse_sstk_groups,
    parse_sstk_segments,
    read_ply_attribute,
    remap_labels,
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
from ..types import FrameBatch, PointBatch, Segmentation3D

IPHONE_RGB_SIZE = (1920, 1440)
IPHONE_DEPTH_SIZE = (256, 192)


def _read_json(path: Path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataFormatError(f"Invalid JSON in {path}.") from exc


def _intrinsic_from_json(data: dict) -> np.ndarray:
    try:
        return np.array(
            [[data["fl_x"], 0, data["cx"]], [0, data["fl_y"], data["cy"]], [0, 0, 1]],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise DataFormatError(
            "Nerfstudio metadata is missing camera intrinsics."
        ) from exc


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


@register_dataset(
    "scannetppv2",
    aliases=("scannet++", "scannetpp", "scannet++v2"),
    description="ScanNet++ v2 scans with iPhone, DSLR, and panoramic captures.",
)
class ScanNetPP(Dataset):
    """ScanNet++ v2 adapter for scans plus iPhone, DSLR, and panoramic captures."""

    def __init__(
        self,
        root_dir,
        *,
        data_dir="data/{scene_id}",
        label_mapping_file="metadata/semantic_benchmark/map_benchmark.csv",
        semantic_classes_file="metadata/semantic_classes.txt",
        instance_classes_file="metadata/instance_classes.txt",
        scene_types_file="metadata/scene_types.json",
        top100_file="metadata/semantic_benchmark/top100.txt",
        top100_instance_file="metadata/semantic_benchmark/top100_instance.txt",
        nvs_sem_train_split_file="splits/nvs_sem_train.txt",
        nvs_sem_val_split_file="splits/nvs_sem_val.txt",
        nvs_test_split_file="splits/nvs_test.txt",
        sem_test_split_file="splits/sem_test.txt",
        nvs_test_small_split_file="splits/nvs_test_small.txt",
        nvs_test_iphone_split_file="splits/nvs_test_iphone.txt",
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
        self.data_dir = data_dir
        self.metadata_files = {
            "label_mapping": label_mapping_file,
            "semantic_classes": semantic_classes_file,
            "instance_classes": instance_classes_file,
            "scene_types": scene_types_file,
            "top100_classes": top100_file,
            "top100_instance_classes": top100_instance_file,
        }
        self.split_files = {
            "nvs_sem_train": nvs_sem_train_split_file,
            "nvs_sem_val": nvs_sem_val_split_file,
            "nvs_test": nvs_test_split_file,
            "sem_test": sem_test_split_file,
            "nvs_test_small": nvs_test_small_split_file,
            "nvs_test_iphone": nvs_test_iphone_split_file,
        }

    def _scene_dir(self, sample_id: str) -> Path:
        return self.root_dir / self.data_dir.format(scene_id=sample_id)

    def _load_splits(self):
        result = {}
        for split, relative_path in self.split_files.items():
            path = self.root_dir / relative_path
            if path.is_file():
                result[split] = path.read_text().splitlines()
        return result

    def _load_metadata(self):
        result = {}
        mapping_path = self.root_dir / self.metadata_files["label_mapping"]
        if mapping_path.is_file():
            with open(mapping_path) as handle:
                mapping = {}
                for row in csv.DictReader(handle):
                    values = {
                        key: value
                        for key, value in row.items()
                        if key != "class" and value
                    }
                    if values:
                        mapping[row["class"]] = values
                result["label_mapping"] = mapping
        for key in (
            "semantic_classes",
            "instance_classes",
            "top100_classes",
            "top100_instance_classes",
        ):
            path = self.root_dir / self.metadata_files[key]
            if path.is_file():
                result[key] = path.read_text().splitlines()
        path = self.root_dir / self.metadata_files["scene_types"]
        if path.is_file():
            result["scene_types"] = _read_json(path)
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
            message="The ScanNet++ root does not exist.",
            expected="<root>/data/<scene_id>/{scans,iphone,dslr,panocam}/...",
            hint="Pass the root produced by the official ScanNet++ download tool.",
        ):
            return builder.finish()
        prefix = self.data_dir.split("{scene_id}", 1)[0].rstrip("/")
        scene_root = self.root_dir / prefix if prefix else self.root_dir
        if not builder.require_directory(
            scene_root,
            code="missing-data-directory",
            message="The configured ScanNet++ scene directory is missing.",
            expected=f"{scene_root}/<scene_id>/...",
            hint="Download and extract at least one ScanNet++ asset group per scene.",
        ):
            archives = archive_candidates(self.root_dir)
            if archives:
                builder.info(
                    "archives-need-extraction",
                    "Downloaded archives were found, but ScanNet++ assets must be extracted.",
                    hint="Use the official downloader/extractor to create data/<scene_id>.",
                )
            return builder.finish()

        recognized = ("scans", "iphone", "dslr", "panocam")
        all_dirs = [path for path in scene_root.iterdir() if path.is_dir()]
        present = {
            path.name
            for path in all_dirs
            if any((path / group).exists() for group in recognized)
        }
        expected = None if sample_ids is not None else load_expected_ids(self, builder)
        selected = check_scene_coverage(
            builder,
            present_ids=present,
            expected_ids=expected or None,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<root>/data/<scene_id>/{scans,iphone,dslr,panocam}/...",
        )
        if not present:
            builder.error(
                "no-scenes",
                "No directory containing a recognized ScanNet++ asset group was found.",
                path=scene_root,
                expected="A scene directory containing scans, iphone, dslr, or panocam.",
                hint="Do not point root_dir at one individual scene or an unextracted archive.",
            )
            return builder.finish()

        malformed = [path for path in all_dirs if path.name not in present]
        if malformed:
            builder.warning(
                "unrecognized-scene-directories",
                f"{len(malformed)} directories contain no recognized asset group. "
                f"Examples: {', '.join(path.name for path in malformed[:5])}",
                expected="Each scene contains at least one of scans/iphone/dslr/panocam.",
            )

        for sid in selected:
            base = self._scene_dir(sid)
            groups = [name for name in recognized if (base / name).exists()]
            builder.stats[f"{sid}_sources"] = ", ".join(groups)
            scans = base / "scans"
            if scans.is_dir() and not (scans / "mesh_aligned_0.05.ply").is_file():
                builder.error(
                    "incomplete-scans-assets",
                    "The scans group is present but its aligned mesh is missing.",
                    path=scans / "mesh_aligned_0.05.ply",
                    expected="scans/mesh_aligned_0.05.ply",
                    hint="Re-download or re-extract the scans asset group for this scene.",
                )
            iphone = base / "iphone"
            if iphone.is_dir():
                missing = [
                    path
                    for path in (iphone / "rgb.mkv", iphone / "pose_intrinsic_imu.json")
                    if not path.is_file()
                ]
                builder.missing_paths(
                    missing,
                    code="incomplete-iphone-assets",
                    label="iPhone stream files",
                    expected="iphone/rgb.mkv and iphone/pose_intrinsic_imu.json",
                    hint="Download the complete iPhone asset group.",
                )
            dslr = base / "dslr"
            if dslr.is_dir() and not any(
                (dslr / folder).is_dir()
                for folder in (
                    "resized_images",
                    "original_images",
                    "resized_undistorted_images",
                )
            ):
                builder.error(
                    "incomplete-dslr-assets",
                    "The DSLR group has no supported image directory.",
                    path=dslr,
                    expected="dslr/resized_images, original_images, or resized_undistorted_images",
                    hint="Download/extract a complete DSLR image asset group.",
                )
            panocam = base / "panocam"
            if panocam.is_dir() and not any(
                (panocam / folder).is_dir() for folder in ("images", "resized_images")
            ):
                builder.error(
                    "incomplete-panocam-assets",
                    "The panoramic group has no supported image directory.",
                    path=panocam,
                    expected="panocam/images or panocam/resized_images",
                    hint="Download/extract the panoramic image asset group.",
                )
        return builder.finish()

    # Scan assets -------------------------------------------------------------
    def get_mesh(self, sample_id, *, mesh_type=None):
        names = {
            "raw": "mesh_aligned_0.05.ply",
            "semantic": "mesh_aligned_0.05_semantic.ply",
        }
        mesh_type = mesh_type or "raw"
        if mesh_type not in names:
            raise ValueError(f"Unknown ScanNet++ mesh type: {mesh_type!r}")
        return load_triangle_mesh(
            self._scene_dir(sample_id) / "scans" / names[mesh_type]
        )

    def get_point_cloud(self, sample_id):
        """Return the aligned laser-scanner point cloud as an Open3D cloud."""

        path = self._scene_dir(sample_id) / "scans/pc_aligned.ply"
        if not path.is_file():
            raise FileNotFoundError(f"ScanNet++ point cloud was not downloaded: {path}")
        point_cloud = o3d.io.read_point_cloud(str(path))
        if not point_cloud.has_points():
            raise DataFormatError(f"File contains no point cloud points: {path}")
        return point_cloud

    def get_points(self, sample_id) -> PointBatch:
        """Return the aligned laser scan as the standard NumPy point batch."""

        cloud = self.get_point_cloud(sample_id)
        colors = np.asarray(cloud.colors)
        normals = np.asarray(cloud.normals)
        return PointBatch(
            xyz=np.asarray(cloud.points),
            rgb=(
                np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
                if len(colors)
                else None
            ),
            normals=normals if len(normals) else None,
            invalid_id=self.invalid_obj_id,
        )

    def get_scanner_poses(self, sample_id) -> np.ndarray:
        """Return the ``(N, 4, 4)`` laser-scanner poses of the scene."""

        path = self._scene_dir(sample_id) / "scans/scanner_poses.json"
        poses = np.asarray(_read_json(path), dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise DataFormatError(f"Scanner poses must have shape (N, 4, 4): {path}")
        return poses

    def get_anonymization_indices(self, sample_id, *, asset="mesh") -> np.ndarray:
        """Return indices of anonymized vertices (``mesh``) or points (``point_cloud``)."""

        names = {
            "mesh": "mesh_aligned_0.05_mask.txt",
            "point_cloud": "pc_aligned_mask.txt",
        }
        if asset not in names:
            raise ValueError("asset must be 'mesh' or 'point_cloud'.")
        path = self._scene_dir(sample_id) / "scans" / names[asset]
        if not path.is_file():
            raise FileNotFoundError(
                f"ScanNet++ anonymization mask was not downloaded: {path}"
            )
        text = path.read_text().strip()
        indices = (
            np.fromstring(text, sep=" ", dtype=np.int64)
            if text
            else np.empty(0, dtype=np.int64)
        )
        if np.any(indices < 0):
            raise DataFormatError(f"Negative anonymization index in {path}.")
        return indices

    def _load_annotation_json(self, sample_id):
        path = self._scene_dir(sample_id) / "scans"
        return _read_json(path / "segments.json"), _read_json(
            path / "segments_anno.json"
        )

    def _mask_segmentation(self, sample_id, labels, invalid_id):
        path = self._scene_dir(sample_id) / "scans/mesh_aligned_0.05_mask.txt"
        if path.is_file():
            masked = self.get_anonymization_indices(sample_id)
            if len(masked) and int(masked.max()) >= len(labels):
                raise DataFormatError(
                    "Mesh anonymization mask references an unknown vertex."
                )
            labels[masked] = invalid_id

    def _instance_segmentation(self, sample_id, *, multilabel):
        segments, annotations = self._load_annotation_json(sample_id)
        vertex_segments = parse_sstk_segments(segments)
        groups = parse_sstk_groups(annotations)
        if not len(vertex_segments):
            shape = (0, 0) if multilabel else (0,)
            return Segmentation3D(
                np.empty(shape, np.int32), {}, "vertex", self.invalid_obj_id
            )
        counts = np.bincount(vertex_segments)
        candidates, names = {}, {}
        for object_id, value in groups.items():
            name = str(value.get("label", ""))
            if name.lower() == "remove":
                continue
            segment_ids = np.asarray(value.get("segments", []), dtype=np.int64)
            if np.any(segment_ids < 0) or np.any(segment_ids >= len(counts)):
                raise DataFormatError(
                    f"Object {object_id} references an unknown segment."
                )
            size = int(counts[segment_ids].sum())
            for segment_id in segment_ids:
                candidates.setdefault(int(segment_id), []).append((size, object_id))
            names[object_id] = name
        width = max((len(values) for values in candidates.values()), default=0)
        segment_labels = np.full(
            (len(counts), width if multilabel else 1),
            self.invalid_obj_id,
            dtype=np.int32,
        )
        for segment_id, values in candidates.items():
            object_ids = [object_id for _, object_id in sorted(values)]
            count = len(object_ids) if multilabel else min(1, len(object_ids))
            segment_labels[segment_id, :count] = object_ids[:count]
        labels = segment_labels[vertex_segments]
        if not multilabel:
            labels = labels[:, 0]
        self._mask_segmentation(sample_id, labels, self.invalid_obj_id)
        return Segmentation3D(labels, names, "vertex", self.invalid_obj_id)

    def get_segmentation(
        self, sample_id, *, segmentation_type="instance", multilabel=False
    ):
        """Return instance or semantic labels, optionally preserving overlaps."""

        if segmentation_type == "instance":
            return self._instance_segmentation(sample_id, multilabel=multilabel)
        if segmentation_type != "semantic":
            raise ValueError("segmentation_type must be 'instance' or 'semantic'.")
        classes = self.metadata.get("semantic_classes", [])
        names = {index: name for index, name in enumerate(classes)}
        if multilabel:
            instance = self._instance_segmentation(sample_id, multilabel=True)
            class_to_id = {name: index for index, name in names.items()}
            object_to_class = {
                object_id: class_to_id.get(name, -100)
                for object_id, name in instance.id_to_name.items()
            }
            labels = remap_labels(instance.labels, object_to_class, -100)
        else:
            path = self._scene_dir(sample_id) / "scans/mesh_aligned_0.05_semantic.ply"
            labels = read_ply_attribute(path, "label").astype(np.int32, copy=False)
            self._mask_segmentation(sample_id, labels, -100)
        return Segmentation3D(labels, names, "vertex", -100)

    def get_boxes(self, sample_id, *, box_type="mobb_gravity"):
        vertices = np.asarray(self.get_mesh(sample_id).vertices)
        keep = np.ones(len(vertices), dtype=bool)
        mask_path = self._scene_dir(sample_id) / "scans/mesh_aligned_0.05_mask.txt"
        if mask_path.is_file():
            keep[self.get_anonymization_indices(sample_id)] = False
        segments, annotations = self._load_annotation_json(sample_id)
        vertex_segments = parse_sstk_segments(segments)
        groups = parse_sstk_groups(annotations)
        members = sstk_object_indices(vertex_segments, groups)
        boxes, names = {}, {}
        for object_id, value in groups.items():
            if str(value["label"]).lower() == "remove":
                continue
            indices = members[object_id]
            points = vertices[indices[keep[indices]]]
            if not len(points):
                raise DataFormatError(
                    f"ScanNet++ object {object_id} has no unmasked vertices."
                )
            boxes[object_id] = construct_box(points, box_type=box_type)
            names[object_id] = value["label"]
        return boxes, names

    # Source metadata ---------------------------------------------------------
    def get_iphone_exif(self, sample_id) -> dict:
        """Return the per-frame iPhone EXIF metadata."""

        return _read_json(self._scene_dir(sample_id) / "iphone/exif.json")

    def get_dslr_train_test_lists(self, sample_id) -> dict:
        """Return the release's filename-based DSLR benchmark split."""

        return _read_json(self._scene_dir(sample_id) / "dslr/train_test_lists.json")

    def get_colmap_cameras(self, sample_id, *, source="dslr"):
        """Return parsed COLMAP cameras while retaining their distortion model."""

        if source not in {"dslr", "iphone_colmap"}:
            raise ValueError("source must be 'dslr' or 'iphone_colmap'.")
        folder = "dslr" if source == "dslr" else "iphone"
        return read_colmap_cameras(
            self._scene_dir(sample_id) / folder / "colmap/cameras.txt"
        )

    def get_colmap_images(self, sample_id, *, source="dslr"):
        """Return parsed COLMAP registered-image poses in aligned coordinates."""

        if source not in {"dslr", "iphone_colmap"}:
            raise ValueError("source must be 'dslr' or 'iphone_colmap'.")
        folder = "dslr" if source == "dslr" else "iphone"
        return read_colmap_images(
            self._scene_dir(sample_id) / folder / "colmap/images.txt"
        )

    def get_nerfstudio_transforms(self, sample_id, *, source="dslr") -> dict:
        """Return the released Nerfstudio ``transforms.json`` for one capture source."""

        names = {
            "dslr": "dslr/nerfstudio/transforms.json",
            "dslr_undistorted": "dslr/nerfstudio/transforms_undistorted.json",
            "iphone_colmap": "iphone/nerfstudio/transforms.json",
        }
        if source not in names:
            raise ValueError(
                f"No Nerfstudio metadata is defined for source {source!r}."
            )
        return _read_json(self._scene_dir(sample_id) / names[source])

    def get_frame_sources(self, sample_id) -> tuple[str, ...]:
        """Return the frame sources downloaded for one scene."""

        base = self._scene_dir(sample_id)
        checks = {
            "iphone": base / "iphone/rgb.mkv",
            "iphone_colmap": base / "iphone/colmap/cameras.txt",
            "dslr": base / "dslr/resized_images",
            "dslr_undistorted": base / "dslr/resized_undistorted_images",
            "dslr_original": base / "dslr/original_images",
            "panocam": base / "panocam/images",
            "panocam_resized": base / "panocam/resized_images",
        }
        return tuple(name for name, path in checks.items() if path.exists())

    def get_colmap_sparse_point_cloud(self, sample_id, *, source="dslr"):
        """Return the COLMAP sparse reconstruction as a colored Open3D cloud."""

        if source not in {"dslr", "iphone_colmap"}:
            raise ValueError("source must be 'dslr' or 'iphone_colmap'.")
        folder = "dslr" if source == "dslr" else "iphone"
        path = self._scene_dir(sample_id) / folder / "colmap/points3D.txt"
        point_ids, points, colors = read_colmap_sparse_points(path)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        return cloud, point_ids

    # iPhone raw stream -------------------------------------------------------
    def _read_iphone_camera(self, sample_id):
        path = self._scene_dir(sample_id) / "iphone/pose_intrinsic_imu.json"
        data = _read_json(path)
        try:
            records = sorted(
                (
                    (int(key.removeprefix("frame_")), key, value)
                    for key, value in data.items()
                ),
                key=lambda value: value[0],
            )
            indices = np.asarray([value[0] for value in records], dtype=np.int64)
            poses = np.asarray(
                [value[2]["aligned_pose"] for value in records], dtype=np.float32
            )
            intrinsics = np.asarray(
                [value[2]["intrinsic"] for value in records], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataFormatError(f"Invalid iPhone camera metadata in {path}.") from exc
        if poses.shape[1:] != (4, 4) or intrinsics.shape[1:] != (3, 3):
            raise DataFormatError(f"Invalid ScanNet++ camera matrices in {path}.")
        return records, indices, np.linalg.inv(poses), intrinsics

    def _read_depth(self, sample_id, indices):
        path = self._scene_dir(sample_id) / "iphone/depth.bin"
        if len(indices) == 0:
            return np.empty(
                (0, IPHONE_DEPTH_SIZE[1], IPHONE_DEPTH_SIZE[0]), dtype=np.float32
            )

        # Current releases store length-prefixed frames. Validate only their
        # headers so requesting two frames never reads a ~450 MB stream.
        file_size = path.stat().st_size
        with open(path, "rb") as stream:
            position, block_count = 0, 0
            while position < file_size:
                size_bytes = stream.read(4)
                if len(size_bytes) != 4:
                    break
                block_size = int.from_bytes(size_bytes, "little")
                position += 4 + block_size
                if block_size <= 0 or position > file_size:
                    break
                stream.seek(block_size, io.SEEK_CUR)
                block_count += 1
        if block_count and position == file_size:
            return self._read_block_depth(path, indices)

        data = path.read_bytes()
        try:
            decoded = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            all_depth = np.frombuffer(decoded, dtype=np.float32).reshape(
                -1, IPHONE_DEPTH_SIZE[1], IPHONE_DEPTH_SIZE[0]
            )
        except (zlib.error, ValueError) as exc:
            raise DataFormatError(
                f"Unknown ScanNet++ depth encoding in {path}."
            ) from exc
        if np.min(indices) < 0 or np.max(indices) >= len(all_depth):
            raise ValueError(
                "Requested ScanNet++ depth frame is outside the source range."
            )
        return all_depth[indices].astype(np.float32, copy=False)

    @staticmethod
    def _decode_depth_block(block, frame_index):
        try:
            decoded = lz4.block.decompress(
                block, uncompressed_size=IPHONE_DEPTH_SIZE[0] * IPHONE_DEPTH_SIZE[1] * 2
            )
            return (
                np.frombuffer(decoded, dtype=np.uint16)
                .reshape(IPHONE_DEPTH_SIZE[1], IPHONE_DEPTH_SIZE[0])
                .astype(np.float32)
                / 1000.0
            )
        except (lz4.block.LZ4BlockError, RuntimeError, ValueError):
            # Some v2 scenes store each frame as raw-DEFLATE float32 metres.
            try:
                decoded = zlib.decompress(block, wbits=-zlib.MAX_WBITS)
                return np.frombuffer(decoded, dtype=np.float32).reshape(
                    IPHONE_DEPTH_SIZE[1], IPHONE_DEPTH_SIZE[0]
                )
            except (zlib.error, ValueError) as exc:
                raise DataFormatError(
                    f"Unknown ScanNet++ depth compression at frame {frame_index}."
                ) from exc

    def _read_block_depth(self, path, indices):
        requested = {int(index): position for position, index in enumerate(indices)}
        frames = [None] * len(indices)
        frame_index = 0
        last_requested = max(requested)
        with open(path, "rb") as stream:
            while frame_index <= last_requested and (size_bytes := stream.read(4)):
                if len(size_bytes) != 4:
                    raise DataFormatError(
                        f"Truncated ScanNet++ depth block header in {path}."
                    )
                block_size = int.from_bytes(size_bytes, byteorder="little")
                if frame_index in requested:
                    block = stream.read(block_size)
                    if len(block) != block_size:
                        raise DataFormatError(
                            f"Truncated ScanNet++ depth block in {path}."
                        )
                    frames[requested[frame_index]] = self._decode_depth_block(
                        block, frame_index
                    )
                else:
                    stream.seek(block_size, io.SEEK_CUR)
                frame_index += 1
        if any(frame is None for frame in frames):
            raise ValueError("Requested ScanNet++ depth frames were not found.")
        return np.stack(frames).astype(np.float32, copy=False)

    @staticmethod
    def _select(available, all_keys, *, indices, step, frame_keys):
        if frame_keys is None:
            return select_frame_indices(available, indices=indices, step=step)
        if indices is not None or step != 1:
            raise ValueError("frame_keys cannot be combined with indices or step != 1.")
        requested = tuple(str(key) for key in frame_keys)
        by_key = {str(key): position for position, key in enumerate(all_keys)}
        missing = [key for key in requested if key not in by_key]
        if missing:
            raise ValueError(f"Unknown frame keys: {missing}")
        positions = np.asarray([by_key[key] for key in requested], dtype=np.int64)
        return np.asarray(available)[positions], positions

    def _get_iphone_frames(
        self, sample_id, *, indices, step, items, output_size, center_crop, frame_keys
    ):
        items = validate_frame_items(
            items,
            {
                "timestamps",
                "rgb",
                "depth",
                "rgb_intrinsics",
                "depth_intrinsics",
                "world_to_camera",
                "frame_mask",
                "imu",
            },
        )
        records, available, world_to_camera, intrinsics = self._read_iphone_camera(
            sample_id
        )
        valid = valid_camera_mask(world_to_camera)
        records = [record for record, keep in zip(records, valid) if keep]
        available, world_to_camera, intrinsics = (
            available[valid],
            world_to_camera[valid],
            intrinsics[valid],
        )
        all_keys = tuple(record[1] for record in records)
        selected, positions = self._select(
            available, all_keys, indices=indices, step=step, frame_keys=frame_keys
        )
        rgb_path = self._scene_dir(sample_id) / "iphone/rgb.mkv"
        source_count = read_video_frame_count(rgb_path)
        transform = build_pixel_transform(
            IPHONE_RGB_SIZE, output_size=output_size, center_crop=center_crop
        )
        selected_records = [records[position][2] for position in positions]
        values = {
            "indices": selected,
            "source_frame_count": source_count,
            "frame_keys": tuple(all_keys[position] for position in positions),
            "camera_model": "PINHOLE",
            "distortion": np.empty((len(selected), 0), dtype=np.float32),
            "image_sizes": np.repeat(
                np.asarray(transform.output_size)[None], len(selected), axis=0
            ),
        }
        if "timestamps" in items:
            try:
                values["timestamps"] = np.asarray(
                    [record["timestamp"] for record in selected_records]
                )
            except KeyError as exc:
                raise DataFormatError(
                    "Requested iPhone timestamps are missing."
                ) from exc
        if "imu" in items:
            try:
                names = tuple(selected_records[0]["imu"]) if selected_records else ()
                values["imu"] = {
                    name: np.asarray(
                        [record["imu"][name] for record in selected_records], np.float32
                    )
                    for name in names
                }
            except KeyError as exc:
                raise DataFormatError("Requested iPhone IMU data is missing.") from exc
        if "world_to_camera" in items:
            values["world_to_camera"] = world_to_camera[positions].astype(np.float32)
        if "rgb_intrinsics" in items:
            values["rgb_intrinsics"] = transform_intrinsics(
                intrinsics[positions], transform
            ).astype(np.float32)
        if "rgb" in items:
            rgb = read_video_frames(rgb_path, frame_indices=selected)
            if len(rgb) and (rgb.shape[2], rgb.shape[1]) != IPHONE_RGB_SIZE:
                raise DataFormatError(
                    "ScanNet++ RGB video has an unexpected resolution."
                )
            values["rgb"] = apply_transform_batch(rgb, transform)
        if "depth" in items:
            depth_transform = build_pixel_transform(
                IPHONE_DEPTH_SIZE, output_size=output_size, center_crop=center_crop
            )
            values["depth"] = apply_transform_batch(
                self._read_depth(sample_id, selected),
                depth_transform,
                resample=Image.Resampling.NEAREST,
            ).astype(np.float32, copy=False)
        if "depth_intrinsics" in items:
            depth_transform = build_pixel_transform(
                IPHONE_RGB_SIZE,
                output_size=output_size or IPHONE_DEPTH_SIZE,
                center_crop=center_crop,
            )
            values["depth_intrinsics"] = transform_intrinsics(
                intrinsics[positions], depth_transform
            ).astype(np.float32)
        if "frame_mask" in items:
            path = self._scene_dir(sample_id) / "iphone/rgb_mask.mkv"
            masks = read_grayscale_video(path, frame_indices=selected)
            mask_transform = build_pixel_transform(
                read_video_frame_size(path),
                output_size=output_size,
                center_crop=center_crop,
            )
            values["frame_mask"] = apply_transform_batch(
                masks, mask_transform, resample=Image.Resampling.NEAREST
            ).astype(bool)
        return FrameBatch(**values)

    # Directory image captures -----------------------------------------------
    def _dslr_records(self, sample_id, source, view_split):
        name = (
            "transforms_undistorted.json"
            if source == "dslr_undistorted"
            else "transforms.json"
        )
        metadata = _read_json(self._scene_dir(sample_id) / "dslr/nerfstudio" / name)
        if view_split == "train":
            records = metadata.get("frames", [])
        elif view_split == "test":
            records = metadata.get("test_frames", [])
        elif view_split == "all":
            records = metadata.get("frames", []) + metadata.get("test_frames", [])
        else:
            raise ValueError("view_split must be 'all', 'train', or 'test'.")
        return metadata, records

    def _get_dslr_frames(
        self,
        sample_id,
        source,
        *,
        indices,
        step,
        items,
        output_size,
        center_crop,
        frame_keys,
        view_split,
        include_bad,
    ):
        items = validate_frame_items(
            items, {"rgb", "rgb_intrinsics", "world_to_camera", "frame_mask", "is_bad"}
        )
        base = self._scene_dir(sample_id)
        metadata, records = self._dslr_records(sample_id, source, view_split)
        if not include_bad:
            records = [record for record in records if not record.get("is_bad", False)]
        keys = tuple(record["file_path"] for record in records)
        available = np.arange(len(records), dtype=np.int64)
        selected, positions = self._select(
            available, keys, indices=indices, step=step, frame_keys=frame_keys
        )
        selected_records = [records[position] for position in positions]

        colmap_dir = base / "dslr/colmap"
        cameras = read_colmap_cameras(colmap_dir / "cameras.txt")
        images = read_colmap_images(colmap_dir / "images.txt")
        try:
            colmap_images = [images[record["file_path"]] for record in selected_records]
            selected_cameras = [cameras[image.camera_id] for image in colmap_images]
        except KeyError as exc:
            raise DataFormatError(
                f"DSLR view is missing from COLMAP: {exc.args[0]}"
            ) from exc

        directories = {
            "dslr": ("resized_images", "resized_anon_masks"),
            "dslr_undistorted": (
                "resized_undistorted_images",
                "resized_undistorted_masks",
            ),
            "dslr_original": ("original_images", "original_anon_masks"),
        }
        image_name, mask_name = directories[source]
        image_dir, mask_dir = base / "dslr" / image_name, base / "dslr" / mask_name
        paths = [image_dir / record["file_path"] for record in selected_records]
        source_size = (
            _image_size(paths[0]) if paths else (int(metadata["w"]), int(metadata["h"]))
        )

        if source == "dslr_undistorted":
            intrinsic = _intrinsic_from_json(metadata)
            model, distortion = "PINHOLE", np.empty(0, dtype=np.float32)
            calibrated_size = (int(metadata["w"]), int(metadata["h"]))
        else:
            camera = (
                selected_cameras[0]
                if selected_cameras
                else next(iter(cameras.values()))
            )
            if any(value.model != camera.model for value in selected_cameras):
                raise DataFormatError("One DSLR batch contains multiple camera models.")
            intrinsic, model = camera.intrinsic.astype(np.float32), camera.model
            distortion = camera.distortion.astype(np.float32)
            calibrated_size = (camera.width, camera.height)
        scale = np.array(
            [
                [source_size[0] / calibrated_size[0], 0, 0],
                [0, source_size[1] / calibrated_size[1], 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsic = scale @ intrinsic
        transform = build_pixel_transform(
            source_size, output_size=output_size, center_crop=center_crop
        )
        values = {
            "indices": selected,
            "source_frame_count": len(records),
            "frame_keys": tuple(keys[position] for position in positions),
            "camera_model": model,
            "distortion": np.repeat(distortion[None], len(selected), axis=0),
            "image_sizes": np.repeat(
                np.asarray(transform.output_size)[None], len(selected), axis=0
            ),
        }
        if "rgb" in items:
            values["rgb"] = apply_transform_batch(
                read_images(paths, rgb=True), transform
            )
        if "rgb_intrinsics" in items:
            intrinsics = np.repeat(intrinsic[None], len(selected), axis=0)
            values["rgb_intrinsics"] = transform_intrinsics(
                intrinsics, transform
            ).astype(np.float32)
        if "world_to_camera" in items:
            values["world_to_camera"] = np.asarray(
                [image.world_to_camera for image in colmap_images], dtype=np.float32
            )
        if "frame_mask" in items:
            mask_paths = [
                mask_dir
                / Path(record.get("mask_path", record["file_path"]))
                .with_suffix(".png")
                .name
                for record in selected_records
            ]
            values["frame_mask"] = apply_transform_batch(
                read_images(mask_paths, rgb=False),
                transform,
                resample=Image.Resampling.NEAREST,
            ).astype(bool)
        if "is_bad" in items:
            values["is_bad"] = np.asarray(
                [record.get("is_bad", False) for record in selected_records], dtype=bool
            )
        return FrameBatch(**values)

    def _get_iphone_colmap_frames(
        self, sample_id, *, indices, step, items, output_size, center_crop, frame_keys
    ):
        items = validate_frame_items(
            items,
            {
                "rgb",
                "depth",
                "rgb_intrinsics",
                "depth_intrinsics",
                "world_to_camera",
                "frame_mask",
                "timestamps",
                "imu",
            },
        )
        base = self._scene_dir(sample_id)
        colmap_dir = base / "iphone/colmap"
        cameras = read_colmap_cameras(colmap_dir / "cameras.txt")
        images = sorted(
            read_colmap_images(colmap_dir / "images.txt").values(),
            key=lambda image: image.image_id,
        )
        keys = tuple(image.name for image in images)
        try:
            available = np.asarray(
                [int(Path(key).stem.removeprefix("frame_")) for key in keys]
            )
        except ValueError as exc:
            raise DataFormatError(
                "iPhone COLMAP names must be frame_NNNNNN images."
            ) from exc
        selected, positions = self._select(
            available, keys, indices=indices, step=step, frame_keys=frame_keys
        )
        selected_images = [images[position] for position in positions]
        selected_cameras = [cameras[image.camera_id] for image in selected_images]
        camera = (
            selected_cameras[0] if selected_cameras else next(iter(cameras.values()))
        )
        if any(value.model != camera.model for value in selected_cameras):
            raise DataFormatError("One iPhone batch contains multiple camera models.")
        transform = build_pixel_transform(
            (camera.width, camera.height),
            output_size=output_size,
            center_crop=center_crop,
        )
        selected_keys = tuple(keys[position] for position in positions)
        values = {
            "indices": selected,
            "source_frame_count": int(available.max()) + 1 if len(available) else 0,
            "frame_keys": selected_keys,
            "camera_model": camera.model,
            "distortion": np.repeat(
                camera.distortion[None], len(selected), axis=0
            ).astype(np.float32),
            "image_sizes": np.repeat(
                np.asarray(transform.output_size)[None], len(selected), axis=0
            ),
        }
        if "rgb" in items:
            paths = [base / "iphone/rgb" / key for key in selected_keys]
            values["rgb"] = apply_transform_batch(
                read_images(paths, rgb=True), transform
            )
        if "rgb_intrinsics" in items:
            intrinsics = np.repeat(camera.intrinsic[None], len(selected), axis=0)
            values["rgb_intrinsics"] = transform_intrinsics(
                intrinsics, transform
            ).astype(np.float32)
        if "world_to_camera" in items:
            values["world_to_camera"] = np.asarray(
                [image.world_to_camera for image in selected_images], dtype=np.float32
            )
        if "depth" in items:
            paths = [
                base / "iphone/depth" / Path(key).with_suffix(".png").name
                for key in selected_keys
            ]
            depth = read_images(paths, rgb=False).astype(np.float32) / 1000.0
            depth_size = (
                (depth.shape[2], depth.shape[1]) if len(depth) else IPHONE_DEPTH_SIZE
            )
            depth_transform = build_pixel_transform(
                depth_size, output_size=output_size, center_crop=center_crop
            )
            values["depth"] = apply_transform_batch(
                depth, depth_transform, resample=Image.Resampling.NEAREST
            )
        if "depth_intrinsics" in items:
            depth_size = IPHONE_DEPTH_SIZE
            depth_path = (
                base / "iphone/depth" / Path(selected_keys[0]).with_suffix(".png").name
                if selected_keys
                else None
            )
            if depth_path is not None and depth_path.is_file():
                depth_size = _image_size(depth_path)
            depth_transform = build_pixel_transform(
                (camera.width, camera.height),
                output_size=output_size or depth_size,
                center_crop=center_crop,
            )
            intrinsics = np.repeat(camera.intrinsic[None], len(selected), axis=0)
            values["depth_intrinsics"] = transform_intrinsics(
                intrinsics, depth_transform
            ).astype(np.float32)
        if "frame_mask" in items:
            paths = [
                base / "iphone/rgb_masks" / Path(key).with_suffix(".png").name
                for key in selected_keys
            ]
            values["frame_mask"] = apply_transform_batch(
                read_images(paths, rgb=False),
                transform,
                resample=Image.Resampling.NEAREST,
            ).astype(bool)
        if "timestamps" in items or "imu" in items:
            raw_records, _, _, _ = self._read_iphone_camera(sample_id)
            by_index = {index: record for index, _, record in raw_records}
            try:
                records = [by_index[int(index)] for index in selected]
            except KeyError as exc:
                raise DataFormatError(
                    f"No iPhone metadata for COLMAP frame {exc.args[0]}."
                ) from exc
            if "timestamps" in items:
                values["timestamps"] = np.asarray(
                    [record["timestamp"] for record in records]
                )
            if "imu" in items:
                names = tuple(records[0]["imu"]) if records else ()
                values["imu"] = {
                    name: np.asarray(
                        [record["imu"][name] for record in records], np.float32
                    )
                    for name in names
                }
        return FrameBatch(**values)

    # Panoramic capture -------------------------------------------------------
    def _get_panocam_frames(
        self,
        sample_id,
        source,
        *,
        indices,
        step,
        items,
        output_size,
        center_crop,
        frame_keys,
    ):
        items = validate_frame_items(
            items,
            {"rgb", "depth", "world_to_camera", "frame_mask", "azimuth", "elevation"},
        )
        base = self._scene_dir(sample_id)
        prefix = "resized_" if source == "panocam_resized" else ""
        image_dir = base / "panocam" / f"{prefix}images"
        paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        if not paths:
            raise FileNotFoundError(f"No ScanNet++ panoramas found in {image_dir}.")
        keys = tuple(path.stem for path in paths)
        available = np.asarray([int(key) for key in keys], dtype=np.int64)
        selected, positions = self._select(
            available, keys, indices=indices, step=step, frame_keys=frame_keys
        )
        selected_paths = [paths[position] for position in positions]
        source_size = _image_size(selected_paths[0] if selected_paths else paths[0])
        transform = build_pixel_transform(
            source_size, output_size=output_size, center_crop=center_crop
        )
        poses = self.get_scanner_poses(sample_id)
        if len(selected) and int(selected.max()) >= len(poses):
            raise DataFormatError("Panocam scan ID is outside scanner_poses.json.")
        selected_keys = tuple(keys[position] for position in positions)
        values = {
            "indices": selected,
            "source_frame_count": len(poses),
            "frame_keys": selected_keys,
            "camera_model": "EQUIRECTANGULAR",
            # Panorama depth is measured along each pixel's viewing ray.
            "depth_mode": "ray_distance" if "depth" in items else None,
            "distortion": np.empty((len(selected), 0), dtype=np.float32),
            "image_sizes": np.repeat(
                np.asarray(transform.output_size)[None], len(selected), axis=0
            ),
        }
        if "rgb" in items:
            values["rgb"] = apply_transform_batch(
                read_images(selected_paths, rgb=True), transform
            )
        folders = {
            "depth": f"{prefix}depth",
            "frame_mask": f"{prefix}anon_mask",
            "azimuth": f"{prefix}azim",
            "elevation": f"{prefix}elev",
        }
        for item in set(items) & folders.keys():
            item_paths = [
                base / "panocam" / folders[item] / f"{key}.png" for key in selected_keys
            ]
            maps = apply_transform_batch(
                read_images(item_paths, rgb=False),
                transform,
                resample=Image.Resampling.NEAREST,
            )
            values[item] = (
                maps.astype(bool)
                if item == "frame_mask"
                else maps.astype(np.float32) / 1000.0
            )
        if "world_to_camera" in items:
            values["world_to_camera"] = np.linalg.inv(poses[selected]).astype(
                np.float32
            )
        return FrameBatch(**values)

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
        source="iphone",
        frame_keys=None,
        view_split="all",
        include_bad=True,
    ):
        """Read frames from one ScanNet++ capture source.

        ScanNet++ is already gravity aligned, so ``rotate_to_up`` is a no-op.
        DSLR ``view_split`` is ``all``, ``train``, or ``test``. String
        ``frame_keys`` are useful for the filename-based DSLR benchmark split.
        """

        del rotate_to_up
        common = dict(
            indices=indices,
            step=step,
            items=items,
            output_size=output_size,
            center_crop=center_crop,
            frame_keys=frame_keys,
        )
        if source == "iphone":
            return self._get_iphone_frames(sample_id, **common)
        if source == "iphone_colmap":
            return self._get_iphone_colmap_frames(sample_id, **common)
        if source in {"dslr", "dslr_undistorted", "dslr_original"}:
            return self._get_dslr_frames(
                sample_id,
                source,
                view_split=view_split,
                include_bad=include_bad,
                **common,
            )
        if source in {"panocam", "panocam_resized"}:
            return self._get_panocam_frames(sample_id, source, **common)
        valid = (
            "iphone",
            "iphone_colmap",
            "dslr",
            "dslr_undistorted",
            "dslr_original",
            "panocam",
            "panocam_resized",
        )
        raise ValueError(
            f"Unknown ScanNet++ frame source {source!r}. Valid sources: {valid}"
        )

    def get_panorama_point_cloud(self, sample_id, scan_id, *, resized=True):
        """Back-project one panoramic RGB-D scan into a colored Open3D point cloud."""

        source = "panocam_resized" if resized else "panocam"
        batch = self.get_frames(
            sample_id,
            source=source,
            indices=[scan_id],
            items=("rgb", "depth", "azimuth", "elevation", "world_to_camera"),
        )
        valid = batch.depth[0] > 0
        distance, azimuth, elevation = (
            batch.depth[0][valid],
            batch.azimuth[0][valid],
            batch.elevation[0][valid],
        )
        points = np.stack(
            [
                distance * np.sin(elevation) * np.cos(azimuth),
                distance * np.sin(elevation) * np.sin(azimuth),
                distance * np.cos(elevation),
            ],
            axis=1,
        )
        camera_to_world = np.linalg.inv(batch.world_to_camera[0])
        points = points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(
            batch.rgb[0][valid].astype(np.float64) / 255.0
        )
        return cloud
