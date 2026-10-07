"""SceneNN adapter."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET

import numpy as np

from ...ops.annotation import read_ply, remap_labels
from ...ops.geometry import load_triangle_mesh
from ..base import DataFormatError, Dataset
from ..registry import register_dataset
from ..types import Segmentation3D


@register_dataset(
    "scenenn",
    description="SceneNN meshes and vertex-domain instance segmentation.",
)
class SceneNN(Dataset):
    def __init__(
        self,
        root_dir,
        *,
        raw_mesh_file="{scene_id}/{scene_id}_color.ply",
        instance_mesh_file="{scene_id}/{scene_id}.ply",
        annotation_file="{scene_id}/{scene_id}.xml",
        room_category_file="https://hkust-vgd.ust.hk/scenenn/main/category.csv",
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
        self.raw_mesh_file = raw_mesh_file
        self.instance_mesh_file = instance_mesh_file
        self.annotation_file = annotation_file
        self.room_category_file = room_category_file

    def _category_rows(self):
        with self.open_file(self.room_category_file, "r") as handle:
            return [row for row in csv.reader(handle) if row]

    def _load_splits(self):
        return {}  # SceneNN has no official splits.

    def _load_sample_ids(self):
        rows = self._category_rows()
        all_row = next(
            (row for row in reversed(rows) if row[0].strip() == "#All"), None
        )
        if all_row is None:
            raise DataFormatError("SceneNN category metadata has no #All row.")
        return [value.strip() for value in all_row[1:] if value.strip()]

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
            message="The SceneNN root does not exist.",
            expected="<root>/<scene_id>/{<scene_id>_color.ply,<scene_id>.ply,<scene_id>.xml}",
            hint="Pass the directory containing extracted SceneNN scene folders.",
        ):
            return builder.finish()
        present = {
            path.name
            for path in self.root_dir.iterdir()
            if path.is_dir() and any(path.glob("*.ply"))
        }
        expected = None if sample_ids is not None else load_expected_ids(self, builder)
        selected = check_scene_coverage(
            builder,
            present_ids=present,
            expected_ids=expected,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<root>/<scene_id>/<scene_id>_color.ply plus instance PLY/XML",
        )
        if not present:
            builder.error(
                "no-scenes",
                "No extracted SceneNN scene directories with PLY files were found.",
                path=self.root_dir,
                expected="One directory per numeric scene ID.",
                hint="Extract SceneNN archives; ONI-only captures cannot serve mesh operations.",
            )
            if archive_candidates(self.root_dir):
                builder.info(
                    "archives-need-extraction",
                    "SceneNN archives must be extracted before use.",
                )
            return builder.finish()
        check_required_scene_paths(
            builder,
            selected,
            {
                "raw_mesh": lambda sid: self.root_dir
                / self.raw_mesh_file.format(scene_id=sid),
                "instance_mesh": lambda sid: self.root_dir
                / self.instance_mesh_file.format(scene_id=sid),
                "annotations": lambda sid: self.root_dir
                / self.annotation_file.format(scene_id=sid),
            },
            expected="<scene_id>_color.ply, <scene_id>.ply, and <scene_id>.xml.",
            hint="Download/extract the processed mesh and annotation files for this scene.",
        )
        return builder.finish()

    def _load_metadata(self):
        categories = {}
        for row in self._category_rows():
            if len(row) > 1 and row[0].strip() != "#All":
                categories[row[0].strip()] = [
                    value.strip() for value in row[1:] if value.strip()
                ]
        return {"room_categories": categories}

    def get_mesh(self, sample_id, *, mesh_type=None):
        mesh_type = mesh_type or "raw"
        instance_path = self.root_dir / self.instance_mesh_file.format(
            scene_id=sample_id
        )
        if mesh_type == "instance":
            return load_triangle_mesh(instance_path)
        if mesh_type != "raw":
            raise ValueError(f"Unknown SceneNN mesh type: {mesh_type!r}")
        mesh = load_triangle_mesh(
            self.root_dir / self.raw_mesh_file.format(scene_id=sample_id)
        )
        instance_mesh = load_triangle_mesh(instance_path)
        if len(mesh.vertices) != len(instance_mesh.vertices):
            raise DataFormatError(
                "SceneNN raw and instance meshes have different vertex counts and cannot be aligned."
            )
        mesh.vertices = instance_mesh.vertices
        return mesh

    def _read_annotations(self, sample_id):
        path = self.root_dir / self.annotation_file.format(scene_id=sample_id)
        root = ET.parse(path).getroot()
        return [
            {
                "color": label.get("color"),
                "name": label.get("text"),
                "id": label.get("id"),
            }
            for label in root.findall("label")
        ]

    def get_segmentation(self, sample_id):
        path = self.root_dir / self.instance_mesh_file.format(scene_id=sample_id)
        ply = read_ply(path)
        vertex = ply["vertex"]
        # Annotations identify objects by vertex colour; pack RGB into one key.
        color_keys = (
            vertex["red"].astype(np.int64) << 16
            | vertex["green"].astype(np.int64) << 8
            | vertex["blue"].astype(np.int64)
        )
        object_by_color = {}
        names = {}
        for fallback_id, annotation in enumerate(self._read_annotations(sample_id)):
            if not annotation["color"] or annotation["name"] is None:
                raise DataFormatError(
                    "SceneNN annotation is missing color or label text."
                )
            object_id = (
                int(annotation["id"]) if annotation["id"] is not None else fallback_id
            )
            color = np.fromstring(annotation["color"], sep=" ", dtype=np.uint8)
            if color.shape != (3,):
                raise DataFormatError(
                    f"Invalid SceneNN annotation color: {annotation['color']!r}"
                )
            red, green, blue = color.astype(np.int64)
            object_by_color[int(red << 16 | green << 8 | blue)] = object_id
            names[object_id] = annotation["name"] or "unknown"
        labels = remap_labels(color_keys, object_by_color, self.invalid_obj_id)
        return Segmentation3D(labels, names, "vertex", self.invalid_obj_id)
