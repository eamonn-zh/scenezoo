"""Annotation parsers shared by scene adapters."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElementParseError

from ..dataset.base import DataFormatError
from ..dataset.types import Segmentation3D


# Declaring fixed-length face lists lets plyfile parse binary meshes with numpy
# instead of one Python call per face (seconds -> milliseconds on large scans).
_TRIANGLE_LISTS = {"face": {"vertex_indices": 3, "vertex_index": 3}}


def read_ply(ply_file_path) -> PlyData:
    """Read a PLY file from disk, using plyfile's fast path for triangle faces."""

    try:
        return PlyData.read(ply_file_path, known_list_len=_TRIANGLE_LISTS)
    except PlyElementParseError:
        # Some faces are not triangles: use the general parser.
        return PlyData.read(ply_file_path)


def read_ply_attribute(
    ply_file_path,
    attribute_name: str,
    *,
    source: str = "vertex",
) -> np.ndarray:
    data = read_ply(ply_file_path)
    try:
        return np.asarray(data[source][attribute_name])
    except (KeyError, ValueError) as exc:
        raise DataFormatError(
            f"PLY attribute {source}.{attribute_name} was not found in {ply_file_path}."
        ) from exc


def group_indices(labels) -> dict[int, np.ndarray]:
    """Map every distinct label to the sorted positions where it occurs."""

    labels = np.asarray(labels)
    groups = pd.Series(labels).groupby(labels, sort=True).indices
    return {int(label): positions for label, positions in groups.items()}


def remap_labels(labels, mapping: dict[int, int], default: int) -> np.ndarray:
    """Replace each label by ``mapping[label]``, or ``default`` when unmapped."""

    values, inverse = np.unique(labels, return_inverse=True)
    lookup = np.array([mapping.get(int(value), default) for value in values], np.int32)
    return lookup[inverse].reshape(np.shape(labels))


def sstk_object_indices(
    vertex_segments: np.ndarray, objects: dict[int, dict]
) -> dict[int, np.ndarray]:
    """Map SSTK object IDs to the sorted vertices of their segments.

    Objects may share segments, so vertices are grouped per segment rather than
    assigned one object label each.
    """

    by_segment = group_indices(vertex_segments)
    empty = np.empty(0, dtype=np.int64)
    return {
        object_id: np.sort(
            np.concatenate(
                [
                    by_segment.get(int(segment), empty)
                    for segment in np.unique(value["segments"])
                ]
                + [empty]
            )
        )
        for object_id, value in objects.items()
    }


def parse_sstk_groups(
    data: dict,
    *,
    object_id_key: str = "objectId",
    segments_key: str = "segments",
) -> dict[int, dict]:
    try:
        groups = data["segGroups"]
    except KeyError as exc:
        raise DataFormatError("SSTK annotation is missing 'segGroups'.") from exc
    result: dict[int, dict] = {}
    for group in groups:
        object_id = int(group[object_id_key])
        if object_id in result:
            raise DataFormatError(
                f"Duplicate object ID in SSTK annotation: {object_id}"
            )
        value = {key: item for key, item in group.items() if key != object_id_key}
        if segments_key in value:
            value[segments_key] = np.asarray(value[segments_key], dtype=np.int64)
        result[object_id] = value
    return result


def parse_sstk_segments(data: dict) -> np.ndarray:
    try:
        segments = np.asarray(data["segIndices"], dtype=np.int64)
    except KeyError as exc:
        raise DataFormatError("SSTK segmentation is missing 'segIndices'.") from exc
    if segments.ndim != 1 or np.any(segments < 0):
        raise DataFormatError(
            "SSTK segIndices must be non-negative and one-dimensional."
        )
    return segments


def parse_sstk_segmentation(
    segment_data: dict,
    group_data: dict,
    *,
    invalid_id: int = -1,
    domain: Literal["vertex", "face"] = "vertex",
    object_id_key: str = "objectId",
    object_name_key: str = "label",
    segments_key: str = "segments",
    skip_remove_label: bool = False,
) -> Segmentation3D:
    segment_ids = parse_sstk_segments(segment_data)
    groups = parse_sstk_groups(
        group_data,
        object_id_key=object_id_key,
        segments_key=segments_key,
    )
    if len(segment_ids) == 0:
        return Segmentation3D(
            labels=np.empty((0,), dtype=np.int32),
            id_to_name={},
            domain=domain,
            invalid_id=invalid_id,
        )
    max_segment = int(segment_ids.max())
    segment_to_object = np.full(max_segment + 1, invalid_id, dtype=np.int32)
    id_to_name: dict[int, str] = {}
    for object_id, value in groups.items():
        try:
            name = value[object_name_key]
            group_segments = value[segments_key]
        except KeyError as exc:
            raise DataFormatError(
                f"Object {object_id} has incomplete SSTK metadata."
            ) from exc
        if skip_remove_label and str(name).lower() == "remove":
            continue
        invalid_segments = group_segments[
            (group_segments < 0) | (group_segments > max_segment)
        ]
        if len(invalid_segments):
            raise DataFormatError(
                f"Object {object_id} references unknown segments: {invalid_segments.tolist()}"
            )
        segment_to_object[group_segments] = object_id
        id_to_name[object_id] = str(name)
    return Segmentation3D(
        labels=segment_to_object[segment_ids],
        id_to_name=id_to_name,
        domain=domain,
        invalid_id=invalid_id,
    )
