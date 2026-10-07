"""Dataset loading, registration, and shared types."""

from .base import (
    DataFormatError,
    Dataset,
    DatasetCheckError,
    SceneZooError,
    UnsupportedOperationError,
)
from .check import CheckIssue, DatasetCheckReport
from .registry import (
    DatasetInfo,
    DatasetRegistry,
    _REGISTRY,
    get_dataset,
    get_dataset_info,
    is_registered,
    list_datasets,
    register_dataset,
)
from .spec import DatasetSpec, FrameSourceSpec
from .types import FrameBatch, PointBatch, Segmentation3D

from .builtins import register_builtin_datasets

# This package exposes adapter classes lazily through ``__getattr__``.
from . import scene as scene

register_builtin_datasets(_REGISTRY)

__all__ = [
    "DataFormatError",
    "CheckIssue",
    "Dataset",
    "DatasetCheckError",
    "DatasetCheckReport",
    "DatasetInfo",
    "DatasetRegistry",
    "DatasetSpec",
    "FrameBatch",
    "FrameSourceSpec",
    "PointBatch",
    "SceneZooError",
    "Segmentation3D",
    "UnsupportedOperationError",
    "get_dataset",
    "get_dataset_info",
    "is_registered",
    "list_datasets",
    "register_dataset",
    "scene",
]
