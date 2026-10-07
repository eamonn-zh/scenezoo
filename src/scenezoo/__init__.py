"""A small, extensible toolkit for 3D dataset preprocessing."""

from .dataset import (
    CheckIssue,
    DataFormatError,
    Dataset,
    DatasetCheckError,
    DatasetCheckReport,
    DatasetInfo,
    DatasetSpec,
    FrameBatch,
    FrameSourceSpec,
    PointBatch,
    SceneZooError,
    Segmentation3D,
    UnsupportedOperationError,
    get_dataset,
    get_dataset_info,
    list_datasets,
    register_dataset,
)

__version__ = "0.2.0"

__all__ = [
    "CheckIssue",
    "DataFormatError",
    "Dataset",
    "DatasetCheckError",
    "DatasetCheckReport",
    "DatasetInfo",
    "DatasetSpec",
    "FrameBatch",
    "FrameSourceSpec",
    "PointBatch",
    "SceneZooError",
    "Segmentation3D",
    "UnsupportedOperationError",
    "get_dataset",
    "get_dataset_info",
    "list_datasets",
    "register_dataset",
]
