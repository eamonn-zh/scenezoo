"""Core dataset abstraction."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from .spec import DatasetSpec

if TYPE_CHECKING:
    from .check import DatasetCheckReport


class SceneZooError(Exception):
    """Base exception raised by SceneZoo."""


class UnsupportedOperationError(SceneZooError):
    """Raised when a dataset does not provide a requested operation or item."""


class DataFormatError(SceneZooError):
    """Raised when source data does not match its documented format."""


class DatasetCheckError(SceneZooError):
    """Raised when an explicitly requested dataset check finds errors."""

    def __init__(self, report: DatasetCheckReport) -> None:
        self.report = report
        super().__init__(report.format())


class Dataset:
    """Minimal base class for 3D scene datasets.

    Optional operations are regular methods rather than abstract methods. This
    lets partial adapters expose exactly the data they really provide.
    """

    name: ClassVar[str] = "dataset"
    spec: ClassVar[DatasetSpec] = DatasetSpec()
    _OPERATIONS: ClassVar[dict[str, str]] = {
        "mesh": "get_mesh",
        "points": "get_points",
        "segmentation": "get_segmentation",
        "boxes": "get_boxes",
        "frames": "get_frames",
    }

    def __init__(
        self,
        root_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        invalid_obj_id: int = -1,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.offline = offline
        self.invalid_obj_id = invalid_obj_id

    @cached_property
    def splits(self) -> dict[str, list[str]]:
        """Sample IDs of each official split, loaded once on first access."""

        splits = self._load_splits()
        if not isinstance(splits, dict):
            raise DataFormatError("_load_splits() must return a dictionary.")
        return splits

    @cached_property
    def metadata(self) -> dict[str, Any]:
        """Dataset-wide metadata such as label maps, loaded once on first access."""

        metadata = self._load_metadata()
        if not isinstance(metadata, dict):
            raise DataFormatError("_load_metadata() must return a dictionary.")
        return metadata

    def _load_splits(self) -> dict[str, list[str]]:
        raise NotImplementedError

    def _load_metadata(self) -> dict[str, Any]:
        return {}

    def _load_sample_ids(self) -> list[str]:
        # Override when the release has samples outside every official split.
        return list(dict.fromkeys(i for ids in self.splits.values() for i in ids))

    def get_ids(self, split: str | None = None) -> list[str]:
        """Return the sample IDs of an official ``split``, or of the whole
        dataset when ``split`` is omitted; see :attr:`splits` for the names."""

        if split is None:
            return self._load_sample_ids()
        try:
            return self.splits[split]
        except KeyError as exc:
            available = ", ".join(sorted(self.splits)) or "none"
            raise KeyError(
                f"Unknown split {split!r}. Official splits: {available}. "
                "Call get_ids() without a split for every sample."
            ) from exc

    def open_file(self, file_path: str | Path, mode: str = "rb"):
        """Open a local or cached remote metadata file using this dataset's policy."""
        from ..io.cache import open_cached_file

        return open_cached_file(
            file_path,
            mode,
            cache_dir=self.cache_dir,
            offline=self.offline,
        )

    def check(
        self,
        *,
        sample_ids: Sequence[str] | None = None,
        require_complete: bool = False,
        raise_on_error: bool = False,
    ) -> DatasetCheckReport:
        """Explicitly inspect the dataset path and return an actionable report.

        Checks are never run by the constructor or by data access methods. Use
        ``sample_ids`` for a focused check, or ``require_complete=True`` to treat
        missing scenes from declared splits as errors rather than warnings.
        """

        report = self._check(
            sample_ids=sample_ids,
            require_complete=require_complete,
        )
        if raise_on_error:
            report.raise_for_errors()
        return report

    def _check(
        self,
        *,
        sample_ids: Sequence[str] | None,
        require_complete: bool,
    ) -> DatasetCheckReport:
        from .check import CheckBuilder

        builder = CheckBuilder(self.name, self.root_dir)
        builder.require_directory(
            self.root_dir,
            code="missing-root",
            message="The configured dataset root does not exist or is not a directory.",
            expected="An existing directory containing the dataset files.",
            hint="Correct root_dir or download and extract the dataset first.",
        )
        return builder.finish()

    @classmethod
    def supports(cls, operation: str) -> bool:
        """Whether this adapter implements a standard operation.

        ``operation`` is one of ``mesh``, ``points``, ``segmentation``,
        ``boxes``, or ``frames``.
        """

        try:
            method_name = cls._OPERATIONS[operation]
        except KeyError as exc:
            valid = ", ".join(sorted(cls._OPERATIONS))
            raise ValueError(
                f"Unknown operation {operation!r}. Valid operations: {valid}"
            ) from exc
        return getattr(cls, method_name) is not getattr(Dataset, method_name)

    @classmethod
    def capabilities(cls) -> tuple[str, ...]:
        """The standard operations this adapter implements."""

        return tuple(name for name in cls._OPERATIONS if cls.supports(name))

    def _unsupported(self, operation: str) -> None:
        raise UnsupportedOperationError(
            f"{self.__class__.__name__} does not support the {operation!r} operation."
        )

    def get_mesh(self, sample_id: str, *, mesh_type: str | None = None):
        """Return an Open3D ``TriangleMesh``.

        ``mesh_type`` selects a dataset-specific variant listed in
        ``spec.mesh_types``; ``None`` selects the adapter's default.
        """

        self._unsupported("mesh")

    def get_points(self, sample_id: str):
        """Return a :class:`~scenezoo.PointBatch` with row-aligned attributes."""

        self._unsupported("points")

    def get_segmentation(self, sample_id: str):
        """Return :class:`~scenezoo.Segmentation3D` labels.

        Labels are defined on the ``domain`` the result reports (points, mesh
        vertices, or mesh faces) and align with the matching geometry.
        """

        self._unsupported("segmentation")

    def get_boxes(self, sample_id: str, *, box_type: str = "mobb_gravity"):
        """Return ``(boxes_by_id, id_to_name)`` with Open3D bounding boxes.

        ``box_type`` is ``obb_gt`` for released annotations, or ``aabb``,
        ``obb``, ``mobb``, or ``mobb_gravity`` to fit boxes to annotated points.
        """

        self._unsupported("boxes")

    def get_frames(
        self,
        sample_id: str,
        *,
        indices=None,
        step: int = 1,
        items=("rgb", "rgb_intrinsics", "world_to_camera"),
        output_size=None,
        center_crop: bool = False,
        rotate_to_up: bool = True,
    ):
        """Return synchronized frames as a :class:`~scenezoo.FrameBatch`.

        Args:
            indices: Source frame numbers to read, in the requested order.
                ``None`` selects every frame with a valid camera.
            step: Sampling stride when ``indices`` is ``None``.
            items: :class:`~scenezoo.FrameBatch` fields to load.
            output_size: Optional ``(width, height)`` for every image item.
            center_crop: Preserve the aspect ratio by resizing then cropping.
            rotate_to_up: Rotate portrait captures upright; the camera roll is
                applied to ``world_to_camera`` so intrinsics stay pinhole.
        """

        self._unsupported("frames")

    def __repr__(self) -> str:
        loaded_splits = self.__dict__.get("splits")
        split_sizes = (
            None
            if loaded_splits is None
            else {key: len(value) for key, value in loaded_splits.items()}
        )
        return (
            f"{self.__class__.__name__}(root_dir={str(self.root_dir)!r}, "
            f"splits={split_sizes})"
        )
