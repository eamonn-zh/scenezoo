"""Static, import-light descriptions of dataset capabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameSourceSpec:
    """Capabilities of one named frame source.

    ``requires`` lists selectors beyond ``sample_id`` that callers must supply,
    such as Structured3D's ``room_id``.
    """

    name: str
    items: tuple[str, ...]
    camera_models: tuple[str, ...] = ()
    configurations: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ValueError("Frame source names must be non-empty.")
        object.__setattr__(self, "name", name)
        for field_name in ("items", "camera_models", "configurations", "requires"):
            values = tuple(str(value).strip() for value in getattr(self, field_name))
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(
                    f"FrameSourceSpec.{field_name} must be unique and non-empty."
                )
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Machine-readable public capabilities without loading an adapter."""

    sample_unit: str = "sample"
    operations: tuple[str, ...] = ()
    mesh_types: tuple[str, ...] = ()
    segmentation_types: tuple[str, ...] = ()
    box_types: tuple[str, ...] = ()
    label_spaces: tuple[str, ...] = ()
    frame_sources: tuple[FrameSourceSpec, ...] = ()

    def __post_init__(self) -> None:
        sample_unit = str(self.sample_unit).strip().lower()
        if not sample_unit:
            raise ValueError("DatasetSpec.sample_unit must be non-empty.")
        object.__setattr__(self, "sample_unit", sample_unit)
        tuple_fields = (
            "operations",
            "mesh_types",
            "segmentation_types",
            "box_types",
            "label_spaces",
        )
        for field_name in tuple_fields:
            values = tuple(str(value).strip() for value in getattr(self, field_name))
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(
                    f"DatasetSpec.{field_name} must be unique and non-empty."
                )
            object.__setattr__(self, field_name, values)
        unknown = set(self.operations) - {
            "mesh",
            "points",
            "segmentation",
            "boxes",
            "frames",
        }
        if unknown:
            raise ValueError(
                "Unknown dataset operations: " + ", ".join(sorted(unknown))
            )
        sources = tuple(self.frame_sources)
        if len({source.name for source in sources}) != len(sources):
            raise ValueError("DatasetSpec frame source names must be unique.")
        if sources and "frames" not in self.operations:
            raise ValueError("DatasetSpec.frame_sources requires the frames operation.")
        object.__setattr__(self, "frame_sources", sources)

    def frame_source(self, name: str) -> FrameSourceSpec:
        """Return one source description or raise a deterministic error."""

        normalized = str(name).strip().lower()
        for source in self.frame_sources:
            if source.name == normalized:
                return source
        available = ", ".join(source.name for source in self.frame_sources)
        raise KeyError(
            f"Unknown frame source {name!r}. Available sources: {available or 'none'}"
        )
