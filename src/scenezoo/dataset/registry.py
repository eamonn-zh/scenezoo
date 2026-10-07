"""The lazy registry for 3D scene-dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, TypeVar

from .base import Dataset
from .spec import DatasetSpec


DatasetType = TypeVar("DatasetType", bound=type[Dataset])


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Import-light registry information for one canonical dataset."""

    name: str
    aliases: tuple[str, ...]
    description: str | None
    spec: DatasetSpec
    import_path: str | None
    _loader: Callable[[], type[Dataset]] = field(repr=False, compare=False)

    @property
    def capabilities(self) -> tuple[str, ...]:
        """The standard operations the adapter implements."""

        return self.spec.operations

    @property
    def dataset_class(self) -> type[Dataset]:
        """Load and return the adapter class on explicit access."""

        return self._loader()


@dataclass(slots=True)
class _RegistryEntry:
    name: str
    aliases: tuple[str, ...]
    description: str | None
    spec: DatasetSpec
    import_path: str | None = None
    dataset_class: type[Dataset] | None = None


class DatasetRegistry:
    """Maps canonical names and aliases to adapters, importing them lazily."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegistryEntry] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset names must be non-empty strings.")
        return name.strip().lower()

    def _registration_values(
        self, name: str, aliases: tuple[str, ...]
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        canonical = self._normalize_name(name)
        normalized_aliases = tuple(self._normalize_name(alias) for alias in aliases)
        all_names = (canonical, *normalized_aliases)
        if len(set(all_names)) != len(all_names):
            raise ValueError("A dataset name and its aliases must be unique.")
        return canonical, normalized_aliases, all_names

    def _add_entry(self, entry: _RegistryEntry) -> None:
        all_names = (entry.name, *entry.aliases)
        conflicts = [name for name in all_names if name in self._entries]
        if conflicts:
            raise ValueError(
                "Dataset names already registered: " + ", ".join(sorted(conflicts))
            )
        for name in all_names:
            self._entries[name] = entry

    def register_lazy(
        self,
        name: str,
        *,
        import_path: str,
        aliases: tuple[str, ...] = (),
        description: str | None = None,
        spec: DatasetSpec | None = None,
    ) -> None:
        """Register an import path without importing its adapter module."""

        canonical, normalized_aliases, _ = self._registration_values(name, aliases)
        module_name, separator, class_name = import_path.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError("import_path must use the 'module:ClassName' form.")
        entry = _RegistryEntry(
            name=canonical,
            aliases=normalized_aliases,
            description=description,
            spec=spec or DatasetSpec(sample_unit="scene"),
            import_path=import_path,
        )
        self._add_entry(entry)

    def register(
        self,
        name: str,
        *,
        aliases: tuple[str, ...] = (),
        description: str | None = None,
        spec: DatasetSpec | None = None,
    ) -> Callable[[DatasetType], DatasetType]:
        """Return a class decorator that registers an adapter under ``name``."""

        canonical, normalized_aliases, all_names = self._registration_values(
            name, aliases
        )

        def decorator(cls: DatasetType) -> DatasetType:
            if not issubclass(cls, Dataset):
                raise TypeError("Registered dataset classes must inherit from Dataset.")
            existing = self._entries.get(canonical)
            expected_path = f"{cls.__module__}:{cls.__name__}"
            if existing is not None:
                matching_lazy_entry = (
                    existing.dataset_class is None
                    and existing.import_path == expected_path
                    and existing.name == canonical
                    and existing.aliases == normalized_aliases
                    and all(self._entries.get(value) is existing for value in all_names)
                )
                if not matching_lazy_entry:
                    conflicts = [value for value in all_names if value in self._entries]
                    raise ValueError(
                        "Dataset names already registered: "
                        + ", ".join(sorted(conflicts))
                    )
                if existing.spec.operations != cls.capabilities():
                    raise RuntimeError(
                        f"Static capabilities for {canonical!r} do not match its adapter."
                    )
                existing.dataset_class = cls
                existing.description = (
                    existing.description or description or cls.__doc__
                )
                cls.name = canonical
                cls.spec = existing.spec
                return cls

            derived_spec = spec or DatasetSpec(
                sample_unit="scene",
                operations=cls.capabilities(),
            )
            if derived_spec.operations != cls.capabilities():
                raise ValueError(
                    "DatasetSpec.operations must match the adapter's standard methods."
                )
            entry = _RegistryEntry(
                name=canonical,
                aliases=normalized_aliases,
                description=description or cls.__doc__,
                spec=derived_spec,
                dataset_class=cls,
            )
            self._add_entry(entry)
            cls.name = canonical
            cls.spec = derived_spec
            return cls

        return decorator

    def _load_class(self, entry: _RegistryEntry) -> type[Dataset]:
        if entry.dataset_class is not None:
            return entry.dataset_class
        if entry.import_path is None:
            raise RuntimeError(f"Dataset {entry.name!r} has no class or import path.")
        module_name, class_name = entry.import_path.split(":", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        if not isinstance(cls, type) or not issubclass(cls, Dataset):
            raise TypeError(
                f"Lazy dataset target {entry.import_path!r} is not a Dataset class."
            )
        if entry.dataset_class is not None and entry.dataset_class is not cls:
            raise RuntimeError(
                f"Dataset module {module_name!r} registered an unexpected class."
            )
        if entry.spec.operations != cls.capabilities():
            raise RuntimeError(
                f"Static capabilities for {entry.name!r} do not match its adapter."
            )
        entry.dataset_class = cls
        cls.name = entry.name
        cls.spec = entry.spec
        return cls

    def create(
        self,
        name: str,
        root_dir: str | Path,
        **options: Any,
    ) -> Dataset:
        """Import the adapter for ``name`` and construct it with ``options``."""

        key = self._normalize_name(name)
        try:
            entry = self._entries[key]
        except KeyError as exc:
            available = ", ".join(self.list())
            raise KeyError(
                f"Unknown dataset {name!r}. Available datasets: {available}"
            ) from exc
        return self._load_class(entry)(root_dir=root_dir, **options)

    def list(self, *, include_aliases: bool = False) -> list[str]:
        """Return sorted canonical names, optionally including aliases."""

        if include_aliases:
            return sorted(self._entries)
        return sorted({entry.name for entry in self._entries.values()})

    def info(self, name: str) -> DatasetInfo:
        """Return import-free catalog information for a name or alias."""

        key = self._normalize_name(name)
        try:
            entry = self._entries[key]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset {name!r}.") from exc
        return DatasetInfo(
            name=entry.name,
            aliases=entry.aliases,
            description=entry.description,
            spec=entry.spec,
            import_path=entry.import_path,
            _loader=lambda: self._load_class(entry),
        )

    def contains(self, name: str) -> bool:
        """Whether ``name`` is a registered canonical name or alias."""

        return self._normalize_name(name) in self._entries


_REGISTRY = DatasetRegistry()


def register_dataset(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    description: str | None = None,
    spec: DatasetSpec | None = None,
):
    """Class decorator registering a :class:`~scenezoo.Dataset` adapter.

    Registration is atomic: the name and every alias are checked for conflicts
    before any is added. Without ``spec``, capabilities are derived from the
    standard methods the class overrides.
    """

    return _REGISTRY.register(
        name,
        aliases=aliases,
        description=description,
        spec=spec,
    )


def get_dataset(name: str, root_dir: str | Path, **options: Any) -> Dataset:
    """Create the adapter registered as ``name`` (or an alias) for ``root_dir``.

    ``options`` are passed to the adapter constructor, for example
    ``offline=True``. Only the selected adapter module is imported.
    """

    return _REGISTRY.create(name, root_dir, **options)


def list_datasets(include_aliases: bool = False) -> list[str]:
    """Return registered dataset names without importing any adapter."""

    return _REGISTRY.list(include_aliases=include_aliases)


def get_dataset_info(name: str) -> DatasetInfo:
    """Return the static :class:`DatasetInfo` for a dataset name or alias."""

    return _REGISTRY.info(name)


def is_registered(name: str) -> bool:
    """Whether ``name`` is a registered dataset name or alias."""

    return _REGISTRY.contains(name)
