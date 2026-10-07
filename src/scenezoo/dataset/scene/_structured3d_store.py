"""Access extracted Structured3D scene directories."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
import re

from ..base import DataFormatError


_SCENE_PATTERN = re.compile(r"^scene_([0-9]{5})$")


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", value)
    )


class Structured3DStore:
    """Resolve logical paths inside an extracted Structured3D release.

    Official ZIP shards are inventoried only for checker diagnostics. Runtime
    access never opens an archive.
    """

    def __init__(self, root: Path, data_dir: str | Path | None = None) -> None:
        if data_dir is not None:
            candidate = Path(data_dir).expanduser()
            self.data_root = candidate if candidate.is_absolute() else root / candidate
        else:
            candidates = (root / "data", root / "Structured3D", root)
            self.data_root = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.is_dir() and next(candidate.glob("scene_*"), None)
                ),
                root,
            )
        self.archive_root = root

    @cached_property
    def zip_paths(self) -> tuple[Path, ...]:
        """Return downloaded release shards without inspecting their contents."""

        if not self.archive_root.is_dir():
            return ()
        directories = (self.archive_root, self.archive_root / "download")
        return tuple(
            sorted(
                path
                for directory in directories
                if directory.is_dir()
                for path in directory.glob("*.zip")
                if "structured3d" in path.name.lower()
            )
        )

    def archive_candidates(self, scene_id: str) -> tuple[Path, ...]:
        """Return downloaded shards whose names may cover one scene."""

        shard = int(scene_id.removeprefix("scene_")) // 200
        selected = []
        for path in self.zip_paths:
            stem = path.stem.lower()
            if "annotation" in stem or "bbox" in stem:
                selected.append(path)
                continue
            suffix = re.search(r"(?:_|-)([0-9]{1,2})$", stem)
            if suffix is None or int(suffix.group(1)) == shard:
                selected.append(path)
        return tuple(selected)

    @staticmethod
    def _scene_from_logical(logical: str) -> str:
        scene_id = logical.replace("\\", "/").split("/", 1)[0]
        if _SCENE_PATTERN.fullmatch(scene_id) is None:
            raise ValueError(
                f"Structured3D path does not begin with a scene ID: {logical}"
            )
        return scene_id

    def _missing(self, logical: str) -> FileNotFoundError:
        scene_id = self._scene_from_logical(logical)
        candidates = self.archive_candidates(scene_id)
        if candidates:
            names = ", ".join(path.name for path in candidates[:4])
            return FileNotFoundError(
                f"Structured3D scene {scene_id} is not fully extracted under "
                f"{self.data_root}. Extract the relevant shard(s), such as {names}, "
                "before using the dataset API."
            )
        return FileNotFoundError(
            f"Structured3D asset was not found under {self.data_root}: {logical}"
        )

    def exists(self, logical: str) -> bool:
        return (self.data_root / logical).exists()

    def read_bytes(self, logical: str) -> bytes:
        logical = logical.replace("\\", "/")
        path = self.data_root / logical
        if not path.is_file():
            raise self._missing(logical)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DataFormatError(f"Cannot read Structured3D asset: {path}") from exc

    def read_text(self, logical: str) -> str:
        try:
            return self.read_bytes(logical).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DataFormatError(f"Structured3D text is not UTF-8: {logical}") from exc

    def listdir(self, logical: str) -> list[str]:
        logical = logical.replace("\\", "/").rstrip("/")
        path = self.data_root / logical
        if not path.is_dir():
            raise self._missing(logical)
        return sorted((child.name for child in path.iterdir()), key=_natural_key)
