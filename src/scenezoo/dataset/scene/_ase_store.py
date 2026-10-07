"""Access extracted Aria Synthetic Environments scene directories."""

from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
import os
from pathlib import Path
import re

from ..base import DataFormatError


_SAMPLE_PATTERN = re.compile(r"^(?:(train|test)/)?([0-9]+)$")
_ARCHIVE_PATTERN = re.compile(r"^(train|test)_chunk_([0-9]{7})\.zip$")


class ASEStore:
    """Resolve ASE sample IDs against extracted directories.

    Chunk ZIPs are inventoried only so the dataset checker can explain which
    scenes still need extraction. Runtime data access never opens an archive.
    """

    def __init__(
        self,
        root: Path,
        data_dir: str | Path | None = None,
        *,
        extracted_split: str = "train",
    ) -> None:
        if extracted_split not in {"train", "test"}:
            raise ValueError("extracted_split must be 'train' or 'test'.")
        if data_dir is None:
            candidate = root / "download"
            self.data_root = candidate if candidate.is_dir() else root
        else:
            candidate = Path(data_dir).expanduser()
            self.data_root = candidate if candidate.is_absolute() else root / candidate
        self.extracted_split = extracted_split

    def normalize_sample_id(self, sample_id: str | int) -> str:
        value = str(sample_id)
        match = _SAMPLE_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                "ASE sample IDs must be '<train|test>/<integer>' or an integer."
            )
        split = match.group(1) or self.extracted_split
        number = int(match.group(2))
        return f"{split}/{number}"

    def parts(self, sample_id: str | int) -> tuple[str, int]:
        split, number = self.normalize_sample_id(sample_id).split("/", 1)
        return split, int(number)

    def scene_directory(self, sample_id: str | int) -> Path | None:
        split, number = self.parts(sample_id)
        namespaced = self.data_root / split / str(number)
        if namespaced.is_dir():
            return namespaced
        bare = self.data_root / str(number)
        if split == self.extracted_split and bare.is_dir():
            return bare
        return None

    def expected_scene_directory(self, sample_id: str | int) -> Path:
        split, number = self.parts(sample_id)
        if split == self.extracted_split:
            return self.data_root / str(number)
        return self.data_root / split / str(number)

    def archive_path(self, sample_id: str | int) -> Path:
        """Return the download archive containing a sample, without opening it."""

        split, number = self.parts(sample_id)
        return self.data_root / f"{split}_chunk_{number // 10:07d}.zip"

    def _require_scene_directory(self, sample_id: str | int) -> Path:
        normalized = self.normalize_sample_id(sample_id)
        directory = self.scene_directory(normalized)
        if directory is not None:
            return directory
        archive = self.archive_path(normalized)
        expected = self.expected_scene_directory(normalized)
        if archive.is_file():
            raise FileNotFoundError(
                f"ASE scene {normalized} is not extracted. Extract {archive} under "
                f"{self.data_root}; the scene must be available at {expected}."
            )
        raise FileNotFoundError(
            f"ASE scene {normalized} was not found at {expected}. Download and extract "
            f"{archive.name} first."
        )

    def exists(self, sample_id: str | int, relative_path: str) -> bool:
        directory = self.scene_directory(sample_id)
        return directory is not None and (directory / relative_path).exists()

    @contextmanager
    def open_binary(self, sample_id: str | int, relative_path: str):
        directory = self._require_scene_directory(sample_id)
        path = directory / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"ASE asset is missing: {path}")
        with path.open("rb") as handle:
            yield handle

    def read_bytes(self, sample_id: str | int, relative_path: str) -> bytes:
        with self.open_binary(sample_id, relative_path) as handle:
            return handle.read()

    def read_many(self, sample_id: str | int, relative_paths: list[str]) -> list[bytes]:
        directory = self._require_scene_directory(sample_id)
        try:
            return [(directory / relative).read_bytes() for relative in relative_paths]
        except OSError as exc:
            raise FileNotFoundError(
                f"A requested ASE asset is missing under {directory}."
            ) from exc

    def read_text(self, sample_id: str | int, relative_path: str) -> str:
        try:
            return self.read_bytes(sample_id, relative_path).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DataFormatError(
                f"ASE text asset is not UTF-8: {sample_id}/{relative_path}"
            ) from exc

    def listdir(self, sample_id: str | int, relative_path: str) -> list[str]:
        directory = self._require_scene_directory(sample_id)
        path = directory / relative_path
        return sorted(child.name for child in path.iterdir()) if path.is_dir() else []

    @cached_property
    def inventory(self) -> tuple[frozenset[str], frozenset[str]]:
        """Return extracted IDs and IDs covered by downloaded chunk ZIPs."""

        extracted: set[str] = set()
        archived: set[str] = set()
        if not self.data_root.is_dir():
            return frozenset(), frozenset()
        with os.scandir(self.data_root) as entries:
            for entry in entries:
                # Avoid a stat call for every scene on large shared filesystems;
                # official directory and archive names are unambiguous.
                if entry.name in {"train", "test"}:
                    with os.scandir(entry.path) as children:
                        extracted.update(
                            f"{entry.name}/{int(child.name)}"
                            for child in children
                            if child.name.isdigit()
                        )
                elif entry.name.isdigit():
                    extracted.add(f"{self.extracted_split}/{int(entry.name)}")
                elif match := _ARCHIVE_PATTERN.fullmatch(entry.name):
                    split, chunk = match.group(1), int(match.group(2))
                    archived.update(
                        f"{split}/{number}"
                        for number in range(chunk * 10, chunk * 10 + 10)
                    )
        return frozenset(extracted), frozenset(archived)

    @property
    def extracted_sample_ids(self) -> frozenset[str]:
        return self.inventory[0]

    @property
    def archive_sample_ids(self) -> frozenset[str]:
        """Return scene IDs in downloaded ZIPs for checker diagnostics only."""

        return self.inventory[1]

    @cached_property
    def available_sample_ids(self) -> tuple[str, ...]:
        """Discover extracted, immediately readable scenes."""

        return tuple(
            sorted(
                self.extracted_sample_ids,
                key=lambda value: (
                    value.split("/")[0],
                    int(value.split("/")[1]),
                ),
            )
        )
