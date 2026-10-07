"""Archive access helpers."""

from __future__ import annotations

import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def locate_zip_members(
    file_path: str | Path,
    basenames: Sequence[str],
) -> dict[str, str]:
    """Resolve unique ZIP member paths by basename without extracting data."""

    requested = set(basenames)
    matches: dict[str, str] = {}
    with zipfile.ZipFile(file_path, "r") as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name not in requested:
                continue
            if name in matches:
                raise ValueError(f"ZIP contains multiple members named {name!r}.")
            matches[name] = member
    missing = requested - set(matches)
    if missing:
        raise KeyError("ZIP members not found: " + ", ".join(sorted(missing)))
    return matches


def read_zip_members(
    file_path: str | Path,
    members: Sequence[str],
) -> dict[str, bytes]:
    """Read only the requested ZIP members into memory."""

    with zipfile.ZipFile(file_path, "r") as archive:
        available = set(archive.namelist())
        missing = [name for name in members if name not in available]
        if missing:
            raise KeyError("ZIP members not found: " + ", ".join(missing))
        return {name: archive.read(name) for name in members}


@contextmanager
def extract_files_from_zip(
    file_path: str | Path,
    members: Sequence[str] | None = None,
    *,
    temp_dir: str | Path | None = None,
) -> Iterator[list[Path]]:
    """Extract selected ZIP members into a temporary directory."""

    with tempfile.TemporaryDirectory(dir=temp_dir) as directory:
        destination = Path(directory)
        with zipfile.ZipFile(file_path, "r") as archive:
            selected = list(archive.namelist() if members is None else members)
            available = set(archive.namelist())
            missing = [name for name in selected if name not in available]
            if missing:
                raise KeyError("ZIP members not found: " + ", ".join(missing))
            archive.extractall(path=destination, members=selected)
        yield [destination / name for name in selected]
