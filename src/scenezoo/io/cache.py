"""Local and cached remote file access."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO, BinaryIO
from urllib.parse import urlparse

import fsspec

from ..config import SCENEZOO_CACHE_DIR


def cached_file_path(file_path: str, cache_dir: str | Path | None = None) -> Path:
    """Return the deterministic local cache location for a remote URL."""

    digest = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    name = Path(urlparse(file_path).path).name or "download"
    root = Path(cache_dir or SCENEZOO_CACHE_DIR).expanduser()
    return root / "files" / f"{digest[:32]}-{name}"


def _legacy_fsspec_cache(path: str, cache_dir: Path, scheme: str) -> Path | None:
    """Find files cached by earlier versions through ``fsspec`` filecache."""

    try:
        filesystem = fsspec.filesystem(
            "filecache",
            target_protocol=scheme,
            cache_storage=str(cache_dir),
            check_files=False,
            expiry_time=None,
        )
        cached = filesystem._check_file(filesystem._strip_protocol(path))
    except Exception:  # Private fsspec internals; absence means "not cached".
        return None
    if not cached:
        return None
    local_path = Path(cached[1])
    return local_path if local_path.is_file() else None


def _download(path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    os.close(handle)
    try:
        # Stream the whole response. Reading through a seekable file object can
        # truncate gzip-encoded HTTP responses to their compressed length.
        filesystem, remote_path = fsspec.core.url_to_fs(path)
        filesystem.get_file(remote_path, temporary)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@contextmanager
def open_cached_file(
    file_path: str | Path,
    mode: str = "rb",
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> Iterator[TextIO | BinaryIO]:
    """Open a local path directly or a remote path through a persistent cache.

    Remote files are downloaded once and reused afterwards. ``offline=True``
    never touches the network and fails if the file has not been cached.
    """

    path = str(file_path)
    scheme = urlparse(path).scheme
    is_remote = bool(scheme and scheme != "file")
    if not is_remote:
        local_path = urlparse(path).path if scheme == "file" else path
        with open(Path(local_path).expanduser(), mode) as handle:
            yield handle
        return

    resolved_cache = Path(cache_dir or SCENEZOO_CACHE_DIR).expanduser()
    local_path = cached_file_path(path, resolved_cache)
    if not local_path.is_file():
        legacy = _legacy_fsspec_cache(path, resolved_cache, scheme)
        if legacy is not None:
            local_path = legacy
        elif offline:
            raise FileNotFoundError(
                f"Remote file is not cached for offline use: {path}"
            )
        else:
            _download(path, local_path)
    with open(local_path, mode) as handle:
        yield handle
