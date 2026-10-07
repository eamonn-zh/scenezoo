"""Image decoding shared by dataset adapters."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..dataset.base import DataFormatError


def read_image(
    source: str | Path | bytes, *, rgb: bool = False, name: str | None = None
) -> np.ndarray:
    """Decode an image file or encoded bytes into a writable array.

    ``rgb=True`` converts any mode to three-channel RGB; otherwise the stored
    mode (for example 16-bit depth or label IDs) is preserved. A missing file
    raises :class:`FileNotFoundError`; undecodable data raises
    :class:`~scenezoo.DataFormatError`.
    """

    label = name or ("image bytes" if isinstance(source, bytes) else str(source))
    try:
        with Image.open(
            io.BytesIO(source) if isinstance(source, bytes) else source
        ) as image:
            return np.array(image.convert("RGB") if rgb else image)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise DataFormatError(f"Cannot decode image: {label}") from exc


def read_images(paths: Sequence[str | Path], *, rgb: bool = False) -> np.ndarray:
    """Decode same-sized image files into one ``(N, H, W[, C])`` array."""

    images = [read_image(path, rgb=rgb) for path in paths]
    if not images:
        return np.empty((0, 0, 0, 3) if rgb else (0, 0, 0), dtype=np.uint8)
    if len({image.shape for image in images}) != 1:
        raise DataFormatError("Requested images do not share one resolution.")
    return np.stack(images)
