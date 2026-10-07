"""Streaming readers for frame-oriented compressed arrays."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ..dataset.base import DataFormatError


def read_raw_deflate_frames(
    path: str | Path,
    frame_indices: Sequence[int] | np.ndarray,
    *,
    frame_shape: tuple[int, ...],
    dtype: np.dtype | str,
    frame_count: int,
    chunk_size: int = 1 << 20,
) -> np.ndarray:
    """Read selected arrays from one continuous raw-DEFLATE stream.

    MultiScan stores all frames in a single stream, so reaching a late frame
    necessarily requires inflating the preceding bytes. This reader discards
    those bytes immediately and retains only the requested frames.
    """

    requested = np.asarray(frame_indices)
    if requested.ndim != 1 or (
        requested.size
        and (
            not np.issubdtype(requested.dtype, np.integer)
            or np.issubdtype(requested.dtype, np.bool_)
        )
    ):
        raise ValueError("frame_indices must be a one-dimensional integer sequence.")
    requested = requested.astype(np.int64, copy=False)
    if len(np.unique(requested)) != len(requested):
        raise ValueError("frame_indices must not contain duplicates.")
    if frame_count < 0 or np.any(requested < 0) or np.any(requested >= frame_count):
        raise ValueError("frame_indices are outside the source frame range.")

    resolved_dtype = np.dtype(dtype)
    if not frame_shape or any(int(size) <= 0 for size in frame_shape):
        raise ValueError("frame_shape must contain positive dimensions.")
    shape = tuple(int(size) for size in frame_shape)
    output = np.empty((len(requested), *shape), dtype=resolved_dtype)
    if len(requested) == 0:
        return output

    frame_bytes = int(np.prod(shape, dtype=np.int64)) * resolved_dtype.itemsize
    sorted_order = np.argsort(requested)
    sorted_indices = requested[sorted_order]
    targets = {
        int(frame_id): output[int(sorted_order[position])].view(np.uint8).reshape(-1)
        for position, frame_id in enumerate(sorted_indices)
    }
    last_byte = (int(sorted_indices[-1]) + 1) * frame_bytes
    produced = 0
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)

    def consume(data: bytes) -> None:
        nonlocal produced
        if not data:
            return
        start = produced
        end = produced + len(data)
        first_frame = start // frame_bytes
        last_frame = min((end - 1) // frame_bytes, int(sorted_indices[-1]))
        for frame_id in range(first_frame, last_frame + 1):
            target = targets.get(frame_id)
            if target is None:
                continue
            frame_start = frame_id * frame_bytes
            overlap_start = max(start, frame_start)
            overlap_end = min(end, frame_start + frame_bytes)
            if overlap_start < overlap_end:
                target[overlap_start - frame_start : overlap_end - frame_start] = (
                    np.frombuffer(
                        data,
                        dtype=np.uint8,
                        count=overlap_end - overlap_start,
                        offset=overlap_start - start,
                    )
                )
        produced = end

    try:
        with open(path, "rb") as handle:
            while produced < last_byte:
                compressed = handle.read(chunk_size)
                if not compressed:
                    break
                pending = compressed
                while pending and produced < last_byte:
                    decoded = decoder.decompress(
                        pending, min(chunk_size, last_byte - produced)
                    )
                    pending = decoder.unconsumed_tail
                    consume(decoded)
                    if not decoded and not pending:
                        break
    except zlib.error as exc:
        raise DataFormatError(f"Invalid raw-DEFLATE stream: {path}") from exc

    if produced < last_byte:
        raise DataFormatError(
            f"Compressed stream {path} ended before frame {int(sorted_indices[-1])}."
        )
    return output
