"""Video metadata and selective, frame-exact decoding with PyAV.

Frames are numbered by presentation order over every coded frame, including
frames hidden by MOV edit lists, so indices match per-frame metadata files
such as ARKitScenes intrinsics. Frames are always returned in their coded
orientation: display-rotation metadata is reported separately by
:func:`read_video_rotation` and is never applied implicitly, so pixels stay
consistent with calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import av
import numpy as np

# Number every coded frame. MOV edit lists can hide leading frames that still
# have per-frame metadata; other demuxers ignore this option.
_OPEN_OPTIONS = {"ignore_editlist": "1"}
_CHANNELS = {"rgb24": 3, "gray": 1}
# A seek flushes the (multithreaded) decoder, so it only pays off when it skips
# at least this many frames that forward decoding would otherwise decode.
_MIN_SEEK_SKIP = 64


@dataclass(frozen=True, slots=True)
class _VideoIndex:
    """Presentation timestamps and seek points of one video stream."""

    pts: np.ndarray
    keyframes: np.ndarray
    size: tuple[int, int]

    def position(self, pts: int | None) -> int:
        position = int(np.searchsorted(self.pts, pts)) if pts is not None else -1
        if pts is None or position >= len(self.pts) or self.pts[position] != pts:
            raise RuntimeError(f"Decoded a frame with an unindexed timestamp {pts}.")
        return position


def _open(path: str | Path):
    try:
        return av.open(str(path), options=_OPEN_OPTIONS)
    except (av.error.FFmpegError, OSError) as exc:
        raise RuntimeError(f"Unable to open video {path}: {exc}") from exc


def _video_stream(container, path):
    if not container.streams.video:
        raise RuntimeError(f"No video stream found in {path}")
    return container.streams.video[0]


@lru_cache(maxsize=32)
def _cached_index(path: str, size: int, mtime_ns: int) -> _VideoIndex:
    del size, mtime_ns  # Only part of the cache key.
    pts, keyframes = [], []
    with _open(path) as container:
        stream = _video_stream(container, path)
        width = stream.codec_context.width
        height = stream.codec_context.height
        for packet in container.demux(stream):
            if packet.size == 0:  # End-of-stream flush packet.
                continue
            if packet.pts is None:
                raise RuntimeError(f"Video packets without timestamps in {path}")
            pts.append(packet.pts)
            if packet.is_keyframe:
                keyframes.append(packet.pts)
    pts = np.sort(np.asarray(pts, dtype=np.int64))
    if len(np.unique(pts)) != len(pts):
        raise RuntimeError(f"Video frames share presentation timestamps in {path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Unable to determine video frame size for {path}")
    keyframe_positions = np.searchsorted(pts, np.sort(np.asarray(keyframes, np.int64)))
    return _VideoIndex(pts, keyframe_positions.astype(np.int64), (width, height))


def _video_index(path: str | Path) -> _VideoIndex:
    resolved = Path(path)
    stat = resolved.stat()
    return _cached_index(str(resolved), stat.st_size, stat.st_mtime_ns)


def read_video_rotation(path: str | Path) -> int:
    """Return the clockwise display rotation stored in video metadata."""

    with _open(path) as container:
        stream = _video_stream(container, path)
        for frame in container.decode(stream):
            # The display matrix rotation is counter-clockwise.
            return int(round(-float(frame.rotation))) % 360
    return 0


def read_video_frame_size(path: str | Path) -> tuple[int, int]:
    """Return the coded ``(width, height)`` of the first video stream."""

    return _video_index(path).size


def read_video_frame_count(path: str | Path) -> int:
    """Return the number of coded video frames without decoding them."""

    return len(_video_index(path).pts)


def _validated_indices(frame_indices) -> np.ndarray:
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or (
        indices.size
        and (
            not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
        )
    ):
        raise ValueError("frame_indices must be a one-dimensional integer sequence.")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0):
        raise ValueError("frame_indices must contain non-negative integers.")
    return indices


def _decode_positions(path, index: _VideoIndex, targets: np.ndarray, pix_fmt: str):
    """Decode sorted, unique presentation positions.

    Decoding continues forward unless seeking to the last keyframe at or before
    the next target skips more than ``_MIN_SEEK_SKIP`` frames.
    Frames are identified by timestamp, never by counting. If a seek ever lands
    past its target, decoding restarts from the beginning of the stream.
    """

    decoded: dict[int, np.ndarray] = {}
    container = _open(path)
    try:
        stream = _video_stream(container, path)
        stream.thread_type = "AUTO"
        frames = None
        position = -1  # Last position decoded by the active run.
        fresh = True  # Nothing has been decoded or sought yet.
        for target in targets.tolist():
            preceding = int(np.searchsorted(index.keyframes, target, side="right")) - 1
            # ``None`` means the target is reached by decoding from the start.
            keyframe = int(index.keyframes[preceding]) if preceding > 0 else None
            if frames is None or (
                keyframe is not None and keyframe - position > _MIN_SEEK_SKIP
            ):
                if keyframe is not None:
                    container.seek(
                        int(index.pts[keyframe]), stream=stream, backward=True
                    )
                elif not fresh:
                    container.close()
                    container = _open(path)
                    stream = _video_stream(container, path)
                    stream.thread_type = "AUTO"
                fresh = False
                frames = container.decode(stream)
                position = -1
            while True:
                frame = next(frames, None)
                if frame is None:
                    raise RuntimeError(f"Video ended before frame {target}: {path}")
                current = index.position(frame.pts)
                if current <= position:
                    raise RuntimeError(f"Video frames decoded out of order: {path}")
                if current > target:
                    if position >= 0 or keyframe is None:
                        raise RuntimeError(
                            f"Unable to decode video frame {target}: {path}"
                        )
                    # The seek overshot: decode this target from the start.
                    container.close()
                    container = _open(path)
                    stream = _video_stream(container, path)
                    stream.thread_type = "AUTO"
                    frames = container.decode(stream)
                    keyframe = None
                    continue
                position = current
                if current == target:
                    decoded[target] = frame.to_ndarray(format=pix_fmt)
                    break
    finally:
        container.close()
    return decoded


def read_video_frames(
    path: str | Path,
    *,
    frame_indices: Sequence[int] | np.ndarray,
    pix_fmt: str = "rgb24",
) -> np.ndarray:
    """Decode selected frames exactly, in the requested order.

    ``pix_fmt`` is ``"rgb24"`` for ``(N, H, W, 3)`` or ``"gray"`` for
    ``(N, H, W)`` uint8 output. Repeated indices are decoded once.
    """

    if pix_fmt not in _CHANNELS:
        raise ValueError("pix_fmt must be 'rgb24' or 'gray'.")
    indices = _validated_indices(frame_indices)
    index = _video_index(path)
    width, height = index.size
    shape = (height, width, 3) if _CHANNELS[pix_fmt] == 3 else (height, width)
    if len(indices) == 0:
        return np.empty((0, *shape), dtype=np.uint8)
    if int(indices.max()) >= len(index.pts):
        raise IndexError(f"Video frame index is outside [0, {len(index.pts)}).")
    decoded = _decode_positions(path, index, np.unique(indices), pix_fmt)
    for position, array in decoded.items():
        if array.shape != shape or array.dtype != np.uint8:
            raise RuntimeError(
                f"Video frame {position} decoded as {array.shape} {array.dtype}; "
                f"expected {shape} uint8: {path}"
            )
    return np.stack([decoded[int(position)] for position in indices])


def read_grayscale_video(
    path: str | Path,
    *,
    frame_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Decode selected frames as ``(N, H, W)`` uint8 luma images."""

    return read_video_frames(path, frame_indices=frame_indices, pix_fmt="gray")
