"""Indexed access to ScanNet ``.sens`` version 4 files."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np

from .image import read_image
from ..dataset.base import DataFormatError


COLOR_COMPRESSION_JPEG = 2
DEPTH_COMPRESSION_ZLIB_USHORT = 1
INVALID_TIMESTAMP = np.iinfo(np.uint64).max


@dataclass(frozen=True, slots=True)
class SensIndex:
    """Small random-access index for one ScanNet sensor stream."""

    path: Path
    sensor_name: str
    color_intrinsic: np.ndarray
    color_extrinsic: np.ndarray
    depth_intrinsic: np.ndarray
    depth_extrinsic: np.ndarray
    color_compression: int
    depth_compression: int
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    depth_shift: float
    camera_to_world: np.ndarray
    color_timestamps: np.ndarray
    depth_timestamps: np.ndarray
    color_offsets: np.ndarray
    color_sizes: np.ndarray
    depth_offsets: np.ndarray
    depth_sizes: np.ndarray
    imu_offset: int
    imu_count: int

    @property
    def frame_count(self) -> int:
        return len(self.camera_to_world)


def _read_exact(handle: BinaryIO, size: int, path: Path) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise DataFormatError(f"Truncated ScanNet .sens file: {path}")
    return payload


def read_sens_index(file_path: str | Path) -> SensIndex:
    """Read metadata and byte offsets without decoding image payloads."""

    path = Path(file_path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        version = struct.unpack("<I", _read_exact(handle, 4, path))[0]
        if version != 4:
            raise DataFormatError(f"Unsupported ScanNet .sens version: {version}")
        name_length = struct.unpack("<Q", _read_exact(handle, 8, path))[0]
        try:
            sensor_name = _read_exact(handle, name_length, path).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataFormatError(f"Invalid sensor name in {path}.") from exc

        matrices = [
            np.frombuffer(_read_exact(handle, 64, path), dtype="<f4")
            .reshape(4, 4)
            .copy()
            for _ in range(4)
        ]
        color_intrinsic, color_extrinsic, depth_intrinsic, depth_extrinsic = matrices
        color_compression, depth_compression = struct.unpack(
            "<ii", _read_exact(handle, 8, path)
        )
        color_width, color_height, depth_width, depth_height = struct.unpack(
            "<IIII", _read_exact(handle, 16, path)
        )
        if min(color_width, color_height, depth_width, depth_height) <= 0:
            raise DataFormatError(f"Invalid image dimensions in {path}.")
        depth_shift = struct.unpack("<f", _read_exact(handle, 4, path))[0]
        if not np.isfinite(depth_shift) or depth_shift <= 0:
            raise DataFormatError(
                f"Invalid ScanNet depth shift in {path}: {depth_shift}"
            )
        frame_count = struct.unpack("<Q", _read_exact(handle, 8, path))[0]

        poses = np.empty((frame_count, 4, 4), dtype=np.float32)
        color_timestamps = np.empty(frame_count, dtype=np.uint64)
        depth_timestamps = np.empty(frame_count, dtype=np.uint64)
        color_offsets = np.empty(frame_count, dtype=np.uint64)
        color_sizes = np.empty(frame_count, dtype=np.uint64)
        depth_offsets = np.empty(frame_count, dtype=np.uint64)
        depth_sizes = np.empty(frame_count, dtype=np.uint64)
        for frame_index in range(frame_count):
            poses[frame_index] = np.frombuffer(
                _read_exact(handle, 64, path), dtype="<f4"
            ).reshape(4, 4)
            color_timestamps[frame_index], depth_timestamps[frame_index] = (
                struct.unpack("<QQ", _read_exact(handle, 16, path))
            )
            color_size, depth_size = struct.unpack("<QQ", _read_exact(handle, 16, path))
            color_offset = handle.tell()
            color_offsets[frame_index] = color_offset
            color_sizes[frame_index] = color_size
            if color_offset + color_size > file_size:
                raise DataFormatError(f"Truncated ScanNet color frame: {path}")
            handle.seek(color_size, 1)
            depth_offset = handle.tell()
            depth_offsets[frame_index] = depth_offset
            depth_sizes[frame_index] = depth_size
            if depth_offset + depth_size > file_size:
                raise DataFormatError(f"Truncated ScanNet depth frame: {path}")
            handle.seek(depth_size, 1)

        imu_count_payload = handle.read(8)
        if len(imu_count_payload) != 8:
            raise DataFormatError(f"Truncated ScanNet IMU header: {path}")
        imu_count = struct.unpack("<Q", imu_count_payload)[0]
        imu_offset = handle.tell()
        expected_end = imu_offset + imu_count * 128
        if file_size < expected_end:
            raise DataFormatError(f"Truncated ScanNet IMU stream: {path}")

    return SensIndex(
        path=path,
        sensor_name=sensor_name,
        color_intrinsic=color_intrinsic,
        color_extrinsic=color_extrinsic,
        depth_intrinsic=depth_intrinsic,
        depth_extrinsic=depth_extrinsic,
        color_compression=color_compression,
        depth_compression=depth_compression,
        color_size=(color_width, color_height),
        depth_size=(depth_width, depth_height),
        depth_shift=float(depth_shift),
        camera_to_world=poses,
        color_timestamps=color_timestamps,
        depth_timestamps=depth_timestamps,
        color_offsets=color_offsets,
        color_sizes=color_sizes,
        depth_offsets=depth_offsets,
        depth_sizes=depth_sizes,
        imu_offset=imu_offset,
        imu_count=imu_count,
    )


def read_sens_frames(
    index: SensIndex,
    frame_indices: Sequence[int] | np.ndarray,
    *,
    rgb: bool = False,
    depth: bool = False,
) -> dict[str, np.ndarray]:
    """Decode only the requested image payloads, in the requested order."""

    requested = np.asarray(frame_indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("frame_indices must be one-dimensional.")
    if np.any(requested < 0) or (
        len(requested) and int(requested.max()) >= index.frame_count
    ):
        raise ValueError("ScanNet frame indices are outside the source frame range.")
    if rgb and index.color_compression != COLOR_COMPRESSION_JPEG:
        raise DataFormatError(
            f"Unsupported ScanNet color compression: {index.color_compression}"
        )
    if depth and index.depth_compression != DEPTH_COMPRESSION_ZLIB_USHORT:
        raise DataFormatError(
            f"Unsupported ScanNet depth compression: {index.depth_compression}"
        )

    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    with index.path.open("rb") as handle:
        for frame_index in requested:
            position = int(frame_index)
            if rgb:
                handle.seek(int(index.color_offsets[position]))
                payload = _read_exact(
                    handle, int(index.color_sizes[position]), index.path
                )
                rgbs.append(
                    read_image(payload, rgb=True, name=f"{index.path} frame {position}")
                )
            if depth:
                handle.seek(int(index.depth_offsets[position]))
                payload = _read_exact(
                    handle, int(index.depth_sizes[position]), index.path
                )
                try:
                    raw = zlib.decompress(payload)
                except zlib.error as exc:
                    raise DataFormatError(
                        f"Invalid ScanNet depth payload for frame {position}."
                    ) from exc
                width, height = index.depth_size
                values = np.frombuffer(raw, dtype="<u2")
                if values.size != width * height:
                    raise DataFormatError(
                        f"ScanNet depth frame {position} has an invalid size."
                    )
                depths.append(values.reshape(height, width))

    output: dict[str, np.ndarray] = {}
    if rgb:
        width, height = index.color_size
        output["rgb"] = (
            np.stack(rgbs) if rgbs else np.empty((0, height, width, 3), dtype=np.uint8)
        )
    if depth:
        width, height = index.depth_size
        output["depth"] = (
            np.stack(depths)
            if depths
            else np.empty((0, height, width), dtype=np.uint16)
        )
    return output


def read_sens_imu(index: SensIndex) -> dict[str, np.ndarray]:
    """Read the raw, independently sampled IMU stream."""

    fields = (
        "rotation_rate",
        "acceleration",
        "magnetic_field",
        "attitude",
        "gravity",
    )
    if index.imu_count == 0:
        result = {name: np.empty((0, 3), dtype=np.float64) for name in fields}
        result["timestamps"] = np.empty((0,), dtype=np.uint64)
        return result
    with index.path.open("rb") as handle:
        handle.seek(index.imu_offset)
        payload = _read_exact(handle, index.imu_count * 128, index.path)
    record_dtype = np.dtype(
        [(name, "<f8", (3,)) for name in fields] + [("timestamps", "<u8")]
    )
    records = np.frombuffer(payload, dtype=record_dtype, count=index.imu_count)
    return {name: np.asarray(records[name]).copy() for name in (*fields, "timestamps")}
