"""Parsers for the RIO/3RScan sequence container metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset.base import DataFormatError


@dataclass(frozen=True, slots=True)
class RIOSequenceInfo:
    """Calibration and dimensions declared by a 3RScan ``_info.txt`` file."""

    sensor_name: str
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    depth_shift: float
    color_intrinsic: np.ndarray
    color_extrinsic: np.ndarray
    depth_intrinsic: np.ndarray
    depth_extrinsic: np.ndarray
    frame_count: int


def parse_rio_sequence_info(payload: bytes | str) -> RIOSequenceInfo:
    """Parse 3RScan sequence metadata by key, independent of line ordering."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataFormatError("3RScan sequence metadata is not UTF-8.") from exc
    else:
        text = payload
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()

    def required(key: str) -> str:
        try:
            return values[key]
        except KeyError as exc:
            raise DataFormatError(
                f"3RScan sequence metadata is missing {key!r}."
            ) from exc

    def integer(key: str) -> int:
        try:
            value = int(required(key))
        except ValueError as exc:
            raise DataFormatError(f"Invalid integer value for {key!r}.") from exc
        return value

    def matrix(key: str) -> np.ndarray:
        try:
            array = np.fromstring(required(key), sep=" ", dtype=np.float32)
        except ValueError as exc:
            raise DataFormatError(f"Invalid matrix value for {key!r}.") from exc
        if array.size != 16:
            raise DataFormatError(f"3RScan {key!r} must contain 16 values.")
        result = array.reshape(4, 4)
        if not np.isfinite(result).all():
            raise DataFormatError(f"3RScan {key!r} contains non-finite values.")
        return result

    color_size = (integer("m_colorWidth"), integer("m_colorHeight"))
    depth_size = (integer("m_depthWidth"), integer("m_depthHeight"))
    frame_count = integer("m_frames.size")
    try:
        depth_shift = float(required("m_depthShift"))
    except ValueError as exc:
        raise DataFormatError("Invalid 3RScan depth shift.") from exc
    if min(*color_size, *depth_size, frame_count) < 0 or 0 in color_size + depth_size:
        raise DataFormatError("3RScan image sizes and frame count must be valid.")
    if not np.isfinite(depth_shift) or depth_shift <= 0:
        raise DataFormatError("3RScan depth shift must be positive and finite.")

    return RIOSequenceInfo(
        sensor_name=values.get("m_sensorName", ""),
        color_size=color_size,
        depth_size=depth_size,
        depth_shift=depth_shift,
        color_intrinsic=matrix("m_calibrationColorIntrinsic"),
        color_extrinsic=matrix("m_calibrationColorExtrinsic"),
        depth_intrinsic=matrix("m_calibrationDepthIntrinsic"),
        depth_extrinsic=matrix("m_calibrationDepthExtrinsic"),
        frame_count=frame_count,
    )
