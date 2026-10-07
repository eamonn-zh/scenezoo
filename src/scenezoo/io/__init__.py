"""Lazily exposed I/O helpers used by dataset adapters."""

from __future__ import annotations

from importlib import import_module


_PUBLIC_SYMBOLS = {
    "ColmapCamera": ("camera", "ColmapCamera"),
    "ColmapImage": ("camera", "ColmapImage"),
    "RIOSequenceInfo": ("rio", "RIOSequenceInfo"),
    "extract_files_from_zip": ("archive", "extract_files_from_zip"),
    "locate_zip_members": ("archive", "locate_zip_members"),
    "open_cached_file": ("cache", "open_cached_file"),
    "parse_rio_sequence_info": ("rio", "parse_rio_sequence_info"),
    "read_colmap_cameras": ("camera", "read_colmap_cameras"),
    "read_colmap_images": ("camera", "read_colmap_images"),
    "read_colmap_sparse_points": ("camera", "read_colmap_sparse_points"),
    "read_grayscale_video": ("video", "read_grayscale_video"),
    "read_image": ("image", "read_image"),
    "read_images": ("image", "read_images"),
    "read_raw_deflate_frames": ("compression", "read_raw_deflate_frames"),
    "read_video_frame_count": ("video", "read_video_frame_count"),
    "read_video_frame_size": ("video", "read_video_frame_size"),
    "read_video_frames": ("video", "read_video_frames"),
    "read_video_rotation": ("video", "read_video_rotation"),
    "read_zip_members": ("archive", "read_zip_members"),
}

__all__ = list(_PUBLIC_SYMBOLS)


def __getattr__(name: str):
    try:
        module_name, symbol_name = _PUBLIC_SYMBOLS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_PUBLIC_SYMBOLS))
