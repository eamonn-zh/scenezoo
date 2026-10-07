"""Lazily exposed annotation, frame, and geometry operations."""

from __future__ import annotations

from importlib import import_module


_PUBLIC_SYMBOLS = {
    "FRAME_ITEMS": ("frame", "FRAME_ITEMS"),
    "PixelTransform": ("frame", "PixelTransform"),
    "apply_image_transform": ("frame", "apply_image_transform"),
    "apply_transform_batch": ("frame", "apply_transform_batch"),
    "build_pixel_transform": ("frame", "build_pixel_transform"),
    "compute_gravity_aligned_box": ("geometry", "compute_gravity_aligned_box"),
    "group_indices": ("annotation", "group_indices"),
    "construct_box": ("geometry", "construct_box"),
    "load_triangle_mesh": ("geometry", "load_triangle_mesh"),
    "parse_sstk_segmentation": ("annotation", "parse_sstk_segmentation"),
    "read_ply": ("annotation", "read_ply"),
    "read_ply_attribute": ("annotation", "read_ply_attribute"),
    "reconstruct_mesh_poisson": ("geometry", "reconstruct_mesh_poisson"),
    "remap_labels": ("annotation", "remap_labels"),
    "select_frame_indices": ("frame", "select_frame_indices"),
    "transform_intrinsics": ("frame", "transform_intrinsics"),
    "valid_camera_mask": ("frame", "valid_camera_mask"),
    "validate_frame_items": ("frame", "validate_frame_items"),
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
