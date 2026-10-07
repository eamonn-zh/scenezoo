"""Lazily exposed built-in scene dataset adapters."""

from __future__ import annotations

from importlib import import_module


_PUBLIC_CLASSES = {
    "ARKitScenes": ("arkitscenes", "ARKitScenes"),
    "AriaSyntheticEnvironments": ("ase", "AriaSyntheticEnvironments"),
    "ASEPointData": ("ase", "ASEPointData"),
    "Matterport3D": ("matterport3d", "Matterport3D"),
    "MultiScan": ("multiscan", "MultiScan"),
    "S3DIS": ("s3dis", "S3DIS"),
    "S3DISPointData": ("s3dis", "S3DISPointData"),
    "ScanNet": ("scannet", "ScanNet"),
    "ScanNetPP": ("scannetpp", "ScanNetPP"),
    "SceneNN": ("scenenn", "SceneNN"),
    "Structured3D": ("structured3d", "Structured3D"),
    "Structured3DPointData": ("structured3d", "Structured3DPointData"),
    "ThreeRScan": ("threerscan", "ThreeRScan"),
}

__all__ = list(_PUBLIC_CLASSES)


def __getattr__(name: str):
    try:
        module_name, class_name = _PUBLIC_CLASSES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), class_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_PUBLIC_CLASSES))
