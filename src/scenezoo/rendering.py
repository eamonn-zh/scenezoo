"""Optional PyTorch3D rendering helpers.

This module is not imported by :mod:`scenezoo`; Torch and PyTorch3D are loaded
only when rasterization is requested.
"""

from __future__ import annotations

import numpy as np


def _pytorch3d():
    try:
        import torch
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
    except ImportError as exc:
        raise ImportError(
            "Rendering requires PyTorch and PyTorch3D. Install scenezoo[render], "
            "then PyTorch3D: "
            "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
        ) from exc
    return (
        torch,
        MeshRasterizer,
        RasterizationSettings,
        Meshes,
        cameras_from_opencv_projection,
    )


def rasterize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    image_size: tuple[int, int],
    batch_size: int = 8,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a mesh with OpenCV cameras and return face IDs and depth.

    Pixels without a visible face have face ID ``-1`` and depth ``0.0``,
    matching :class:`~scenezoo.FrameBatch` depth conventions.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    height, width = (int(value) for value in image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive (height, width) values.")
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float32)
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None], len(world_to_camera), axis=0)
    if intrinsics.shape != (len(world_to_camera), 3, 3):
        raise ValueError("intrinsics must have shape (3, 3) or (N, 3, 3).")
    if world_to_camera.shape[1:] != (4, 4):
        raise ValueError("world_to_camera must have shape (N, 4, 4).")

    torch, MeshRasterizer, RasterizationSettings, Meshes, cameras_from_opencv = (
        _pytorch3d()
    )
    target = torch.device(device)
    vertices_t = torch.as_tensor(vertices, device=target)
    faces_t = torch.as_tensor(faces, device=target)
    base_mesh = Meshes(verts=[vertices_t], faces=[faces_t])
    rasterizer = MeshRasterizer(
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            cull_to_frustum=True,
        )
    ).to(target)
    image_sizes = torch.tensor([height, width], dtype=torch.int32, device=target)
    face_ids = np.empty((len(world_to_camera), height, width), dtype=np.int32)
    depths = np.empty((len(world_to_camera), height, width), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(world_to_camera), batch_size):
            stop = min(start + batch_size, len(world_to_camera))
            count = stop - start
            extrinsics = torch.as_tensor(world_to_camera[start:stop], device=target)
            camera_matrix = torch.as_tensor(intrinsics[start:stop], device=target)
            cameras = cameras_from_opencv(
                R=extrinsics[:, :3, :3],
                tvec=extrinsics[:, :3, 3],
                camera_matrix=camera_matrix,
                image_size=image_sizes.expand(count, -1),
            ).to(target)
            fragments = rasterizer(base_mesh.extend(count), cameras=cameras)
            ids = fragments.pix_to_face[..., 0]
            valid = ids >= 0
            offsets = torch.arange(count, device=target, dtype=ids.dtype)[:, None, None]
            ids = torch.where(valid, ids - offsets * len(faces), ids)
            face_ids[start:stop] = ids.cpu().numpy().astype(np.int32, copy=False)
            depth = fragments.zbuf[..., 0]
            depths[start:stop] = torch.where(valid, depth, 0).cpu().numpy()
    return face_ids, depths


def project_vertex_values(
    face_ids: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    *,
    invalid_value=-1,
) -> np.ndarray:
    """Project the first vertex value of each visible face."""

    face_ids = np.asarray(face_ids)
    invalid = face_ids < 0
    safe_ids = np.where(invalid, 0, face_ids)
    projected = np.asarray(vertex_values)[np.asarray(faces)[safe_ids, 0]].copy()
    projected[invalid] = invalid_value
    return projected


def project_face_values(
    face_ids: np.ndarray,
    face_values: np.ndarray,
    *,
    invalid_value=-1,
) -> np.ndarray:
    """Project one value per mesh face into rasterized images."""

    face_ids = np.asarray(face_ids)
    invalid = face_ids < 0
    projected = np.asarray(face_values)[np.where(invalid, 0, face_ids)].copy()
    projected[invalid] = invalid_value
    return projected
