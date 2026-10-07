"""Validated geometry operations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError


def load_triangle_mesh(
    file_path,
    *,
    enable_post_processing: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Load a non-empty triangle mesh with deterministic source errors."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")
    mesh = o3d.io.read_triangle_mesh(
        str(path), enable_post_processing=enable_post_processing
    )
    if not mesh.has_vertices() or not mesh.has_triangles():
        from ..dataset.base import DataFormatError

        raise DataFormatError(
            f"File does not contain a non-empty triangle mesh: {path}"
        )
    return mesh


def _minimum_area_axis(hull_points: np.ndarray) -> np.ndarray:
    edges = np.roll(hull_points, -1, axis=0) - hull_points
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > np.finfo(hull_points.dtype).eps
    directions = edges[valid] / lengths[valid, None]
    perpendiculars = np.stack([directions[:, 1], -directions[:, 0]], axis=1)
    projected = hull_points @ directions.T
    projected_perpendicular = hull_points @ perpendiculars.T
    widths = np.ptp(projected, axis=0)
    heights = np.ptp(projected_perpendicular, axis=0)
    best = int(np.argmin(widths * heights))
    return directions[best] if widths[best] >= heights[best] else perpendiculars[best]


def compute_gravity_aligned_box(
    points: np.ndarray,
    *,
    up_axis: Sequence[float] = (0.0, 0.0, 1.0),
    stats_nb_neighbors: int = 0,
    stats_std_ratio: float = 0,
    radius_nb_points: int = 0,
    radius_radius: float = 0,
):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("At least three 3D points are required.")
    if not np.isfinite(points).all():
        raise ValueError("Bounding-box points must be finite.")

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if 0 < radius_nb_points < len(points) and radius_radius > 0:
        cloud, _ = cloud.remove_radius_outlier(radius_nb_points, radius_radius)
        points = np.asarray(cloud.points)
    if 0 < stats_nb_neighbors < len(points) and stats_std_ratio > 0:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud, _ = cloud.remove_statistical_outlier(stats_nb_neighbors, stats_std_ratio)
        points = np.asarray(cloud.points)
    if len(points) < 3:
        raise ValueError("Too few points remain after outlier removal.")

    up = np.asarray(up_axis, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError("up_axis must be a finite, non-zero 3-vector.")
    up /= np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    plane_x = np.cross(up, seed)
    plane_x /= np.linalg.norm(plane_x)
    plane_y = np.cross(up, plane_x)
    plane_points = np.stack([points @ plane_x, points @ plane_y], axis=1)
    if np.linalg.matrix_rank(plane_points - plane_points.mean(axis=0)) < 2:
        raise ValueError(
            "Projected points are collinear; an oriented box is undefined."
        )
    try:
        hull = ConvexHull(plane_points)
    except QhullError as exc:
        raise ValueError(
            "Unable to compute a convex hull for the supplied points."
        ) from exc
    axis_2d = _minimum_area_axis(plane_points[hull.vertices])
    primary = plane_x * axis_2d[0] + plane_y * axis_2d[1]
    primary /= np.linalg.norm(primary)
    secondary = np.cross(up, primary)
    rotation = np.stack([primary, secondary, up], axis=1)
    local = points @ rotation
    minimum = local.min(axis=0)
    maximum = local.max(axis=0)
    center = rotation @ ((minimum + maximum) / 2)
    return o3d.geometry.OrientedBoundingBox(center, rotation, maximum - minimum)


def construct_box(points: np.ndarray, *, box_type: str = "mobb_gravity", **options):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Bounding-box input must be a non-empty (N, 3) array.")
    valid_types = {"aabb", "obb", "mobb", "mobb_gravity"}
    if box_type not in valid_types:
        raise ValueError(f"Unknown box type: {box_type!r}")
    if not np.isfinite(points).all():
        raise ValueError("Bounding-box points must be finite.")
    vector = o3d.utility.Vector3dVector(points)
    if box_type == "aabb":
        if options:
            raise ValueError("AABB construction does not accept extra options.")
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(vector)
    rank = np.linalg.matrix_rank(points - points.mean(axis=0))
    if len(points) < 3 or rank < 2:
        raise ValueError("Oriented bounding boxes require non-collinear 3D points.")
    if box_type == "obb":
        if options:
            raise ValueError("OBB construction does not accept extra options.")
        if len(points) < 4 or rank < 3:
            raise ValueError("A 3D OBB requires at least four non-coplanar points.")
        return o3d.geometry.OrientedBoundingBox.create_from_points(vector)
    if box_type == "mobb":
        if options:
            raise ValueError("Minimal OBB construction does not accept extra options.")
        if len(points) < 4 or rank < 3:
            raise ValueError(
                "A minimal 3D OBB requires at least four non-coplanar points."
            )
        return o3d.geometry.OrientedBoundingBox.create_from_points_minimal(vector)
    if box_type == "mobb_gravity":
        return compute_gravity_aligned_box(points, **options)


def reconstruct_mesh_poisson(
    point_cloud,
    *,
    depth: int = 9,
    density_filter_quantile: float = 0.01,
    linear_fit: bool = True,
    n_threads: int = -1,
):
    if not 0 <= density_filter_quantile <= 1:
        raise ValueError("density_filter_quantile must be between 0 and 1.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth,
        linear_fit=linear_fit,
        n_threads=n_threads,
    )
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_filter_quantile)
    mesh.remove_vertices_by_mask(densities < threshold)
    return mesh
