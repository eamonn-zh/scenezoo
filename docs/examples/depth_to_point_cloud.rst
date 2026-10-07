Turn depth frames into a point cloud
====================================

This recipe back-projects the depth of several frames into world coordinates,
colors the points from the RGB images, and saves one fused point cloud. It is a
quick way to look at a capture in 3D or to check that depth and camera poses
agree with the released mesh.

.. code-block:: python

   import numpy as np
   import open3d as o3d

   from scenezoo import get_dataset

   dataset = get_dataset("scannetppv2", "/path/to/scannetpp")
   scene_id = dataset.get_ids("nvs_sem_val")[0]

   # Read RGB at the depth resolution so every depth pixel has a color.
   frames = dataset.get_frames(
       scene_id,
       source="iphone",
       step=120,
       items=("rgb", "depth", "depth_intrinsics", "world_to_camera"),
       output_size=(256, 192),
   )

   points, colors = [], []
   for rgb, depth, K, world_to_camera in zip(
       frames.rgb, frames.depth, frames.depth_intrinsics, frames.world_to_camera
   ):
       v, u = np.nonzero(depth > 0)
       pixels = np.stack([u, v, np.ones_like(u)])               # (3, M)
       camera_points = (np.linalg.inv(K) @ pixels) * depth[v, u]
       camera_to_world = np.linalg.inv(world_to_camera)
       R, t = camera_to_world[:3, :3], camera_to_world[:3, 3]
       points.append(camera_points.T @ R.T + t)
       colors.append(rgb[v, u] / 255.0)

   cloud = o3d.geometry.PointCloud()
   cloud.points = o3d.utility.Vector3dVector(np.concatenate(points))
   cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
   cloud = cloud.voxel_down_sample(voxel_size=0.02)

   o3d.io.write_point_cloud(f"{scene_id}_fused.ply", cloud)
   print(f"{len(frames)} frames -> {len(cloud.points)} points")

Open the PLY in any point-cloud viewer, or overlay it on the scene mesh from
``dataset.get_mesh(scene_id)``; the two should line up.

Notes
-----

- ``step=120`` uses every 120th frame (two seconds of this 60 FPS video).
  Use a smaller step for a denser cloud.
- This works for every pinhole frame source. For Aria Synthetic Environments
  and for panoramas, depth is measured along each pixel's ray instead of the
  camera Z axis, so the formula above does not apply. See
  :doc:`../guide/frames`.
