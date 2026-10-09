Working with frames
===================

:meth:`~scenezoo.Dataset.get_frames` returns images, depth maps, and cameras
for selected frames of a capture. The result is a
:class:`~scenezoo.FrameBatch` whose arrays all start with the frame dimension
``N``, so ``frames.rgb[i]``, ``frames.depth[i]``, and
``frames.world_to_camera[i]`` always describe the same frame.

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("scannetppv2", "/path/to/scannetpp")
   scene_id = dataset.get_ids("nvs_sem_val")[0]

   frames = dataset.get_frames(
       scene_id,
       source="iphone",
       indices=[0, 30, 60],
       items=("rgb", "depth", "depth_intrinsics", "world_to_camera"),
   )

Choosing frames
---------------

``source``
   Many datasets record several streams, for example ScanNet++'s iPhone video,
   DSLR photos, and panoramas. Each dataset page lists its sources, and
   ``dataset.spec.frame_sources`` lists them in code. Datasets with a single
   stream need no ``source``.

``indices``
   Frame numbers in the source, such as video frame numbers. The result keeps
   your order. Requesting a frame without a valid camera pose raises
   :class:`ValueError`.

``step``
   Without ``indices``, every frame with a valid camera is used. ``step=10``
   keeps every tenth of them. ``indices`` and ``step`` cannot be combined.

``items``
   The :class:`~scenezoo.FrameBatch` fields to load. Only these are read from
   disk, so ask for what you need. The default is
   ``("rgb", "rgb_intrinsics", "world_to_camera")``.

Some datasets add selectors. Structured3D requires ``room_id=``, ScanNet++
DSLR accepts filenames through ``frame_keys=``, and ScanNet can return 2D
labels from its ``raw`` or ``filtered`` annotation archives. These are
documented on the dataset pages.

What a frame batch contains
---------------------------

Fields that were not requested, or that a dataset does not have, are ``None``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Content
   * - ``indices``
     - ``(N,)`` source frame numbers of the returned frames.
   * - ``frame_keys``
     - Source names, such as image filenames, when the dataset has them.
   * - ``rgb``
     - ``(N, H, W, 3)`` ``uint8`` color images.
   * - ``depth``
     - ``(N, H, W)`` ``float32`` depth in **metres**; ``0`` means no
       measurement.
   * - ``rgb_intrinsics``, ``depth_intrinsics``
     - ``(N, 3, 3)`` pinhole camera matrices for the RGB and depth images.
   * - ``world_to_camera``
     - ``(N, 4, 4)`` camera poses (see `Camera conventions`_).
   * - ``timestamps``
     - Capture times in the source's units.
   * - ``instance_maps``, ``semantic_maps``
     - ``(N, H, W)`` integer 2D labels, where the dataset provides them.
   * - ``confidence_maps``
     - ``(N, H, W)`` depth confidence (ARKitScenes, MultiScan).
   * - ``frame_mask``, ``is_bad``
     - Valid-pixel masks and per-frame quality flags, where available.
   * - ``camera_model``, ``distortion``, ``image_sizes``
     - Camera model name (``"PINHOLE"`` for most sources), distortion
       coefficients for non-pinhole sources, and image sizes.
   * - ``albedo``, ``normal_maps``, ``imu``, ``exposure_durations``, ``azimuth``, ``elevation``
     - Extra source-specific data, described on the dataset pages.

Camera conventions
------------------

All datasets are converted to the same conventions:

- **World coordinates** are z-up and in metres for every dataset; see
  :ref:`coordinate-system`.
- **Poses** are ``world_to_camera`` matrices in the OpenCV convention: the
  camera looks along +Z, +X points right, and +Y points down in the image. Use
  ``np.linalg.inv(world_to_camera)`` for the camera-to-world pose (the camera
  position is its last column).
- **Intrinsics** are 3x3 pinhole matrices ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``
  for the returned image size. Integer pixel coordinates are pixel centres.
- **Depth** is ``float32`` metres along the camera Z axis for pinhole
  cameras. For Aria's fisheye camera and for panoramas, which have no single
  Z axis, depth is the distance along each pixel's viewing ray, and
  ``frames.depth_mode`` is ``"ray_distance"``.
- **Images** are top-left origin, row-major arrays, as usual for NumPy.
- **Normal maps** (``normal_maps``) hold unit normals in the same OpenCV
  camera axes, pointing towards the camera, whatever axes the dataset stores
  them in.

With these conventions, a pixel ``(u, v)`` with depth ``d`` is at camera point
``d * inv(K) @ [u, v, 1]``, and a world point ``X`` lands on pixel
``K @ (R @ X + t)`` after dividing by depth:

.. code-block:: python

   import numpy as np

   K = frames.depth_intrinsics[0]
   world_to_camera = frames.world_to_camera[0]
   depth = frames.depth[0]

   # Back-project every valid depth pixel to world coordinates.
   v, u = np.nonzero(depth > 0)
   pixels = np.stack([u, v, np.ones_like(u)])
   camera_points = (np.linalg.inv(K) @ pixels) * depth[v, u]
   camera_to_world = np.linalg.inv(world_to_camera)
   R_cw, t_cw = camera_to_world[:3, :3], camera_to_world[:3, 3]
   world_points = camera_points.T @ R_cw.T + t_cw

   # Project them back: we recover the original pixels.
   R, t = world_to_camera[:3, :3], world_to_camera[:3, 3:]
   projected = K @ (R @ world_points.T + t)
   print(np.abs(projected[:2] / projected[2] - np.stack([u, v])).max())  # ~0

Resizing, cropping, and rotating
--------------------------------

``output_size=(width, height)``
   Resizes every image item, including depth and label maps, and updates the
   intrinsics to match. Color uses high-quality interpolation; depth and label
   maps use nearest-neighbor so values are never blended.

``center_crop=True``
   With ``output_size``, keeps the aspect ratio by resizing and then cropping
   the center instead of stretching.

``rotate_to_up=True`` (default)
   Some captures are stored sideways, for example portrait phone videos. With
   this option the images are rotated upright. The rotation is applied to the
   camera too, so ``world_to_camera`` changes with the images and the
   intrinsics stay a plain pinhole matrix. Pass ``rotate_to_up=False`` to get
   the images exactly as stored. Datasets that are already upright ignore this
   option.

.. code-block:: python

   frames = dataset.get_frames(
       scene_id,
       source="iphone",
       indices=[0, 30],
       items=("rgb", "depth", "rgb_intrinsics", "depth_intrinsics"),
       output_size=(320, 240),
       center_crop=True,
   )
   print(frames.rgb.shape, frames.depth.shape)  # (2, 240, 320, 3) (2, 240, 320)

Non-pinhole cameras
-------------------

Some sources do not use an ideal pinhole camera. Their frames keep the source's
camera model instead of approximating it:

- ``camera_model`` names the model, for example ``"OPENCV"`` (Matterport3D raw
  images), ``"OPENCV_FISHEYE"`` (ScanNet++ DSLR), ``"FISHEYE624"`` (Aria), or
  ``"EQUIRECTANGULAR"`` (panoramas).
- ``distortion`` holds the coefficients in the order the model defines, and
  ``image_sizes`` holds the image size of each frame.
- Panoramas have no 3x3 intrinsics; ScanNet++ panoramas provide per-pixel
  ``azimuth`` and ``elevation`` instead.

If you need pinhole images, prefer an undistorted source where the dataset has
one, such as ScanNet++ ``dslr_undistorted`` or Matterport3D ``undistorted``.
