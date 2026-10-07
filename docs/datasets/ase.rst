Aria Synthetic Environments
===========================

Meta's large synthetic dataset of indoor scenes seen through a Project Aria
headset: fisheye RGB, depth, instance images, ground-truth trajectories,
semi-dense SLAM points, and a procedural description of each room layout.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``ase`` (aliases ``aria-ase``, ``aria-synthetic-environments``)
   * - Samples
     - Sequences, named by split and number, for example ``train/83788``
   * - Splits
     - ``train``, ``test`` (extracted sequences only)
   * - Data
     - Semi-dense points, wall/door/window layout boxes, fisheye RGB-D frames
       with instance images and poses (no meshes or 3D segmentation)

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("ase", "/path/to/ase")
   sequence_id = dataset.get_ids("train")[0]

   frames = dataset.get_frames(
       sequence_id,
       indices=[0, 20],
       items=("rgb", "depth", "instance_maps", "world_to_camera"),
   )
   points = dataset.get_points(sequence_id, max_inverse_distance_std=0.005)
   boxes, names = dataset.get_boxes(sequence_id)  # walls, doors, and windows
   print(frames.camera_model, frames.depth_mode)  # FISHEYE624 ray_distance

Download layout
---------------

The official downloader fetches chunk ZIPs of ten sequences each, such as
``train_chunk_0008378.zip``. **Extract them** before use; SceneZoo does not
read sequences from the ZIPs:

.. code-block:: text

   ase/
   ├── train/83788/                  # or a bare 83788/ (see below)
   │   ├── rgb/vignette0000000.jpg
   │   ├── depth/depth0000000.png
   │   ├── instances/instance0000000.png
   │   ├── trajectory.csv
   │   ├── semidense_points.csv.gz
   │   ├── semidense_observations.csv.gz
   │   ├── object_instances_to_classes.json
   │   └── ase_scene_language.txt
   └── test/20/

Train and test both use small numbers, so SceneZoo always names sequences
with their split: ``train/83788``, ``test/20``. The official extractor creates
bare numbered folders; these are treated as training data unless you pass
``extracted_split="test"``. To keep both splits in one place, use ``train/`` and
``test/`` subfolders as above. ``dataset.check()`` reports sequences that are
still only zipped.

Test sequences have only RGB, trajectories, and semi-dense data. Depth,
instance images, and layouts are training-only; requesting them for a test
sequence raises :class:`~scenezoo.UnsupportedOperationError`.

Fisheye frames and depth
------------------------

Frames come from Aria's fisheye RGB camera (``camera_model="FISHEYE624"``).
``rgb_intrinsics`` holds the focal length and principal point, and
``distortion`` holds the twelve fisheye coefficients; ``frame_mask`` marks the
valid image circle. ``get_calibration()`` returns the full fixed calibration.

Depth is in metres but measured **along each pixel's ray**, not along the
camera Z axis (``frames.depth_mode == "ray_distance"``), so pinhole
back-projection does not apply.

The stored images are rotated 90 degrees. With ``rotate_to_up=True`` (default)
they are turned upright, and the pose, intrinsics, and distortion are rotated
with them so the calibration still matches the pixels exactly. Pass
``rotate_to_up=False`` for the images as stored.

Points, layout, and trajectories
--------------------------------

- ``get_points`` returns the semi-dense SLAM points with their uncertainties;
  ``max_inverse_distance_std`` and ``max_distance_std`` drop uncertain points.
- ``get_semidense_observations`` returns which frames observed which points,
  optionally filtered by ``frame_indices`` or ``point_ids``.
- ``get_boxes`` turns the walls, doors, and windows of ``ase_scene_language.txt``
  into oriented boxes; ``get_scene_commands`` returns the parsed commands.
- ``get_trajectory`` returns timestamps, device poses, and camera poses.
- ``get_instance_classes`` maps instance-image IDs to class names.

Official resources
------------------

* `Dataset homepage and access <https://www.projectaria.com/datasets/ase/>`__
* `Data format <https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format>`__
* `Tools and visualization <https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_tools>`__
* `Project Aria Tools <https://github.com/facebookresearch/projectaria_tools>`__
* `SceneScript paper (ECCV 2024) <https://arxiv.org/abs/2403.13064>`__

API reference
-------------

**Options.** ``data_dir`` selects the folder of extracted sequences and ZIPs;
``data_root`` is the resolved folder. ``manifest_file`` locates the official
download manifest, and ``extracted_split`` sets the split of bare numbered
folders.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Method
     - Returns
   * - ``get_frames``
     - Fisheye RGB, ray depth, instance images, masks, and poses.
   * - ``get_calibration``
     - The fixed Fisheye624 calibration and camera extrinsic.
   * - ``get_points``
     - Semi-dense points (:class:`~scenezoo.dataset.scene.ASEPointData`).
   * - ``get_semidense_observations``
     - Point observations per frame.
   * - ``get_boxes``, ``get_scene_commands``
     - Layout boxes; the parsed scene language.
   * - ``get_trajectory``
     - Timestamps, device poses, and camera poses.
   * - ``get_instance_classes``
     - Instance ID to class name.
   * - ``get_available_ids``
     - Sequences that are extracted on disk.

.. autoclass:: scenezoo.dataset.scene.AriaSyntheticEnvironments
   :members:
   :inherited-members:

.. autoclass:: scenezoo.dataset.scene.ASEPointData
   :members:
