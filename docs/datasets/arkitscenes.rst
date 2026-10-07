ARKitScenes
===========

Apple's indoor RGB-D dataset captured with iPad and iPhone LiDAR, with 3D
object boxes, high-resolution depth, and registered laser scans.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``arkitscenes``
   * - Samples
     - Video captures, for example ``40753679``
   * - Splits
     - ``train_raw``, ``val_raw``, ``train_threedod``, ``val_threedod``,
       ``train_depth_upsampling``, ``val_depth_upsampling``: the official
       Training and Validation lists of each of the three releases
   * - Data
     - Meshes, box-derived vertex segmentation, 3D boxes, RGB-D frames from six
       camera sources, laser scans

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("arkitscenes", "/path/to/ARKitScenes")
   video_id = dataset.get_ids("val_raw")[0]
   print(dataset.get_frame_sources(video_id))

   mesh = dataset.get_mesh(video_id)
   boxes, names = dataset.get_boxes(video_id)
   frames = dataset.get_frames(
       video_id,
       source="mov",
       step=300,
       items=("rgb", "depth", "depth_intrinsics", "world_to_camera"),
       output_size=(640, 480),
   )

Download layout
---------------

Pass the folder that holds the downloaded ARKitScenes parts, or one of the
``raw``, ``threedod``, or ``depth_upsampling`` folders directly:

.. code-block:: text

   ARKitScenes/
   ├── raw/
   │   ├── metadata.csv
   │   ├── Training/<video_id>/
   │   │   ├── <video_id>.mov                  # high-resolution RGB video
   │   │   ├── <video_id>_3dod_mesh.ply
   │   │   ├── <video_id>_3dod_annotation.json
   │   │   ├── lowres_wide.traj                # camera trajectory
   │   │   ├── lowres_wide_intrinsics/
   │   │   └── lowres_depth/, confidence/, wide/, ...
   │   └── Validation/<video_id>/
   ├── laser_scanner_point_clouds/<visit_id>/     # optional FARO scans
   └── laser_scanner_point_clouds_mapping.csv

Download only the parts you need; ``get_frame_sources(video_id)`` lists the
camera sources present for a capture. Extract the downloads before use, and run
``dataset.check()`` to confirm the captures you need are complete.

With ``raw/metadata.csv`` present, splits are read locally; otherwise the
official split lists are downloaded once.

Frame sources
-------------

Pass ``source=`` to ``get_frames``. Frame indices count frames within the
chosen source, and ``frame_keys`` keeps the original file names.

.. list-table::
   :header-rows: 1
   :widths: 18 34 32 16

   * - Source
     - RGB
     - Also available
     - Rate
   * - ``mov`` (default)
     - High-resolution video
     - low-resolution depth, confidence, cameras
     - 60 FPS
   * - ``lowres_wide``
     - 256x192 PNG
     - low-resolution depth, confidence, cameras
     - 60 FPS
   * - ``wide``
     - 1920x1440 PNG
     - high- and low-resolution depth, confidence, cameras
     - 10 FPS
   * - ``ultrawide``
     - 640x480 PNG
     - cameras
     - 10 FPS
   * - ``vga_wide``
     - 640x480 PNG
     - cameras
     - 30 FPS
   * - ``threedod``
     - prepared 3DOD frames
     - densified depth, cameras
     - 10 FPS

- ``wide`` returns high-resolution depth by default; pass
  ``depth_type="lowres"`` for the iPad LiDAR depth instead. ``mov`` returns
  low-resolution depth and accepts ``depth_type="highres"``.
- Depth is converted from millimetres to metres; confidence keeps the original
  values 0 to 2.
- Images and cameras are matched by timestamp within ``timestamp_tolerance``
  (default 5 ms).
- The standalone depth-upsampling download has no cameras; request only
  ``("rgb", "depth", "confidence_maps")`` from it.

Upright images
--------------

Captures were recorded in different device orientations. With
``rotate_to_up=True`` (default) images are rotated upright and the camera poses
are rotated with them. The rotation comes from the capture's ``sky_direction``.
Apple's metadata contains some wrong or missing directions, so SceneZoo ships
a human-reviewed correction table: ``metadata[video_id]["sky_direction"]`` is
the value used, ``official_sky_direction`` keeps Apple's value, and
``has_direction_correction`` tells whether it was corrected.

Meshes, boxes, and labels
-------------------------

- ``get_mesh`` returns the 3DOD mesh (the only mesh type, ``"raw"``).
- ``get_boxes`` returns the annotated oriented boxes (``box_type="obb_gt"``).
- ``get_segmentation`` labels each mesh vertex with the box that contains it;
  where boxes overlap, the smaller box wins.
- ``get_annotations`` returns the complete annotation JSON.

Laser scans and depth upsampling
--------------------------------

``get_laser_scanner_ids``, ``get_laser_scanner_poses``, and
``get_laser_scanner_point_clouds`` read the registered FARO scans of the
capture's venue (pass ``scan_ids=`` to select some).
``get_depth_upsampling_attributes`` reads the attributes table of the depth
upsampling benchmark.

Official resources
------------------

* `Project repository <https://github.com/apple-aiml-research/ARKitScenes>`__
* `Data download and format <https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md>`__
* `Raw capture documentation <https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/README.md>`__
* `Paper (NeurIPS 2021 Datasets) <https://openreview.net/forum?id=tjZjv_qh_CE>`__
* `License <https://github.com/apple-aiml-research/ARKitScenes/blob/main/LICENSE>`__

API reference
-------------

**Options.** ``data_dir`` selects the raw folder; ``data_root`` and
``dataset_root`` are the resolved raw and top-level folders, and
``depth_upsampling_root`` the depth-upsampling folder. File templates are
stored in ``metadata_file``, ``mesh_file``, ``rgb_video_file``,
``trajectory_file``, ``annotation_file``, ``laser_mapping_file``,
``laser_scanner_dir``, and ``depth_upsampling_attributes_file``.
``split_files`` holds the split list locations and ``timestamp_tolerance`` the
matching tolerance in seconds.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - ``get_mesh``
     - The 3DOD mesh.
   * - ``get_boxes``, ``get_annotations``
     - Annotated oriented boxes; the full annotation JSON.
   * - ``get_segmentation``
     - Box-derived vertex instance labels.
   * - ``get_frames``, ``get_frame_sources``
     - Frames of one camera source; the sources present for a capture.
   * - ``get_laser_scanner_ids``, ``get_laser_scanner_poses``, ``get_laser_scanner_point_clouds``
     - FARO laser scans of the capture's venue.
   * - ``get_depth_upsampling_attributes``
     - Depth-upsampling benchmark attributes.

.. autoclass:: scenezoo.dataset.scene.ARKitScenes
   :members:
   :inherited-members:
