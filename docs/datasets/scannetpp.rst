ScanNet++ v2
============

High-fidelity indoor scenes with laser scans, DSLR photos, iPhone RGB-D video,
panoramas, and dense semantic and instance annotations.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``scannetppv2`` (aliases ``scannet++``, ``scannetpp``, ``scannet++v2``)
   * - Samples
     - Scenes, for example ``39f36da05b``
   * - Splits
     - ``nvs_sem_train``, ``nvs_sem_val``, ``nvs_test``, ``sem_test``,
       ``nvs_test_small``, ``nvs_test_iphone``; read from the release's
       ``splits/`` folder (files that are absent are skipped)
   * - Data
     - Meshes, laser point clouds, vertex segmentation, boxes, and frames from
       iPhone, DSLR, and panorama cameras

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("scannetppv2", "/path/to/scannetpp")
   scene_id = dataset.get_ids("nvs_sem_val")[0]
   print(dataset.get_frame_sources(scene_id))  # sources downloaded here

   mesh = dataset.get_mesh(scene_id)
   semantics = dataset.get_segmentation(scene_id, segmentation_type="semantic")
   iphone = dataset.get_frames(
       scene_id,
       source="iphone",
       indices=[0, 60],
       items=("rgb", "depth", "world_to_camera", "timestamps"),
   )
   dslr = dataset.get_frames(
       scene_id,
       source="dslr_undistorted",
       indices=[0, 1],
       items=("rgb", "rgb_intrinsics", "world_to_camera"),
   )

Download layout
---------------

Pass the folder that contains ``data/``, ``metadata/``, and ``splits/``, as
created by the official download script:

.. code-block:: text

   scannetpp/
   ├── data/<scene_id>/
   │   ├── scans/       # meshes, laser scan, annotations
   │   ├── iphone/      # rgb.mkv, depth.bin, rgb_mask.mkv, poses, COLMAP
   │   ├── dslr/        # images, COLMAP cameras, Nerfstudio transforms
   │   └── panocam/     # panoramas (optional)
   ├── metadata/        # label maps and class lists
   └── splits/          # split lists

Download only the capture types you need; ``get_frame_sources(scene_id)`` tells
you which are present. Extract the official archives before use, and run
``dataset.check()`` to confirm the scenes you need are complete.

Meshes, points, and labels
--------------------------

- ``get_mesh`` returns the aligned ``"raw"`` mesh (default) or the
  ``"semantic"`` mesh colored by class.
- ``get_points`` returns the laser scan as a :class:`~scenezoo.PointBatch`;
  ``get_point_cloud`` returns it as an Open3D point cloud.
- ``get_segmentation`` returns per-vertex ``"instance"`` (default) or
  ``"semantic"`` labels. ScanNet++ objects can overlap; pass
  ``multilabel=True`` to keep every label of a vertex (``labels`` then has one
  column per label, padded with ``invalid_id``).
- ``get_boxes`` fits a box to every annotated object.
- ``get_anonymization_indices(scene_id, asset="mesh")`` returns the vertices
  (or, with ``asset="point_cloud"``, the laser points) that were blurred for
  privacy; they carry no labels.

Frame sources
-------------

Pass ``source=`` to ``get_frames``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Source
     - Frames
   * - ``iphone``
     - iPhone video at 60 FPS with LiDAR depth (256x192), poses, timestamps,
       IMU, and valid-pixel masks. Default.
   * - ``iphone_colmap``
     - The iPhone frames registered by COLMAP, with their distortion model.
   * - ``dslr``
     - Resized DSLR photos with their fisheye camera model.
   * - ``dslr_undistorted``
     - The same photos undistorted to a pinhole camera.
   * - ``dslr_original``
     - Full-resolution DSLR photos.
   * - ``panocam``, ``panocam_resized``
     - Panoramas with depth and per-pixel ``azimuth`` and ``elevation``.

For DSLR sources:

- ``view_split="train"`` or ``"test"`` selects the official novel-view
  benchmark views (default ``"all"``).
- ``frame_keys=["DSC01234.JPG", ...]`` selects photos by filename instead of
  ``indices``.
- ``include_bad=False`` drops photos the release marks as bad; the ``is_bad``
  item returns the flags.

Non-pinhole sources keep their camera model in ``camera_model`` and
``distortion``; see :doc:`../guide/frames`. ``get_panorama_point_cloud``
back-projects one panorama into a colored point cloud.

Other files
-----------

``get_scanner_poses``, ``get_iphone_exif``, ``get_dslr_train_test_lists``,
``get_colmap_cameras``, ``get_colmap_images``, ``get_colmap_sparse_point_cloud``,
and ``get_nerfstudio_transforms`` read the corresponding release files.
``metadata`` holds the label maps, class lists, scene types, and top-100
benchmark classes.

Official resources
------------------

* `Dataset homepage and access <https://scannetpp.mlsg.cit.tum.de/scannetpp/>`__
* `Data documentation <https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation>`__
* `Official toolkit <https://github.com/scannetpp/scannetpp>`__
* `Paper (ICCV 2023) <https://openaccess.thecvf.com/content/ICCV2023/html/Yeshwanth_ScanNet_A_High-Fidelity_Dataset_of_3D_Indoor_Scenes_ICCV_2023_paper.html>`__

API reference
-------------

**Options.** ``data_dir`` is the scene folder template (default
``data/{scene_id}``). ``metadata_files`` and ``split_files`` store the label
and split file locations, which can be overridden with ``get_dataset``
keyword arguments such as ``train_split_file=``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - ``get_mesh``
     - ``raw`` (default) or ``semantic`` mesh.
   * - ``get_points``, ``get_point_cloud``
     - The aligned laser scan.
   * - ``get_segmentation``
     - Instance or semantic vertex labels, optionally multilabel.
   * - ``get_boxes``
     - Boxes fitted to annotated objects.
   * - ``get_frames``, ``get_frame_sources``
     - Frames of one capture source; the sources present for a scene.
   * - ``get_panorama_point_cloud``
     - One panorama back-projected into 3D.
   * - ``get_anonymization_indices``
     - Anonymized mesh vertices or laser points.
   * - ``get_scanner_poses``
     - ``(N, 4, 4)`` laser scanner poses.
   * - ``get_iphone_exif``
     - iPhone EXIF metadata.
   * - ``get_dslr_train_test_lists``
     - Official DSLR novel-view split.
   * - ``get_colmap_cameras``, ``get_colmap_images``, ``get_colmap_sparse_point_cloud``
     - COLMAP cameras, image poses, and sparse points.
   * - ``get_nerfstudio_transforms``
     - Nerfstudio ``transforms.json``.

.. autoclass:: scenezoo.dataset.scene.ScanNetPP
   :members:
   :inherited-members:
