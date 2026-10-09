ScanNet v2
==========

RGB-D videos of 1,500+ indoor scenes with reconstructed meshes, camera poses,
and instance and semantic annotations.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``scannetv2`` (alias ``scannet``)
   * - Samples
     - Scenes, for example ``scene0000_00``
   * - Splits
     - ``train`` (1,201), ``val`` (312), ``test`` (100); the official v2 lists
       are downloaded once from the ScanNet repository
   * - Coordinates
     - z-up, metres, as released; see :ref:`coordinate-system`
   * - Data
     - Meshes, vertex segmentation, boxes, RGB-D frames with 2D labels and IMU

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("scannetv2", "/path/to/scannet")
   scene_id = dataset.get_ids("val")[0]

   mesh = dataset.get_mesh(scene_id)                       # decimated mesh
   semantics = dataset.get_segmentation(scene_id, segmentation_type="semantic")
   frames = dataset.get_frames(
       scene_id,
       indices=[0, 100],
       items=("rgb", "depth", "world_to_camera", "semantic_maps"),
       semantic_label_space="nyu40",
   )

Download layout
---------------

Use the folder created by the official ScanNet download script:

.. code-block:: text

   scannet/
   └── scans/
       └── scene0000_00/
           ├── scene0000_00.sens                       # RGB-D video and cameras
           ├── scene0000_00.txt                        # scene metadata
           ├── scene0000_00_vh_clean.ply               # raw mesh
           ├── scene0000_00_vh_clean_2.ply             # decimated mesh
           ├── scene0000_00_vh_clean_2.labels.ply      # semantic mesh
           ├── scene0000_00.aggregation.json           # instance annotations
           ├── scene0000_00_vh_clean_2.0.010000.segs.json
           ├── scene0000_00_2d-label-filt.zip          # 2D labels (optional)
           └── scene0000_00_2d-instance-filt.zip       # 2D instances (optional)

Frames are read directly from each ``.sens`` file; you do not need the
separate ``scannet_frames_25k`` export, and the 2D label ZIPs stay compressed.
Run ``dataset.check()`` to confirm that the scenes you need are complete.

SceneZoo reads every scene from ``scans/``. To use test scenes, move or
symlink their folders into ``scans/``. Test scenes have no annotations.

Meshes and labels
-----------------

``get_mesh`` returns the ``"decimated"`` mesh by default; ``"raw"`` (full
resolution) and ``"semantic"`` (decimated, colored by class) are also
available.

``get_segmentation`` returns per-vertex labels for the raw or decimated mesh
(``mesh_type=``). ``segmentation_type="instance"`` (default) gives object IDs,
and ``"semantic"`` gives NYU40 class IDs. ``get_boxes`` fits a box to each
annotated object, on the same choice of mesh.

.. code-block:: python

   instances = dataset.get_segmentation(scene_id, mesh_type="raw")
   boxes, names = dataset.get_boxes(scene_id, box_type="aabb")

Frames
------

``get_frames`` reads RGB, depth, both intrinsics, poses, timestamps, IMU, and
the official 2D instance and semantic label images:

- ``annotation_variant="filtered"`` (default) or ``"raw"`` selects which 2D
  label archives are used.
- 2D instance IDs match the 3D object IDs; background becomes
  ``invalid_obj_id``.
- 2D semantic labels are ScanNet label IDs; pass
  ``semantic_label_space="nyu40"`` to get NYU40 classes.
- ``timestamps`` has shape ``(N, 2)``: the color and depth timestamps in
  microseconds.
- ``imu`` holds the IMU sample nearest to each frame and a ``valid`` mask;
  ``get_imu(scene_id)`` returns the complete IMU stream.

ScanNet scenes are already upright, so ``rotate_to_up`` has no effect.

Calibration and metadata
------------------------

- ``get_scene_info(scene_id)`` parses the scene's ``.txt`` file.
- ``get_alignment(scene_id)`` returns its 4x4 ``axisAlignment`` matrix, which
  aligns the scene with the coordinate axes.
- ``get_calibration(scene_id)`` returns the native intrinsics, image sizes,
  depth scale, and the RGB-to-depth transform.
- The label map ``scannetv2-labels.combined.tsv`` is read from the dataset
  folder (or its ``tasks/`` folder) when present, and downloaded once
  otherwise. ``metadata["label_mapping"]`` holds its rows by category name.

Official resources
------------------

* `Dataset homepage <http://www.scan-net.org/ScanNet/>`__
* `Benchmark and data documentation <https://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`__
* `Official tools and data format <https://github.com/ScanNet/ScanNet>`__
* `Paper (CVPR 2017) <https://openaccess.thecvf.com/content_cvpr_2017/html/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.html>`__

API reference
-------------

**Options.** Every path is a template relative to the dataset folder and can
be changed through ``get_dataset`` keyword arguments; the adapter stores them
as ``mesh_files``, ``aggregation_files``, ``segmentation_files``,
``alignment_file``, ``sens_file``, and ``projection_archives``.
``label_mapping_file`` and ``split_files`` override the label map and split
lists.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Returns
   * - ``get_mesh``
     - ``raw``, ``decimated`` (default), or ``semantic`` mesh.
   * - ``get_segmentation``
     - Instance or NYU40 semantic vertex labels.
   * - ``get_boxes``
     - Boxes fitted to annotated objects.
   * - ``get_frames``
     - RGB-D frames, cameras, timestamps, IMU, and 2D labels.
   * - ``get_scene_info``
     - All fields of the scene metadata file.
   * - ``get_alignment``
     - The scene's 4x4 axis-alignment matrix.
   * - ``get_calibration``
     - Native intrinsics, image sizes, depth scale, and sensor transforms.
   * - ``get_imu``
     - The complete IMU stream.

.. autoclass:: scenezoo.dataset.scene.ScanNet
   :members:
   :inherited-members:
