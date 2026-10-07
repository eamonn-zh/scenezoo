MultiScan
=========

Indoor scenes scanned repeatedly with an iPad, annotated with objects, object
parts, and how parts move (articulations).

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``multiscan``
   * - Samples
     - Scans, for example ``scene_00000_00`` (scene 0, scan 0)
   * - Splits
     - ``train``, ``val``, ``test``; the official list is installed with
       SceneZoo
   * - Data
     - Meshes, face segmentation of objects and parts, boxes, articulations,
       RGB-D frames with confidence

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("multiscan", "/path/to/multiscan")
   scan_id = dataset.get_ids("val")[0]

   mesh = dataset.get_mesh(scan_id)
   parts = dataset.get_segmentation(scan_id, segmentation_type="part_semantic")
   boxes, names = dataset.get_boxes(scan_id)
   frames = dataset.get_frames(
       scan_id,
       indices=[0, 100],
       items=("rgb", "depth", "confidence_maps", "world_to_camera"),
   )

Download layout
---------------

One folder per scan, named after the scan:

.. code-block:: text

   multiscan/
   └── scene_00000_00/
       ├── scene_00000_00.ply                 # mesh with face annotations
       ├── scene_00000_00.annotations.json    # objects, parts, articulations
       ├── scene_00000_00.mp4                 # RGB video
       ├── scene_00000_00.depth.zlib          # depth stream
       ├── scene_00000_00.confidence.zlib     # depth confidence stream
       ├── scene_00000_00.jsonl               # per-frame cameras
       ├── scene_00000_00.json                # capture metadata
       ├── scene_00000_00.align.json          # alignment transforms
       └── textured_mesh/scene_00000_00.obj   # optional textured mesh

Extract the released scan archives and keep the original file names. Run
``dataset.check()`` to confirm the scans you need are complete.

Meshes and labels
-----------------

- ``get_mesh`` returns the annotated ``"ply"`` mesh (default) or the
  ``"textured"`` OBJ.
- ``get_segmentation`` labels each **face** of the PLY mesh. Choose
  ``segmentation_type`` from ``"object_instance"`` (default),
  ``"part_instance"``, ``"object_semantic"``, and ``"part_semantic"``.
  Unlabeled faces have ID ``0``.
- ``get_boxes`` fits a box to every object; ``box_type="obb_gt"`` returns the
  annotated oriented boxes instead.
- ``get_annotations`` returns objects, parts, and the scan's bounding box, and
  ``get_articulations`` returns each movable part with its motion type, axis,
  and origin.

Frames
------

``get_frames`` reads RGB from the MP4 video and depth and confidence from their
compressed streams, together with intrinsics, poses, timestamps, and exposure
durations. Depth is in metres; confidence keeps the original integer values.

Depth and confidence are each stored as one continuous compressed stream, so
reading a late frame still has to decompress everything before it. Reading
frames in increasing order is fastest.

``rotate_to_up=True`` (default) turns sideways captures upright using the
scan's recorded device orientation. ``get_camera_metadata`` returns the raw
per-frame camera records, including orientation quaternions and Euler angles.

Alignment and metadata
----------------------

- ``get_alignment(scan_id)`` returns the 4x4 transform from the raw mesh to the
  gravity-aligned scene.
- ``get_reference_alignment(scan_id)`` returns ``(reference_scan_id,
  transform)`` relating a rescan to its reference scan, or ``None``.
- ``get_scene_info(scan_id)`` returns the capture metadata.
- ``metadata`` holds the object and part label maps and the device orientation
  of every scan.

Official resources
------------------

* `Project page <https://3dlg-hcvc.github.io/multiscan/>`__
* `Official repository <https://github.com/smartscenes/multiscan>`__
* `Dataset download <https://huggingface.co/datasets/3dlg-hcvc/MultiScan>`__
* `Data format documentation <https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/index.html>`__
* `Paper (NeurIPS 2022) <https://openreview.net/forum?id=YxUdazpgweG>`__

API reference
-------------

**Options.** File templates are stored in ``mesh_files``, ``annotation_file``,
``rgb_video_file``, ``depth_file``, ``confidence_file``, ``alignment_file``,
``camera_metadata_file``, and ``camera_file``. ``split_file``,
``object_label_file``, and ``part_label_file`` default to copies included in
SceneZoo.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Returns
   * - ``get_mesh``
     - ``ply`` (default) or ``textured`` mesh.
   * - ``get_segmentation``
     - Object or part, instance or semantic face labels.
   * - ``get_boxes``
     - Fitted or annotated object boxes.
   * - ``get_frames``
     - RGB-D frames, confidence, cameras, timestamps, exposure.
   * - ``get_camera_metadata``
     - Raw per-frame camera records.
   * - ``get_annotations``, ``get_articulations``
     - Object, part, and articulation annotations.
   * - ``get_alignment``, ``get_reference_alignment``
     - Alignment to the upright scene and to the reference scan.
   * - ``get_scene_info``
     - Capture metadata.

.. autoclass:: scenezoo.dataset.scene.MultiScan
   :members:
   :inherited-members:
