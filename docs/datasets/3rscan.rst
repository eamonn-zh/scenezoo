3RScan
======

Indoor rooms scanned several times over months, with object instances matched
across the scans and the changes between them annotated.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``3rscan`` (alias ``threerscan``)
   * - Samples
     - Scans, identified by UUID, for example
       ``02b33dfb-be2b-2d54-92d2-cd012b2b3c40``
   * - Splits
     - ``train`` (1,178), ``val`` (157), ``test`` (147), from ``3RScan.json``
   * - Coordinates
     - z-up, metres, as released; see :ref:`coordinate-system`
   * - Data
     - Meshes, vertex segmentation in six label spaces, boxes, RGB-D frames,
       scan-to-reference alignment

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("3rscan", "/path/to/3rscan")
   scan_id = dataset.get_ids("val")[0]

   mesh = dataset.get_mesh(scan_id)
   labels = dataset.get_segmentation(scan_id, segmentation_type="nyu40")
   boxes, names = dataset.get_boxes(scan_id)
   frames = dataset.get_frames(
       scan_id,
       indices=[0, 2],
       items=("rgb", "depth", "depth_intrinsics", "world_to_camera"),
   )

Download layout
---------------

Pass the folder that contains ``3RScan.json`` and ``download/``, or the
``download/`` folder itself:

.. code-block:: text

   3rscan/
   ├── 3RScan.json                            # scan index and changes
   └── download/<scan_id>/
       ├── mesh.refined.v2.obj                # textured mesh
       ├── labels.instances.annotated.v2.ply  # instance mesh
       ├── labels.instances.align.annotated.v2.ply
       ├── semseg.v2.json                     # instance annotations
       ├── mesh.refined.0.010000.segs.json    # mesh segments
       └── sequence.zip                       # RGB-D frames and poses

Extract the downloaded archives, but **keep** ``sequence.zip`` compressed:
SceneZoo reads frames from it directly. If ``3RScan.json`` is missing, it is
downloaded once. Run ``dataset.check()`` to confirm the scans you need are
complete.

Scans and rescans
-----------------

Each room has one **reference** scan and several **rescans** taken later. The
official splits in ``3RScan.json`` include both. (The ``splits/*.txt`` files in
the official repository list only reference scans and are not used.)

- ``get_reference_id(scan_id)`` returns the room's reference scan, and
  ``get_scene_info(scan_id)`` reports ``is_reference`` and ``has_annotations``.
- ``get_scene_alignment(scan_id)`` returns the 4x4 transform (in metres) from a
  rescan to its reference scan.
- ``get_changes(scan_id)`` returns the annotated object changes: moved,
  deformed, and removed objects, and ambiguities.
- ``get_scene_info(scan_id)`` returns the split, reference, and annotation
  status of a scan.
- ``get_scene_graph(scan_id)`` returns the scan's 3DSSG scene graph, which the
  3RScan authors publish separately: each object's label, attributes (color,
  shape, state, ...), and affordances, keyed by the instance IDs of
  ``get_segmentation``, and ``(subject_id, object_id, predicate_id, predicate)``
  relationships such as ``"standing on"``. The two JSON files are downloaded
  once from the 3DSSG site; ``scene_graph_files`` holds their locations.

The 101 hidden test rescans have no published annotations or alignment;
requesting those raises :class:`~scenezoo.UnsupportedOperationError`, while
their meshes and frames remain available.

Meshes and labels
-----------------

- ``get_mesh`` returns the ``"instance"`` mesh by default (or ``"textured"``
  where no annotations exist). ``"instance_aligned"`` is the instance mesh in
  the reference scan's coordinates.
- ``get_segmentation`` labels the vertices of the instance mesh. Choose
  ``segmentation_type`` from ``"instance"`` (default), ``"global"``,
  ``"nyu40"``, ``"eigen13"``, ``"rio27"``, and ``"rio7"``. The label mapping is
  the official table linked from the 3RScan repository, downloaded once.
- ``get_boxes`` fits boxes to objects; ``box_type="obb_gt"`` returns the
  annotated oriented boxes.
- ``get_annotations`` and ``get_oversegmentation`` return the raw instance
  annotations and mesh segments.

Frames
------

``get_frames`` reads RGB, depth (converted to metres), both intrinsics, and
poses from ``sequence.zip``. Frames are stored sideways; ``rotate_to_up=True``
(default) turns them upright and rotates the poses to match. Pass
``coordinate_space="reference"`` to express poses in the reference scan's
coordinates. ``get_calibration`` returns the native image sizes, depth scale,
intrinsics, and extrinsics.

Official resources
------------------

* `Dataset homepage <https://vmnavab26.in.tum.de/3RScan/>`__
* `Data format and access <https://vmnavab26.in.tum.de/3RScan/documentation.php>`__
* `Official tools <https://github.com/WaldJohannaU/3RScan>`__
* `Paper (ICCV 2019) <https://openaccess.thecvf.com/content_ICCV_2019/html/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.html>`__
* `FAQ <https://github.com/WaldJohannaU/3RScan/blob/master/FAQ.md>`__
* `3DSSG scene graphs <https://3dssg.github.io/>`__ and `paper (CVPR 2020) <https://arxiv.org/abs/2004.03967>`__

API reference
-------------

**Options.** ``scans_dir`` is the detected ``download`` folder. File templates
are stored in ``mesh_files``, ``aggregation_file``, ``oversegmentation_file``,
and ``sequence_archive``. ``dataset_metadata_file`` locates ``3RScan.json`` and
``label_mapping_file`` the semantic label mapping. Unlabeled vertices have ID
``0`` (``invalid_obj_id``).

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Returns
   * - ``get_mesh``
     - ``instance`` (default), ``instance_aligned``, or ``textured`` mesh.
   * - ``get_segmentation``
     - Vertex labels in one of six label spaces.
   * - ``get_boxes``
     - Fitted or annotated object boxes.
   * - ``get_frames``
     - RGB-D frames and cameras from ``sequence.zip``.
   * - ``get_calibration``
     - Native image sizes, depth scale, intrinsics, and extrinsics.
   * - ``get_annotations``, ``get_oversegmentation``
     - Raw instance annotations and mesh segments.
   * - ``get_scene_info``, ``get_reference_id``
     - Split, reference, and annotation status.
   * - ``get_scene_alignment``
     - Rescan-to-reference transform.
   * - ``get_changes``
     - Object change annotations.
   * - ``get_scene_graph``
     - 3DSSG objects, attributes, and relationships.

.. autoclass:: scenezoo.dataset.scene.ThreeRScan
   :members:
   :inherited-members:
