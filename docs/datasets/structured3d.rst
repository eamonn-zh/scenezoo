Structured3D
============

3,500 synthetic house designs with photorealistic perspective and panoramic
renderings, dense 2D labels, 3D room-layout annotations, and object boxes.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``structured3d`` (alias ``structure3d``)
   * - Samples
     - Scenes, for example ``scene_00000``; rooms inside a scene are selected
       with ``room_id``
   * - Splits
     - ``train`` (scenes 0--2999), ``val`` (3000--3249), ``test``
       (3250--3499)
   * - Data
     - Room-layout meshes, point clouds fused from renderings, point
       segmentation, object boxes, rendered RGB-D views with 2D labels

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("structured3d", "/path/to/Structured3D")
   scene_id = dataset.get_ids("train")[0]
   room_id = dataset.get_room_ids(scene_id)[0]

   frames = dataset.get_frames(
       scene_id,
       room_id=room_id,
       source="perspective",
       items=("rgb", "depth", "semantic_maps", "world_to_camera"),
   )
   points = dataset.get_points(scene_id, room_id, voxel_size=0.02)
   layout = dataset.get_mesh(scene_id, room_id=room_id)
   boxes, names = dataset.get_boxes(scene_id)

Download layout
---------------

Extract all downloaded ZIP parts into one folder of scene directories:

.. code-block:: text

   Structured3D/
   └── data/scene_00000/                   # or Structured3D/scene_00000/
       ├── annotation_3d.json              # room structure
       ├── bbox_3d.json                    # object boxes
       └── 2D_rendering/<room_id>/
           ├── perspective/{empty,full}/<position>/
           └── panorama/{empty,simple,full}/

The ZIP files themselves are not read. Pass ``data_dir=`` if your scene folders
live somewhere else, and run ``dataset.check()`` to confirm the scenes you need
are extracted.

The splits follow the official scene ranges. Six of those scenes are listed as
invalid by the official errata; pass ``include_invalid=False`` to leave them out.

Rendered views
--------------

``get_frames`` requires ``room_id``; ``get_frame_sources(scene_id, room_id)``
lists what is available:

- ``source="perspective"`` (default) for pinhole views, ``"panorama"`` for
  equirectangular panoramas (no intrinsics; see :doc:`../guide/frames`).
- ``configuration`` is ``"full"`` (default, furnished), ``"empty"``, or, for
  panoramas, ``"simple"``.
- ``lighting`` is ``"raw"`` (default) or, for panoramas, ``"cold"`` or
  ``"warm"``.
- Items include ``rgb``, ``depth`` (metres), ``albedo``, ``normal_maps``
  (unit vectors), ``semantic_maps`` and ``instance_maps`` (instance background
  is ``65535``), intrinsics (perspective only), and ``world_to_camera``.

Scenes are rendered upright, so ``rotate_to_up`` has no effect.
``get_layout(scene_id, room_id)`` returns the official 2D room layout of a
view.

Points, labels, and room geometry
---------------------------------

There is no scanned mesh, so 3D data is derived from the renderings and the
structure annotation:

- ``get_points(scene_id, room_id)`` back-projects the room's RGB-D views into
  one colored point cloud with normals, semantic and instance labels, and the
  source view of every point. ``sources=("panorama", "perspective")`` selects
  views, ``voxel_size`` keeps one point per voxel, and ``min_view_cosine``
  drops points seen at grazing angles. ``get_point_cloud`` returns the same
  as an Open3D cloud, and ``get_point_data`` is an alias.
- ``get_segmentation(scene_id, room_id=...)`` returns the matching
  point labels: ``"semantic"`` (default) or ``"instance"``.
  ``semantic_label_space`` is ``"nyu40"`` (default) or ``"pointcept25"``, the
  25 classes used by Pointcept (``valid_class_ids_25``, ``class_names_25``).
- ``get_mesh(scene_id, room_id=...)`` builds the room's floor, walls, and
  ceiling (``include_ceiling=``) from the structure annotation. Without
  ``room_id`` it combines all rooms. It is not a furniture mesh.
- ``get_boxes(scene_id)`` returns the official oriented object boxes in metres.
  Their class names are inferred from the panorama labels; objects never seen
  are named ``"unknown"``. ``infer_labels=False`` skips this step.
- ``get_structure_annotations(scene_id)`` returns the full
  ``annotation_3d.json``.

``metadata`` contains the room types, label names, both label spaces, and the
official errata lists.

Official resources
------------------

* `Dataset homepage <https://structured3d-dataset.org/>`__
* `Official repository <https://github.com/bertjiazheng/Structured3D>`__
* `Data organization <https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md>`__
* `Paper (ECCV 2020) <https://arxiv.org/abs/1908.00222>`__
* `Official errata <https://github.com/bertjiazheng/Structured3D/issues/31>`__

API reference
-------------

**Options.** ``dataset_root`` is the resolved folder of scene directories
(``data_dir`` overrides it). ``room_type_file``, ``label_name_file``, and
``errata_file`` default to copies installed with SceneZoo, and
``include_invalid=False`` drops the invalid scenes from the splits.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - ``get_room_ids``, ``get_frame_sources``
     - Rooms of a scene; rendered sources of a room.
   * - ``get_frames``, ``get_layout``
     - Rendered views; 2D room layouts.
   * - ``get_points``, ``get_point_cloud``, ``get_point_data``
     - Point cloud fused from a room's renderings.
   * - ``get_segmentation``
     - Semantic or instance labels of those points.
   * - ``get_mesh``
     - Room-layout mesh.
   * - ``get_boxes``
     - Official object boxes with inferred names.
   * - ``get_structure_annotations``
     - The complete 3D structure annotation.

.. autoclass:: scenezoo.dataset.scene.Structured3D
   :members:
   :inherited-members:

.. autoclass:: scenezoo.dataset.scene.Structured3DPointData
   :members:
