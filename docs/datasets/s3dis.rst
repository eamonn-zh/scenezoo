S3DIS
=====

The Stanford Large-Scale 3D Indoor Spaces dataset: colored laser-scan point
clouds of 272 rooms in six building areas, with instance and 13-class semantic
labels.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``s3dis`` (aliases ``stanford3d``, ``stanford-3d``)
   * - Samples
     - Rooms, for example ``Area_5/office_1``
   * - Splits
     - ``area_1`` to ``area_6``; see `Area folds`_
   * - Coordinates
     - z-up, metres, as released; see :ref:`coordinate-system`
   * - Data
     - Point clouds with RGB, point segmentation, boxes (no meshes or camera
       frames)

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("s3dis", "/path/to/S3DIS")
   train_ids, test_ids = dataset.get_cross_validation_split(test_area=5)
   room_id = test_ids[0]

   points = dataset.get_points(room_id)  # xyz, rgb, and labels per point
   semantics = dataset.get_segmentation(room_id, segmentation_type="semantic")
   boxes, names = dataset.get_boxes(
       room_id, classes=dataset.detection_class_names
   )
   print(points.xyz.shape, points.semantic_labels.shape)

Download layout
---------------

Extract ``Stanford3dDataset_v1.2`` or ``Stanford3dDataset_v1.2_Aligned_Version``
and pass either the extracted folder or its parent:

.. code-block:: text

   Stanford3dDataset_v1.2_Aligned_Version/
   └── Area_1/
       └── office_1/
           ├── office_1.txt              # the whole room as XYZRGB
           └── Annotations/
               ├── chair_1.txt           # one file per object instance
               └── ...

SceneZoo reads these text files directly; no NumPy or HDF5 conversion is
needed. Exports that dropped the ``Annotations`` folders cannot be used. Run
``dataset.check()`` to confirm the rooms you need are complete.

Points and labels
-----------------

``get_points`` concatenates a room's annotation files in a fixed, natural order
(``chair_2`` before ``chair_10``) and returns XYZ (``float32``), RGB
(``uint8``), and ``int32`` semantic and instance labels for every point.
``get_segmentation`` and ``get_boxes`` use the same order, so their labels
always match ``points.xyz``. The room-level ``.txt`` file may be ordered
differently; ``get_raw_room_points`` returns it unchanged as an ``(N, 6)``
array.

Parsing the text files takes a few seconds per room. When you need several
outputs of one room, call ``get_points`` once and use its label arrays, or
``points.to_point_cloud()`` for Open3D.

The 13 semantic classes follow the common PointNet order:

.. csv-table::
   :header: "ID", "Class", "ID", "Class"
   :widths: 8, 20, 8, 20

   0, ceiling, 7, table
   1, floor, 8, chair
   2, wall, 9, sofa
   3, beam, 10, bookcase
   4, column, 11, board
   5, window, 12, clutter
   6, door, ,

Objects with other names, such as the release's few ``stairs`` files, are
counted as ``unknown_class`` (``"clutter"`` by default). Two known defects of
the release, a stray NUL character in ``Area_5/hallway_6`` and the
``copy_Room_1.txt`` spelling in ``Area_6``, are handled without editing your
files.

``get_boxes`` returns an axis-aligned box for every instance by default.
``classes=dataset.detection_class_names`` keeps the five classes commonly used
for 3D detection (table, chair, sofa, bookcase, and board).

Coordinates
-----------

Coordinates are returned exactly as stored unless you ask otherwise:

- ``align_to_axes=True`` rotates the room around its center by the official
  alignment angle, so walls line up with the X and Y axes (as in Pointcept).
  ``get_alignment_angle(room_id)`` returns that angle in degrees.
- ``shift_to_origin=True`` moves the room so its minimum corner is at the
  origin (as in PointNet and MMDetection3D).

Both options are accepted by ``get_points``, ``get_point_cloud``,
``get_raw_room_points``, and (``align_to_axes`` only) ``get_boxes``.

Area folds
----------

S3DIS has no single official train/test split. Each area is a split
(``area_1`` to ``area_6``); ``get_cross_validation_split(test_area=5)`` returns
``(train_ids, test_ids)`` for the common Area 5 benchmark, and any other area
gives the folds of six-fold cross-validation.

The related Stanford 2D-3D-S release, which adds panoramas, depth, and meshes,
is a different dataset and is not read by this adapter.

Official resources
------------------

* `Dataset page <http://buildingparser.stanford.edu/dataset.html>`__
* `Stanford Computer Vision Lab resources <https://cs.stanford.edu/groups/cvgl/resources.html>`__
* `Paper (CVPR 2016) <https://openaccess.thecvf.com/content_cvpr_2016/html/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.html>`__
* `PointNet preprocessing <https://github.com/charlesq34/pointnet/tree/master/sem_seg>`__

API reference
-------------

**Options.** ``dataset_dir`` selects the extracted folder when it cannot be
found automatically; ``dataset_root`` is the resolved folder.
``annotation_dir`` and ``room_file`` are the per-room file names, and
``unknown_class`` the class used for unrecognized objects. ``class_names`` and
``detection_class_names`` list the classes. ``metadata`` is keyed by room ID
and holds each room's area, name, type, annotation count, and alignment angle.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Method
     - Returns
   * - ``get_points``, ``get_point_data``
     - :class:`~scenezoo.dataset.scene.S3DISPointData` with XYZ, RGB, and labels.
   * - ``get_point_cloud``
     - The same points as an Open3D cloud.
   * - ``get_segmentation``
     - Instance (default) or semantic point labels.
   * - ``get_boxes``
     - One box per instance, optionally filtered by class.
   * - ``get_raw_room_points``
     - The room-level text file as ``(N, 6)`` XYZRGB.
   * - ``get_cross_validation_split``
     - Train and test rooms for one held-out area.
   * - ``get_alignment_angle``
     - The official alignment angle of a room.

.. autoclass:: scenezoo.dataset.scene.S3DIS
   :members:
   :inherited-members:

.. autoclass:: scenezoo.dataset.scene.S3DISPointData
   :members:
