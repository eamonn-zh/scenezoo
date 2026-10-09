Core concepts
=============

This page explains the few ideas that are shared by every dataset in
SceneZoo. Once you know them, the API of each dataset looks the same.

Datasets and names
------------------

Each supported dataset has an **adapter**: a class that knows the dataset's
file layout and converts it to the common format. You never need to import an
adapter directly. Create one by name with :func:`~scenezoo.get_dataset`:

.. code-block:: python

   from scenezoo import get_dataset, get_dataset_info

   dataset = get_dataset("scannetppv2", "/path/to/scannetpp")

Names are case-insensitive, and most datasets also accept aliases; for example
``"scannet++"`` and ``"scannetpp"`` both mean ``"scannetppv2"``. To see what a
dataset offers without opening it, use :func:`~scenezoo.get_dataset_info`:

.. code-block:: python

   info = get_dataset_info("scannet++")
   print(info.name)              # 'scannetppv2'
   print(info.spec.operations)   # ('mesh', 'points', 'segmentation', ...)
   print(info.spec.mesh_types)   # ('raw', 'semantic')
   print([source.name for source in info.spec.frame_sources])

``get_dataset`` also accepts the adapter's keyword options, such as
``offline=True`` or dataset-specific path overrides; they are listed on each
:doc:`dataset page <../datasets/index>`.

Sample IDs and splits
---------------------

A **sample** is the unit a dataset is organized by, usually one scanned scene.
Every method takes a sample ID as its first argument. IDs keep the dataset's
own naming:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Dataset
     - Sample ID example
     - Splits
   * - ScanNet v2
     - ``scene0000_00``
     - ``train``, ``val``, ``test``
   * - ScanNet++ v2
     - ``39f36da05b``
     - ``nvs_sem_train``, ``nvs_sem_val``, ``nvs_test``, ``sem_test``, ...
   * - ARKitScenes
     - ``40753679`` (video ID)
     - ``train_raw``, ``val_raw``, ``train_threedod``, ...
   * - MultiScan
     - ``scene_00000_00``
     - ``train``, ``val``, ``test``
   * - Aria Synthetic Environments
     - ``train/83788``
     - ``train``, ``test`` (extracted scenes only)
   * - 3RScan
     - scan UUID
     - ``train``, ``val``, ``test``
   * - S3DIS
     - ``Area_5/office_1`` (room)
     - ``area_1`` ... ``area_6``
   * - Matterport3D
     - ``17DRP5sb8fy`` (house)
     - ``train``, ``val``, ``test``
   * - SceneNN
     - ``005``
     - none
   * - Structured3D
     - ``scene_00000``
     - ``train``, ``val``, ``test``
   * - 3D-FRONT
     - house UUID
     - none

Splits are the official ones, under their official names; SceneZoo adds none
of its own. ``dataset.splits`` is a dictionary of them, ``dataset.get_ids(split)``
returns one, and ``dataset.get_ids()`` returns every sample of the release,
including any that the official splits leave out. These lists may include
scenes you have not downloaded; use :doc:`dataset.check() <checking>` to find
out what is present.

.. _coordinate-system:

Coordinate system
-----------------

All scene data uses one world convention, whichever dataset it comes from:

- **z-up, in metres.** Meshes, point clouds, boxes, and camera poses share a
  right-handed world frame in which +Z points up, against gravity, so floors
  lie in an XY plane.
- **Datasets that are not z-up are rotated on read.** 3D-FRONT and SceneNN are
  y-up in their release; SceneZoo rotates them with
  ``(x, y, z) -> (x, -z, y)``. The rotation is applied the same way by every
  operation, so meshes, labels, boxes, and cameras stay consistent. The other
  datasets are already z-up and are returned as released.
- **Only the up axis is fixed.** The origin and the horizontal axes stay as
  released. Some datasets also provide an alignment to the room's walls, for
  example ScanNet's ``get_alignment``.

Each dataset page states the source convention.

The five standard operations
----------------------------

Every adapter can offer up to five standard operations. They have the same
names, arguments, and return types for all datasets:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Returns
   * - ``get_mesh(sample_id)``
     - An Open3D ``TriangleMesh``. ``mesh_type=`` selects a variant, for
       example a raw or a decimated mesh.
   * - ``get_points(sample_id)``
     - A :class:`~scenezoo.PointBatch` of NumPy arrays: ``xyz`` and, when
       available, ``rgb``, ``normals``, and labels for every point.
   * - ``get_segmentation(sample_id)``
     - A :class:`~scenezoo.Segmentation3D` with one label per point, vertex,
       or face (see `Label domains`_).
   * - ``get_boxes(sample_id)``
     - ``(boxes, names)``: two dictionaries keyed by object ID, holding Open3D
       bounding boxes and class names (see `Bounding boxes`_).
   * - ``get_frames(sample_id)``
     - A :class:`~scenezoo.FrameBatch` of images, depth, and cameras. See
       :doc:`frames`.

Not every dataset has every kind of data. S3DIS, for example, has point
clouds but no meshes or camera frames. Check before calling, especially in code
that handles several datasets:

.. code-block:: python

   scene_id = dataset.get_ids("nvs_sem_val")[0]
   if dataset.supports("frames"):
       frames = dataset.get_frames(scene_id, indices=[0])

   print(dataset.capabilities())  # all operations this dataset supports

Calling an unsupported operation raises
:class:`~scenezoo.UnsupportedOperationError`. Besides the standard
operations, many adapters offer **dataset-specific methods**, such as
ScanNet's ``get_calibration()`` or ARKitScenes' ``get_laser_scanner_point_clouds()``.
They are listed on each dataset page.

Label domains
-------------

:class:`~scenezoo.Segmentation3D` always tells you what its labels are
attached to through ``domain``:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - ``domain``
     - ``labels[i]`` belongs to
     - Used by
   * - ``"vertex"``
     - ``mesh.vertices[i]`` of ``get_mesh()``
     - ScanNet, ScanNet++, ARKitScenes, 3RScan, SceneNN
   * - ``"face"``
     - ``mesh.triangles[i]`` of ``get_mesh()``
     - MultiScan, Matterport3D, 3D-FRONT
   * - ``"point"``
     - ``points.xyz[i]`` of ``get_points()``
     - S3DIS, Structured3D

The other fields are ``id_to_name`` (label ID to class or object name) and
``invalid_id``, the value used for unlabeled entries. Most adapters accept a
``segmentation_type`` option to choose between instance and semantic labels,
and some add a label space such as NYU40; see the dataset pages. A
two-dimensional ``labels`` array (``is_multilabel``) stores overlapping labels
padded with ``invalid_id``.

Bounding boxes
--------------

``get_boxes()`` returns ``(boxes, names)``. ``box_type`` chooses how boxes are
produced:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``box_type``
     - Box
   * - ``"obb_gt"``
     - The oriented boxes published with the dataset (where available).
   * - ``"aabb"``
     - Axis-aligned box around the object's points.
   * - ``"obb"``
     - Oriented box from a principal-component fit.
   * - ``"mobb"``
     - Minimal-volume oriented box.
   * - ``"mobb_gravity"``
     - Minimal box that may only rotate around the vertical (gravity) axis.

The default is the most natural choice for each dataset: published boxes when
the dataset has them, otherwise ``"mobb_gravity"`` (``"aabb"`` for S3DIS).
Boxes are Open3D ``OrientedBoundingBox`` or ``AxisAlignedBoundingBox`` objects
with ``center``, ``extent``, and, for oriented boxes, rotation ``R``.

Common attributes and methods
-----------------------------

All adapters inherit these from :class:`scenezoo.Dataset`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Meaning
   * - ``root_dir``
     - The dataset folder you passed in, as a :class:`pathlib.Path`.
   * - ``name``
     - The canonical dataset name, for example ``"scannetv2"``.
   * - ``spec``
     - A :class:`~scenezoo.DatasetSpec` listing operations, mesh, box, and
       segmentation types, label spaces, and frame sources.
   * - ``splits``, ``get_ids(split=None)``
     - The official splits as a dictionary, or the sample IDs of one split
       (every sample when ``split`` is omitted). Loaded on
       first use.
   * - ``metadata``
     - Dataset-wide information such as label maps. Loaded on first use; the
       keys are described on each dataset page.
   * - ``invalid_obj_id``
     - The label value used for background or unlabeled data.
   * - ``offline``, ``cache_dir``
     - How remote metadata files are fetched and cached (see below).
   * - ``supports(op)``, ``capabilities()``
     - Whether one standard operation is available, or all available ones.
   * - ``check()``
     - Inspect the download. See :doc:`checking`.
   * - ``open_file(path)``
     - Open a local or cached remote metadata file under the same offline
       policy.

Errors
------

All SceneZoo errors derive from :class:`~scenezoo.SceneZooError`:

- :class:`~scenezoo.UnsupportedOperationError`: the dataset (or this sample,
  for example a test scan without annotations) does not provide the requested
  data.
- :class:`~scenezoo.DataFormatError`: a file exists but does not match the
  documented format.
- :class:`~scenezoo.DatasetCheckError`: raised by
  ``check(raise_on_error=True)``.

Missing files raise Python's usual :class:`FileNotFoundError`, and invalid
arguments raise :class:`ValueError`.

.. _cache-and-offline:

Cache and offline use
---------------------

Some adapters read small official metadata files that are not part of every
download, such as split lists, label maps, or the 3RScan index. They are
downloaded from the official repository on first use and cached in
``~/.cache/scenezoo`` (or ``$SCENEZOO_CACHE_DIR`` if set). SceneZoo itself
ships only its own corrections to the official data (for example image
directions for ARKitScenes and MultiScan, and the 3D-FUTURE models stored in
centimetres).

- ``get_dataset(..., offline=True)`` never accesses the network; it uses the
  cache and fails with a clear error if a file was never downloaded.
- ``get_dataset(..., cache_dir="...")`` uses a different cache folder.
- Where the official file is also part of the download, such as ScanNet's
  ``scannetv2-labels.combined.tsv``, 3RScan's ``3RScan.json``, or ARKitScenes'
  ``metadata.csv``, the local copy is used and nothing is downloaded.
