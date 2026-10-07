Matterport3D
============

90 whole buildings captured with panoramic RGB-D cameras, with textured
meshes, room (region) and object annotations, and calibrated images.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``matterport3d``
   * - Samples
     - Houses, for example ``17DRP5sb8fy``; rooms inside a house are selected
       with ``region_id``
   * - Splits
     - ``train`` (61), ``val`` (11), ``test`` (18)
   * - Data
     - House, room, and Poisson meshes; face segmentation in four label spaces;
       annotated object boxes; raw and undistorted RGB-D images

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("matterport3d", "/path/to/matterport3d")
   house_id = dataset.get_ids("val")[0]
   region_id = dataset.get_region_ids(house_id)[0]

   room_mesh = dataset.get_mesh(house_id, mesh_type="region", region_id=region_id)
   room_labels = dataset.get_segmentation(
       house_id,
       region_id=region_id,
       segmentation_type="semantic",
       label_space="mpcat40",
   )
   boxes, names = dataset.get_boxes(house_id, region_id=region_id)
   frames = dataset.get_frames(
       house_id,
       source="undistorted",
       indices=[0, 12],
       items=("rgb", "depth", "depth_intrinsics", "world_to_camera"),
       output_size=(640, 512),
   )

Download layout
---------------

Pass the folder that contains ``v1/scans/`` (``scans/`` or the scans folder
itself also work). Each official ZIP bundle must be extracted into a folder
of the same name:

.. code-block:: text

   matterport3d/v1/scans/<house_id>/
   ├── house_segmentations/            # annotated house mesh and .house file
   ├── region_segmentations/           # per-room meshes and annotations
   ├── matterport_mesh/                # textured house mesh
   ├── poisson_meshes/                 # Poisson reconstructions
   ├── undistorted_color_images/       # undistorted RGB-D and cameras
   ├── undistorted_depth_images/
   ├── undistorted_camera_parameters/
   ├── matterport_color_images/        # raw RGB-D and cameras
   ├── matterport_depth_images/
   ├── matterport_camera_intrinsics/
   └── matterport_camera_poses/

For example ``region_segmentations.zip`` becomes ``region_segmentations/``.
Download only the bundles you need, and run ``dataset.check()`` to confirm what
is present. The official split lists and label mapping are downloaded once.

Meshes, rooms, and labels
-------------------------

- ``get_mesh`` returns the annotated ``"house"`` mesh by default. Other types
  are ``"region"`` (one room, requires ``region_id``), ``"raw"`` (the textured
  OBJ), and ``"poisson"`` (choose a reconstruction with ``poisson_level=``).
- ``get_region_ids(house_id)`` lists the rooms.
- ``get_segmentation`` labels the **faces** of the house or room mesh with
  ``"instance"`` (default) or ``"semantic"`` labels. ``label_space`` is
  ``"raw"`` (default), ``"mpcat40"``, ``"nyu40"``, or ``"eigen13"``.
- ``get_boxes`` returns the annotated oriented object boxes (``"obb_gt"``),
  optionally of one room. Objects without a category, or without a class in
  the chosen ``label_space``, are left out.
- ``get_house_info`` returns the parsed ``.house`` file: levels, rooms,
  categories, objects, and images.

Images and cameras
------------------

``source="undistorted"`` (default) returns pinhole images. ``source="raw"``
returns the original distorted images with ``camera_model="OPENCV"`` and the
coefficients ``(k1, k2, p1, p2, k3)`` in ``distortion``. Both return depth in
metres and poses as OpenCV ``world_to_camera`` matrices.

Frame indices number the images of a house in a fixed order;
``get_frame_keys(house_id)`` returns their official file names, and
``get_frame_sources(house_id)`` lists the downloaded sources. Matterport images
are already upright, so ``rotate_to_up`` has no effect.

Official resources
------------------

* `Dataset homepage <https://niessner.github.io/Matterport/>`__
* `Official repository <https://github.com/niessner/Matterport>`__
* `Data organization <https://github.com/niessner/Matterport/blob/master/data_organization.md>`__
* `Paper (3DV 2017) <https://arxiv.org/abs/1709.06158>`__
* `Academic-use agreement <https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data>`__

API reference
-------------

**Options.** ``scans_dir`` is the detected scans folder. Bundle folder
templates are stored in ``mesh_dir``, ``house_segmentation_dir``,
``region_segmentation_dir``, ``poisson_dir``, and ``frame_dirs``.
``label_mapping_file`` and ``split_files`` locate the category mapping and
split lists.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Returns
   * - ``get_mesh``
     - ``house`` (default), ``region``, ``raw``, or ``poisson`` mesh.
   * - ``get_region_ids``
     - The rooms of a house.
   * - ``get_segmentation``
     - Instance or semantic face labels of a house or room.
   * - ``get_boxes``
     - Annotated object boxes.
   * - ``get_house_info``
     - Parsed ``.house`` file.
   * - ``get_frames``, ``get_frame_keys``, ``get_frame_sources``
     - RGB-D images and cameras; image names; downloaded sources.

.. autoclass:: scenezoo.dataset.scene.Matterport3D
   :members:
   :inherited-members:
