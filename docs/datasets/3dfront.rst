3D-FRONT
========

Professionally designed, furnished houses: about 6,800 floor plans whose rooms
are filled with textured furniture models from 3D-FUTURE. SceneZoo supports the
latest release of each: 3D-FRONT of January 2022 and 3D-FUTURE-model of
April 2021.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``3dfront`` (alias ``3d-front``)
   * - Samples
     - Houses, for example ``00004f89-9aa5-43c2-ae3c-129586be8aaa``; rooms
       inside a house are selected with ``room_id``
   * - Splits
     - None (no official split); ``get_ids()`` lists every house
   * - Coordinates
     - y-up in the release; rotated to z-up (metres); see :ref:`coordinate-system`
   * - Data
     - Textured house, room, layout, and furniture meshes; face segmentation
       of furniture instances, furniture categories, and layout elements;
       furniture boxes

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("3dfront", "/path/to/3dfront")
   house_id = dataset.get_ids()[0]
   room_id = dataset.get_room_ids(house_id)[0]

   house = dataset.get_mesh(house_id)                   # textured, z-up, metres
   room = dataset.get_mesh(house_id, room_id=room_id)   # one room
   labels = dataset.get_segmentation(house_id, segmentation_type="semantic")
   boxes, names = dataset.get_boxes(house_id, room_id=room_id)
   print(dataset.get_scene_info(house_id)["rooms"][room_id]["type"])

Download layout
---------------

3D-FRONT stores houses as JSON files that reference furniture by its 3D-FUTURE
model ID, so both downloads are needed. SceneZoo expects the folder names used
by the official toolbox (``3D-FRONT``, ``3D-FUTURE-model``, and
``3D-FRONT-texture``, with ``model_info.json`` inside ``3D-FUTURE-model``):

.. code-block:: text

   3dfront/
   ├── 3D-FRONT/
   │   └── <house_id>.json             # floor plan, rooms, furniture placement
   ├── 3D-FUTURE-model/
   │   ├── model_info.json             # categories, styles, materials
   │   └── <model_id>/
   │       ├── raw_model.obj           # furniture mesh, in metres
   │       ├── model.mtl
   │       └── texture.png
   └── 3D-FRONT-texture/               # optional: wall and floor textures
       └── <material_id>/texture.png

If your archives extract to other names, pass ``scene_dir=``, ``model_dir=``,
``texture_dir=``, or ``model_info_file=`` (relative to the root). Without
``3D-FRONT-texture/``, walls and floors use the flat colors stored in the house
JSON. Run ``dataset.check()`` to confirm the folders are in place; with
``sample_ids=``, it also confirms that every furniture model those houses use
is present.

Coordinates and meshes
----------------------

- **Coordinates are converted to z-up.** 3D-FRONT itself is y-up; SceneZoo
  rotates every mesh and box with ``(x, y, z) -> (x, -z, y)`` so that the floor
  is the XY plane, as in the other datasets. Units are metres.
- ``get_mesh`` places each furniture model with the official position,
  quaternion rotation, and scale, and returns one ``TriangleMesh`` with its
  textures: ``triangle_uvs``, ``textures``, and ``triangle_material_ids``.
  ``mesh_type`` is ``"full"`` (default), ``"layout"`` (walls, floors,
  ceilings, doors, windows, and other architectural elements), or
  ``"furniture"``. Pass ``room_id`` for a single room.
- Following the official toolbox, only furniture that the house JSON marks
  ``"valid": true`` is placed; a valid entry whose model file is missing raises
  ``FileNotFoundError`` (``dataset.check(sample_ids=...)`` lists them).
- A few 3D-FUTURE model files are in centimetres while houses are in metres,
  so they would be placed 100 times too large. SceneZoo keeps a list of these
  model IDs (``scenezoo/metadata/3dfuture_centimetre_models.txt``, found by
  comparing each model with the furniture size stored in the house JSONs) and
  scales them by 0.01. Pass ``fix_model_units=False`` to keep the files as
  they are, or ``centimetre_models_file=`` to use your own list.
- ``get_object_mesh(model_id)`` returns one textured 3D-FUTURE model in its own
  coordinates (converted to z-up), and ``metadata["models"]`` holds the
  ``model_info.json`` record of every model.
- ``get_room_ids(house_id)`` lists the rooms; ``get_scene_info(house_id)``
  gives each room's official type (``"Bedroom"``, ``"Kitchen"``, ...) and the
  model IDs placed in it.

Labels and boxes
----------------

- ``get_segmentation`` labels the **faces** of the mesh returned by
  ``get_mesh`` with the same ``mesh_type`` and ``room_id``.

  - ``"instance"`` (default): one ID per furniture instance, numbered across
    the house; layout faces are ``invalid_obj_id``.
  - ``"semantic"``: an index into ``metadata["semantic_classes"][label_space]``.
    Furniture uses its 3D-FUTURE category, and layout elements their official
    type, such as ``"Floor"`` or ``"WallInner"``.

  ``label_space`` is ``"category"`` (default, for example ``"Dining Chair"``)
  or ``"super_category"`` (for example ``"Chair"``).
- ``get_boxes`` returns one box per furniture instance, with the same IDs as
  the instance segmentation. ``"obb_gt"`` (default) is the model's bounding box
  under the instance's placement; ``"aabb"``, ``"obb"``, ``"mobb"``, and
  ``"mobb_gravity"`` are fitted to that box's corners.

3D-FRONT ships no camera images, so ``get_frames`` is not available.

Official resources
------------------

Both datasets are distributed by Alibaba through Tianchi after you accept the
terms of use; the papers link to the current download pages.

* `3D-FRONT paper (ICCV 2021) <https://arxiv.org/abs/2011.09127>`__
* `3D-FUTURE paper (IJCV 2021) <https://arxiv.org/abs/2009.09633>`__
* `3D-FRONT paper at ICCV open access <https://openaccess.thecvf.com/content/ICCV2021/html/Fu_3D-FRONT_3D_Furnished_Rooms_With_layOuts_and_semaNTics_ICCV_2021_paper.html>`__
* `Official toolbox <https://github.com/3D-FRONT-FUTURE/3D-FRONT-ToolBox>`__

API reference
-------------

**Options.** ``scene_dir``, ``model_dir``, and ``texture_dir`` are the house
JSON, 3D-FUTURE model, and layout texture folders; ``model_info_file`` locates
``model_info.json``. Paths are relative to the root. ``fix_model_units``
(default ``True``) rescales the models listed in ``centimetre_models_file``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Returns
   * - ``get_mesh``
     - Textured ``full`` (default), ``layout``, or ``furniture`` mesh of a
       house or room.
   * - ``get_segmentation``
     - Instance or semantic face labels aligned with ``get_mesh``.
   * - ``get_boxes``
     - Furniture boxes and category names.
   * - ``get_room_ids``, ``get_scene_info``
     - The rooms of a house; room types and placed models.
   * - ``get_object_mesh``
     - One textured 3D-FUTURE model.

.. autoclass:: scenezoo.dataset.scene.ThreeDFront
   :members:
   :inherited-members:
