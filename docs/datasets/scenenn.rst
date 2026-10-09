SceneNN
=======

About 100 reconstructed indoor scenes with per-vertex instance annotations.

.. list-table::
   :widths: 22 78
   :stub-columns: 1

   * - Name
     - ``scenenn``
   * - Samples
     - Scenes, for example ``005``
   * - Splits
     - None (no official split); ``get_ids()`` lists every scene
   * - Coordinates
     - y-up in the release; rotated to z-up (metres); see :ref:`coordinate-system`
   * - Data
     - Meshes and vertex instance segmentation

Quick example
-------------

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("scenenn", "/path/to/scenenn")
   scene_id = dataset.get_ids()[0]

   mesh = dataset.get_mesh(scene_id)               # colored mesh
   instances = dataset.get_segmentation(scene_id)  # one object ID per vertex
   print(instances.id_to_name)

Download layout
---------------

One folder per scene:

.. code-block:: text

   scenenn/
   └── 005/
       ├── 005_color.ply      # colored mesh
       ├── 005.ply            # same mesh, colored by instance
       └── 005.xml            # instance labels

Extract the downloads before use. The ONI videos of the release are not needed.
Run ``dataset.check()`` to confirm the scenes you need are complete.

Meshes and labels
-----------------

- ``get_mesh`` returns the ``"raw"`` colored mesh (default) or the
  ``"instance"`` mesh colored by object. SceneNN meshes are y-up; both are
  rotated to z-up (see :ref:`coordinate-system`).
- ``get_segmentation`` returns per-vertex instance IDs and names. The release
  identifies objects by color in the instance mesh; SceneZoo matches those
  colors to the labels in the XML file. Unlabeled vertices get
  ``invalid_obj_id`` (``-1``).
- ``metadata["room_categories"]`` maps room categories to scene IDs. The
  category list is downloaded once.

SceneNN also contains RGB-D videos and camera trajectories in ONI format;
SceneZoo does not read them yet, so ``get_frames`` and ``get_boxes`` are not
available.

Official resources
------------------

* `Dataset homepage <https://scenenn.net/>`__
* `Official code and downloads <https://github.com/hkust-vgd/scenenn>`__
* `Paper (3DV 2016) <https://scenenn.net/pdf/dataset_3dv16.pdf>`__

API reference
-------------

**Options.** ``raw_mesh_file``, ``instance_mesh_file``, and
``annotation_file`` are path templates; ``room_category_file`` locates the room
category list.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Returns
   * - ``get_mesh``
     - ``raw`` (default) or ``instance`` mesh.
   * - ``get_segmentation``
     - Vertex instance labels and names.

.. autoclass:: scenezoo.dataset.scene.SceneNN
   :members:
   :inherited-members:
