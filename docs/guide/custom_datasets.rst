Adding your own dataset
=======================

You can register your own dataset so that it works with
:func:`~scenezoo.get_dataset` and with code written for the built-in
datasets. An adapter is a subclass of :class:`~scenezoo.Dataset` that
implements only the operations your data supports.

A minimal adapter
-----------------

Suppose each scene is a folder with a ``mesh.ply`` file, and ``splits.json``
lists the scene names:

.. code-block:: text

   my_scenes/
   ├── splits.json        {"train": ["kitchen"], "val": ["office"]}
   ├── kitchen/mesh.ply
   └── office/mesh.ply

.. code-block:: python

   import json

   from scenezoo import Dataset, register_dataset
   from scenezoo.ops import load_triangle_mesh


   @register_dataset("my-scenes", aliases=("mine",))
   class MyScenes(Dataset):
       """Meshes of my own scanned rooms."""

       def _load_splits(self):
           with open(self.root_dir / "splits.json") as handle:
               return json.load(handle)

       def get_mesh(self, sample_id, *, mesh_type=None):
           return load_triangle_mesh(self.root_dir / sample_id / "mesh.ply")

Now it behaves like any other dataset:

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("my-scenes", "/path/to/my_scenes")
   print(dataset.capabilities())          # ('mesh',)
   mesh = dataset.get_mesh(dataset.get_ids("train")[0])

How it works
------------

- ``_load_splits()`` returns ``{split_name: [sample_id, ...]}``. It runs once,
  the first time splits are needed.
- ``get_ids()`` without a split returns every sample in the splits. If some
  samples belong to no split, override ``_load_sample_ids()`` to list them all.
- ``_load_metadata()`` (optional) returns a dictionary for
  ``dataset.metadata``, such as label names.
- Override only the standard operations your data supports: ``get_mesh``,
  ``get_points``, ``get_segmentation``, ``get_boxes``, and ``get_frames``.
  ``supports()`` detects them automatically, and the others raise
  :class:`~scenezoo.UnsupportedOperationError`. Return the standard types
  described in :doc:`concepts`.
- Add constructor options as explicit keyword arguments and pass the common
  ones on to the base class:

  .. code-block:: python

     def __init__(
         self, root_dir, *, mesh_file="mesh.ply", cache_dir=None, offline=False
     ):
         super().__init__(root_dir, cache_dir=cache_dir, offline=offline)
         self.mesh_file = mesh_file

- ``register_dataset`` checks the name and every alias for conflicts before
  registering anything, so a clash never leaves a half-registered dataset.

Supporting ``check()``
----------------------

Every dataset inherits a basic ``check()`` that verifies the root folder
exists. To report missing files in your layout, override ``_check()`` and
collect findings with :class:`~scenezoo.dataset.check.CheckBuilder`:

.. code-block:: python

   from scenezoo.dataset.check import CheckBuilder


   class MyScenes(Dataset):
       ...

       def _check(self, *, sample_ids, require_complete):
           builder = CheckBuilder(self.name, self.root_dir)
           scenes = sample_ids or self.get_ids("train")
           builder.missing_paths(
               (
                   self.root_dir / scene / "mesh.ply"
                   for scene in scenes
                   if not (self.root_dir / scene / "mesh.ply").is_file()
               ),
               code="missing-mesh",
               label="scene meshes",
               expected="<root>/<scene>/mesh.ply",
               hint="Copy the scene folders into the dataset root.",
           )
           return builder.finish()

To contribute a dataset to SceneZoo itself, see :doc:`../development/contributing`.
