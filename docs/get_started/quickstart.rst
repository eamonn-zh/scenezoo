Quick start
===========

This page walks through the main features in about ten minutes, using
ScanNet++ v2 as the example. Every other dataset works the same way; only the
dataset name, the path, and a few dataset-specific options change.

You need SceneZoo installed (see :doc:`installation`) and a downloaded
dataset. Replace ``/path/to/scannetpp`` with the folder that contains
ScanNet++'s ``data/`` directory.

1. Open a dataset
-----------------

:func:`~scenezoo.get_dataset` takes a dataset name and the folder where you
downloaded it:

.. code-block:: python

   from scenezoo import get_dataset, list_datasets

   print(list_datasets())
   # ['3rscan', 'arkitscenes', 'ase', 'matterport3d', 'multiscan', 's3dis',
   #  'scannetppv2', 'scannetv2', 'scenenn', 'structured3d']

   dataset = get_dataset("scannetppv2", "/path/to/scannetpp")

Opening a dataset is instant: nothing is read until you ask for data.

2. Find scenes
--------------

Every dataset keeps its official splits under their official names. Each split
is a list of sample IDs, which you pass to all other methods:

.. code-block:: python

   print(list(dataset.splits))      # ['nvs_sem_train', 'nvs_sem_val', ...]
   print(len(dataset.get_ids()))    # every scene, whatever its split
   scene_id = dataset.get_ids("nvs_sem_val")[0]
   print(scene_id)                  # for example '7b6477cb95'

Before a long job, check that the scenes you need are complete. This only looks
at file names, so it is fast:

.. code-block:: python

   report = dataset.check(sample_ids=[scene_id])
   print(report)  # "PASSED", or what is missing and how to fix it

3. Load 3D data
---------------

Datasets provide different kinds of data. ``capabilities()`` lists which of
the five standard operations a dataset offers, and ``supports("mesh")`` checks
one of them:

.. code-block:: python

   print(dataset.capabilities())
   # ('mesh', 'points', 'segmentation', 'boxes', 'frames')

   mesh = dataset.get_mesh(scene_id)  # open3d.geometry.TriangleMesh
   segmentation = dataset.get_segmentation(scene_id)  # instance labels
   boxes, names = dataset.get_boxes(scene_id)  # {id: box}, {id: class name}

   print(segmentation.domain)        # 'vertex'
   print(segmentation.labels.shape)  # (num_vertices,)
   print(len(boxes), "objects, e.g.", next(iter(names.values())))

Segmentation labels are aligned with the geometry they belong to. Here the
``domain`` is ``"vertex"``, so ``segmentation.labels[i]`` is the object ID of
``mesh.vertices[i]`` and ``segmentation.id_to_name`` gives its class name.
Other datasets label faces or points instead; :doc:`../guide/concepts` explains
the three domains.

4. Load RGB-D frames and cameras
--------------------------------

:meth:`~scenezoo.Dataset.get_frames` reads only the frames you ask for and
returns them as one :class:`~scenezoo.FrameBatch`. Choose the fields with
``items``:

.. code-block:: python

   frames = dataset.get_frames(
       scene_id,
       source="iphone",       # ScanNet++ also has DSLR and panoramas
       indices=[0, 60, 120],  # frame numbers in the source video
       items=(
           "rgb", "depth", "rgb_intrinsics", "depth_intrinsics", "world_to_camera"
       ),
   )

   print(frames.rgb.shape)              # (3, 1440, 1920, 3) uint8
   print(frames.depth.shape)            # (3, 192, 256) float32, metres
   print(frames.world_to_camera.shape)  # (3, 4, 4), OpenCV camera convention

To get smaller images, pass ``output_size=(width, height)``. The intrinsics
are adjusted to match, so projections stay correct:

.. code-block:: python

   small = dataset.get_frames(
       scene_id,
       source="iphone",
       indices=[0, 60, 120],
       items=("rgb", "depth", "rgb_intrinsics", "depth_intrinsics"),
       output_size=(256, 192),
   )
   print(small.rgb.shape, small.depth.shape)  # both 192 x 256 now

Instead of ``indices`` you can sample regularly with ``step``. For example,
``step=60`` gives one frame per second of this 60 FPS video.

5. Save results
---------------

Results are plain NumPy arrays and Open3D objects, so you can save them with the
tools you already use:

.. code-block:: python

   import numpy as np
   import open3d as o3d

   o3d.io.write_triangle_mesh(f"{scene_id}_mesh.ply", mesh)
   np.savez_compressed(
       f"{scene_id}_frames.npz",
       rgb=frames.rgb,
       depth=frames.depth,
       intrinsics=frames.rgb_intrinsics,
       world_to_camera=frames.world_to_camera,
   )

Next steps
----------

- :doc:`../guide/concepts` explains sample IDs, splits, return types, and
  label domains.
- :doc:`../guide/frames` covers camera conventions, frame sources, and image
  resizing in detail.
- :doc:`../examples/index` has complete recipes, such as extracting an RGB-D
  sequence at 1 FPS or turning depth into a point cloud.
- Each page under :doc:`../datasets/index` lists the options and extra methods
  of one dataset.
