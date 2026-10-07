Write one loop for several datasets
===================================

Because every dataset shares the same API, code written once can process all
of them. This recipe summarizes the first validation scene of each dataset you
have downloaded, using ``supports()`` to skip data a dataset does not have.

.. code-block:: python

   from scenezoo import get_dataset

   # Point each dataset name to where you downloaded it.
   roots = {
       "scannetppv2": "/path/to/scannetpp",
       "multiscan": "/path/to/multiscan",
       "3rscan": "/path/to/3rscan",
       "matterport3d": "/path/to/matterport3d",
   }

   for name, root in roots.items():
       dataset = get_dataset(name, root)
       scene_id = dataset.get_ids()[0]
       summary = [f"{name:13s}"]

       if dataset.supports("mesh"):
           mesh = dataset.get_mesh(scene_id)
           summary.append(f"{len(mesh.vertices):>9,} vertices")
       if dataset.supports("boxes"):
           boxes, names = dataset.get_boxes(scene_id)
           summary.append(f"{len(boxes):>4} objects")
       if dataset.supports("frames"):
           frames = dataset.get_frames(
               scene_id, step=500, items=("world_to_camera",)
           )
           summary.append(f"{len(frames):>3} sampled cameras")

       print(" | ".join(summary))

Example output:

.. code-block:: text

   scannetppv2   | 1,317,897 vertices |   93 objects |  15 sampled cameras
   multiscan     |    93,123 vertices |   28 objects |  14 sampled cameras
   3rscan        |    27,027 vertices |   12 objects |   1 sampled cameras
   matterport3d  | 4,847,541 vertices |  862 objects |   8 sampled cameras

``get_frames`` uses each dataset's default frame source here. Pass ``source=``
to choose another one; ``dataset.spec.frame_sources`` lists them.

Datasets with different split names
-----------------------------------

Split names are the official ones, so they differ between datasets (for
example ScanNet++ uses ``nvs_sem_val`` and S3DIS uses ``area_5``). To stay
generic, call ``get_ids()`` without a split, or list ``dataset.splits``:

.. code-block:: python

   for name, root in roots.items():
       dataset = get_dataset(name, root)
       sizes = {split: len(ids) for split, ids in dataset.splits.items()}
       print(name, len(dataset.get_ids()), sizes)
