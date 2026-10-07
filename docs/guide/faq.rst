FAQ
===

Which folder do I pass to ``get_dataset``?
------------------------------------------

The folder you downloaded the dataset into. Each
:doc:`dataset page <../datasets/index>` shows the expected layout under
*Download layout*, and several adapters also accept a few neighbouring levels
(for example both ``3RScan/`` and ``3RScan/download/``). If you are unsure,
run ``dataset.check()``: it reports what it expected to find and where.

Can SceneZoo read data straight from the downloaded ZIP files?
---------------------------------------------------------------

Only where the ZIP is part of the dataset's own format: 3RScan reads frames
from each scan's ``sequence.zip``, and ScanNet reads 2D labels from its
per-scene label ZIPs. All other download archives must be extracted first;
``dataset.check()`` lists archives that still need extracting.

Why does ``get_frames`` reject a frame index?
---------------------------------------------

Frames without a valid camera pose are skipped, because their images cannot be
placed in 3D. Requesting such a frame by index raises :class:`ValueError` that
names it. To get every usable frame, omit ``indices`` (optionally with
``step``) and read ``frames.indices`` to see which source frames you got.

How do I get the camera position or a camera-to-world pose?
-----------------------------------------------------------

Invert ``world_to_camera``:

.. code-block:: python

   import numpy as np

   camera_to_world = np.linalg.inv(frames.world_to_camera)  # (N, 4, 4)
   camera_positions = camera_to_world[:, :3, 3]             # (N, 3)

See :doc:`frames` for the full conventions.

Why are my images rotated, or why did the pose change when I set ``rotate_to_up``?
-----------------------------------------------------------------------------------

Some captures, such as portrait phone videos, are stored sideways.
``rotate_to_up=True`` (the default) turns them upright and rotates the camera
pose with them, so 3D projections stay correct. Pass ``rotate_to_up=False`` to
get images and poses exactly as stored.

Loading frames uses a lot of memory
-----------------------------------

Full-resolution frames are large: one 1920x1440 RGB image is about 8 MB. Load
fewer frames at a time with ``indices`` or ``step``, request only the
``items`` you need, and use ``output_size=(width, height)`` to downscale while
reading.

How do I use SceneZoo with a PyTorch ``DataLoader``?
-----------------------------------------------------

Dataset objects can be sent to ``DataLoader`` worker processes, so a thin
wrapper is enough:

.. code-block:: python

   import torch

   from scenezoo import get_dataset


   class FrameDataset(torch.utils.data.Dataset):
       def __init__(self, root, scene_id, indices):
           self.dataset = get_dataset("scannetppv2", root)
           self.scene_id = scene_id
           self.indices = list(indices)

       def __len__(self):
           return len(self.indices)

       def __getitem__(self, i):
           frames = self.dataset.get_frames(
               self.scene_id,
               source="iphone",
               indices=[self.indices[i]],
               items=("rgb", "depth", "depth_intrinsics", "world_to_camera"),
               output_size=(256, 192),
           )
           return {
               "rgb": torch.from_numpy(frames.rgb[0]),
               "depth": torch.from_numpy(frames.depth[0]),
               "intrinsics": torch.from_numpy(frames.depth_intrinsics[0]),
               "world_to_camera": torch.from_numpy(frames.world_to_camera[0]),
           }


   loader = torch.utils.data.DataLoader(
       FrameDataset("/path/to/scannetpp", "7b6477cb95", range(0, 600, 60)),
       batch_size=4,
       num_workers=2,
   )

Reading a single frame from a long video is slower than reading frames in
order. For training, it is often faster to extract the frames you need once
(see :doc:`../examples/extract_rgbd_sequence`) and train from those files.

My machine has no internet access
---------------------------------

Run once with internet access to fill the cache, or copy the cache folder from
another machine, then pass ``offline=True``. See :ref:`cache-and-offline`.

``ImportError: libEGL.so.1`` on a server
----------------------------------------

Open3D needs a few system graphics libraries even without a display. On
Debian or Ubuntu: ``sudo apt-get install libegl1 libgl1 libgomp1``.
