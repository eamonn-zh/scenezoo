Extract an RGB-D sequence at 1 FPS
==================================

This recipe exports one frame per second of a ScanNet++ iPhone capture to a
folder of images and camera files, a layout most reconstruction and SLAM tools
can read:

.. code-block:: text

   7b6477cb95_1fps/
   ├── color/000000.jpg                1920x1440 RGB
   ├── depth/000000.png                256x192, 16-bit, millimetres (0 = no depth)
   ├── pose/000000.txt                 4x4 camera-to-world matrix (OpenCV axes)
   ├── color_intrinsics/000000.txt     3x3
   └── depth_intrinsics/000000.txt     3x3

The same code works for any dataset with frames; change the dataset name, the
path, and ``source``.

.. code-block:: python

   from pathlib import Path

   import numpy as np
   from PIL import Image

   from scenezoo import get_dataset

   dataset = get_dataset("scannetppv2", "/path/to/scannetpp")
   scene_id = dataset.get_ids("nvs_sem_val")[0]

   # 1. Pick the first frame of every second, using the frame timestamps.
   timeline = dataset.get_frames(scene_id, source="iphone", items=("timestamps",))
   seconds = np.floor(timeline.timestamps - timeline.timestamps[0]).astype(int)
   _, first_of_each_second = np.unique(seconds, return_index=True)
   indices = timeline.indices[first_of_each_second]

   # 2. Read those frames.
   frames = dataset.get_frames(
       scene_id,
       source="iphone",
       indices=indices,
       items=(
           "rgb", "depth", "rgb_intrinsics", "depth_intrinsics", "world_to_camera"
       ),
   )

   # 3. Write them out.
   out = Path(f"{scene_id}_1fps")
   folders = ("color", "depth", "pose", "color_intrinsics", "depth_intrinsics")
   for folder in folders:
       (out / folder).mkdir(parents=True, exist_ok=True)

   camera_to_world = np.linalg.inv(frames.world_to_camera)
   for i in range(len(frames)):
       name = f"{i:06d}"
       color = Image.fromarray(frames.rgb[i])
       color.save(out / "color" / f"{name}.jpg", quality=95)
       depth_mm = np.round(frames.depth[i] * 1000).astype(np.uint16)
       Image.fromarray(depth_mm).save(out / "depth" / f"{name}.png")
       np.savetxt(out / "pose" / f"{name}.txt", camera_to_world[i])
       K_color, K_depth = frames.rgb_intrinsics[i], frames.depth_intrinsics[i]
       np.savetxt(out / "color_intrinsics" / f"{name}.txt", K_color)
       np.savetxt(out / "depth_intrinsics" / f"{name}.txt", K_depth)

   print(f"Saved {len(frames)} frames to {out}")

Notes
-----

- Choosing frames by timestamp gives exactly one frame per second even when
  some frames were skipped for having no valid pose. With a fixed rate you can
  also use ``step=60`` (this video is 60 FPS).
- Intrinsics are saved per frame because they can change within a capture;
  ScanNet++ iPhone focal lengths vary by several pixels due to autofocus.
- Add ``output_size=(640, 480)`` to the second ``get_frames`` call to save
  smaller images; the intrinsics are adjusted automatically.
