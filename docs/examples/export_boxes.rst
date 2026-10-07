Export 3D bounding boxes to JSON
================================

``get_boxes()`` returns Open3D box objects. This recipe saves them as plain JSON
(center, size, rotation, and class name) that any detection pipeline can read.

.. code-block:: python

   import json

   import numpy as np
   import open3d as o3d

   from scenezoo import get_dataset


   def boxes_to_json(boxes, names):
       records = []
       for object_id, box in boxes.items():
           if isinstance(box, o3d.geometry.OrientedBoundingBox):
               center, size, rotation = box.center, box.extent, box.R
           else:  # AxisAlignedBoundingBox
               center, size = box.get_center(), box.get_extent()
               rotation = np.eye(3)
           records.append(
               {
                   "id": int(object_id),
                   "label": names[object_id],
                   "center": np.asarray(center).tolist(),
                   "size": np.asarray(size).tolist(),
                   "rotation": np.asarray(rotation).tolist(),
               }
           )
       return records


   dataset = get_dataset("3rscan", "/path/to/3rscan")
   scan_id = dataset.get_ids("val")[0]
   boxes, names = dataset.get_boxes(scan_id)

   with open(f"{scan_id}_boxes.json", "w") as handle:
       json.dump(boxes_to_json(boxes, names), handle, indent=2)
   print(f"Saved {len(boxes)} boxes, e.g. {next(iter(names.values()))}")

``size`` is the full edge length along each box axis, and the box axes are the
columns of ``rotation``. All values are in the dataset's world coordinates,
in metres.

Choosing the box type
---------------------

``box_type`` controls how boxes are made (see :doc:`../guide/concepts`):

.. code-block:: python

   official, _ = dataset.get_boxes(scan_id, box_type="obb_gt")  # published
   upright, _ = dataset.get_boxes(scan_id, box_type="mobb_gravity")  # fitted
   axis_aligned, _ = dataset.get_boxes(scan_id, box_type="aabb")

``"mobb_gravity"`` only rotates boxes around the vertical axis, which is what
most indoor 3D detection benchmarks expect.
