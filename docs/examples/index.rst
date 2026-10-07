Examples
========

Short, complete recipes for common tasks. Each one runs as-is once you replace
``/path/to/...`` with your dataset folder.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Recipe
     - What it does
   * - :doc:`extract_rgbd_sequence`
     - Export color, depth, poses, and intrinsics of a capture at 1 FPS.
   * - :doc:`depth_to_point_cloud`
     - Fuse depth frames into a colored point cloud and save it as PLY.
   * - :doc:`export_labeled_points`
     - Turn vertex, face, or point labels into ``(xyz, label)`` training data.
   * - :doc:`export_boxes`
     - Save 3D bounding boxes with class names as JSON.
   * - :doc:`multiple_datasets`
     - Process several datasets with the same loop.

.. toctree::
   :hidden:

   extract_rgbd_sequence
   depth_to_point_cloud
   export_labeled_points
   export_boxes
   multiple_datasets
