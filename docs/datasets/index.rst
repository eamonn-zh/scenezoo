Datasets
========

SceneZoo supports ten indoor-scene datasets. Each page below explains the
expected download layout, the data you can load, and the dataset's own options
and extra methods.

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 18 10 30

   * - Dataset (name)
     - Mesh
     - Points
     - Segmentation
     - Boxes
     - Frames
   * - :doc:`ScanNet v2 <scannet>` (``scannetv2``)
     - yes
     -
     - vertex
     - fitted
     - RGB-D, 2D labels, IMU
   * - :doc:`ScanNet++ v2 <scannetpp>` (``scannetppv2``)
     - yes
     - laser scan
     - vertex
     - fitted
     - iPhone RGB-D, DSLR, panoramas
   * - :doc:`ARKitScenes <arkitscenes>` (``arkitscenes``)
     - yes
     -
     - vertex
     - annotated
     - RGB-D from six cameras
   * - :doc:`MultiScan <multiscan>` (``multiscan``)
     - yes
     -
     - face (objects and parts)
     - fitted or annotated
     - RGB-D with confidence
   * - :doc:`Aria Synthetic Environments <ase>` (``ase``)
     -
     - semi-dense
     -
     - room layout
     - fisheye RGB-D
   * - :doc:`3RScan <3rscan>` (``3rscan``)
     - yes
     -
     - vertex
     - fitted or annotated
     - RGB-D
   * - :doc:`S3DIS <s3dis>` (``s3dis``)
     -
     - laser scan
     - point
     - fitted
     -
   * - :doc:`Matterport3D <matterport3d>` (``matterport3d``)
     - yes
     -
     - face
     - annotated
     - RGB-D (raw and undistorted)
   * - :doc:`SceneNN <scenenn>` (``scenenn``)
     - yes
     -
     - vertex
     -
     -
   * - :doc:`Structured3D <structured3d>` (``structured3d``)
     - room layout
     - fused renderings
     - point
     - annotated
     - rendered RGB-D, panoramas

"Fitted" boxes are computed from annotated objects; "annotated" boxes are
published with the dataset (``box_type="obb_gt"``). The label *domains*
(vertex, face, point) are explained in :doc:`../guide/concepts`.

Which dataset should I use?
---------------------------

- **Real RGB-D video with camera poses:** ScanNet, ScanNet++ (iPhone),
  ARKitScenes, MultiScan, and 3RScan.
- **High-quality geometry and dense semantics:** ScanNet++ (laser scans) and
  Matterport3D (whole buildings).
- **Object parts and articulation:** MultiScan.
- **Scene changes over time:** 3RScan.
- **Synthetic data with perfect labels:** Structured3D and Aria Synthetic
  Environments.

.. toctree::
   :hidden:

   scannet
   scannetpp
   arkitscenes
   multiscan
   ase
   3rscan
   s3dis
   matterport3d
   scenenn
   structured3d
