Export labeled points for training
==================================

Semantic-segmentation models usually train on points with one label each.
Datasets attach labels to mesh vertices, mesh faces, or points (the label
*domain*, see :doc:`../guide/concepts`). This recipe turns any of them into the
same ``(xyz, label)`` arrays.

.. code-block:: python

   import numpy as np

   from scenezoo import get_dataset


   def labeled_points(dataset, sample_id, **options):
       """Return (xyz, segmentation) with one position per label."""

       segmentation = dataset.get_segmentation(sample_id, **options)
       if segmentation.domain == "point":
           xyz = dataset.get_points(sample_id, **options).xyz
       else:
           mesh = dataset.get_mesh(sample_id)
           vertices = np.asarray(mesh.vertices, dtype=np.float32)
           if segmentation.domain == "vertex":
               xyz = vertices
           else:  # "face": use the centre of every triangle
               xyz = vertices[np.asarray(mesh.triangles)].mean(axis=1)
       return xyz, segmentation


   dataset = get_dataset("multiscan", "/path/to/multiscan")
   scan_id = dataset.get_ids("val")[0]
   xyz, segmentation = labeled_points(
       dataset, scan_id, segmentation_type="object_semantic"
   )
   print(xyz.shape, segmentation.labels.shape)       # (num_faces, 3) (num_faces,)
   print(segmentation.id_to_name)                    # {label_id: class_name}

   keep = segmentation.labels != segmentation.invalid_id  # drop unlabeled points
   np.savez_compressed(
       f"{scan_id}.npz", xyz=xyz[keep], labels=segmentation.labels[keep]
   )

Options such as ``segmentation_type`` are passed through, so the same function
works for other datasets:

.. code-block:: python

   scannetpp = get_dataset("scannetppv2", "/path/to/scannetpp")
   xyz, segmentation = labeled_points(
       scannetpp, scannetpp.get_ids("nvs_sem_val")[0], segmentation_type="semantic"
   )

   structured3d = get_dataset("structured3d", "/path/to/Structured3D")
   scene_id = structured3d.get_ids("train")[0]
   room_id = structured3d.get_room_ids(scene_id)[0]
   xyz, segmentation = labeled_points(
       structured3d, scene_id, room_id=room_id, voxel_size=0.05
   )

Notes
-----

- Unlabeled entries carry ``segmentation.invalid_id``. Its value depends on
  the dataset and label type, so always compare against this field.
- For colors, use ``mesh.vertex_colors`` (vertex domain) or
  ``points.rgb`` (point domain).
