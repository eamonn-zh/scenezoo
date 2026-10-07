Core API
========

The classes you get back from datasets, and the base class for writing your own.
All of them are importable from the top-level ``scenezoo`` package.

Dataset
-------

.. autoclass:: scenezoo.Dataset
   :members:

Return types
------------

.. autoclass:: scenezoo.PointBatch
   :members:

.. autoclass:: scenezoo.Segmentation3D
   :members:

.. autoclass:: scenezoo.FrameBatch
   :members:

Checking
--------

.. autoclass:: scenezoo.DatasetCheckReport
   :members:

.. autoclass:: scenezoo.CheckIssue

.. autoclass:: scenezoo.dataset.check.CheckBuilder
   :members:

Errors
------

.. autoexception:: scenezoo.SceneZooError

.. autoexception:: scenezoo.UnsupportedOperationError

.. autoexception:: scenezoo.DataFormatError

.. autoexception:: scenezoo.DatasetCheckError
