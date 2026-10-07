Checking a download
===================

Large datasets are rarely downloaded in one piece, and a missing or still-zipped
file usually only shows up hours into a job. ``dataset.check()`` finds these
problems up front. It only looks at file names, never decodes meshes, images,
or videos, and so finishes in seconds.

Checking never happens automatically: opening a dataset or reading data does not
run it. Call it yourself when you want to know.

Check the scenes you are about to use
-------------------------------------

Pass the sample IDs your job needs:

.. code-block:: python

   from scenezoo import get_dataset

   dataset = get_dataset("scenenn", "/path/to/scenenn")
   report = dataset.check(sample_ids=["005", "009"])
   print(report)

The printed report says what is wrong, where, what was expected, and what to do:

.. code-block:: text

   scenenn dataset check: FAILED
   Root: /path/to/scenenn
   Summary:
     - scenes found: 104
     - scenes requested: 2
   Findings:
     ERROR [missing-annotations] 1 required annotations files missing. Examples: /path/to/scenenn/009/009.xml
       Expected: <scene_id>_color.ply, <scene_id>.ply, and <scene_id>.xml.
       Suggested action: Download/extract the processed mesh and annotation files for this scene.

Use the report in code
----------------------

A :class:`~scenezoo.DatasetCheckReport` has these fields:

- ``ok``: ``True`` when there are no errors (warnings are allowed).
- ``errors``, ``warnings``, ``issues``: lists of
  :class:`~scenezoo.CheckIssue`, each with a stable ``code``, a ``message``,
  and, where relevant, the ``path``, the ``expected`` layout, and a ``hint``.
- ``stats``: counts such as ``scenes_found``.

.. code-block:: python

   if not report.ok:
       for issue in report.errors:
           print(issue.code, issue.path, issue.hint)

To stop a script on any error, use ``raise_on_error=True``. It raises
:class:`~scenezoo.DatasetCheckError`, whose ``report`` attribute holds the same
report:

.. code-block:: python

   dataset.check(sample_ids=["005"], raise_on_error=True)

Check the whole download
------------------------

Without ``sample_ids``, ``check()`` compares what is on disk with the official
splits. Scenes that are missing from the splits are reported as **warnings**,
because downloading only part of a dataset is common. To treat them as errors,
for example before archiving a complete copy, use ``require_complete=True``:

.. code-block:: python

   report = dataset.check()                       # partial downloads are OK
   dataset.check(require_complete=True, raise_on_error=True)

What is checked
---------------

- The dataset folder exists and has the expected structure.
- The requested (or all expected) scenes are present.
- Each scene has the files its operations need, following any custom paths you
  passed to ``get_dataset``.
- Archives that must be extracted are reported as not yet extracted, while
  archives the dataset reads directly (for example 3RScan's ``sequence.zip`` or
  ScanNet's 2D label ZIPs) are accepted.

A passed check means the files are where SceneZoo expects them. A file that is
present but corrupted is still reported when it is read.
