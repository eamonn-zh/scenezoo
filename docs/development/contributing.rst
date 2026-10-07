Contributing
============

Set up a development environment and run the checks:

.. code-block:: bash

   git clone https://github.com/eamonn-zh/scenezoo.git
   cd scenezoo
   pip install -e ".[dev]" -r docs/requirements.txt

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
   ruff check src test
   ruff format --check src test
   python -m sphinx -W -b html docs docs/_build/html

Unit tests build tiny synthetic datasets and need no downloads. Tests against
real data are opt-in: set ``SCENEZOO_TEST_<DATASET>_ROOT`` (for example
``SCENEZOO_TEST_SCANNETPP_ROOT=/path/to/scannetpp``) and run
``pytest test/test_integration.py``. They include a check that depth, camera
intrinsics, and poses reproject onto the released mesh; run them after changing
frame or camera code.

Adding a built-in dataset
-------------------------

Start from :doc:`../guide/custom_datasets`, then:

#. Put the adapter in ``src/scenezoo/dataset/scene/`` and decorate it with
   :func:`scenezoo.register_dataset`. Use explicit keyword arguments for
   constructor options.
#. Add its name, aliases, import path, and :class:`scenezoo.DatasetSpec` to
   ``src/scenezoo/dataset/builtins.py``, and expose the class in
   ``src/scenezoo/dataset/scene/__init__.py``. Listing datasets must not
   import adapter modules.
#. Implement ``_check()`` so that ``dataset.check()`` explains missing or
   unextracted files. Missing scenes of an intentional subset are warnings;
   explicitly requested scenes that are missing are errors.
#. Follow the shared conventions: depth in metres, OpenCV ``world_to_camera``
   poses, pinhole intrinsics that match the returned images, and labels aligned
   with their geometry. See :doc:`../guide/frames`.
#. Add synthetic unit tests and an environment-gated integration test.
#. Add a page under ``docs/datasets/`` and a row to the table in
   ``docs/datasets/index.rst`` and the README.

Writing documentation
---------------------

- Dataset pages follow one layout: a summary table, *Quick example*,
  *Download layout*, dataset-specific sections, *Official resources*, and
  *API reference*. ``test/test_dataset_docs.py`` checks that every public
  method and attribute of an adapter is mentioned on its page.
- Use ``/path/to/<dataset>`` for dataset folders in examples, and run every
  example on real data before publishing it.
- Prefer short sentences and concrete examples over complete lists of
  internals; the API reference is generated from docstrings.

Releasing
---------

#. Bump ``__version__`` in ``src/scenezoo/__init__.py`` and merge to ``main``.
#. Create a GitHub release with tag ``v<version>`` (for example ``v0.2.0``).
   Publishing the release runs ``.github/workflows/publish.yml``, which builds
   the package, checks that the tag matches the version, and uploads it to
   PyPI through trusted publishing.
