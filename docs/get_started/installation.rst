Installation
============

Requirements
------------

- Python 3.10 to 3.14 on Linux, macOS, or Windows.
- The datasets you want to use, downloaded under their official license terms.
  SceneZoo does not download datasets for you; each
  :doc:`dataset page <../datasets/index>` links to the official source.

Install
-------

Install SceneZoo with pip:

.. code-block:: bash

   pip install scenezoo

All required dependencies, including Open3D and PyAV for video decoding, are
installed automatically. No system FFmpeg is needed.

To work on SceneZoo itself, install an editable checkout with the
development tools:

.. code-block:: bash

   git clone https://github.com/eamonn-zh/scenezoo.git
   cd scenezoo
   pip install -e ".[dev]"

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - Adds
   * - ``dev``
     - pytest and ruff for development.
   * - ``render``
     - PyTorch for :mod:`scenezoo.rendering` (mesh rasterization into camera
       views). PyTorch3D is needed as well; install it following the
       `PyTorch3D instructions <https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md>`__.

For example, ``pip install -e ".[render]"``.

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import scenezoo; print(scenezoo.list_datasets())"

This prints the names of all built-in datasets. Listing datasets does not
need any dataset on disk.

Troubleshooting
---------------

``ImportError: libEGL.so.1`` (or ``libGL.so.1``) on a headless Linux server
   Open3D needs a few system graphics libraries even without a display. On
   Debian or Ubuntu, install them with
   ``sudo apt-get install libegl1 libgl1 libgomp1``.

Downloads of label maps or split lists fail on an offline machine
   A few adapters fetch small metadata files on first use and cache them. See
   :ref:`cache-and-offline` for how to prefill the cache or run fully offline.
