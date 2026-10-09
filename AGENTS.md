# AGENTS Guide for scenezoo

## What this repo is
- `scenezoo` is a registry-driven adapter toolkit for 3D scene datasets. Each
  adapter exposes the same optional operations: `get_ids`, `get_mesh`,
  `get_points`, `get_segmentation`, `get_boxes`, `get_frames`, and `check`.
- It is a public library: never embed default data paths, and never assume the
  datasets on a development machine match the official download layout.
  Derive layouts from official tools, download scripts, and docs, and keep
  folder names configurable.
- The README and `docs/` are the user-facing contract. Each built-in dataset has
  a page under `docs/datasets/`.

## Layout
- `src/scenezoo/dataset/`
  - `base.py`: `Dataset` base class and exceptions. Operations are ordinary
    methods; `supports()` detects which ones an adapter overrides.
  - `spec.py`: `DatasetSpec` / `FrameSourceSpec` capability descriptions.
  - `registry.py` + `builtins.py`: lazy registry. `builtins.py` is an
    import-light catalog of the built-in adapters and their specs; adapters
    are imported only when `get_dataset()` or `DatasetInfo.dataset_class`
    needs them.
  - `types.py`: `FrameBatch`, `PointBatch`, `Segmentation3D` return types.
  - `check.py`: explicit, never-automatic layout checks (`CheckBuilder`).
  - `scene/*.py`: one adapter per built-in dataset, exposed lazily by
    `scene/__init__.py`.
- `src/scenezoo/io/`: file access (`cache.py`, `archive.py`, `sens.py`,
  `video.py`, ...). `src/scenezoo/ops/`: frame transforms, geometry, and
  annotation parsers. Both namespaces export helpers lazily.
- `src/scenezoo/metadata/`: only SceneZoo's own corrections to official data
  (for example image directions, or 3D-FUTURE models stored in centimetres).
  Files that an official repository or download already provides (split
  lists, label maps, errata) are read from their official URL with
  `Dataset.open_file`, which caches them, and every such path is a constructor
  option. Never copy official files into the package. Record known data defects
  as fixed lists here, not as runtime heuristics.

## Conventions that must hold
- World: every scene is z-up in metres. Rotate datasets that are not
  (currently SceneNN and 3D-FRONT, `(x, y, z) -> (x, -z, y)`) on read,
  identically in every operation, and state the source convention in the
  dataset page's "Coordinates" row.
- Splits: only the official splits, under their official names. `get_ids()`
  without a split returns every sample; override `_load_sample_ids()` when
  samples fall outside the official splits.
- Frames: depth is `float32` metres (invalid = 0); poses are OpenCV
  `world_to_camera`; pixel coordinates put integer values at pixel centres;
  `normal_maps` are unit normals in OpenCV camera axes, facing the camera.
  Determine a dataset's source axes empirically (for example against normals
  derived from depth) before converting.
- Build image/intrinsic changes with `ops.frame.build_pixel_transform`, apply
  them with `transform_intrinsics`, and pass poses through
  `transform_world_to_camera` whenever a rotation is involved. Rotations are
  camera rolls, so intrinsics must stay upper-triangular pinhole matrices.
- Video is decoded with `io.video.read_video_frames` (PyAV): frames are
  identified by presentation timestamp, returned in coded orientation, and
  numbered with MOV edit lists ignored so indices match per-frame metadata.
  Never apply container display rotation implicitly. After changing
  `io/video.py`, compare it pixel-for-pixel against a no-seek reference decode
  on real ARKitScenes MOV, ScanNet++ MKV, and MultiScan MP4 files.
- `builtins.py` specs must match each adapter's overridden operations (the
  registry raises if they drift); keep their frame sources and items in sync
  with `get_frames` by hand.
- Adapter constructors use explicit keyword parameters (no `**kwargs`).
- Prefer maintained PyPI libraries over hand-written helpers, and keep code
  concise.

## Adding or changing an adapter
- Register it in `builtins.py` (name, aliases, import path, `DatasetSpec`),
  decorate the class with the same `@register_dataset` name and aliases, and
  add it to `scene/__init__.py`.
- Implement `_check()` so `dataset.check()` explains missing or unextracted
  files.
- Add synthetic unit tests, a `SCENEZOO_TEST_<DATASET>_ROOT` integration test,
  and the adapter to `DATASETS` in `test/test_dataset_docs.py`.
- Add or update the page in `docs/datasets/` (summary table with a
  "Coordinates" row, *Quick example*, *Download layout*, *Official resources*,
  *API reference*), the tables in `docs/datasets/index.rst` and
  `docs/guide/concepts.rst`, and the README table.
- Preserve existing aliases, split names, and return conventions unless the
  docs and tests are updated together; such changes are breaking.
- Verify on real data, not only unit tests, and run every code snippet of the
  pages you touched.

## Workflows
```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check src test && ruff format --check src test
pip install -r docs/requirements.txt && python -m sphinx -W -b html docs docs/_build/html
```
- Unit tests are hermetic: they build tiny synthetic datasets in `tmp_path` and
  must not use the network. Official metadata files come from the small
  stand-ins in `test/conftest.py`, passed explicitly to the adapter.
- Real-data integration tests are opt-in via `SCENEZOO_TEST_<DATASET>_ROOT`
  environment variables (see `test/test_integration.py`). They include a
  depth-to-mesh back-projection check that catches camera-convention errors;
  run them after touching frame or camera code.
- Video decoding uses PyAV (`av`), which bundles FFmpeg. A few video tests also
  use the `ffmpeg`/`ffprobe` executables to build fixtures and skip without them.
- Releasing: bump `__version__` in `src/scenezoo/__init__.py` (the only version
  source), push to `main`, wait for CI, then publish a GitHub release tagged
  `v<version>`; `.github/workflows/publish.yml` uploads it to PyPI. Never upload
  by hand, and never rewrite history that a release tag points to.
