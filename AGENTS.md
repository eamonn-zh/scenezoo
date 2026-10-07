# AGENTS Guide for scenezoo

## What this repo is
- `scenezoo` is a registry-driven adapter toolkit for 3D scene datasets. Each
  adapter exposes the same optional operations: `get_ids`, `get_mesh`,
  `get_points`, `get_segmentation`, `get_boxes`, `get_frames`, and `check`.
- The README and `docs/` are the user-facing contract. Each built-in dataset has
  a page under `docs/datasets/`.

## Layout
- `src/scenezoo/dataset/`
  - `base.py`: `Dataset` base class and exceptions. Operations are ordinary
    methods; `supports()` detects which ones an adapter overrides.
  - `registry.py` + `builtins.py`: lazy registry. `builtins.py` holds an
    import-light `DatasetSpec` catalog, and adapters are imported only when
    `get_dataset()` or `DatasetInfo.dataset_class` needs them.
  - `types.py`: `FrameBatch`, `PointBatch`, `Segmentation3D` return types.
  - `check.py`: explicit, never-automatic layout checks (`CheckBuilder`).
  - `scene/*.py`: one adapter per dataset (ScanNet, ScanNet++, ARKitScenes,
    MultiScan, ASE, 3RScan, S3DIS, Matterport3D, SceneNN, Structured3D).
- `src/scenezoo/io/`: file access (`cache.py`, `archive.py`, `sens.py`,
  `video.py`, ...). `src/scenezoo/ops/`: frame transforms, geometry, and
  annotation parsers. Both namespaces export helpers lazily.
- `src/scenezoo/metadata/`: packaged split lists, label maps, and corrections.

## Conventions that must hold
- Frames: depth is `float32` metres (invalid = 0); poses are OpenCV
  `world_to_camera`; pixel coordinates put integer values at pixel centres.
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
- `builtins.py` specs must match each adapter's overridden operations; the
  registry raises if they drift.
- Adapter constructors use explicit keyword parameters (no `**kwargs`).

## Adding or changing an adapter
- Register it in `builtins.py` (name, aliases, import path, `DatasetSpec`) and
  decorate the class with the same `@register_dataset` name and aliases.
- Preserve existing aliases, split names, and return conventions unless docs,
  examples, and tests are updated together.
- Update the dataset page in `docs/datasets/` and the README table.

## Workflows
```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check src test && ruff format --check src test
pip install -r docs/requirements.txt && python -m sphinx -W -b html docs docs/_build/html
```
- Unit tests are hermetic and build tiny synthetic datasets in `tmp_path`.
- Real-data integration tests are opt-in via `SCENEZOO_TEST_<DATASET>_ROOT`
  environment variables (see `test/test_integration.py`). They include a
  depth-to-mesh back-projection check that catches camera-convention errors;
  run them after touching frame or camera code.
- Video decoding uses PyAV (`av`), which bundles FFmpeg. A few video tests also
  use the `ffmpeg`/`ffprobe` executables to build fixtures and skip without them.
