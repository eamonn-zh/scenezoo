<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/eamonn-zh/scenezoo/main/docs/_static/logo-dark.png">
    <img alt="SceneZoo" src="https://raw.githubusercontent.com/eamonn-zh/scenezoo/main/docs/_static/logo.png" width="336">
  </picture>
</p>

<p align="center">
  <a href="https://scenezoo.readthedocs.io/en/latest/">Documentation</a> •
  <a href="https://scenezoo.readthedocs.io/en/latest/get_started/installation.html">Installation</a> •
  <a href="https://scenezoo.readthedocs.io/en/latest/get_started/quickstart.html">Quick start</a> •
  <a href="https://scenezoo.readthedocs.io/en/latest/examples/index.html">Examples</a> •
  <a href="https://scenezoo.readthedocs.io/en/latest/datasets/index.html">Datasets</a> •
  <a href="https://scenezoo.readthedocs.io/en/latest/api/core.html">API reference</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/scenezoo/"><img alt="PyPI" src="https://img.shields.io/pypi/v/scenezoo?logo=pypi&logoColor=white&color=0a7cff&cacheSeconds=3600"></a>
  <a href="https://scenezoo.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/readthedocs/scenezoo?logo=readthedocs&logoColor=white&label=docs"></a>
  <a href="https://github.com/eamonn-zh/scenezoo/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/eamonn-zh/scenezoo/ci.yml?branch=main&logo=githubactions&logoColor=white&label=CI"></a>
  <a href="https://github.com/eamonn-zh/scenezoo/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-0a7cff"></a>
</p>

**SceneZoo** reads popular 3D indoor-scene datasets (such as ScanNet,
ScanNet++, ARKitScenes, and Matterport3D) through one Python API.
Point it at a dataset you have downloaded, and get meshes, point clouds,
segmentation labels, 3D boxes, and synchronized RGB-D frames with cameras as
NumPy arrays and Open3D objects, without writing a new loader for every
dataset.

## Installation

SceneZoo supports Python 3.10 to 3.14.

```bash
pip install scenezoo
```

For development, clone the repository and install it in editable mode with the
test tools:

```bash
git clone https://github.com/eamonn-zh/scenezoo.git
cd scenezoo
pip install -e ".[dev]"
```

Video frame sources (ARKitScenes MOV, MultiScan, and ScanNet++ iPhone captures)
are decoded frame-exactly with [PyAV](https://github.com/PyAV-Org/PyAV), whose
wheels bundle FFmpeg; no system `ffmpeg` installation is needed.

## Unified API

```python
from scenezoo import get_dataset, get_dataset_info, list_datasets

dataset = get_dataset("scannetv2", "/path/to/scannet")
scene_id = dataset.get_ids("val")[0]

if dataset.supports("mesh"):
    mesh = dataset.get_mesh(scene_id)

if dataset.supports("points"):
    points = dataset.get_points(scene_id)

if dataset.supports("segmentation"):
    segmentation = dataset.get_segmentation(scene_id)
    print(segmentation.domain, segmentation.labels.shape)

if dataset.supports("boxes"):
    boxes_by_id, id_to_name = dataset.get_boxes(scene_id)

if dataset.supports("frames"):
    frames = dataset.get_frames(
        scene_id,
        step=30,
        items=("rgb", "depth", "rgb_intrinsics", "world_to_camera"),
        output_size=(640, 480),
        center_crop=True,
    )

print(list_datasets())
print(get_dataset_info("scannetv2"))  # canonical release metadata
```

`list_datasets()` and `get_dataset_info()` use a static built-in catalog, so
they do not import Open3D, OpenCV, video decoders, or adapter modules. Inspect
`get_dataset_info(name).spec` for operations, mesh/box/label variants, and frame
sources. Accessing `.dataset_class` or calling `get_dataset()` intentionally
loads only the selected adapter. The `scenezoo.io` and `scenezoo.ops`
namespaces also resolve re-exported helpers lazily, so importing cache or frame
utilities does not initialize unrelated video, annotation, or geometry runtimes.

The seven common dataset calls are:

- `check(*, sample_ids=None, require_complete=False, raise_on_error=False)`
- `get_ids(split=None)`
- `get_mesh(sample_id, *, mesh_type=None)`
- `get_points(sample_id)`
- `get_segmentation(sample_id)`
- `get_boxes(sample_id, *, box_type=...)`
- `get_frames(sample_id, *, indices=None, step=1, items=..., output_size=None, center_crop=False, rotate_to_up=True)`

Unsupported operations and unsupported frame items raise
`UnsupportedOperationError`. Unknown operation names, frame item names, and
invalid argument combinations raise `ValueError`.

## Check a download before using it

Dataset checking is explicit and never runs during construction or normal data
access. It checks paths and file groups without decoding meshes, images, or
videos:

```python
dataset = get_dataset("scannetv2", "/path/to/scannet", offline=True)
report = dataset.check()
print(report)

# Check only the scenes a job is about to use. Missing requested IDs are errors.
focused = dataset.check(sample_ids=("scene0000_00", "scene0001_00"))

# Treat every missing ID from the official splits as an error and raise once.
dataset.check(require_complete=True, raise_on_error=True)
```

`DatasetCheckReport` exposes `ok`, `errors`, `warnings`, `issues`, and `stats`.
Each `CheckIssue` contains a severity, stable code, explanation, relevant path,
expected layout, and suggested action. An intentionally partial download reports
missing split scenes as warnings by default; explicitly requested scenes are
always required. `raise_on_error=True` raises `DatasetCheckError` carrying the
same report.

The checkers understand each release's archive policy. ScanNet, ScanNet++,
ARKitScenes, MultiScan, ASE, Matterport3D, Structured3D, S3DIS, and SceneNN need
their download archives extracted. ScanNet keeps its official 2D annotation
ZIPs, while 3RScan keeps each native `sequence.zip` sensor stream compressed.

## Return semantics

`get_points()` returns a `PointBatch`. Its `xyz` array is always `float32` with
shape `(N, 3)`; optional RGB, normals, semantic/instance labels, and source-view
provenance are row-aligned. Call `to_point_cloud()` only when an Open3D object
is needed.

`get_segmentation()` returns `Segmentation3D` with `labels`, `id_to_name`,
`domain` (`"point"`, `"vertex"`, or `"face"`), and `invalid_id`.

`get_boxes()` always returns `(boxes_by_id, id_to_name)`.

`get_frames()` returns a slot-based `FrameBatch`. `indices` are original video
frame numbers (or stable source positions for filename-only captures), and
`frame_keys` retains source names such as DSLR filenames. `source_frame_count`
is the source sequence length. Every requested per-frame field has `N` as its
first dimension, including constant intrinsics, which are expanded to
`(N, 3, 3)`.

The frame conventions are intentionally strict:

- depth is `float32` in metres; missing, negative, and non-finite values are `0.0`;
- extrinsics are OpenCV-style `world_to_camera` matrices with shape `(N, 4, 4)`;
- pixel coordinates follow OpenCV: integer coordinates are pixel centres;
- RGB, albedo, depth, normal/instance/semantic/confidence maps, and intrinsics use the same resize/crop transform;
- `rotate_to_up=True` rotates pixels by a right angle and applies the equivalent camera roll to `world_to_camera`, so intrinsics always stay upper-triangular pinhole matrices;
- RGB uses high-quality interpolation, while depth and discrete maps use nearest-neighbor;
- explicit `indices` preserve the requested order and cannot be combined with `step != 1`;
- default sampling filters frames with invalid camera matrices before applying `step`.

Adapters for non-pinhole captures additionally return `camera_model`,
`distortion`, and `image_sizes`. ScanNet++ uses these fields for its
`OPENCV_FISHEYE`, `OPENCV`, `PINHOLE`, and `EQUIRECTANGULAR` sources instead of
discarding the source camera model.

## Built-in datasets

| Canonical name | Aliases | Mesh | Points | Segmentation | Boxes | Frames |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| [`scannetv2`](https://scenezoo.readthedocs.io/en/latest/datasets/scannet.html) | `scannet` | raw/decimated/semantic | no | instance + NYU40, vertex | yes | RGB-D, camera, 2D labels, timestamps, IMU |
| [`scannetppv2`](https://scenezoo.readthedocs.io/en/latest/datasets/scannetpp.html) | `scannet++`, `scannetpp`, `scannet++v2` | yes | laser scan | vertex, multilabel | yes | iPhone, DSLR, panorama |
| [`arkitscenes`](https://scenezoo.readthedocs.io/en/latest/datasets/arkitscenes.html) | — | yes | no | vertex (box-derived) | yes | multi-source RGB-D, confidence |
| [`multiscan`](https://scenezoo.readthedocs.io/en/latest/datasets/multiscan.html) | — | PLY/textured | no | object/part instance + semantic, face | yes | RGB-D, confidence, camera metadata |
| [`ase`](https://scenezoo.readthedocs.io/en/latest/datasets/ase.html) | `aria-ase`, `aria-synthetic-environments` | no | semi-dense SLAM points | no | wall/door/window layout OBB | fisheye RGB, ray-depth, instances, poses |
| [`3rscan`](https://scenezoo.readthedocs.io/en/latest/datasets/3rscan.html) | `threerscan` | textured/instance | no | instance + Global/NYU40/Eigen13/RIO27/RIO7, vertex | yes | RGB-D and camera |
| [`s3dis`](https://scenezoo.readthedocs.io/en/latest/datasets/s3dis.html) | `stanford3d`, `stanford-3d` | no | XYZRGB + labels | instance + 13-class semantic, point | yes | no |
| [`matterport3d`](https://scenezoo.readthedocs.io/en/latest/datasets/matterport3d.html) | — | textured/house/region/Poisson | no | instance + semantic, face | official OBB | raw + undistorted RGB-D |
| [`scenenn`](https://scenezoo.readthedocs.io/en/latest/datasets/scenenn.html) | — | yes | no | vertex | no | no |
| [`structured3d`](https://scenezoo.readthedocs.io/en/latest/datasets/structured3d.html) | `structure3d` | reconstructed room layout | RGB-D fusion | semantic + instance, point | official OBB | perspective + panorama RGB-D and labels |

Each dataset page linked above documents the expected download layout,
dataset-specific options (label spaces, capture sources, region and room
selectors), and extra methods such as calibration, IMU, and scene metadata.

## Registering an adapter

Registration is atomic: the canonical name and all aliases are checked for
conflicts before any entry is added.

```python
from scenezoo import Dataset, register_dataset


@register_dataset("my-scenes", aliases=("mine",))
class MyScenes(Dataset):
    def _load_splits(self):
        return {"train": ["scene-001"]}

    def _load_metadata(self):
        return {"coordinate_system": "z-up"}

    def get_mesh(self, sample_id, *, mesh_type=None):
        return load_my_mesh(self.root_dir / sample_id)
```

`splits` and `metadata` are lazy `cached_property` values. Implement only the
standard methods backed reliably by the dataset; `supports()` detects overridden
methods automatically. Adapter constructors should use explicit keyword
parameters rather than a catch-all `**kwargs`. Custom adapters may override the
protected `_check()` hook to add source-specific layout diagnostics; normal users
call only `dataset.check()`.

## Cache and offline mode

Remote split and metadata files are cached under `SCENEZOO_CACHE_DIR` (or
`~/.cache/scenezoo`). Pass `offline=True` to forbid network access while allowing
previously cached remote files to be reused.

## Development

```bash
pip install -e ".[dev]" -r docs/requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check src test && ruff format --check src test
python -m sphinx -W -b html docs docs/_build/html
```

Real-data integration tests are opt-in through `SCENEZOO_TEST_*_ROOT` environment
variables; no dataset path is embedded in the test suite. The
documentation is built from [`docs/`](https://github.com/eamonn-zh/scenezoo/tree/main/docs) and published at
<https://scenezoo.readthedocs.io/en/latest/>.

## License

See [LICENSE](https://github.com/eamonn-zh/scenezoo/blob/main/LICENSE).

## Citation

Please cite SceneZoo if you use it in your research.

```bib
@misc{Zhang2026SceneZoo,
    author       = {Yiming Zhang},
    title        = {{SceneZoo}: A Unified {Python} Interface for {3D} Indoor-Scene Datasets},
    year         = {2026},
    howpublished = {\url{https://github.com/eamonn-zh/scenezoo}},
}
```
