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

<p align="center">
  <img alt="RGB, depth, normals, and segmentation of one ScanNet++ frame; the rendered mesh; and the 3D mesh (textured and wireframe), camera trajectory, and 3D boxes" src="https://raw.githubusercontent.com/eamonn-zh/scenezoo/main/docs/_static/teaser.jpg" width="100%">
</p>

## Installation

SceneZoo supports Python 3.10 to 3.14.

```bash
pip install scenezoo
```

## Quick start

Every dataset is read through the same calls. Each call returns NumPy arrays
or Open3D objects, with the same conventions for every dataset.

```python
from scenezoo import get_dataset

dataset = get_dataset("scannetppv2", "/path/to/scannetpp")
scene_id = dataset.get_ids("nvs_sem_val")[0]

mesh = dataset.get_mesh(scene_id)                  # Open3D TriangleMesh
labels = dataset.get_segmentation(scene_id)        # per-vertex labels
boxes, names = dataset.get_boxes(scene_id)         # 3D boxes and class names
frames = dataset.get_frames(
    scene_id,
    source="iphone",
    indices=[0, 60, 120],
    items=("rgb", "depth", "rgb_intrinsics", "world_to_camera"),
)
print(frames.rgb.shape, frames.world_to_camera.shape)
```

See the [quick start](https://scenezoo.readthedocs.io/en/latest/get_started/quickstart.html) and the
[user guide](https://scenezoo.readthedocs.io/en/latest/guide/concepts.html) for splits, frames and camera
conventions, checking a download, and more.

## Built-in datasets

SceneZoo currently has built-in adapters for the datasets below. Other datasets
can be registered locally and used through the same API; see
[Adding your own dataset](#adding-your-own-dataset).

| Canonical name | Aliases | Mesh | Points | Segmentation | Boxes | Frames |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| [`scannetv2`](https://scenezoo.readthedocs.io/en/latest/datasets/scannet.html) | `scannet` | raw/decimated/semantic | no | instance + NYU40, vertex | yes | RGB-D, camera, 2D labels, timestamps, IMU |
| [`scannetppv2`](https://scenezoo.readthedocs.io/en/latest/datasets/scannetpp.html) | `scannet++`, `scannetpp`, `scannet++v2` | yes | laser scan | vertex, multilabel | yes | iPhone, DSLR, panorama |
| [`arkitscenes`](https://scenezoo.readthedocs.io/en/latest/datasets/arkitscenes.html) | — | yes | no | vertex (box-derived) | yes | multi-source RGB-D, confidence |
| [`multiscan`](https://scenezoo.readthedocs.io/en/latest/datasets/multiscan.html) | — | PLY/textured | no | object/part instance + semantic, face | yes | RGB-D, confidence, camera metadata |
| [`ase`](https://scenezoo.readthedocs.io/en/latest/datasets/ase.html) | `aria-ase`, `aria-synthetic-environments` | no | semi-dense SLAM points | no | wall/door/window layout OBB | fisheye RGB, ray-depth, instances, poses |
| [`3rscan`](https://scenezoo.readthedocs.io/en/latest/datasets/3rscan.html) | `threerscan` | textured/instance | no | instance + Global/NYU40/Eigen13/RIO27/RIO7, vertex | yes | RGB-D and camera |
| [`s3dis`](https://scenezoo.readthedocs.io/en/latest/datasets/s3dis.html) | `stanford3d`, `stanford-3d` | no | XYZRGB + labels | instance + 13-class semantic, point | yes | no |
| [`matterport3d`](https://scenezoo.readthedocs.io/en/latest/datasets/matterport3d.html) | — | textured/house/region/Poisson | no | instance + semantic, face | official OBB | raw + undistorted RGB-D, normal maps |
| [`scenenn`](https://scenezoo.readthedocs.io/en/latest/datasets/scenenn.html) | — | yes | no | vertex | no | no |
| [`structured3d`](https://scenezoo.readthedocs.io/en/latest/datasets/structured3d.html) | `structure3d` | reconstructed room layout | RGB-D fusion | semantic + instance, point | official OBB | perspective + panorama RGB-D and labels |
| [`3dfront`](https://scenezoo.readthedocs.io/en/latest/datasets/3dfront.html) | `3d-front` | textured house/room/layout/furniture | no | furniture instance + category, layout type, face | furniture OBB | no |

## Adding your own dataset

Subclass `Dataset`, register it under a name, and implement the operations
your data supports:

```python
from scenezoo import Dataset, get_dataset, register_dataset


@register_dataset("my-scenes")
class MyScenes(Dataset):
    def _load_splits(self):
        return {"train": ["scene-001"]}

    def get_mesh(self, sample_id, *, mesh_type=None):
        return load_my_mesh(self.root_dir / sample_id)


dataset = get_dataset("my-scenes", "/path/to/my_scenes")
```

The [guide](https://scenezoo.readthedocs.io/en/latest/guide/custom_datasets.html) walks through a complete example.

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
