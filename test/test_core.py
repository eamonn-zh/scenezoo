from __future__ import annotations

import fsspec
import numpy as np
import pytest
import subprocess
import sys

from scenezoo.dataset import (
    Dataset,
    DatasetSpec,
    FrameSourceSpec,
    PointBatch,
    UnsupportedOperationError,
    get_dataset_info,
)
from scenezoo.dataset.registry import DatasetRegistry
from scenezoo.dataset.types import FrameBatch
from scenezoo.io.cache import open_cached_file


class TinyDataset(Dataset):
    split_loads = 0
    metadata_loads = 0

    def _load_splits(self):
        type(self).split_loads += 1
        return {"train": ["a", "b"]}

    def _load_metadata(self):
        type(self).metadata_loads += 1
        return {"version": 1}

    def get_mesh(self, sample_id, *, mesh_type=None):
        return sample_id, mesh_type


def test_dataset_is_lazy_and_detects_capabilities(tmp_path):
    TinyDataset.split_loads = TinyDataset.metadata_loads = 0
    dataset = TinyDataset(tmp_path)
    assert TinyDataset.split_loads == TinyDataset.metadata_loads == 0
    assert dataset.get_ids("train") == ["a", "b"]
    assert dataset.get_ids("train") == ["a", "b"]
    assert dataset.metadata == {"version": 1}
    assert dataset.metadata == {"version": 1}
    assert TinyDataset.split_loads == TinyDataset.metadata_loads == 1
    assert TinyDataset.supports("mesh")
    assert not TinyDataset.supports("frames")
    with pytest.raises(UnsupportedOperationError):
        dataset.get_frames("a")
    with pytest.raises(ValueError, match="Unknown operation"):
        TinyDataset.supports("typo")


def test_registry_aliases_and_registration_are_atomic(tmp_path):
    registry = DatasetRegistry()
    registry.register("alpha", aliases=("a",))(TinyDataset)
    assert registry.create("A", tmp_path).__class__ is TinyDataset
    assert registry.list() == ["alpha"]
    assert registry.list(include_aliases=True) == ["a", "alpha"]
    assert registry.info("a").name == "alpha"

    class OtherDataset(Dataset):
        def _load_splits(self):
            return {}

    with pytest.raises(ValueError, match="already registered"):
        registry.register("beta", aliases=("a", "unused"))(OtherDataset)
    assert not registry.contains("beta")
    assert not registry.contains("unused")


def test_registry_rejects_duplicate_aliases_before_mutating():
    registry = DatasetRegistry()
    with pytest.raises(ValueError, match="must be unique"):
        registry.register("same", aliases=("SAME",))(TinyDataset)
    assert registry.list() == []


def test_builtin_catalog_is_import_light_and_machine_readable():
    script = """
import json
import sys
import scenezoo
info = scenezoo.get_dataset_info('scannet++')
print(json.dumps({
    'names': scenezoo.list_datasets(),
    'canonical': info.name,
    'operations': info.spec.operations,
    'sources': [source.name for source in info.spec.frame_sources],
    'heavy': [name for name in ('open3d', 'pandas', 'av') if name in sys.modules],
    'adapter_loaded': 'scenezoo.dataset.scene.scannetpp' in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    values = json.loads(result.stdout)
    assert "scannetppv2" in values["names"]
    assert "scannetv2" in values["names"]
    assert "scannetpp" not in values["names"]
    assert "scannet" not in values["names"]
    assert values["canonical"] == "scannetppv2"
    assert "frames" in values["operations"]
    assert "dslr" in values["sources"]
    assert values["heavy"] == []
    assert not values["adapter_loaded"]


@pytest.mark.parametrize(
    "canonical, aliases",
    [
        ("scannetv2", ("scannet",)),
        ("scannetppv2", ("scannetpp", "scannet++", "scannet++v2")),
    ],
)
def test_scannet_release_names(canonical, aliases, tmp_path):
    from scenezoo import get_dataset

    dataset = get_dataset(canonical, tmp_path, offline=True)
    assert dataset.name == canonical
    assert "v2" in get_dataset_info(canonical).description
    for alias in aliases:
        assert get_dataset_info(alias).name == canonical
        assert type(get_dataset(alias, tmp_path, offline=True)) is type(dataset)


def test_io_and_ops_submodules_do_not_import_unrelated_heavy_dependencies():
    script = """
import json
import sys
import scenezoo.ops as ops
from scenezoo.io.cache import open_cached_file
print(json.dumps({
    'symbols': [open_cached_file.__name__, 'select_frame_indices' in ops.__all__],
    'unexpected': [
        name for name in ('av', 'open3d', 'plyfile') if name in sys.modules
    ],
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    values = json.loads(result.stdout)
    assert values["symbols"] == ["open_cached_file", True]
    assert values["unexpected"] == []


def test_dataset_specs_describe_frame_sources():
    spec = DatasetSpec(
        sample_unit="scene",
        operations=("frames",),
        frame_sources=(FrameSourceSpec("rgbd", ("rgb", "depth")),),
    )
    assert spec.frame_source(" RGBD ").items == ("rgb", "depth")
    with pytest.raises(KeyError, match="Available sources: rgbd"):
        spec.frame_source("missing")
    assert get_dataset_info("structure3d").spec.sample_unit == "scene"
    with pytest.raises(ValueError, match="Unknown dataset operations"):
        DatasetSpec(operations=("typo",))
    with pytest.raises(ValueError, match="requires the frames operation"):
        DatasetSpec(frame_sources=(FrameSourceSpec("rgb", ("rgb",)),))
    with pytest.raises(ValueError, match="must match"):
        DatasetRegistry().register("wrong", spec=DatasetSpec(operations=("frames",)))(
            TinyDataset
        )


def test_frame_batch_schema_and_depth_normalization():
    batch = FrameBatch(
        indices=[2, 0],
        source_frame_count=3,
        depth=np.array([[[1.0, np.nan]], [[-1.0, 2.0]]], dtype=np.float64),
        rgb_intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
        world_to_camera=np.repeat(np.eye(4)[None], 2, axis=0),
        depth_mode="ray_distance",
    )
    assert batch.indices.dtype == np.int64
    assert batch.depth.dtype == np.float32
    assert batch.depth_mode == "ray_distance"
    assert batch.depth.tolist() == [[[1.0, 0.0]], [[0.0, 2.0]]]
    with pytest.raises(ValueError, match="outside"):
        FrameBatch(indices=[3], source_frame_count=3)
    with pytest.raises(ValueError, match="expected 1"):
        FrameBatch(indices=[0], source_frame_count=1, rgb=np.empty((2, 1, 1, 3)))


def test_point_batch_validates_aligned_attributes():
    batch = PointBatch(
        xyz=[[0, 1, 2], [3, 4, 5]],
        rgb=[[0, 127, 255], [3, 4, 5]],
        semantic_labels=[1, 2],
        view_ids=[0, 1],
        view_keys=("left", "right"),
    )
    assert len(batch) == 2
    assert batch.xyz.dtype == np.float32
    assert batch.rgb.dtype == np.uint8
    assert batch.semantic_labels.dtype == np.int32
    with pytest.raises(ValueError, match="shape"):
        PointBatch(xyz=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unknown view"):
        PointBatch(xyz=np.zeros((1, 3)), view_ids=[1], view_keys=("only",))


def test_remote_cache_can_be_reused_offline(tmp_path):
    url = "memory://scenezoo-tests/split.txt"
    filesystem = fsspec.filesystem("memory")
    filesystem.pipe("/scenezoo-tests/split.txt", b"one\ntwo\n")
    with open_cached_file(url, "r", cache_dir=tmp_path) as handle:
        assert handle.read() == "one\ntwo\n"
    filesystem.rm("/scenezoo-tests/split.txt")
    with open_cached_file(url, "r", cache_dir=tmp_path, offline=True) as handle:
        assert handle.read() == "one\ntwo\n"
    with pytest.raises(FileNotFoundError):
        with open_cached_file(
            "memory://scenezoo-tests/missing.txt",
            "r",
            cache_dir=tmp_path,
            offline=True,
        ):
            pass


def test_file_uri_is_opened_as_a_local_file(tmp_path):
    path = tmp_path / "metadata.txt"
    path.write_text("local")
    with open_cached_file(path.as_uri(), "r") as handle:
        assert handle.read() == "local"
