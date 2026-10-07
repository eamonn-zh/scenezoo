from __future__ import annotations

import numpy as np
import pytest

import scenezoo.dataset.scene.scenenn as scenenn_module
from scenezoo.dataset.scene import SceneNN


def _write_splits(tmp_path):
    paths = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.txt"
        path.write_text("scene\n" if split == "train" else "")
        paths[split] = path
    return paths


def _write_scenenn_ply(path):
    path.write_text(
        """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 1
property list uchar int vertex_indices
end_header
0 0 0 255 0 0
1 0 0 0 255 0
0 1 0 255 0 0
3 0 1 2
"""
    )


def test_scenenn_metadata_mesh_alignment_and_vertex_segmentation(tmp_path, monkeypatch):
    categories = tmp_path / "category.csv"
    categories.write_text("office,scene\n#All,scene\n")
    scene = tmp_path / "scene"
    scene.mkdir()
    _write_scenenn_ply(scene / "scene.ply")
    (scene / "scene_color.ply").write_text("placeholder")
    (scene / "scene.xml").write_text(
        '<root><label id="3" color="255 0 0" text="chair" /></root>'
    )

    class Mesh:
        def __init__(self, count):
            self.vertices = np.zeros((count, 3))

    monkeypatch.setattr(scenenn_module, "load_triangle_mesh", lambda path: Mesh(3))
    dataset = SceneNN(tmp_path, room_category_file=categories)
    assert dataset.splits == {}
    assert dataset.get_ids() == ["scene"]
    assert dataset.metadata == {"room_categories": {"office": ["scene"]}}
    assert len(dataset.get_mesh("scene").vertices) == 3
    segmentation = dataset.get_segmentation("scene")
    assert segmentation.domain == "vertex"
    assert segmentation.labels.tolist() == [3, -1, 3]
    assert segmentation.id_to_name == {3: "chair"}

    monkeypatch.setattr(
        scenenn_module,
        "load_triangle_mesh",
        lambda path: Mesh(2 if path.name.endswith("_color.ply") else 3),
    )
    with pytest.raises(scenenn_module.DataFormatError, match="different vertex counts"):
        dataset.get_mesh("scene")
