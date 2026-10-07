from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "docs").is_dir():
    pytest.skip("needs a source checkout with docs/", allow_module_level=True)
DATASETS = {
    "ase": ("ase.py", "AriaSyntheticEnvironments"),
    "arkitscenes": ("arkitscenes.py", "ARKitScenes"),
    "matterport3d": ("matterport3d.py", "Matterport3D"),
    "multiscan": ("multiscan.py", "MultiScan"),
    "s3dis": ("s3dis.py", "S3DIS"),
    "scannet": ("scannet.py", "ScanNet"),
    "scannetpp": ("scannetpp.py", "ScanNetPP"),
    "scenenn": ("scenenn.py", "SceneNN"),
    "structured3d": ("structured3d.py", "Structured3D"),
    "3rscan": ("threerscan.py", "ThreeRScan"),
}


def _public_api(source_name: str, class_name: str):
    source = ROOT / "src/scenezoo/dataset/scene" / source_name
    tree = ast.parse(source.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    constructor = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    attributes = set()
    for node in ast.walk(constructor):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        )
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and not target.attr.startswith("_")
            ):
                attributes.add(target.attr)
    return methods, attributes


@pytest.mark.parametrize(("slug", "adapter"), DATASETS.items())
def test_dataset_page_covers_public_api_and_official_links(slug, adapter):
    source_name, class_name = adapter
    page = (ROOT / "docs/datasets" / f"{slug}.rst").read_text()
    methods, attributes = _public_api(source_name, class_name)

    for heading in (
        "Quick example",
        "Download layout",
        "Official resources",
        "API reference",
    ):
        assert heading in page, f"{slug}.rst has no {heading!r} section"
    assert "dataset.check()" in page
    assert page.count("https://") >= 3
    assert f".. autoclass:: scenezoo.dataset.scene.{class_name}" in page
    for name in methods | attributes:
        assert name in page, f"{slug}.rst does not document {name}"


def test_common_dataset_api_is_documented():
    concepts = (ROOT / "docs/guide/concepts.rst").read_text()
    for name in (
        "root_dir",
        "cache_dir",
        "offline",
        "invalid_obj_id",
        "splits",
        "metadata",
        "spec",
        "name",
        "get_ids",
        "check",
        "supports",
        "capabilities",
        "open_file",
    ):
        assert f"``{name}" in concepts, f"concepts.rst does not document {name}"
