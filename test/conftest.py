"""Small stand-ins for official metadata files, so unit tests stay offline.

The adapters download these files from the official repositories on first
use; tests pass these local copies explicitly instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def multiscan_metadata(tmp_path_factory):
    """MultiScan benchmark split and label maps in the official CSV formats."""

    root = tmp_path_factory.mktemp("multiscan_metadata")
    splits = ["scanId,split"]
    for index in range(273):  # 257 scans in official splits, 16 with none
        split = ("train", "val", "test")[index % 3] if index < 257 else ""
        splits.append(f"scene_{index:05d}_00,{split}")
    (root / "scans_split.csv").write_text("\n".join(splits) + "\n")
    (root / "object_semantic_label_map.csv").write_text(
        "objectName,objectSemanticName,objectSemanticId\n"
        "floor,floor,1\n"
        "cabinet,cabinet,7\n"
        "wall_cabinet,cabinet,7\n"
        "cabinet_otherroom,cabinet,7\n"
    )
    (root / "part_semantic_label_map.csv").write_text(
        "partName,partSemanticName,partSemanticId\ndoor,door,2\ndrawer,drawer,3\n"
    )
    return {
        "split_file": root / "scans_split.csv",
        "object_label_file": root / "object_semantic_label_map.csv",
        "part_label_file": root / "part_semantic_label_map.csv",
    }


@pytest.fixture
def structured3d_metadata(tmp_path_factory):
    """Structured3D label IDs, room types, and errata in the official formats."""

    root = tmp_path_factory.mktemp("structured3d_metadata")
    names = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door"]
    names += [f"class{index}" for index in range(9, 40)] + ["otherprop"]
    (root / "labelids.txt").write_text(
        "\n".join(f"{index}\t{name}" for index, name in enumerate(names, 1))
    )
    (root / "room_types.txt").write_text("living room\nkitchen\nbedroom\nbathroom\n")
    (root / "errata.txt").write_text(
        "# invalid scene\n"
        + "".join(f"scene_{n:05d}\n" for n in (1155, 1714, 1816, 3398, 1192, 1852))
        + "# a pair of junctions are not aligned along the x-axis\n"
        "scene_01778_room_858455\n"
    )
    return {
        "label_name_file": root / "labelids.txt",
        "room_type_file": root / "room_types.txt",
        "errata_file": root / "errata.txt",
    }


@pytest.fixture
def rio_label_mapping(tmp_path_factory):
    """A two-class excerpt of the official 3RScan class mapping table."""

    path = tmp_path_factory.mktemp("3rscan_metadata") / "mapping.csv"
    path.write_text(
        '"Classes of 3RScan with mappings to other semantics.",,,,,,,,,,,,,\n'
        "Global ID,Label,,NYU40 Mapping,,Eigen Mapping,,RIO27 Mapping,,"
        "RIO7 Mapping,#sum,#train,#test,#val\n"
        "1,air conditioner,40,otherprop,7,Objects,26,object,0,-,32,32,3,0\n"
        "4,armchair,5,chair,4,Chair,5,chair,1,seating,375,332,38,43\n"
    )
    return path


@pytest.fixture
def scannet_splits(tmp_path_factory):
    """Official-format ScanNet v2 split lists with a few scene IDs each."""

    root = tmp_path_factory.mktemp("scannet_splits")
    scenes = {
        "train": ["scene0000_00", "scene0000_01"],
        "val": ["scene0011_00"],
        "test": ["scene0707_00"],
    }
    paths = {}
    for split, ids in scenes.items():
        paths[f"{split}_split_file"] = root / f"scannetv2_{split}.txt"
        paths[f"{split}_split_file"].write_text("\n".join(ids) + "\n")
    return paths
