from __future__ import annotations

import io
import json
import shutil
import zipfile

import numpy as np
import pytest
from PIL import Image

import scenezoo.dataset.scene.matterport3d as matterport_module
from scenezoo.dataset.scene import Matterport3D


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_files(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload)


def _ply(*, annotated=True):
    properties = (
        """
property int material_id
property int segment_id
property int category_id"""
        if annotated
        else ""
    )
    suffix = " 10 0 2" if annotated else ""
    return f"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices{properties}
end_header
0 0 0
1 0 0
0 1 0
0 0 1
3 0 1 2{suffix}
3 0 2 3{suffix}
"""


def _house(*, unlabeled_object=False):
    header = [
        "H",
        "scene",
        "-",
        "0",
        "0",
        "0",
        "0",
        "0",
        "2" if unlabeled_object else "1",
        "1",
        "1",
        "0",
        "1",
        *(["0"] * 5),
        "0",
        "0",
        "0",
        "2",
        "2",
        "2",
        *(["0"] * 5),
    ]
    level = [
        "L",
        "0",
        "1",
        "-",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "2",
        "2",
        "2",
        *(["0"] * 5),
    ]
    region = [
        "R",
        "0",
        "0",
        "0",
        "0",
        "b",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "2",
        "2",
        "2",
        "2",
        *(["0"] * 4),
    ]
    category = ["C", "0", "2", "chair", "3", "chair", *(["0"] * 5)]
    obj = [
        "O",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "0",
        "0",
        "0",
        "1",
        "0",
        "0.5",
        "1",
        "1.5",
        *(["0"] * 8),
    ]
    rows = [header, level, region, category, obj]
    if unlabeled_object:
        # Official houses mark unlabeled objects with category index -1.
        rows.append(["O", "1", "0", "-1", *obj[4:]])
    return "ASCII 1.1\n" + "\n".join(" ".join(row) for row in rows) + "\n"


@pytest.fixture
def matterport_root(tmp_path):
    splits = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.txt"
        path.write_text("scene\n\n" if split == "train" else "")
        splits[split] = path
    labels = tmp_path / "category_mapping.tsv"
    labels.write_text(
        "index\traw_category\tmpcat40index\tmpcat40\tnyu40id\tnyu40class\t"
        "eigen13id\teigen13class\n"
        "2\tchair\t3\tchair\t5\tchair\t4\tObject\n"
    )
    scene = tmp_path / "v1" / "scans" / "scene"
    scene.mkdir(parents=True)
    _write_files(
        scene / "house_segmentations",
        {"scene.house": _house(), "scene.ply": _ply()},
    )
    _write_files(
        scene / "region_segmentations",
        {
            "region0.ply": _ply(),
            "region1.ply": _ply(annotated=False),
            "region1.fsegs.json": json.dumps({"segIndices": [5, 6]}),
            "region1.semseg.json": json.dumps(
                {"segGroups": [{"label": "chair", "segments": [5]}]}
            ),
        },
    )
    _write_files(scene / "matterport_mesh", {"scene.obj": "o placeholder\n"})
    _write_files(scene / "poisson_meshes", {"scene_9.ply": _ply()})

    rgb0 = np.asarray(
        [[[10, 0, 0], [20, 0, 0]], [[30, 0, 0], [40, 0, 0]]],
        dtype=np.uint8,
    )
    rgb1 = rgb0 + 1
    depth0 = np.asarray([[0, 4000], [8000, 12000]], dtype=np.uint16)
    depth1 = depth0 + 4000
    identity = " ".join(map(str, np.eye(4).reshape(-1)))
    translated = np.eye(4)
    translated[0, 3] = 1
    config = (
        "dataset matterport\n"
        "n_images 2\n"
        "intrinsics_matrix 2 0 0.5 0 2 0 0 0 1\n"
        f"scan pano_d0_0.png pano_i0_0.jpg {identity}\n"
        f"scan pano_d0_1.png pano_i0_1.jpg {' '.join(map(str, translated.reshape(-1)))}\n"
    )
    _write_files(scene / "undistorted_camera_parameters", {"scene.conf": config})
    _write_files(
        scene / "undistorted_color_images",
        {"pano_i0_0.jpg": _png(rgb0), "pano_i0_1.jpg": _png(rgb1)},
    )
    _write_files(
        scene / "undistorted_depth_images",
        {"pano_d0_0.png": _png(depth0), "pano_d0_1.png": _png(depth1)},
    )

    _write_files(scene / "matterport_color_images", {"raw_i0_0.jpg": _png(rgb0)})
    _write_files(scene / "matterport_depth_images", {"raw_d0_0.png": _png(depth0)})
    _write_files(scene / "matterport_camera_poses", {"raw_pose_0_0.txt": identity})
    _write_files(
        scene / "matterport_camera_intrinsics",
        {"raw_intrinsics_0.txt": "2 2 2 2 0.5 0.5 0.1 0.2 0.01 0.02 0.3"},
    )
    return tmp_path, labels, splits


def _dataset(fixture):
    root, labels, splits = fixture
    return Matterport3D(
        root,
        label_mapping_file=labels,
        train_split_file=splits["train"],
        val_split_file=splits["val"],
        test_split_file=splits["test"],
    )


def test_official_layout_regions_meshes_and_metadata(matterport_root, monkeypatch):
    dataset = _dataset(matterport_root)
    assert dataset.scans_dir == matterport_root[0] / "v1" / "scans"
    assert dataset.get_ids("train") == ["scene"]
    assert dataset.get_ids() == ["scene"]
    assert dataset.get_region_ids("scene") == ["0", "1"]
    assert dataset.get_frame_sources("scene") == ("raw", "undistorted")
    assert dataset.metadata["label_spaces"]["mpcat40"] == {3: "chair"}
    assert dataset.get_house_info("scene")["regions"][0]["label"] == "b"

    calls = []
    monkeypatch.setattr(
        matterport_module,
        "load_triangle_mesh",
        lambda path, **options: calls.append((path.name, options)) or "mesh",
    )
    assert dataset.get_mesh("scene") == "mesh"
    assert dataset.get_mesh("scene", mesh_type="region", region_id="region1") == "mesh"
    assert dataset.get_mesh("scene", mesh_type="poisson", poisson_level=9) == "mesh"
    assert dataset.get_mesh("scene", mesh_type="raw") == "mesh"
    assert [call[0] for call in calls] == [
        "scene.ply",
        "region1.ply",
        "scene_9.ply",
        "scene.obj",
    ]


def test_face_instance_semantic_and_json_label_fallback(matterport_root):
    dataset = _dataset(matterport_root)
    instances = dataset.get_segmentation("scene", region_id=0)
    assert instances.domain == "face"
    assert instances.labels.tolist() == [0, 0]
    assert instances.id_to_name == {0: "chair"}

    semantic = dataset.get_segmentation(
        "scene", region_id=0, segmentation_type="semantic", label_space="nyu40"
    )
    assert semantic.labels.tolist() == [5, 5]
    assert semantic.id_to_name == {5: "chair"}

    fallback = dataset.get_segmentation("scene", region_id=1)
    assert fallback.labels.tolist() == [0, -1]
    assert fallback.id_to_name == {0: "chair"}


def test_house_boxes(matterport_root):
    dataset = _dataset(matterport_root)
    boxes, names = dataset.get_boxes("scene", box_type="obb_gt", region_id=0)
    assert names == {0: "chair"}
    np.testing.assert_allclose(boxes[0].center, [1, 1, 1])
    np.testing.assert_allclose(boxes[0].extent, [1, 2, 3])


def test_unlabeled_house_objects_have_no_box(matterport_root):
    house = matterport_root[0] / "v1/scans/scene/house_segmentations/scene.house"
    house.write_text(_house(unlabeled_object=True))
    dataset = _dataset(matterport_root)
    boxes, names = dataset.get_boxes("scene")
    assert names == {0: "chair"}
    np.testing.assert_allclose(boxes[0].center, [1, 1, 1])
    np.testing.assert_allclose(boxes[0].extent, [1, 2, 3])


def test_undistorted_frames_keep_images_and_convert_pose_axes(matterport_root):
    # Undistorted images are stored top-left like ordinary images; only the
    # -Z-forward, +Y-up pose convention differs from OpenCV. Verified against
    # real Matterport3D depth and meshes.
    dataset = _dataset(matterport_root)
    frames = dataset.get_frames(
        "scene",
        source="undistorted",
        indices=[1],
        items=("rgb", "depth", "rgb_intrinsics", "depth_intrinsics", "world_to_camera"),
    )
    assert frames.indices.tolist() == [1]
    assert frames.source_frame_count == 2
    assert frames.frame_keys == ("pano_i0_1.jpg",)
    assert frames.rgb.dtype == np.uint8
    assert frames.rgb[0, 0, 0, 0] == 11
    assert frames.depth.dtype == np.float32
    np.testing.assert_allclose(frames.depth[0], [[1, 2], [3, 4]])
    np.testing.assert_allclose(
        frames.rgb_intrinsics[0], [[2, 0, 0.5], [0, 2, 0], [0, 0, 1]]
    )
    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    expected[0, 3] = -1
    np.testing.assert_allclose(frames.world_to_camera[0], expected)


@pytest.mark.parametrize("image_item", ["depth", None])
def test_intrinsics_without_matching_image(matterport_root, image_item):
    dataset = _dataset(matterport_root)
    items = ("rgb_intrinsics", "depth_intrinsics", "world_to_camera")
    frames = dataset.get_frames(
        "scene",
        source="undistorted",
        indices=[1],
        items=items if image_item is None else (image_item, *items),
    )
    expected = [[2, 0, 0.5], [0, 2, 0], [0, 0, 1]]
    np.testing.assert_allclose(frames.rgb_intrinsics[0], expected)
    np.testing.assert_allclose(frames.depth_intrinsics[0], expected)


def test_raw_frames_keep_opencv_distortion(matterport_root):
    dataset = _dataset(matterport_root)
    frames = dataset.get_frames(
        "scene",
        source="raw",
        items=("rgb", "depth", "rgb_intrinsics", "world_to_camera"),
    )
    assert frames.camera_model == "OPENCV"
    np.testing.assert_allclose(frames.distortion[0], [0.1, 0.2, 0.01, 0.02, 0.3])
    np.testing.assert_allclose(frames.depth[0], [[0, 1], [2, 3]])
    np.testing.assert_allclose(frames.world_to_camera[0], np.eye(4))


def test_explicit_indices_cannot_use_step(matterport_root):
    dataset = _dataset(matterport_root)
    with pytest.raises(ValueError, match="step must be 1"):
        dataset.get_frames("scene", indices=[0], step=2)


def test_zip_bundle_requires_extraction(matterport_root):
    scene = matterport_root[0] / "v1" / "scans" / "scene"
    directory = scene / "matterport_color_images"
    archive_path = scene / "matterport_color_images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in directory.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(scene))
    shutil.rmtree(directory)

    dataset = _dataset(matterport_root)
    report = dataset.check(sample_ids=("scene",))
    assert not report.ok
    assert any(issue.code == "incomplete-frame-bundle-group" for issue in report.errors)
    assert any(issue.code == "archives-need-extraction" for issue in report.warnings)
    with pytest.raises(FileNotFoundError, match="not extracted"):
        dataset.get_frames("scene", source="raw", items=("rgb",))
