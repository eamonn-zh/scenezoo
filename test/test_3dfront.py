from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from scenezoo import get_dataset
from scenezoo.dataset.scene import ThreeDFront


def _write_dataset(root):
    model = root / "3D-FUTURE-model" / "chair-model"
    model.mkdir(parents=True)
    # One textured triangle in the model's own (y-up) coordinates.
    (model / "raw_model.obj").write_text(
        "mtllib model.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "usemtl wood\nf 1/1 2/2 3/3\n"
    )
    (model / "model.mtl").write_text("newmtl wood\nmap_Kd texture.png\n")
    Image.fromarray(np.full((2, 2, 3), 90, np.uint8)).save(model / "texture.png")
    (root / "3D-FUTURE-model" / "model_info.json").write_text(
        json.dumps(
            [
                {
                    "model_id": "chair-model",
                    "category": "Dining Chair",
                    "super-category": "Chair",
                    "style": "Modern",
                    "theme": None,
                    "material": "Wood",
                }
            ]
        )
    )
    house = {
        "uid": "house",
        "furniture": [
            {"uid": "f1", "jid": "chair-model", "valid": True},
            {"uid": "f2", "jid": "unreleased-model"},  # not valid: skipped
        ],
        "mesh": [
            {
                "uid": "floor",
                "type": "Floor",
                "xyz": [0, 0, 0, 4, 0, 0, 4, 0, 3, 0, 0, 3],
                "faces": [0, 1, 2, 0, 2, 3],
                "uv": [0, 0, 1, 0, 1, 1, 0, 1],
                "material": "paint",
            }
        ],
        "material": [
            {"uid": "paint", "jid": "red", "color": [255, 0, 0, 255], "useColor": True}
        ],
        "scene": {
            "room": [
                {
                    "instanceid": "Bedroom-1",
                    "type": "Bedroom",
                    "children": [
                        {"ref": "floor"},
                        {
                            "ref": "f1",
                            "pos": [1, 0, 2],
                            # 90 degrees about +y, as an (x, y, z, w) quaternion.
                            "rot": [0, np.sqrt(0.5), 0, np.sqrt(0.5)],
                            "scale": [2, 1, 1],
                        },
                        {"ref": "f2", "pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
                    ],
                },
                {"instanceid": "Kitchen-2", "type": "Kitchen", "children": []},
            ]
        },
    }
    (root / "3D-FRONT").mkdir()
    (root / "3D-FRONT" / "house.json").write_text(json.dumps(house))


def test_3dfront_places_furniture_in_z_up_house_with_textures(tmp_path):
    _write_dataset(tmp_path)
    dataset = get_dataset("3dfront", tmp_path)
    assert isinstance(dataset, ThreeDFront)
    assert dataset.splits == {}
    assert dataset.get_ids() == ["house"]
    assert dataset.get_room_ids("house") == ["Bedroom-1", "Kitchen-2"]
    assert dataset.get_scene_info("house")["rooms"]["Bedroom-1"] == {
        "type": "Bedroom",
        "model_ids": ["chair-model"],
    }

    mesh = dataset.get_mesh("house")
    assert len(mesh.triangles) == 3  # two floor faces, one furniture face
    vertices = np.asarray(mesh.vertices)
    # The y-up floor (y = 0) becomes the z = 0 plane; z maps to -y.
    np.testing.assert_allclose(vertices[:4, 2], 0)
    np.testing.assert_allclose(vertices[2], [4, -3, 0])
    # scale -> rotate 90 deg about y -> translate, then y-up to z-up.
    np.testing.assert_allclose(
        vertices[4:], [[1, -2, 0], [1, 0, 0], [1, -2, 1]], atol=1e-9
    )
    assert len(mesh.textures) == 2
    assert np.asarray(mesh.triangle_material_ids).tolist() == [0, 0, 1]
    assert len(mesh.triangle_uvs) == 9
    np.testing.assert_array_equal(np.asarray(mesh.textures[0])[0, 0], [255, 0, 0])

    assert len(dataset.get_mesh("house", mesh_type="layout").triangles) == 2
    assert len(dataset.get_mesh("house", mesh_type="furniture").triangles) == 1
    with pytest.raises(KeyError, match="Unknown 3D-FRONT room"):
        dataset.get_mesh("house", room_id="Garage-9")
    with pytest.raises(ValueError, match="mesh type"):
        dataset.get_mesh("house", mesh_type="raw")

    obj = np.asarray(dataset.get_object_mesh("chair-model").vertices)
    np.testing.assert_allclose(obj, [[0, 0, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)


def test_3dfront_face_labels_and_boxes_follow_the_mesh(tmp_path):
    _write_dataset(tmp_path)
    dataset = ThreeDFront(tmp_path)

    instance = dataset.get_segmentation("house")
    assert instance.domain == "face"
    assert instance.labels.tolist() == [-1, -1, 0]
    assert instance.id_to_name == {0: "Dining Chair"}

    semantic = dataset.get_segmentation(
        "house", segmentation_type="semantic", label_space="super_category"
    )
    classes = dataset.metadata["semantic_classes"]["super_category"]
    assert [classes[i] for i in semantic.labels] == ["Floor", "Floor", "Chair"]
    room = dataset.get_segmentation("house", room_id="Bedroom-1", mesh_type="layout")
    assert room.labels.tolist() == [-1, -1]

    boxes, names = dataset.get_boxes("house")
    assert names == {0: "Dining Chair"}
    np.testing.assert_allclose(boxes[0].center, [1, -1, 0.5], atol=1e-9)
    np.testing.assert_allclose(boxes[0].extent, [2, 1, 0], atol=1e-9)
    assert dataset.get_boxes("house", room_id="Kitchen-2") == ({}, {})
    assert set(dataset.get_boxes("house", box_type="aabb")[0]) == {0}


def test_3dfront_check_reports_missing_furniture_models(tmp_path):
    _write_dataset(tmp_path)
    dataset = ThreeDFront(tmp_path)
    report = dataset.check(sample_ids=["house"])
    assert report.ok, report.format()
    assert "missing-textures" in {issue.code for issue in report.issues}

    (tmp_path / "3D-FUTURE-model" / "chair-model" / "raw_model.obj").unlink()
    report = ThreeDFront(tmp_path).check(sample_ids=["house"])
    assert "missing-furniture-models" in {issue.code for issue in report.errors}


def test_3dfront_rescales_listed_centimetre_models(tmp_path):
    _write_dataset(tmp_path)
    listed = tmp_path / "centimetre_models.txt"
    listed.write_text("# model IDs stored in centimetres\nchair-model\n")

    fixed = ThreeDFront(tmp_path, centimetre_models_file=listed)
    raw = ThreeDFront(tmp_path, centimetre_models_file=listed, fix_model_units=False)
    a = np.asarray(fixed.get_mesh("house").vertices)[4:]
    b = np.asarray(raw.get_mesh("house").vertices)[4:]
    np.testing.assert_allclose(a - a[0], (b - b[0]) * 0.01, atol=1e-12)
    np.testing.assert_allclose(
        fixed.get_boxes("house")[0][0].extent, [0.02, 0.01, 0], atol=1e-12
    )
    # Models that are not listed are left alone.
    assert np.allclose(ThreeDFront(tmp_path).get_boxes("house")[0][0].extent, [2, 1, 0])


def test_3dfront_paint_colors_are_not_shared_through_material_ids(tmp_path):
    _write_dataset(tmp_path)
    path = tmp_path / "3D-FRONT" / "house.json"
    house = json.loads(path.read_text())
    # A second floor whose paint reuses the jid "red" with another color.
    second = dict(house["mesh"][0], uid="floor2", material="paint2")
    house["mesh"].append(second)
    house["material"].append(
        {"uid": "paint2", "jid": "red", "color": [0, 0, 255, 255], "useColor": True}
    )
    house["scene"]["room"][1]["children"].append({"ref": "floor2"})
    path.write_text(json.dumps(house))

    mesh = ThreeDFront(tmp_path).get_mesh("house", mesh_type="layout")
    colors = [np.asarray(texture)[0, 0].tolist() for texture in mesh.textures]
    assert colors == [[255, 0, 0], [0, 0, 255]]
