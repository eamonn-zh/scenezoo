"""3D-FRONT houses assembled with their 3D-FUTURE furniture models."""

from __future__ import annotations

import json
import operator
from functools import cached_property
from importlib.resources import files

import numpy as np
import open3d as o3d
from cachetools import LRUCache, cachedmethod
from scipy.spatial.transform import Rotation

from ...ops.geometry import construct_box, load_triangle_mesh
from ..base import DataFormatError, Dataset
from ..registry import register_dataset
from ..types import Segmentation3D

# 3D-FRONT is y-up; SceneZoo returns z-up coordinates: (x, y, z) -> (x, -z, y).
_Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
_LABEL_SPACES = ("category", "super_category")
_BOX_TYPES = ("obb_gt", "aabb", "obb", "mobb", "mobb_gravity")
_MESH_TYPES = ("full", "layout", "furniture")
_SOLID_GRAY = o3d.geometry.Image(np.full((1, 1, 3), 200, np.uint8))
# Architectural element types in the 2022 release; other values become "Other".
_LAYOUT_TYPES = (
    "Back", "Baseboard", "BayWindow", "Beam", "Cabinet", "Cabinet/LightBand",
    "Ceiling", "Column", "Cornice", "CustomizedBackgroundModel",
    "CustomizedCeiling", "CustomizedFeatureWall", "CustomizedFixedFurniture",
    "CustomizedFurniture", "CustomizedPersonalizedModel", "CustomizedPlatform",
    "Customized_wainscot", "Door", "ExtrusionCustomizedBackgroundWall",
    "ExtrusionCustomizedCeilingModel", "Floor", "Flue", "Front", "Hole",
    "LightBand", "Other", "Pocket", "SewerPipe", "SlabBottom", "SlabSide",
    "SlabTop", "SmartCustomizedCeiling", "WallBottom", "WallInner", "WallOuter",
    "WallTop", "Window",
)  # fmt: skip


@register_dataset(
    "3dfront",
    aliases=("3d-front",),
    description="3D-FRONT houses with 3D-FUTURE furniture, rooms, and labels.",
)
class ThreeDFront(Dataset):
    """Read 3D-FRONT house JSONs and place their 3D-FUTURE furniture models."""

    def __init__(
        self,
        root_dir,
        *,
        scene_dir="3D-FRONT",
        model_dir="3D-FUTURE-model",
        texture_dir="3D-FRONT-texture",
        model_info_file="3D-FUTURE-model/model_info.json",
        fix_model_units=True,
        centimetre_models_file=None,
        invalid_obj_id=-1,
        cache_dir=None,
        offline=False,
    ):
        super().__init__(
            root_dir,
            cache_dir=cache_dir,
            offline=offline,
            invalid_obj_id=invalid_obj_id,
        )
        self.scene_dir = self.root_dir / scene_dir
        self.model_dir = self.root_dir / model_dir
        self.texture_dir = self.root_dir / texture_dir
        self.model_info_file = self.root_dir / model_info_file
        self.fix_model_units = bool(fix_model_units)
        self.centimetre_models_file = centimetre_models_file or (
            files("scenezoo.metadata") / "3dfuture_centimetre_models.txt"
        )
        self._house_cache = LRUCache(maxsize=8)
        self._model_cache = LRUCache(maxsize=256)
        self._texture_cache = LRUCache(maxsize=512)

    # ------------------------------------------------------------------ ids

    def _load_splits(self):
        return {}  # 3D-FRONT has no official splits.

    def _load_sample_ids(self):
        return sorted(path.stem for path in self.scene_dir.glob("*.json"))

    def _load_metadata(self):
        with open(self.model_info_file, encoding="utf-8") as handle:
            records = json.load(handle)
        models = {
            str(record["model_id"]): {
                "category": _name(record.get("category")),
                "super_category": _name(record.get("super-category")),
                "style": record.get("style"),
                "theme": record.get("theme"),
                "material": record.get("material"),
            }
            for record in records
        }
        classes = {
            space: (
                sorted({model[space] for model in models.values()} | {"unknown"})
                + list(_LAYOUT_TYPES)
            )
            for space in _LABEL_SPACES
        }
        return {"models": models, "semantic_classes": classes, "up_axis": "z"}

    # ---------------------------------------------------------------- house

    @cachedmethod(operator.attrgetter("_house_cache"))
    def _house(self, sample_id):
        path = self.scene_dir / f"{sample_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"3D-FRONT house JSON does not exist: {path}")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        try:
            # Like the official toolbox, place only furniture marked valid.
            furniture = {
                item["uid"]: str(item["jid"])
                for item in data.get("furniture", [])
                if item.get("valid")
            }
            meshes = {item["uid"]: item for item in data.get("mesh", [])}
            materials = {item["uid"]: item for item in data.get("material", [])}
            rooms = {}
            instances = []
            for room in data["scene"]["room"]:
                room_id = str(room["instanceid"])
                layout, objects = [], []
                for child in room.get("children", []):
                    ref = child["ref"]
                    if ref in furniture:
                        instance = {
                            "id": len(instances),
                            "room_id": room_id,
                            "model_id": furniture[ref],
                            "pos": np.asarray(child.get("pos", (0, 0, 0)), float),
                            "rot": np.asarray(child.get("rot", (0, 0, 0, 1)), float),
                            "scale": np.asarray(child.get("scale", (1, 1, 1)), float),
                        }
                        instances.append(instance)
                        objects.append(instance)
                    elif ref in meshes:
                        layout.append(ref)
                rooms[room_id] = {
                    "type": str(room["type"]),
                    "layout": layout,
                    "objects": objects,
                }
        except (KeyError, TypeError) as exc:
            raise DataFormatError(f"Invalid 3D-FRONT house JSON: {path}") from exc
        return {"rooms": rooms, "meshes": meshes, "materials": materials}

    def get_room_ids(self, sample_id) -> list[str]:
        """Return the room instance IDs of one house, for example ``Bedroom-1234``."""

        return list(self._house(str(sample_id))["rooms"])

    def get_scene_info(self, sample_id) -> dict:
        """Return each room's type and the 3D-FUTURE model IDs placed in it."""

        rooms = self._house(str(sample_id))["rooms"]
        return {
            "rooms": {
                room_id: {
                    "type": room["type"],
                    "model_ids": [item["model_id"] for item in room["objects"]],
                }
                for room_id, room in rooms.items()
            }
        }

    def _selected_rooms(self, sample_id, room_id):
        rooms = self._house(str(sample_id))["rooms"]
        if room_id is None:
            return list(rooms.values())
        try:
            return [rooms[str(room_id)]]
        except KeyError as exc:
            raise KeyError(
                f"Unknown 3D-FRONT room {room_id!r}; see get_room_ids()."
            ) from exc

    # --------------------------------------------------------------- meshes

    @cachedmethod(operator.attrgetter("_model_cache"))
    def _model(self, model_id):
        """Textured 3D-FUTURE model in its own (y-up) coordinates."""

        folder = self.model_dir / model_id
        mesh = load_triangle_mesh(folder / "raw_model.obj")
        # Every material of a 3D-FUTURE model maps the folder's one texture.png;
        # Open3D loads it once per material plus an empty default, so keep one.
        texture = folder / "texture.png"
        mesh.textures = [
            o3d.io.read_image(str(texture)) if texture.is_file() else _SOLID_GRAY
        ]
        mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(len(mesh.triangles), np.int32)
        )
        mesh.compute_vertex_normals()
        return mesh

    def get_object_mesh(self, model_id) -> o3d.geometry.TriangleMesh:
        """Return a textured 3D-FUTURE model, converted to z-up, unplaced."""

        mesh = o3d.geometry.TriangleMesh(self._model(str(model_id)))
        return mesh.rotate(_Y_UP_TO_Z_UP, center=(0, 0, 0))

    @cached_property
    def _centimetre_models(self) -> frozenset[str]:
        with self.open_file(self.centimetre_models_file, "r") as handle:
            lines = (line.split("#", 1)[0].strip() for line in handle)
            return frozenset(line for line in lines if line)

    def _scale(self, instance):
        """Instance scale; models listed as stored in centimetres get x0.01."""

        scale = instance["scale"]
        if self.fix_model_units and instance["model_id"] in self._centimetre_models:
            scale = scale * 0.01
        return scale

    def _placed_model(self, instance):
        mesh = o3d.geometry.TriangleMesh(self._model(instance["model_id"]))
        rotation = _Y_UP_TO_Z_UP @ Rotation.from_quat(instance["rot"]).as_matrix()
        vertices = np.asarray(mesh.vertices) * self._scale(instance)
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices @ rotation.T + _Y_UP_TO_Z_UP @ instance["pos"]
        )
        if np.prod(instance["scale"]) < 0:  # mirrored: keep faces outward
            triangles = np.asarray(mesh.triangles)[:, ::-1]
            mesh.triangles = o3d.utility.Vector3iVector(triangles.copy())
            uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)[:, ::-1]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs.reshape(-1, 2))
        mesh.compute_vertex_normals()
        return mesh

    def _texture(self, material):
        """Texture image of a layout material: its file, or a 1x1 solid color."""

        material = material or {}
        jid = material.get("jid")
        color = tuple(material.get("color") or (200, 200, 200))[:3]
        path = None if jid is None else self.texture_dir / jid / "texture.png"
        textured = path is not None and not material.get("useColor") and path.is_file()
        # Materials sharing a jid can differ in color, so key on both.
        key = (jid, None) if textured else (None, color)
        if key not in self._texture_cache:
            self._texture_cache[key] = (
                o3d.io.read_image(str(path))
                if textured
                else o3d.geometry.Image(np.asarray(color, np.uint8).reshape(1, 1, 3))
            )
        return self._texture_cache[key]

    def _layout_mesh(self, house, uid):
        record = house["meshes"][uid]
        vertices = np.asarray(record["xyz"], np.float64).reshape(-1, 3)
        triangles = np.asarray(record["faces"], np.int64).reshape(-1, 3)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices @ _Y_UP_TO_Z_UP.T),
            o3d.utility.Vector3iVector(triangles),
        )
        uv = np.asarray(record.get("uv") or [], np.float64).reshape(-1, 2)
        if len(uv) != len(vertices):
            uv = np.zeros((len(vertices), 2))
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uv[triangles].reshape(-1, 2))
        mesh.textures = [self._texture(house["materials"].get(record.get("material")))]
        mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(len(triangles), np.int32)
        )
        mesh.compute_vertex_normals()
        layout_type = record.get("type") or "Other"
        return mesh, layout_type if layout_type in _LAYOUT_TYPES else "Other"

    def _parts(self, sample_id, room_id, mesh_type):
        """Yield (mesh, instance, layout_type) in a fixed order."""

        if mesh_type not in _MESH_TYPES:
            raise ValueError(
                f"Unknown 3D-FRONT mesh type {mesh_type!r}; use {', '.join(_MESH_TYPES)}."
            )
        house = self._house(str(sample_id))
        for room in self._selected_rooms(sample_id, room_id):
            if mesh_type in {"full", "layout"}:
                for uid in room["layout"]:
                    mesh, layout_type = self._layout_mesh(house, uid)
                    if len(mesh.triangles):
                        yield mesh, None, layout_type
            if mesh_type in {"full", "furniture"}:
                for instance in room["objects"]:
                    yield self._placed_model(instance), instance, None

    def get_mesh(self, sample_id, *, mesh_type=None, room_id=None):
        """Return the textured house (or one room), z-up, in metres.

        ``mesh_type`` is ``"full"`` (default), ``"layout"`` (walls, floors,
        ceilings, doors, windows, ...) or ``"furniture"``.
        """

        parts = [
            part for part, _, _ in self._parts(sample_id, room_id, mesh_type or "full")
        ]
        if not parts:
            raise DataFormatError(f"3D-FRONT house {sample_id} has no geometry.")
        return _merge_textured(parts)

    # --------------------------------------------------------------- labels

    def get_segmentation(
        self,
        sample_id,
        *,
        segmentation_type="instance",
        label_space="category",
        mesh_type=None,
        room_id=None,
    ):
        """Return face labels aligned with ``get_mesh`` for the same options.

        Instance labels number furniture instances across the house; layout
        faces are ``invalid_obj_id``. Semantic labels index
        ``metadata["semantic_classes"][label_space]``: 3D-FUTURE categories
        for furniture and official element types (``Floor``, ``WallInner``,
        ...) for the layout.
        """

        if segmentation_type not in {"instance", "semantic"}:
            raise ValueError(
                "3D-FRONT segmentation_type must be 'instance' or 'semantic'."
            )
        if label_space not in _LABEL_SPACES:
            raise ValueError(
                f"Unknown 3D-FRONT label space {label_space!r}; "
                f"use {', '.join(_LABEL_SPACES)}."
            )
        classes = self.metadata["semantic_classes"][label_space]
        class_index = {name: index for index, name in enumerate(classes)}
        labels, names = [], {}
        for part, instance, layout_type in self._parts(
            sample_id, room_id, mesh_type or "full"
        ):
            name = layout_type or self._model_label(instance, label_space)
            if segmentation_type == "semantic":
                value = class_index[name]
                names[value] = name
            elif instance is None:
                value = self.invalid_obj_id
            else:
                value = instance["id"]
                names[value] = name
            labels.append(np.full(len(part.triangles), value, np.int32))
        labels = np.concatenate(labels) if labels else np.empty(0, np.int32)
        return Segmentation3D(labels, names, "face", self.invalid_obj_id)

    def _model_label(self, instance, label_space):
        model = self.metadata["models"].get(instance["model_id"], {})
        return model.get(label_space, "unknown")

    def get_boxes(
        self, sample_id, *, box_type="obb_gt", label_space="category", room_id=None
    ):
        """Return one box per furniture instance and its category name.

        ``"obb_gt"`` is the model's bounding box placed with the instance's
        rotation, scale, and position; other types are fitted to its corners.
        """

        if box_type not in _BOX_TYPES:
            raise ValueError(
                f"Unknown 3D-FRONT box type {box_type!r}; use {', '.join(_BOX_TYPES)}."
            )
        if label_space not in _LABEL_SPACES:
            raise ValueError(
                f"Unknown 3D-FRONT label space {label_space!r}; "
                f"use {', '.join(_LABEL_SPACES)}."
            )
        boxes, names = {}, {}
        for room in self._selected_rooms(sample_id, room_id):
            for instance in room["objects"]:
                vertices = np.asarray(self._model(instance["model_id"]).vertices)
                low, high = vertices.min(0), vertices.max(0)
                scale = self._scale(instance)
                rotation = (
                    _Y_UP_TO_Z_UP @ Rotation.from_quat(instance["rot"]).as_matrix()
                )
                center = rotation @ ((low + high) / 2 * scale)
                box = o3d.geometry.OrientedBoundingBox(
                    center + _Y_UP_TO_Z_UP @ instance["pos"],
                    rotation,
                    (high - low) * np.abs(scale),
                )
                if box_type != "obb_gt":
                    box = construct_box(
                        np.asarray(box.get_box_points()), box_type=box_type
                    )
                boxes[instance["id"]] = box
                names[instance["id"]] = self._model_label(instance, label_space)
        return boxes, names

    # ---------------------------------------------------------------- check

    def _check(self, *, sample_ids, require_complete):
        from ..check import CheckBuilder, check_scene_coverage

        builder = CheckBuilder(self.name, self.root_dir)
        if not builder.require_directory(
            self.scene_dir,
            code="missing-scenes",
            message="The 3D-FRONT house JSON folder does not exist.",
            expected="<root>/3D-FRONT/<house_id>.json",
            hint="Extract 3D-FRONT.zip under the root, or pass scene_dir=.",
        ):
            return builder.finish()
        if not builder.require_directory(
            self.model_dir,
            code="missing-models",
            message="The 3D-FUTURE model folder does not exist.",
            expected="<root>/3D-FUTURE-model/<model_id>/raw_model.obj",
            hint="Extract 3D-FUTURE-model.zip under the root, or pass model_dir=.",
        ):
            return builder.finish()
        if not self.model_info_file.is_file():
            builder.error(
                "missing-model-info",
                "3D-FUTURE model_info.json is missing.",
                path=self.model_info_file,
                hint="It ships inside 3D-FUTURE-model.zip; or pass model_info_file=.",
            )
        if not self.texture_dir.is_dir():
            builder.warning(
                "missing-textures",
                "3D-FRONT-texture is missing; walls and floors use flat colors.",
                path=self.texture_dir,
            )
        present = set(self._load_sample_ids())
        selected = check_scene_coverage(
            builder,
            present_ids=present,
            expected_ids=None,
            sample_ids=sample_ids,
            require_complete=require_complete,
            expected_layout="<root>/3D-FRONT/<house_id>.json",
        )
        if sample_ids is not None:
            models = set()
            for sid in selected:
                try:
                    rooms = self._house(sid)["rooms"].values()
                except (DataFormatError, OSError, ValueError) as exc:
                    builder.error(
                        "invalid-house-json",
                        f"Cannot read house {sid}: {exc}",
                        path=self.scene_dir / f"{sid}.json",
                    )
                    continue
                models.update(
                    self.model_dir / instance["model_id"] / "raw_model.obj"
                    for room in rooms
                    for instance in room["objects"]
                )
            missing = sorted(path for path in models if not path.is_file())
            builder.missing_paths(
                missing,
                code="missing-furniture-models",
                label="3D-FUTURE furniture models",
                expected="<root>/3D-FUTURE-model/<model_id>/raw_model.obj",
                hint="Re-extract 3D-FUTURE-model.zip; houses reference these models.",
            )
        return builder.finish()


def _merge_textured(parts) -> o3d.geometry.TriangleMesh:
    """Concatenate meshes, keeping every texture (Open3D's ``+=`` drops them)."""

    vertices, normals, triangles, uvs, material_ids, textures = [], [], [], [], [], []
    offset = 0
    for part in parts:
        count = len(part.triangles)
        vertices.append(np.asarray(part.vertices))
        normals.append(np.asarray(part.vertex_normals))
        triangles.append(np.asarray(part.triangles) + offset)
        offset += len(part.vertices)
        part_uvs = np.asarray(part.triangle_uvs)
        uvs.append(part_uvs if len(part_uvs) == 3 * count else np.zeros((3 * count, 2)))
        part_textures = list(part.textures) or [_SOLID_GRAY]
        ids = np.asarray(part.triangle_material_ids)
        ids = ids if len(ids) == count else np.zeros(count, np.int32)
        material_ids.append(np.clip(ids, 0, len(part_textures) - 1) + len(textures))
        textures.extend(part_textures)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(vertices)),
        o3d.utility.Vector3iVector(np.concatenate(triangles)),
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.concatenate(normals))
    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.concatenate(uvs))
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.concatenate(material_ids).astype(np.int32)
    )
    mesh.textures = textures
    return mesh


def _name(value) -> str:
    text = "" if value is None else str(value).strip()
    return text if text and text != "None" else "unknown"
