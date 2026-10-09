"""Import-light catalog for built-in dataset adapters."""

from __future__ import annotations

from .spec import DatasetSpec, FrameSourceSpec

_RGBD_ITEMS = (
    "rgb",
    "depth",
    "rgb_intrinsics",
    "depth_intrinsics",
    "world_to_camera",
)


BUILTIN_DATASETS = (
    {
        "name": "scannetv2",
        "aliases": ("scannet",),
        "import_path": "scenezoo.dataset.scene.scannet:ScanNet",
        "description": "ScanNet v2 indoor RGB-D scenes.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "segmentation", "boxes", "frames"),
            mesh_types=("raw", "decimated", "semantic"),
            segmentation_types=("instance", "semantic"),
            box_types=("aabb", "obb", "mobb", "mobb_gravity"),
            label_spaces=("scannet", "nyu40"),
            frame_sources=(
                FrameSourceSpec(
                    "sens",
                    _RGBD_ITEMS
                    + ("instance_maps", "semantic_maps", "timestamps", "imu"),
                    ("PINHOLE",),
                ),
            ),
        ),
    },
    {
        "name": "scannetppv2",
        "aliases": ("scannet++", "scannetpp", "scannet++v2"),
        "import_path": "scenezoo.dataset.scene.scannetpp:ScanNetPP",
        "description": "ScanNet++ v2 scans with iPhone, DSLR, and panoramic captures.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "points", "segmentation", "boxes", "frames"),
            mesh_types=("raw", "semantic"),
            segmentation_types=("instance", "semantic"),
            box_types=("aabb", "obb", "mobb", "mobb_gravity"),
            frame_sources=(
                FrameSourceSpec(
                    "iphone",
                    _RGBD_ITEMS + ("frame_mask", "timestamps", "imu"),
                    ("PINHOLE",),
                ),
                FrameSourceSpec(
                    "iphone_colmap",
                    _RGBD_ITEMS + ("frame_mask", "timestamps", "imu"),
                    ("SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "OPENCV_FISHEYE"),
                ),
                FrameSourceSpec(
                    "dslr",
                    (
                        "rgb",
                        "rgb_intrinsics",
                        "world_to_camera",
                        "frame_mask",
                        "is_bad",
                    ),
                    ("SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "OPENCV_FISHEYE"),
                ),
                FrameSourceSpec(
                    "dslr_undistorted",
                    (
                        "rgb",
                        "rgb_intrinsics",
                        "world_to_camera",
                        "frame_mask",
                        "is_bad",
                    ),
                    ("PINHOLE",),
                ),
                FrameSourceSpec(
                    "dslr_original",
                    (
                        "rgb",
                        "rgb_intrinsics",
                        "world_to_camera",
                        "frame_mask",
                        "is_bad",
                    ),
                    ("SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "OPENCV_FISHEYE"),
                ),
                FrameSourceSpec(
                    "panocam",
                    (
                        "rgb",
                        "depth",
                        "world_to_camera",
                        "frame_mask",
                        "azimuth",
                        "elevation",
                    ),
                    ("EQUIRECTANGULAR",),
                ),
                FrameSourceSpec(
                    "panocam_resized",
                    (
                        "rgb",
                        "depth",
                        "world_to_camera",
                        "frame_mask",
                        "azimuth",
                        "elevation",
                    ),
                    ("EQUIRECTANGULAR",),
                ),
            ),
        ),
    },
    {
        "name": "arkitscenes",
        "aliases": (),
        "import_path": "scenezoo.dataset.scene.arkitscenes:ARKitScenes",
        "description": "ARKitScenes scans, mobile RGB-D streams, and FARO captures.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "segmentation", "boxes", "frames"),
            mesh_types=("raw",),
            segmentation_types=("instance",),
            box_types=("obb_gt",),
            frame_sources=tuple(
                FrameSourceSpec(
                    name,
                    _RGBD_ITEMS + ("confidence_maps", "timestamps"),
                    ("PINHOLE",),
                )
                for name in ("mov", "lowres_wide", "wide")
            )
            + tuple(
                FrameSourceSpec(
                    name,
                    ("rgb", "rgb_intrinsics", "world_to_camera", "timestamps"),
                    ("PINHOLE",),
                )
                for name in ("ultrawide", "vga_wide")
            )
            + (
                FrameSourceSpec(
                    "threedod",
                    _RGBD_ITEMS + ("timestamps",),
                    ("PINHOLE",),
                ),
            ),
        ),
    },
    {
        "name": "multiscan",
        "aliases": (),
        "import_path": "scenezoo.dataset.scene.multiscan:MultiScan",
        "description": "MultiScan indoor scenes.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "segmentation", "boxes", "frames"),
            mesh_types=("ply", "textured"),
            segmentation_types=(
                "object_instance",
                "part_instance",
                "object_semantic",
                "part_semantic",
            ),
            box_types=("aabb", "obb", "mobb", "mobb_gravity", "obb_gt"),
            frame_sources=(
                FrameSourceSpec(
                    "capture",
                    _RGBD_ITEMS
                    + ("confidence_maps", "timestamps", "exposure_durations"),
                    ("PINHOLE",),
                ),
            ),
        ),
    },
    {
        "name": "ase",
        "aliases": ("aria-ase", "aria-synthetic-environments"),
        "import_path": ("scenezoo.dataset.scene.ase:AriaSyntheticEnvironments"),
        "description": "Aria Synthetic Environments egocentric synthetic scenes.",
        "spec": DatasetSpec(
            sample_unit="sequence",
            operations=("points", "boxes", "frames"),
            box_types=("obb_gt",),
            frame_sources=(
                FrameSourceSpec(
                    "rgb",
                    _RGBD_ITEMS + ("instance_maps", "frame_mask", "timestamps"),
                    ("FISHEYE624",),
                ),
            ),
        ),
    },
    {
        "name": "3rscan",
        "aliases": ("threerscan",),
        "import_path": "scenezoo.dataset.scene.threerscan:ThreeRScan",
        "description": "3RScan indoor scene sequences.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "segmentation", "boxes", "frames"),
            mesh_types=("textured", "instance", "instance_aligned"),
            segmentation_types=(
                "instance",
                "global",
                "nyu40",
                "eigen13",
                "rio27",
                "rio7",
            ),
            box_types=("aabb", "obb", "mobb", "mobb_gravity", "obb_gt"),
            label_spaces=("global", "nyu40", "eigen13", "rio27", "rio7"),
            frame_sources=(FrameSourceSpec("sequence", _RGBD_ITEMS, ("PINHOLE",)),),
        ),
    },
    {
        "name": "s3dis",
        "aliases": ("stanford3d", "stanford-3d"),
        "import_path": "scenezoo.dataset.scene.s3dis:S3DIS",
        "description": "S3DIS room point clouds, point-domain labels, and boxes.",
        "spec": DatasetSpec(
            sample_unit="room",
            operations=("points", "segmentation", "boxes"),
            segmentation_types=("instance", "semantic"),
            box_types=("aabb", "obb", "mobb", "mobb_gravity"),
            label_spaces=("s3dis13",),
        ),
    },
    {
        "name": "matterport3d",
        "aliases": (),
        "import_path": "scenezoo.dataset.scene.matterport3d:Matterport3D",
        "description": "Matterport3D houses with region annotations and RGB-D views.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "segmentation", "boxes", "frames"),
            mesh_types=("raw", "house", "region", "poisson"),
            segmentation_types=("instance", "semantic"),
            box_types=("obb_gt", "aabb", "obb", "mobb", "mobb_gravity"),
            label_spaces=("raw", "mpcat40", "nyu40", "eigen13"),
            frame_sources=(
                FrameSourceSpec(
                    "raw",
                    _RGBD_ITEMS,
                    ("OPENCV",),
                ),
                FrameSourceSpec(
                    "undistorted",
                    _RGBD_ITEMS + ("normal_maps",),
                    ("PINHOLE",),
                ),
            ),
        ),
    },
    {
        "name": "scenenn",
        "aliases": (),
        "import_path": "scenezoo.dataset.scene.scenenn:SceneNN",
        "description": "SceneNN meshes and vertex-domain instance segmentation.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "segmentation"),
            mesh_types=("raw", "instance"),
            segmentation_types=("instance",),
        ),
    },
    {
        "name": "structured3d",
        "aliases": ("structure3d",),
        "import_path": "scenezoo.dataset.scene.structured3d:Structured3D",
        "description": "Structured3D renderings, layouts, boxes, and fused points.",
        "spec": DatasetSpec(
            sample_unit="scene",
            operations=("mesh", "points", "segmentation", "boxes", "frames"),
            mesh_types=("layout",),
            segmentation_types=("instance", "semantic"),
            box_types=("obb_gt",),
            label_spaces=("nyu40", "pointcept25"),
            frame_sources=(
                FrameSourceSpec(
                    "perspective",
                    _RGBD_ITEMS
                    + ("semantic_maps", "instance_maps", "albedo", "normal_maps"),
                    ("PINHOLE",),
                    ("empty", "full"),
                    ("room_id",),
                ),
                FrameSourceSpec(
                    "panorama",
                    (
                        "rgb",
                        "depth",
                        "world_to_camera",
                        "semantic_maps",
                        "instance_maps",
                        "albedo",
                        "normal_maps",
                    ),
                    ("EQUIRECTANGULAR",),
                    ("empty", "simple", "full"),
                    ("room_id",),
                ),
            ),
        ),
    },
    {
        "name": "3dfront",
        "aliases": ("3d-front",),
        "import_path": "scenezoo.dataset.scene.threedfront:ThreeDFront",
        "description": "3D-FRONT houses with 3D-FUTURE furniture, rooms, and labels.",
        "spec": DatasetSpec(
            sample_unit="house",
            operations=("mesh", "segmentation", "boxes"),
            mesh_types=("full", "layout", "furniture"),
            segmentation_types=("instance", "semantic"),
            box_types=("obb_gt", "aabb", "obb", "mobb", "mobb_gravity"),
            label_spaces=("category", "super_category"),
        ),
    },
)


def register_builtin_datasets(registry) -> None:
    """Populate a registry without importing any adapter modules."""

    for entry in BUILTIN_DATASETS:
        registry.register_lazy(**entry)
