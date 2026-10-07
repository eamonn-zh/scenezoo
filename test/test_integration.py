from __future__ import annotations

import os

import numpy as np
import pytest

from scenezoo import UnsupportedOperationError, get_dataset


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "variable"),
    [
        ("scannetv2", "SCENEZOO_TEST_SCANNET_ROOT"),
        ("scannetppv2", "SCENEZOO_TEST_SCANNETPP_ROOT"),
        ("arkitscenes", "SCENEZOO_TEST_ARKITSCENES_ROOT"),
        ("multiscan", "SCENEZOO_TEST_MULTISCAN_ROOT"),
        ("ase", "SCENEZOO_TEST_ASE_ROOT"),
        ("3rscan", "SCENEZOO_TEST_3RSCAN_ROOT"),
        ("matterport3d", "SCENEZOO_TEST_MATTERPORT3D_ROOT"),
        ("scenenn", "SCENEZOO_TEST_SCENENN_ROOT"),
        ("s3dis", "SCENEZOO_TEST_S3DIS_ROOT"),
    ],
)
def test_configured_dataset_smoke(name, variable):
    root = os.getenv(variable)
    if not root:
        pytest.skip(f"{variable} is not configured")
    dataset = get_dataset(name, root)
    sample_id = _first_sample(dataset)
    mesh = None
    if dataset.supports("mesh"):
        mesh = dataset.get_mesh(sample_id)
        assert mesh is not None
    if dataset.supports("segmentation"):
        segmentation = dataset.get_segmentation(sample_id)
        assert segmentation.domain in {"point", "vertex", "face"}
        if segmentation.domain == "point":
            expected = len(dataset.get_point_cloud(sample_id).points)
        elif segmentation.domain == "vertex":
            assert mesh is not None
            expected = len(mesh.vertices)
        else:
            assert mesh is not None
            expected = len(mesh.triangles)
        assert len(segmentation.labels) == expected
    if dataset.supports("boxes"):
        boxes, names = dataset.get_boxes(sample_id)
        assert set(boxes) <= set(names)


def _first_sample(dataset):
    for ids in dataset.splits.values():
        if ids:
            return ids[0]
    pytest.skip("Every split of the configured dataset root is empty")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "variable", "options"),
    [
        ("scannetv2", "SCENEZOO_TEST_SCANNET_ROOT", {}),
        ("scannetppv2", "SCENEZOO_TEST_SCANNETPP_ROOT", {"source": "iphone"}),
        ("arkitscenes", "SCENEZOO_TEST_ARKITSCENES_ROOT", {"source": "mov"}),
        ("multiscan", "SCENEZOO_TEST_MULTISCAN_ROOT", {}),
        ("3rscan", "SCENEZOO_TEST_3RSCAN_ROOT", {}),
        (
            "matterport3d",
            "SCENEZOO_TEST_MATTERPORT3D_ROOT",
            {"source": "undistorted"},
        ),
    ],
)
@pytest.mark.parametrize("rotate_to_up", [False, True])
def test_configured_depth_backprojects_onto_mesh(name, variable, options, rotate_to_up):
    """Depth, intrinsics, and poses must agree with the released geometry."""

    from scipy.spatial import cKDTree

    root = os.getenv(variable)
    if not root:
        pytest.skip(f"{variable} is not configured")
    dataset = get_dataset(name, root)
    sample_id = _first_sample(dataset)
    try:
        frames = dataset.get_frames(
            sample_id,
            step=10_000,
            items=("depth", "depth_intrinsics", "world_to_camera"),
            output_size=(320, 240),
            center_crop=True,
            rotate_to_up=rotate_to_up,
            **options,
        )
    except (FileNotFoundError, UnsupportedOperationError) as exc:
        pytest.skip(f"Frames are unavailable for {sample_id}: {exc}")
    if not len(frames):
        pytest.skip(f"No valid frames for {sample_id}")
    mesh = dataset.get_mesh(sample_id)
    surface = cKDTree(np.asarray(mesh.sample_points_uniformly(1_000_000).points))

    for depth, intrinsic, world_to_camera in zip(
        frames.depth, frames.depth_intrinsics, frames.world_to_camera
    ):
        # Intrinsics must remain pinhole matrices even after upright rotation.
        assert intrinsic[0, 1] == 0 and np.all(intrinsic[1:, 0] == 0)
        assert intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0
        rows, columns = np.nonzero(depth > 0)
        if len(rows) < 100:
            continue
        z = depth[rows, columns].astype(np.float64)
        x = (columns - intrinsic[0, 2]) / intrinsic[0, 0] * z
        y = (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
        camera_to_world = np.linalg.inv(world_to_camera.astype(np.float64))
        points = np.c_[x, y, z] @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        distance = np.median(surface.query(points)[0])
        assert distance < 0.05, f"{name} {sample_id}: median error {distance:.3f} m"


@pytest.mark.integration
def test_configured_scannetpp_capture_sources():
    root = os.getenv("SCENEZOO_TEST_SCANNETPP_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_SCANNETPP_ROOT is not configured")
    dataset = get_dataset("scannetppv2", root)
    sample_id = dataset.get_ids("nvs_sem_val")[0]
    sources = dataset.get_frame_sources(sample_id)
    assert "iphone" in sources
    iphone = dataset.get_frames(
        sample_id,
        indices=(0, 2),
        items=("timestamps", "imu", "rgb_intrinsics", "world_to_camera"),
    )
    assert iphone.camera_model == "PINHOLE"
    assert iphone.timestamps.shape == (2,)
    assert iphone.imu
    if "dslr_undistorted" in sources:
        dslr = dataset.get_frames(
            sample_id,
            source="dslr_undistorted",
            indices=(0, 2),
            items=("rgb", "rgb_intrinsics", "world_to_camera", "is_bad"),
            output_size=(320, 240),
        )
        assert dslr.camera_model == "PINHOLE"
        assert dslr.rgb.shape == (2, 240, 320, 3)
        assert len(dslr.frame_keys) == 2


@pytest.mark.integration
def test_configured_ase_complete_sources():
    root = os.getenv("SCENEZOO_TEST_ASE_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_ASE_ROOT is not configured")
    dataset = get_dataset("ase", root, offline=True)
    available = set(dataset.get_available_ids("train"))
    sample_id = next(
        (value for value in dataset.get_ids("train") if value in available), None
    )
    if sample_id is None:
        pytest.skip("No downloaded ASE train scene was found")
    report = dataset.check(sample_ids=(sample_id,))
    assert report.ok, report.format()
    frames = dataset.get_frames(
        sample_id,
        indices=(0, 1),
        items=(
            "rgb",
            "depth",
            "instance_maps",
            "timestamps",
            "rgb_intrinsics",
            "world_to_camera",
        ),
        output_size=(352, 352),
    )
    assert frames.rgb.shape == (2, 352, 352, 3)
    assert frames.depth.shape == frames.instance_maps.shape == (2, 352, 352)
    assert frames.depth.dtype == np.float32
    assert frames.depth_mode == "ray_distance"
    assert frames.camera_model == "FISHEYE624"
    assert frames.world_to_camera.shape == (2, 4, 4)
    points = dataset.get_points(sample_id, max_inverse_distance_std=0.005)
    assert points.xyz.shape[1:] == (3,)


@pytest.mark.integration
def test_configured_scannet_complete_sources():
    root = os.getenv("SCENEZOO_TEST_SCANNET_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_SCANNET_ROOT is not configured")
    dataset = get_dataset("scannetv2", root, offline=True)
    sample_id = dataset.get_ids("train")[0]

    raw_mesh = dataset.get_mesh(sample_id, mesh_type="raw")
    raw_instances = dataset.get_segmentation(sample_id, mesh_type="raw")
    raw_semantics = dataset.get_segmentation(
        sample_id, segmentation_type="semantic", mesh_type="raw"
    )
    assert raw_instances.domain == raw_semantics.domain == "vertex"
    assert (
        len(raw_instances.labels) == len(raw_semantics.labels) == len(raw_mesh.vertices)
    )

    frames = dataset.get_frames(
        sample_id,
        indices=(100, 0),
        items=(
            "rgb",
            "depth",
            "timestamps",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
            "instance_maps",
            "semantic_maps",
            "imu",
        ),
        output_size=(320, 240),
        semantic_label_space="nyu40",
    )
    assert frames.indices.tolist() == [100, 0]
    assert frames.rgb.shape == (2, 240, 320, 3)
    assert frames.rgb.dtype == np.uint8
    assert frames.depth.shape == (2, 240, 320)
    assert frames.depth.dtype == np.float32
    assert frames.timestamps.shape == (2, 2)
    assert frames.rgb_intrinsics.shape == frames.depth_intrinsics.shape == (2, 3, 3)
    assert frames.world_to_camera.shape == (2, 4, 4)
    assert frames.instance_maps.shape == frames.semantic_maps.shape == (2, 240, 320)
    assert frames.instance_maps.dtype == frames.semantic_maps.dtype == np.int32
    assert frames.imu["valid"].shape == (2,)

    info = dataset.get_scene_info(sample_id)
    calibration = dataset.get_calibration(sample_id)
    assert info["numColorFrames"] == frames.source_frame_count
    assert calibration["rgb_intrinsic"].shape == (4, 4)
    assert calibration["depth_intrinsic"].shape == (4, 4)


@pytest.mark.integration
def test_configured_matterport3d_complete_sources():
    root = os.getenv("SCENEZOO_TEST_MATTERPORT3D_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_MATTERPORT3D_ROOT is not configured")
    dataset = get_dataset("matterport3d", root)
    sample_id = dataset.get_ids("train")[0]

    region_id = dataset.get_region_ids(sample_id)[0]
    region_mesh = dataset.get_mesh(sample_id, mesh_type="region", region_id=region_id)
    instances = dataset.get_segmentation(sample_id, region_id=region_id)
    semantics = dataset.get_segmentation(
        sample_id,
        region_id=region_id,
        segmentation_type="semantic",
        label_space="mpcat40",
    )
    assert instances.domain == semantics.domain == "face"
    assert len(instances.labels) == len(semantics.labels) == len(region_mesh.triangles)
    boxes, names = dataset.get_boxes(sample_id, region_id=region_id)
    assert set(boxes) == set(names)

    sources = dataset.get_frame_sources(sample_id)
    if not sources:
        pytest.skip("No Matterport3D RGB-D source was downloaded for the first scene")
    source = "undistorted" if "undistorted" in sources else sources[0]
    count = len(dataset.get_frame_keys(sample_id, source=source))
    selected = tuple(dict.fromkeys((0, min(1, count - 1))))
    frames = dataset.get_frames(
        sample_id,
        source=source,
        indices=selected,
        items=(
            "rgb",
            "depth",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        output_size=(160, 128),
    )
    assert frames.indices.tolist() == list(selected)
    assert frames.rgb.shape == (len(selected), 128, 160, 3)
    assert frames.depth.shape == (len(selected), 128, 160)
    assert frames.depth.dtype == np.float32
    assert frames.rgb_intrinsics.shape == (len(selected), 3, 3)
    assert frames.depth_intrinsics.shape == (len(selected), 3, 3)
    assert frames.world_to_camera.shape == (len(selected), 4, 4)


@pytest.mark.integration
def test_configured_arkitscenes_complete_sources():
    root = os.getenv("SCENEZOO_TEST_ARKITSCENES_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_ARKITSCENES_ROOT is not configured")
    dataset = get_dataset("arkitscenes", root, offline=True)
    sample_id = dataset.get_ids("train_threedod")[0]
    metadata = dataset.metadata[sample_id]
    assert metadata["official_sky_direction"] in {"Up", "Down", "Left", "Right", "NA"}
    assert metadata["sky_direction"] in {"Up", "Down", "Left", "Right"}
    assert isinstance(metadata["is_in_threedod"], bool)

    camera = dataset.get_frames(
        sample_id,
        source="mov",
        step=60,
        items=("timestamps", "world_to_camera"),
    )
    assert camera.timestamps.dtype == np.float64
    assert camera.world_to_camera.shape[1:] == (4, 4)
    selected = camera.indices[:2]
    if len(selected):
        frames = dataset.get_frames(
            sample_id,
            source="mov",
            indices=selected,
            items=("depth", "depth_intrinsics", "world_to_camera"),
            output_size=(320, 240),
        )
        assert frames.indices.tolist() == selected.tolist()
        assert frames.depth.shape == (len(selected), 240, 320)
        assert frames.depth.dtype == np.float32
        assert frames.depth_intrinsics.shape == (len(selected), 3, 3)

    faro_sample = next(
        (
            video_id
            for video_id, value in dataset.metadata.items()
            if value["has_laser_scanner_point_clouds"]
            and dataset.get_laser_scanner_ids(video_id)
            and all(
                (
                    dataset.dataset_root
                    / "laser_scanner_point_clouds"
                    / str(value["visit_id"])
                    / f"{scan_id}_pose.txt"
                ).is_file()
                for scan_id in dataset.get_laser_scanner_ids(video_id)
            )
        ),
        None,
    )
    if faro_sample is not None:
        poses = dataset.get_laser_scanner_poses(faro_sample)
        assert poses
        assert all(pose.shape == (4, 4) for pose in poses.values())


@pytest.mark.integration
def test_configured_multiscan_complete_sources():
    root = os.getenv("SCENEZOO_TEST_MULTISCAN_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_MULTISCAN_ROOT is not configured")
    dataset = get_dataset("multiscan", root, offline=True)
    sample_id = dataset.get_ids("train")[0]

    for segmentation_type in (
        "object_instance",
        "part_instance",
        "object_semantic",
        "part_semantic",
    ):
        segmentation = dataset.get_segmentation(
            sample_id, segmentation_type=segmentation_type
        )
        assert segmentation.domain == "face"
        assert segmentation.labels.dtype == np.int32

    frames = dataset.get_frames(
        sample_id,
        indices=(100, 0),
        items=(
            "rgb",
            "depth",
            "confidence_maps",
            "timestamps",
            "exposure_durations",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        output_size=(320, 240),
    )
    assert frames.indices.tolist() == [100, 0]
    assert frames.rgb.shape == (2, 240, 320, 3)
    assert frames.rgb.dtype == np.uint8
    assert frames.depth.shape == frames.confidence_maps.shape == (2, 240, 320)
    assert frames.depth.dtype == np.float32
    assert frames.confidence_maps.dtype == np.uint8
    assert set(np.unique(frames.confidence_maps)) <= {0, 1, 2}
    assert frames.timestamps.shape == frames.exposure_durations.shape == (2,)
    assert frames.rgb_intrinsics.shape == frames.depth_intrinsics.shape == (2, 3, 3)
    assert frames.world_to_camera.shape == (2, 4, 4)

    info = dataset.get_scene_info(sample_id)
    camera = dataset.get_camera_metadata(sample_id, indices=(100, 0))
    assert info["device"]["type"].startswith(("iPad", "iPhone"))
    assert camera["quaternions"].shape == (2, 4)
    assert camera["euler_angles"].shape == (2, 3)
    assert camera["quaternion_format"] == "wxyz"


@pytest.mark.integration
def test_configured_3rscan_complete_sources():
    root = os.getenv("SCENEZOO_TEST_3RSCAN_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_3RSCAN_ROOT is not configured")
    dataset = get_dataset("3rscan", root, offline=True)
    assert {
        split: len(dataset.get_ids(split)) for split in ("train", "val", "test")
    } == {
        "train": 1178,
        "val": 157,
        "test": 147,
    }
    assert len(dataset.get_ids()) == 1482

    sample_id = dataset.get_ids("train")[0]
    instance = dataset.get_segmentation(sample_id)
    for segmentation_type in ("global", "nyu40", "eigen13", "rio27", "rio7"):
        semantics = dataset.get_segmentation(
            sample_id, segmentation_type=segmentation_type
        )
        assert semantics.labels.shape == instance.labels.shape
        assert semantics.labels.dtype == np.int32

    frames = dataset.get_frames(
        sample_id,
        indices=(2, 0),
        items=(
            "rgb",
            "depth",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        output_size=(160, 120),
    )
    assert frames.indices.tolist() == [2, 0]
    assert frames.rgb.shape == (2, 120, 160, 3)
    assert frames.rgb.dtype == np.uint8
    assert frames.depth.shape == (2, 120, 160)
    assert frames.depth.dtype == np.float32
    assert frames.rgb_intrinsics.shape == frames.depth_intrinsics.shape == (2, 3, 3)
    assert frames.world_to_camera.shape == (2, 4, 4)
    assert frames.camera_model == "PINHOLE"

    rescan_id = next(
        scan_id
        for scan_id in dataset.get_ids("train")
        if not dataset.get_scene_info(scan_id)["is_reference"]
    )
    alignment = dataset.get_scene_alignment(rescan_id)
    assert alignment.shape == (4, 4)
    assert np.isfinite(alignment).all()


@pytest.mark.integration
def test_configured_structured3d_complete_sources():
    root = os.getenv("SCENEZOO_TEST_STRUCTURED3D_ROOT")
    if not root:
        pytest.skip("SCENEZOO_TEST_STRUCTURED3D_ROOT is not configured")
    dataset = get_dataset("structured3d", root, offline=True)
    assert {
        split: len(dataset.get_ids(split)) for split in ("train", "val", "test")
    } == {"train": 3000, "val": 250, "test": 250}

    sample_id = "scene_00000"
    room_id = dataset.get_room_ids(sample_id)[0]
    assert {"perspective", "panorama"} <= set(
        dataset.get_frame_sources(sample_id, room_id)
    )

    perspective = dataset.get_frames(
        sample_id,
        room_id=room_id,
        source="perspective",
        indices=(2, 0),
        items=(
            "rgb",
            "depth",
            "albedo",
            "normal_maps",
            "semantic_maps",
            "instance_maps",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        output_size=(160, 90),
    )
    assert perspective.indices.tolist() == [2, 0]
    assert perspective.rgb.shape == perspective.albedo.shape == (2, 90, 160, 3)
    assert (
        perspective.depth.shape
        == perspective.semantic_maps.shape
        == (
            2,
            90,
            160,
        )
    )
    assert perspective.normal_maps.shape == (2, 90, 160, 3)
    assert perspective.depth.dtype == np.float32
    assert (
        perspective.semantic_maps.dtype == perspective.instance_maps.dtype == np.int32
    )
    assert perspective.rgb_intrinsics.shape == (2, 3, 3)
    assert perspective.world_to_camera.shape == (2, 4, 4)
    assert perspective.camera_model == "PINHOLE"

    panorama = dataset.get_frames(
        sample_id,
        room_id=room_id,
        source="panorama",
        items=("rgb", "depth", "world_to_camera"),
        output_size=(160, 80),
    )
    assert panorama.rgb.shape == (1, 80, 160, 3)
    assert panorama.depth.shape == (1, 80, 160)
    assert panorama.camera_model == "EQUIRECTANGULAR"
    assert panorama.rgb_intrinsics is None

    mesh = dataset.get_mesh(sample_id, room_id=room_id)
    assert len(mesh.vertices) >= 8
    assert len(mesh.triangles) >= 12
    boxes, names = dataset.get_boxes(sample_id)
    assert boxes and set(boxes) == set(names)

    points = dataset.get_points(
        sample_id,
        room_id,
        sources=("panorama",),
        semantic_label_space="pointcept25",
        voxel_size=0.08,
        min_view_cosine=0.15,
    )
    segmentation = dataset.get_segmentation(
        sample_id,
        room_id=room_id,
        semantic_label_space="pointcept25",
        voxel_size=0.08,
        min_view_cosine=0.15,
    )
    assert len(points) == len(segmentation.labels)
    assert segmentation.domain == "point"
