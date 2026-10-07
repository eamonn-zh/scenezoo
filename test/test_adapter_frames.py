from __future__ import annotations

import io
import json
import zipfile
import zlib

import numpy as np
import pandas as pd
import lz4.block
from PIL import Image

from scenezoo.dataset.scene import ARKitScenes, MultiScan, ScanNetPP, ThreeRScan
import scenezoo.dataset.scene.arkitscenes as arkitscenes_module
import scenezoo.dataset.scene.scannetpp as scannetpp_module


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def test_scannetpp_camera_and_depth_keep_requested_order(tmp_path, monkeypatch):
    scene = "sample"
    directory = tmp_path / "data" / scene / "iphone"
    directory.mkdir(parents=True)
    camera = {}
    for index in range(3):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index
        camera[f"frame_{index}"] = {
            "aligned_pose": pose.tolist(),
            "intrinsic": np.eye(3, dtype=np.float32).tolist(),
        }
    (directory / "pose_intrinsic_imu.json").write_text(json.dumps(camera))
    depths = np.stack(
        [np.full((192, 256), index + 1, dtype=np.float32) for index in range(3)]
    )
    (directory / "depth.bin").write_bytes(_raw_deflate(depths.tobytes()))
    monkeypatch.setattr(scannetpp_module, "read_video_frame_count", lambda _: 3)

    batch = ScanNetPP(tmp_path).get_frames(
        scene,
        indices=[2, 0],
        items=("depth", "depth_intrinsics", "world_to_camera"),
        output_size=(4, 3),
    )
    assert batch.indices.tolist() == [2, 0]
    assert batch.depth.shape == (2, 3, 4)
    assert np.all(batch.depth[:, 0, 0] == [3, 1])
    assert batch.depth_intrinsics.shape == (2, 3, 3)
    assert batch.world_to_camera[:, 0, 3].tolist() == [-2, 0]


def test_scannetpp_block_depth_decodes_only_requested_frames(tmp_path):
    scene = "sample"
    directory = tmp_path / "data" / scene / "iphone"
    directory.mkdir(parents=True)
    requested = np.full((192, 256), 2500, dtype=np.uint16)
    blocks = [
        b"not-compressed",
        b"also-not-compressed",
        lz4.block.compress(requested.tobytes(), store_size=False),
    ]
    payload = b"".join(len(block).to_bytes(4, "little") + block for block in blocks)
    (directory / "depth.bin").write_bytes(payload)
    depth = ScanNetPP(tmp_path)._read_depth(scene, np.array([2], dtype=np.int64))
    assert depth.shape == (1, 192, 256)
    assert depth.dtype == np.float32
    assert np.all(depth == 2.5)


def test_scannetpp_block_depth_accepts_raw_deflate_frames(tmp_path):
    # Some released v2 scenes store length-prefixed raw-DEFLATE float32 frames,
    # which lz4 rejects with LZ4BlockError rather than RuntimeError/ValueError.
    scene = "sample"
    directory = tmp_path / "data" / scene / "iphone"
    directory.mkdir(parents=True)
    blocks = [
        _raw_deflate(np.full((192, 256), value, dtype=np.float32).tobytes())
        for value in (1.25, 2.5)
    ]
    payload = b"".join(len(block).to_bytes(4, "little") + block for block in blocks)
    (directory / "depth.bin").write_bytes(payload)
    depth = ScanNetPP(tmp_path)._read_depth(scene, np.array([1, 0], dtype=np.int64))
    assert depth.shape == (2, 192, 256)
    assert np.all(depth[0] == 2.5) and np.all(depth[1] == 1.25)


def test_scannetpp_split_names_are_not_swapped(tmp_path):
    split_paths = {}
    for name in (
        "nvs_sem_train",
        "nvs_sem_val",
        "nvs_test",
        "sem_test",
        "nvs_test_small",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(name + "-scene\n")
        split_paths[name] = str(path.relative_to(tmp_path))
    dataset = ScanNetPP(
        tmp_path,
        nvs_sem_train_split_file=split_paths["nvs_sem_train"],
        nvs_sem_val_split_file=split_paths["nvs_sem_val"],
        nvs_test_split_file=split_paths["nvs_test"],
        sem_test_split_file=split_paths["sem_test"],
        nvs_test_small_split_file=split_paths["nvs_test_small"],
    )
    assert dataset.get_ids("nvs_sem_train") == ["nvs_sem_train-scene"]
    assert dataset.get_ids("nvs_test") == ["nvs_test-scene"]
    assert dataset.get_ids("sem_test") == ["sem_test-scene"]


def test_arkitscenes_timestamps_intrinsics_and_explicit_indices(tmp_path, monkeypatch):
    scene = "123"
    split = tmp_path / "split.csv"
    pd.DataFrame({"video_id": [scene], "fold": ["Training"]}).to_csv(split, index=False)
    scene_dir = tmp_path / "Training" / scene
    intrinsic_dir = scene_dir / "lowres_wide_intrinsics"
    intrinsic_dir.mkdir(parents=True)
    for timestamp in ("0.000", "1.000"):
        (intrinsic_dir / f"{scene}_{timestamp}.pincam").write_text(
            "4 3 2 2 1.5 1.0 0 0"
        )
    (scene_dir / "lowres_wide.traj").write_text(
        "0.000 0 0 0 0 0 0\n1.000 0 0 0 1 0 0\n"
    )
    dataset = ARKitScenes(
        tmp_path,
        threedod_split_file=str(split),
        raw_split_file=str(split),
        depth_upsampling_split_file=str(split),
    )
    monkeypatch.setattr(arkitscenes_module, "read_video_frame_size", lambda _: (4, 3))
    batch = dataset.get_frames(
        scene,
        indices=[1, 0],
        items=("timestamps", "rgb_intrinsics", "world_to_camera"),
        output_size=(2, 2),
        rotate_to_up=False,
    )
    assert batch.indices.tolist() == [1, 0]
    assert batch.timestamps.dtype == np.float64
    assert batch.timestamps.tolist() == [1.0, 0.0]
    assert batch.rgb_intrinsics.shape == (2, 3, 3)
    assert batch.world_to_camera[:, 0, 3].tolist() == [1, 0]
    monkeypatch.setattr(
        arkitscenes_module,
        "read_video_frames",
        lambda *args, **kwargs: np.empty((0, 3, 4, 3), dtype=np.uint8),
    )
    (scene_dir / f"{scene}.mov").touch()
    empty = dataset.get_frames(
        scene,
        indices=[],
        items=("rgb", "rgb_intrinsics"),
        rotate_to_up=False,
    )
    assert empty.rgb.shape == (0, 3, 4, 3)
    assert empty.rgb_intrinsics.shape == (0, 3, 3)


def test_multiscan_jsonl_camera_alignment(tmp_path):
    scene = "scene_00000_00"
    directory = tmp_path / scene
    directory.mkdir()
    (directory / f"{scene}.json").write_text(
        json.dumps(
            {
                "streams": [
                    {
                        "type": "color_camera",
                        "resolution": [3, 4],
                        "number_of_frames": 2,
                    }
                ]
            }
        )
    )
    identity3 = np.eye(3).reshape(-1, order="F").tolist()
    records = []
    for index in range(2):
        transform = np.eye(4)
        transform[0, 3] = index
        records.append(
            {
                "intrinsics": identity3,
                "transform": transform.reshape(-1, order="F").tolist(),
                "timestamp": 100 + index,
                "exposure_duration": 10 + index,
                "quaternion": [1, 0, 0, 0],
                "euler_angles": [0, 0, 0],
            }
        )
    with open(directory / f"{scene}.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (directory / f"{scene}.align.json").write_text(
        json.dumps({"coordinate_transform": np.eye(4).reshape(-1, order="F").tolist()})
    )
    batch = MultiScan(tmp_path).get_frames(
        scene,
        indices=[1, 0],
        items=("rgb_intrinsics", "world_to_camera"),
        output_size=(2, 2),
        rotate_to_up=False,
    )
    assert batch.indices.tolist() == [1, 0]
    assert batch.rgb_intrinsics.shape == (2, 3, 3)
    assert batch.world_to_camera[:, 0, 3].tolist() == [-1, 0]


def test_3rscan_zip_reads_only_requested_frames(tmp_path):
    scene = "scan"
    directory = tmp_path / scene
    directory.mkdir()
    archive_path = directory / "sequence.zip"
    identity = " ".join(map(str, np.eye(4).reshape(-1)))
    lines = [
        "m_colorWidth = 4",
        "m_depthHeight = 2",
        f"m_calibrationDepthExtrinsic = {identity}",
        "m_colorHeight = 3",
        "m_depthShift = 1000",
        f"m_calibrationColorIntrinsic = {identity}",
        "m_depthWidth = 3",
        f"m_calibrationColorExtrinsic = {identity}",
        "m_frames.size = 3",
        f"m_calibrationDepthIntrinsic = {identity}",
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("_info.txt", "\n".join(lines))
        for index in range(3):
            image = Image.fromarray(np.full((3, 4, 3), index * 100, dtype=np.uint8))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            archive.writestr(f"frame-{index:06d}.color.jpg", buffer.getvalue())
            depth = Image.fromarray(np.full((2, 3), (index + 1) * 1000, dtype=np.int32))
            buffer = io.BytesIO()
            depth.save(buffer, format="PPM")
            archive.writestr(f"frame-{index:06d}.depth.pgm", buffer.getvalue())
            pose = np.eye(4)
            pose[0, 3] = index
            if index == 1:
                pose[0, 0] = np.nan
            archive.writestr(
                f"frame-{index:06d}.pose.txt",
                "\n".join(" ".join(map(str, row)) for row in pose),
            )
    batch = ThreeRScan(tmp_path).get_frames(
        scene,
        indices=[2, 0],
        items=(
            "rgb",
            "depth",
            "rgb_intrinsics",
            "depth_intrinsics",
            "world_to_camera",
        ),
        rotate_to_up=False,
    )
    assert batch.indices.tolist() == [2, 0]
    assert batch.rgb.shape == (2, 3, 4, 3)
    assert batch.depth.shape == (2, 2, 3)
    assert batch.depth.dtype == np.float32
    assert batch.depth[:, 0, 0].tolist() == [3.0, 1.0]
    assert batch.rgb_intrinsics.shape == (2, 3, 3)
    assert batch.depth_intrinsics.shape == (2, 3, 3)
    assert batch.camera_model == "PINHOLE"
    assert batch.world_to_camera[:, 0, 3].tolist() == [-2, 0]
    sampled = ThreeRScan(tmp_path).get_frames(
        scene, step=2, items=("world_to_camera",), rotate_to_up=False
    )
    assert sampled.indices.tolist() == [0]
