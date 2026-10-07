from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction

import av
import numpy as np
import pytest

import scenezoo.io.video as video

needs_ffmpeg_cli = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the ffmpeg/ffprobe executables are not installed",
)


def _encode(path, frame_count, *, gop=10, size=(32, 16)):
    """Write lossless H.264 with B-frames whose frame n has brightness 2 * n."""

    width, height = size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuvj444p"  # Full range keeps gray levels exact.
        stream.options = {"qp": "0", "g": str(gop), "bf": "2", "sc_threshold": "0"}
        for index in range(frame_count):
            gray = np.full((height, width), 2 * index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(gray, format="gray")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def _frame_ids(frames):
    return np.round(frames.reshape(len(frames), -1).mean(axis=1) / 2).astype(int)


@pytest.fixture(autouse=True)
def _fresh_index_cache():
    video._cached_index.cache_clear()
    yield
    video._cached_index.cache_clear()


@pytest.mark.parametrize("suffix", [".mp4", ".mkv", ".mov"])
@pytest.mark.parametrize("min_seek_skip", [0, 64, 10_000])
def test_selected_frames_are_exact_and_ordered(
    tmp_path, monkeypatch, suffix, min_seek_skip
):
    path = tmp_path / f"clip{suffix}"
    _encode(path, 120)
    # 0 seeks whenever possible, 10_000 always decodes forward.
    monkeypatch.setattr(video, "_MIN_SEEK_SKIP", min_seek_skip)
    assert video.read_video_frame_count(path) == 120
    assert video.read_video_frame_size(path) == (32, 16)
    assert video.read_video_rotation(path) == 0

    requested = [119, 0, 57, 58, 3, 90, 60, 57, 1, 10, 9, 11]
    frames = video.read_video_frames(path, frame_indices=requested)
    assert frames.shape == (len(requested), 16, 32, 3)
    assert frames.dtype == np.uint8
    assert _frame_ids(frames).tolist() == requested

    stepped = np.arange(5, 120, 7)
    gray = video.read_grayscale_video(path, frame_indices=stepped)
    assert gray.shape == (len(stepped), 16, 32)
    assert _frame_ids(gray).tolist() == stepped.tolist()

    every = video.read_grayscale_video(path, frame_indices=np.arange(120))
    assert _frame_ids(every).tolist() == list(range(120))


def test_seek_overshoot_restarts_from_the_beginning(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    _encode(path, 80)
    opened = video._open

    class OvershootingContainer:
        """Simulate a demuxer whose seek lands on a later keyframe."""

        def __init__(self, container):
            self._container = container

        def __getattr__(self, name):
            return getattr(self._container, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._container.close()

        def seek(self, offset, **options):
            # Land on the next keyframe instead of the requested one.
            self._container.seek(offset + 1, **{**options, "backward": False})
            seeks.append(offset)

    monkeypatch.setattr(video, "_MIN_SEEK_SKIP", 0)
    monkeypatch.setattr(video, "_open", lambda p: OvershootingContainer(opened(p)))
    seeks = []
    requested = [45, 2, 65, 33]  # Keyframes every 10 frames; 70 is the last.
    frames = video.read_video_frames(path, frame_indices=requested)
    assert _frame_ids(frames).tolist() == requested
    assert seeks  # The fallback was exercised.


def test_empty_out_of_range_and_invalid_requests(tmp_path):
    path = tmp_path / "clip.mp4"
    _encode(path, 12)
    assert video.read_video_frames(path, frame_indices=[]).shape == (0, 16, 32, 3)
    assert video.read_grayscale_video(path, frame_indices=[]).shape == (0, 16, 32)
    with pytest.raises(IndexError):
        video.read_video_frames(path, frame_indices=[12])
    with pytest.raises(ValueError, match="non-negative"):
        video.read_video_frames(path, frame_indices=[-1])
    with pytest.raises(ValueError, match="integer"):
        video.read_video_frames(path, frame_indices=[0.5])
    with pytest.raises(ValueError, match="pix_fmt"):
        video.read_video_frames(path, frame_indices=[0], pix_fmt="bgr24")
    with pytest.raises(FileNotFoundError):
        video.read_video_frames(tmp_path / "missing.mp4", frame_indices=[0])
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="Unable to open"):
        video.read_video_frames(corrupt, frame_indices=[0])


def test_index_cache_follows_file_changes(tmp_path):
    path = tmp_path / "clip.mp4"
    _encode(path, 12)
    assert video.read_video_frame_count(path) == 12
    _encode(path, 20)
    assert video.read_video_frame_count(path) == 20


def _ffmpeg(*arguments):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *map(str, arguments)],
        check=True,
        capture_output=True,
    )


def _ffprobe_clockwise_rotation(path) -> int:
    """Read rotation independently, the way ffprobe reports it."""

    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams"]
        + ["-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    stream = json.loads(output)["streams"][0]
    if "rotate" in stream.get("tags", {}):
        return int(stream["tags"]["rotate"]) % 360
    for side_data in stream.get("side_data_list", ()):
        if "rotation" in side_data:
            return int(round(-float(side_data["rotation"]))) % 360
    return 0


@needs_ffmpeg_cli
def test_rotation_metadata_is_reported_but_not_applied(tmp_path):
    source = tmp_path / "clip.mp4"
    _encode(source, 12)
    path = tmp_path / "rotated.mp4"
    # FFmpeg < 6 writes the legacy tag on a stream-copy remux; newer releases
    # take the display matrix as an input option.
    for options in (
        ["-i", source, "-c", "copy", "-metadata:s:v:0", "rotate=90"],
        ["-display_rotation:v:0", "-90", "-i", source, "-c", "copy"],
    ):
        try:
            _ffmpeg(*options, path)
        except subprocess.CalledProcessError:
            continue
        if _ffprobe_clockwise_rotation(path):
            break
    else:
        pytest.skip("this FFmpeg build cannot write rotation metadata")
    # FFmpeg versions disagree on the sign written for "rotate=90", so compare
    # with ffprobe's own reading rather than the requested value.
    rotation = video.read_video_rotation(path)
    assert rotation in {90, 270}
    assert rotation == _ffprobe_clockwise_rotation(path)
    assert video.read_video_frame_size(path) == (32, 16)
    frames = video.read_video_frames(path, frame_indices=[11, 4])
    assert frames.shape == (2, 16, 32, 3)
    assert _frame_ids(frames).tolist() == [11, 4]


@needs_ffmpeg_cli
def test_frames_hidden_by_mov_edit_lists_are_numbered(tmp_path):
    source = tmp_path / "clip.mp4"
    _encode(source, 60)
    path = tmp_path / "trimmed.mov"
    # A stream-copy trim inside a GOP keeps the GOP from keyframe 10 and hides
    # its first frames with an edit list. Every coded frame is still numbered.
    _ffmpeg("-ss", "0.5", "-i", source, "-c", "copy", path)
    count = video.read_video_frame_count(path)
    frames = video.read_grayscale_video(path, frame_indices=np.arange(count))
    ids = _frame_ids(frames)
    assert ids[0] == 10
    assert ids.tolist() == list(range(10, 10 + count))
    assert ids[-1] == 59
