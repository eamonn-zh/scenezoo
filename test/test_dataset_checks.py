from __future__ import annotations

import zipfile

import pytest

from scenezoo import Dataset, DatasetCheckError, get_dataset


BUILTINS = (
    "ase",
    "scannetv2",
    "scannetppv2",
    "arkitscenes",
    "multiscan",
    "3rscan",
    "s3dis",
    "matterport3d",
    "scenenn",
    "structured3d",
)


def test_base_check_is_explicit_structured_and_optionally_raises(tmp_path):
    missing = tmp_path / "not-downloaded"
    dataset = Dataset(missing)
    assert "splits" not in dataset.__dict__

    report = dataset.check()
    assert not report.ok
    assert report.errors[0].code == "missing-root"
    assert report.root_dir == missing
    assert "Expected:" in report.format()
    assert "Suggested action:" in str(report)
    assert "splits" not in dataset.__dict__

    with pytest.raises(DatasetCheckError) as caught:
        dataset.check(raise_on_error=True)
    assert caught.value.report == report


def test_base_check_passes_for_an_existing_root(tmp_path):
    report = Dataset(tmp_path).check(raise_on_error=True)
    assert report.ok
    assert not report.errors


@pytest.mark.parametrize("name", BUILTINS)
def test_each_builtin_explains_an_empty_layout_without_remote_access(name, tmp_path):
    dataset = get_dataset(name, tmp_path, offline=True)
    report = dataset.check(sample_ids=("definitely-missing-scene",))

    assert not report.ok
    rendered = report.format()
    assert "FAILED" in rendered
    assert "Expected:" in rendered
    assert "Suggested action:" in rendered
    assert "splits" not in dataset.__dict__


def test_scannet_focused_check_accepts_minimal_core_scene(tmp_path):
    scene_id = "scene0000_00"
    scene = tmp_path / "scans" / scene_id
    scene.mkdir(parents=True)
    for name in (
        f"{scene_id}_vh_clean_2.ply",
        f"{scene_id}.sens",
        f"{scene_id}.txt",
    ):
        (scene / name).touch()

    report = get_dataset("scannetv2", tmp_path).check(sample_ids=(scene_id,))
    assert report.ok, report.format()
    assert report.stats["scenes_found"] == 1
    assert report.stats["scenes_requested"] == 1


def test_s3dis_checker_reports_malformed_rooms(tmp_path):
    room = tmp_path / "Area_1" / "office_1"
    room.mkdir(parents=True)
    report = get_dataset("s3dis", tmp_path).check(sample_ids=("Area_1/office_1",))

    assert not report.ok
    assert any(
        issue.code == "missing-annotation-directories" for issue in report.errors
    )
    assert any(issue.code == "missing-areas" for issue in report.warnings)


def test_structured3d_checker_requires_official_zip_extraction(
    tmp_path, structured3d_metadata
):
    archive_path = tmp_path / "Structured3D_0.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Structured3D/scene_00000/annotation_3d.json", "{}")

    dataset = get_dataset("structured3d", tmp_path, **structured3d_metadata)
    report = dataset.check(sample_ids=("scene_00000",))
    assert not report.ok
    assert report.stats["zip_shards_found"] == 1
    assert report.stats["extracted_scenes"] == 0
    assert any(issue.code == "scene-not-extracted" for issue in report.errors)
    assert any(issue.code == "archives-need-extraction" for issue in report.warnings)
