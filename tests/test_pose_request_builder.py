from pathlib import Path

import pytest

from corporidoc.domain import VideoAsset
from corporidoc.pose import MockPoseBackend
from corporidoc.pose.request_builder import build_inference_request


def video(source: Path, managed: Path, *, video_id: int | None = 9) -> VideoAsset:
    return VideoAsset(
        id=video_id,
        patient_id=3,
        source_path=str(source),
        managed_path=str(managed),
        filename=source.name,
        file_sha256="a" * 64,
        file_size_bytes=100,
        extension=".mp4",
        duration_seconds=1,
        fps=25,
        frame_count=25,
        width=640,
        height=480,
    )


def test_request_builder_prefers_managed_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    managed = tmp_path / "managed.mp4"
    source.write_bytes(b"source")
    managed.write_bytes(b"managed")
    backend = MockPoseBackend()

    request = build_inference_request(video(source, managed), tmp_path / "data", backend.info)

    assert request.video_path == managed.resolve()
    assert request.backend == backend.info
    assert request.output_directory == (
        (tmp_path / "data").resolve()
        / "patients"
        / "patient-000003"
        / "inference"
        / "video-000009"
    )


def test_request_builder_rejects_unpersisted_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match="数据库 ID"):
        build_inference_request(
            video(source, tmp_path / "missing.mp4", video_id=None),
            tmp_path / "data",
            MockPoseBackend().info,
        )
