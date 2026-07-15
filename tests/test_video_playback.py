from pathlib import Path

import pytest

from corporidoc.data import VideoPlaybackError, resolve_video_playback_source
from corporidoc.domain import VideoAsset


def video(source_path: Path, managed_path: Path | None = None) -> VideoAsset:
    return VideoAsset(
        patient_id=1,
        source_path=str(source_path),
        filename=source_path.name,
        file_sha256="a" * 64,
        file_size_bytes=1024,
        extension=source_path.suffix,
        duration_seconds=10,
        fps=25,
        frame_count=250,
        width=1920,
        height=1080,
        managed_path=str(managed_path) if managed_path else "",
    )


def test_playback_prefers_managed_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    managed = tmp_path / "managed.mp4"
    source.write_bytes(b"source")
    managed.write_bytes(b"managed")

    resolved = resolve_video_playback_source(video(source, managed))

    assert resolved.path == managed.resolve()
    assert resolved.label == "应用副本"


def test_playback_falls_back_to_source_path(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    resolved = resolve_video_playback_source(video(source, tmp_path / "missing.mp4"))

    assert resolved.path == source.resolve()
    assert resolved.label == "原路径（应用副本缺失）"


def test_playback_rejects_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(VideoPlaybackError, match="均不可用"):
        resolve_video_playback_source(
            video(tmp_path / "missing-source.mp4", tmp_path / "missing-managed.mp4")
        )
