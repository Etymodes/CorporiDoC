from pathlib import Path

from corporidoc.data import VideoMetadata
from corporidoc.domain import assess_video_quality, decode_quality_warnings


def metadata(
    *,
    width: int = 1920,
    height: int = 1080,
    fps: float = 25.0,
    frame_count: int = 250,
    duration_seconds: float = 10.0,
) -> VideoMetadata:
    return VideoMetadata(
        source_path=Path("demo.mp4"),
        filename="demo.mp4",
        file_sha256="a" * 64,
        file_size_bytes=1024,
        extension=".mp4",
        duration_seconds=duration_seconds,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def test_quality_passes_complete_baseline_metadata() -> None:
    assessment = assess_video_quality(metadata())

    assert assessment.summary == "基础检查通过"
    assert assessment.warnings == ()
    assert decode_quality_warnings(assessment.to_json()) == ()


def test_quality_warns_for_low_resolution_and_frame_rate() -> None:
    assessment = assess_video_quality(metadata(width=320, height=240, fps=10.0))

    assert assessment.summary == "需复核（2）"
    assert "分辨率低于 640×480" in assessment.warnings[0]
    assert "帧率低于 15 FPS" in assessment.warnings[1]


def test_quality_warns_when_timing_metadata_is_missing() -> None:
    assessment = assess_video_quality(metadata(fps=0, frame_count=0, duration_seconds=0))

    assert len(assessment.warnings) == 3
    assert "未能读取帧率" in assessment.warnings[0]
    assert "未能读取总帧数" in assessment.warnings[1]
    assert "未能估算视频时长" in assessment.warnings[2]


def test_invalid_quality_json_is_reported() -> None:
    assert decode_quality_warnings("not-json") == ("质控记录无法解析",)
