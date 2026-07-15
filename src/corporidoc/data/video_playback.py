from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from corporidoc.domain import VideoAsset


class VideoPlaybackError(FileNotFoundError):
    pass


@dataclass(frozen=True, slots=True)
class VideoPlaybackSource:
    path: Path
    label: str


def resolve_video_playback_source(video: VideoAsset) -> VideoPlaybackSource:
    if video.managed_path:
        managed = Path(video.managed_path).expanduser().resolve()
        if managed.is_file():
            return VideoPlaybackSource(managed, "应用副本")

    source = Path(video.source_path).expanduser().resolve()
    if source.is_file():
        return VideoPlaybackSource(source, "原路径（应用副本缺失）")

    raise VideoPlaybackError("应用副本和原路径均不可用，无法播放该视频")
