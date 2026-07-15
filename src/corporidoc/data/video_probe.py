from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


class VideoProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    source_path: Path
    filename: str
    file_sha256: str
    file_size_bytes: int
    extension: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoProbe:
    def __init__(self, capture_factory: Callable[[str], Capture] = cv2.VideoCapture) -> None:
        self.capture_factory = capture_factory

    def inspect(self, source: Path) -> VideoMetadata:
        path = source.expanduser().resolve()
        if not path.is_file():
            raise VideoProbeError("视频文件不存在或不是普通文件")
        if path.stat().st_size == 0:
            raise VideoProbeError("视频文件为空")

        capture = self.capture_factory(str(path))
        try:
            if not capture.isOpened():
                raise VideoProbeError("OpenCV 无法打开该视频；文件可能损坏或编码不受支持")

            fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
            frame_count = max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = max(0, round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(0, round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            decoded, frame = capture.read()
            if not decoded or frame is None:
                raise VideoProbeError("视频已打开，但第一帧无法解码")
        finally:
            capture.release()

        if width <= 0 or height <= 0:
            raise VideoProbeError("未能读取有效的视频分辨率")

        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return VideoMetadata(
            source_path=path,
            filename=path.name,
            file_sha256=sha256_file(path),
            file_size_bytes=path.stat().st_size,
            extension=path.suffix.lower(),
            duration_seconds=duration,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
