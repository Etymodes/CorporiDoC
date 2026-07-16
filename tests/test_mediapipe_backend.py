import csv
import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import cv2
import pytest

from corporidoc.domain import ModelAsset
from corporidoc.pose import (
    ArtifactKind,
    InferenceRequest,
    InferenceStatus,
    MEDIAPIPE_POSE_33,
    MediaPipePoseBackend,
)


@dataclass
class Landmark:
    x: float = 0.25
    y: float = 0.5
    z: float = -0.1
    visibility: float = 0.9
    presence: float = 0.8


class Frame:
    shape = (480, 640, 3)


class FakeCapture:
    def __init__(self, _: str) -> None:
        self.frames = [Frame(), Frame()]

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 2.0,
        }.get(property_id, 0.0)

    def read(self) -> tuple[bool, Frame | None]:
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self) -> None:
        pass


class FakeResult:
    def __init__(self, detected: bool) -> None:
        points = [Landmark() for _ in MEDIAPIPE_POSE_33]
        self.pose_landmarks = [points] if detected else []
        self.pose_world_landmarks = [points] if detected else []


class FakeLandmarker:
    def __init__(self) -> None:
        self.results = [FakeResult(True), FakeResult(False)]

    def __enter__(self) -> "FakeLandmarker":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def detect_for_video(self, _: object, timestamp_ms: int) -> FakeResult:
        assert timestamp_ms >= 0
        return self.results.pop(0)


class FakePoseLandmarker:
    @staticmethod
    def create_from_options(_: object) -> FakeLandmarker:
        return FakeLandmarker()


class FakeVision:
    class RunningMode:
        VIDEO = "video"

    class PoseLandmarkerOptions:
        def __init__(self, **values: object) -> None:
            self.values = values

    PoseLandmarker = FakePoseLandmarker


class FakeTaskPython:
    class BaseOptions:
        def __init__(self, **values: object) -> None:
            self.values = values


class FakeMediaPipe:
    class ImageFormat:
        SRGB = "srgb"

    class Image:
        def __init__(self, **values: object) -> None:
            self.values = values


class FakeVideoWriter:
    def __init__(self, path: str, *_: object) -> None:
        self.path = Path(path)
        self.path.write_bytes(b"fake labeled video")

    def isOpened(self) -> bool:
        return True

    def write(self, _: object) -> None:
        pass

    def release(self) -> None:
        pass


def test_mediapipe_backend_exports_detected_and_missing_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "models" / "pose.task"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = ModelAsset(
        name="Pose Full",
        backend_name="mediapipe-pose-landmarker",
        model_version="full-test",
        file_path=str(model_path),
        file_sha256=model_hash,
        file_size_bytes=model_path.stat().st_size,
        license_name="Apache-2.0",
        source_url="https://example.invalid/pose.task",
    )
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
    backend = MediaPipePoseBackend(
        model,
        package_version="test",
        runtime_loader=lambda: (FakeMediaPipe, FakeTaskPython, FakeVision),
    )
    request = InferenceRequest.create(
        patient_id=1,
        video_asset_id=1,
        video_path=video_path,
        video_sha256=video_hash,
        output_directory=tmp_path / "outputs",
        backend=backend.info,
        requested_artifacts=(ArtifactKind.KEYPOINTS, ArtifactKind.LABELED_VIDEO),
        parameters=backend.parameters,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _: frame)
    monkeypatch.setattr(cv2, "VideoWriter", FakeVideoWriter)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *_: 0)
    monkeypatch.setattr(cv2, "line", lambda *args: None)
    monkeypatch.setattr(cv2, "circle", lambda *args: None)
    monkeypatch.setattr(cv2, "putText", lambda *args: None)

    result = backend.analyze(request)

    assert result.status is InferenceStatus.SUCCEEDED
    assert result.processed_frames == 2
    assert [artifact.kind for artifact in result.artifacts] == [
        ArtifactKind.KEYPOINTS,
        ArtifactKind.LABELED_VIDEO,
    ]
    assert result.artifacts[1].path.read_bytes() == b"fake labeled video"
    assert "1/2 帧未检出" in "；".join(result.warnings)
    with result.artifacts[0].path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 66
    assert rows[0]["keypoint"] == "nose"
    assert rows[0]["detected"] == "1"
    assert rows[0]["x_pixels"] == "160.000"
    assert rows[33]["detected"] == "0"
    assert rows[33]["visibility"] == ""
