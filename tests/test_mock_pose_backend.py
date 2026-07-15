import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

from corporidoc.pose import (
    ArtifactKind,
    CancellationToken,
    InferenceRequest,
    InferenceStatus,
    MockPoseBackend,
    PoseBackend,
)


class FakeFrame:
    shape = (100, 200, 3)


class FakeCapture:
    def __init__(self, frame_count: int = 3, opened: bool = True) -> None:
        self.frame_count = frame_count
        self.opened = opened
        self.current = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        return 25.0 if property_id == 5 else float(self.frame_count)

    def read(self) -> tuple[bool, FakeFrame | None]:
        if self.current >= self.frame_count:
            return False, None
        self.current += 1
        return True, FakeFrame()

    def release(self) -> None:
        self.released = True


def install_fake_cv2(monkeypatch, capture: FakeCapture) -> None:
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        VideoCapture=lambda _: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)


def request_for(tmp_path: Path, backend: MockPoseBackend) -> InferenceRequest:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video")
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    return InferenceRequest.create(
        patient_id=1,
        video_asset_id=2,
        video_path=video,
        video_sha256=video_hash,
        output_directory=tmp_path / "outputs",
        backend=backend.info,
    )


def test_mock_backend_generates_deterministic_keypoints(tmp_path: Path, monkeypatch) -> None:
    backend = MockPoseBackend()
    capture = FakeCapture()
    install_fake_cv2(monkeypatch, capture)
    updates = []

    result = backend.analyze(request_for(tmp_path, backend), progress=updates.append)

    assert isinstance(backend, PoseBackend)
    assert result.status is InferenceStatus.SUCCEEDED
    assert result.processed_frames == 3
    assert capture.released
    assert result.artifacts[0].kind is ArtifactKind.KEYPOINTS
    rows = result.artifacts[0].path.read_text().splitlines()
    assert len(rows) == 10
    assert rows[1] == "0,0.000000,nose,100.000,20.000,1.000,mock-not-clinical"
    assert updates[-1].fraction == 1.0


def test_mock_backend_cancels_without_artifact(tmp_path: Path, monkeypatch) -> None:
    backend = MockPoseBackend()
    install_fake_cv2(monkeypatch, FakeCapture())
    token = CancellationToken()
    token.cancel()

    result = backend.analyze(request_for(tmp_path, backend), cancellation=token)

    assert result.status is InferenceStatus.CANCELLED
    assert result.artifacts == ()
    assert not list(tmp_path.rglob("*.csv"))


def test_mock_backend_rejects_changed_video(tmp_path: Path, monkeypatch) -> None:
    backend = MockPoseBackend()
    install_fake_cv2(monkeypatch, FakeCapture())
    request = request_for(tmp_path, backend)
    request.video_path.write_bytes(b"changed after registration")

    result = backend.analyze(request)

    assert result.status is InferenceStatus.FAILED
    assert "SHA-256" in result.error_message
    assert result.artifacts == ()
