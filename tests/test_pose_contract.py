from datetime import datetime, timezone
from pathlib import Path

import pytest

from corporidoc.pose import (
    ArtifactKind,
    BackendInfo,
    CancellationToken,
    InferenceCancelled,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    PoseBackend,
    ProgressUpdate,
)


VIDEO_SHA256 = "a" * 64


def backend_info() -> BackendInfo:
    return BackendInfo(
        name="mock",
        version="1.0",
        model_name="stationary-points",
        model_version="1",
    )


def test_request_create_and_validate(tmp_path: Path) -> None:
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"test video placeholder")

    request = InferenceRequest.create(
        patient_id=1,
        video_asset_id=2,
        video_path=video,
        video_sha256=VIDEO_SHA256,
        output_directory=tmp_path / "outputs",
        backend=backend_info(),
        requested_artifacts=(ArtifactKind.KEYPOINTS, ArtifactKind.LABELED_VIDEO),
        parameters={"confidence_threshold": 0.5},
    )

    assert len(request.request_id) == 32
    assert request.requested_at.tzinfo is not None
    assert request.parameters["confidence_threshold"] == 0.5
    assert request.validation_errors() == ()


def test_request_reports_invalid_input(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied")
    request = InferenceRequest(
        request_id="",
        patient_id=0,
        video_asset_id=-1,
        video_path=tmp_path / "missing.mp4",
        video_sha256="invalid",
        output_directory=output_file,
        backend=backend_info(),
        requested_artifacts=(),
        requested_at=datetime(2026, 1, 1),
    )

    assert len(request.validation_errors()) == 8


def test_progress_and_cancellation() -> None:
    assert ProgressUpdate(25, 100).fraction == 0.25
    assert ProgressUpdate(120, 100).fraction == 1.0
    assert ProgressUpdate(0, 0).fraction is None

    token = CancellationToken()
    token.cancel()
    with pytest.raises(InferenceCancelled):
        token.raise_if_cancelled()


def test_result_requires_terminal_consistent_state() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="终态"):
        InferenceResult("job-1", InferenceStatus.RUNNING, now, now)
    with pytest.raises(ValueError, match="错误信息"):
        InferenceResult("job-1", InferenceStatus.FAILED, now, now)


def test_protocol_accepts_backend_shape() -> None:
    class ExampleBackend:
        info = backend_info()

        def analyze(
            self,
            request: InferenceRequest,
            *,
            progress=None,
            cancellation=None,
        ) -> InferenceResult:
            now = datetime.now(timezone.utc)
            return InferenceResult(request.request_id, InferenceStatus.SUCCEEDED, now, now)

    assert isinstance(ExampleBackend(), PoseBackend)
