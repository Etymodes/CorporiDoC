import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from corporidoc.data import PatientRepository
from corporidoc.domain import Patient, VideoAsset
from corporidoc.pose import (
    ArtifactKind,
    InferenceArtifact,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    MockPoseBackend,
)


def registered_video(repository: PatientRepository, tmp_path: Path) -> VideoAsset:
    patient = repository.create_patient(Patient(patient_code="INFERENCE-001"))
    assert patient.id is not None
    source = tmp_path / "patients" / "patient-000001" / "videos" / "video.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"registered video")
    video_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    return repository.create_video_asset(
        VideoAsset(
            patient_id=patient.id,
            source_path=str(source),
            managed_path=str(source),
            filename=source.name,
            file_sha256=video_hash,
            file_size_bytes=source.stat().st_size,
            extension=".mp4",
            duration_seconds=1,
            fps=25,
            frame_count=25,
            width=640,
            height=480,
        )
    )


def inference_request(
    repository: PatientRepository,
    video: VideoAsset,
) -> InferenceRequest:
    assert video.id is not None
    return InferenceRequest.create(
        patient_id=video.patient_id,
        video_asset_id=video.id,
        video_path=Path(video.managed_path),
        video_sha256=video.file_sha256,
        output_directory=repository.database_path.parent / "patients" / "outputs",
        backend=MockPoseBackend().info,
        parameters={"confidence_threshold": 0.5},
    )


def test_successful_inference_run_and_artifact_are_persisted(tmp_path: Path) -> None:
    repository = PatientRepository(tmp_path / "corporidoc.sqlite3")
    video = registered_video(repository, tmp_path)
    request = inference_request(repository, video)

    running = repository.create_inference_run(request)
    artifact_path = request.output_directory / "keypoints.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("frame,x,y\n0,1,2\n")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    result = InferenceResult(
        request.request_id,
        InferenceStatus.SUCCEEDED,
        now,
        now,
        processed_frames=1,
        artifacts=(
            InferenceArtifact(ArtifactKind.KEYPOINTS, artifact_path, artifact_hash),
        ),
    )

    finished = repository.finish_inference_run(result)

    assert running.status == "running"
    assert finished.status == "succeeded"
    assert finished.processed_frames == 1
    assert finished.artifacts[0].sha256 == artifact_hash
    assert repository.list_inference_runs(video.patient_id) == [finished]
    assert [event["action"] for event in repository.audit_events()][-2:] == [
        "START_INFERENCE",
        "FINISH_INFERENCE",
    ]
    with pytest.raises(ValueError, match="证据链"):
        repository.delete_video_asset(video.id)


def test_failed_run_has_no_artifacts(tmp_path: Path) -> None:
    repository = PatientRepository(tmp_path / "corporidoc.sqlite3")
    video = registered_video(repository, tmp_path)
    request = inference_request(repository, video)
    repository.create_inference_run(request)
    now = datetime.now(timezone.utc)

    finished = repository.finish_inference_run(
        InferenceResult(
            request.request_id,
            InferenceStatus.FAILED,
            now,
            now,
            error_message="decoder failed",
        )
    )

    assert finished.status == "failed"
    assert finished.error_message == "decoder failed"
    assert finished.artifacts == ()


def test_failed_run_cannot_register_artifact(tmp_path: Path) -> None:
    repository = PatientRepository(tmp_path / "corporidoc.sqlite3")
    video = registered_video(repository, tmp_path)
    request = inference_request(repository, video)
    repository.create_inference_run(request)
    artifact_path = request.output_directory / "partial.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("partial")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="不能登记完成产物"):
        repository.finish_inference_run(
            InferenceResult(
                request.request_id,
                InferenceStatus.FAILED,
                now,
                now,
                artifacts=(
                    InferenceArtifact(
                        ArtifactKind.KEYPOINTS,
                        artifact_path,
                        artifact_hash,
                    ),
                ),
                error_message="decoder failed",
            )
        )


def test_reopen_marks_interrupted_run_failed(tmp_path: Path) -> None:
    database_path = tmp_path / "corporidoc.sqlite3"
    repository = PatientRepository(database_path)
    video = registered_video(repository, tmp_path)
    request = inference_request(repository, video)
    repository.create_inference_run(request)

    reopened = PatientRepository(database_path)
    recovered = reopened.get_inference_run(request.request_id)

    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.error_message == "应用上次退出前任务未完成"
    assert reopened.audit_events()[-1]["action"] == "RECOVER_INFERENCE"
