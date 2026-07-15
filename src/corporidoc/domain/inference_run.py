from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceArtifactRecord:
    kind: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class InferenceRunRecord:
    id: int
    request_id: str
    patient_id: int
    video_asset_id: int
    status: str
    backend_name: str
    backend_version: str
    model_name: str
    model_version: str
    weights_sha256: str
    keypoint_schema_version: str
    video_sha256: str
    requested_artifacts_json: str
    parameters_json: str
    requested_at: str
    started_at: str
    finished_at: str
    processed_frames: int
    warnings_json: str
    error_message: str
    artifacts: tuple[InferenceArtifactRecord, ...] = ()
