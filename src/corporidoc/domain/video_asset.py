from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VideoAsset:
    """Immutable registration record for a managed video and its source path."""

    patient_id: int
    source_path: str
    filename: str
    file_sha256: str
    file_size_bytes: int
    extension: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    managed_path: str = ""
    camera_view: str = "未记录"
    body_side: str = "未记录"
    capture_protocol: str = ""
    video_notes: str = ""
    quality_rule_version: str = ""
    quality_warnings_json: str = "[]"
    id: int | None = None
    imported_at: str = ""
