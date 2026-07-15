from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VideoAsset:
    """Immutable registration record for a source video file."""

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
    id: int | None = None
    imported_at: str = ""
