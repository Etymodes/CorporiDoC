import sqlite3
from pathlib import Path

import cv2
import pytest

from corporidoc.data import DuplicateVideoError, PatientRepository, VideoProbe, sha256_file
from corporidoc.domain import Patient, VideoAsset


class FakeCapture:
    def __init__(self, _: str) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 250.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        }.get(property_id, 0.0)

    def read(self) -> tuple[bool, object]:
        return True, object()

    def release(self) -> None:
        self.released = True


def test_probe_hashes_and_reads_metadata(tmp_path: Path) -> None:
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"CorporiDoC demo video bytes")

    metadata = VideoProbe(capture_factory=FakeCapture).inspect(source)

    assert metadata.file_sha256 == sha256_file(source)
    assert metadata.duration_seconds == 10.0
    assert metadata.fps == 25.0
    assert metadata.frame_count == 250
    assert (metadata.width, metadata.height) == (1920, 1080)


def test_repository_rejects_duplicate_video_content(tmp_path: Path) -> None:
    repository = PatientRepository(tmp_path / "corporidoc.sqlite3")
    patient = repository.create_patient(Patient(patient_code="DEMO-VIDEO-001"))
    assert patient.id is not None
    video = VideoAsset(
        patient_id=patient.id,
        source_path=str(tmp_path / "demo.mp4"),
        filename="demo.mp4",
        file_sha256="a" * 64,
        file_size_bytes=1024,
        extension=".mp4",
        duration_seconds=10.0,
        fps=25.0,
        frame_count=250,
        width=1920,
        height=1080,
    )

    created = repository.create_video_asset(video)
    assert created.id is not None
    assert repository.list_video_assets(patient.id)[0].file_sha256 == "a" * 64

    with pytest.raises(DuplicateVideoError):
        repository.create_video_asset(video)

    assert repository.audit_events()[-1]["action"] == "IMPORT_VIDEO"


def test_existing_m1_database_is_migrated(tmp_path: Path) -> None:
    database = tmp_path / "corporidoc.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL DEFAULT '',
                sex TEXT NOT NULL DEFAULT '未知',
                date_of_birth TEXT NOT NULL DEFAULT '',
                etiology TEXT NOT NULL DEFAULT '',
                injury_date TEXT NOT NULL DEFAULT '',
                current_diagnosis TEXT NOT NULL DEFAULT '待评估',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id INTEGER,
                summary TEXT NOT NULL
            );
            """
        )

    repository = PatientRepository(database)

    assert repository.list_video_assets(patient_id=1) == []
