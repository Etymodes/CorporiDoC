import shutil
import sqlite3
from pathlib import Path

import cv2
import pytest

from corporidoc.data import (
    DuplicateVideoError,
    ManagedVideoStore,
    PatientRepository,
    VideoProbe,
    VideoStorageError,
    sha256_file,
)
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


def test_video_is_copied_into_patient_managed_storage(tmp_path: Path) -> None:
    source = tmp_path / "incoming" / "demo.mp4"
    source.parent.mkdir()
    source.write_bytes(b"CorporiDoC managed video")
    metadata = VideoProbe(capture_factory=FakeCapture).inspect(source)

    managed = ManagedVideoStore(tmp_path / "app-data").archive(metadata, patient_id=7)
    source.unlink()

    assert managed.is_file()
    assert managed.parent == tmp_path / "app-data" / "patients" / "patient-000007" / "videos"
    assert managed.name == f"{metadata.file_sha256}.mp4"
    assert sha256_file(managed) == metadata.file_sha256


def test_managed_copy_can_be_removed_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "incoming" / "demo.mp4"
    source.parent.mkdir()
    source.write_bytes(b"CorporiDoC managed video")
    metadata = VideoProbe(capture_factory=FakeCapture).inspect(source)
    store = ManagedVideoStore(tmp_path / "app-data")
    managed = store.archive(metadata, patient_id=7)

    assert store.remove_managed_copy(str(managed)) is True
    assert not managed.exists()
    assert source.is_file()


def test_managed_store_refuses_to_delete_external_file(tmp_path: Path) -> None:
    external = tmp_path / "external.mp4"
    external.write_bytes(b"do not delete")

    with pytest.raises(VideoStorageError, match="患者目录之外"):
        ManagedVideoStore(tmp_path / "app-data").remove_managed_copy(str(external))

    assert external.is_file()


def test_failed_copy_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"CorporiDoC managed video")
    metadata = VideoProbe(capture_factory=FakeCapture).inspect(source)
    store = ManagedVideoStore(tmp_path / "app-data")

    def fail_copy(_: Path, __: Path) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(shutil, "copy2", fail_copy)

    with pytest.raises(VideoStorageError, match="无法复制视频到应用目录"):
        store.archive(metadata, patient_id=7)

    patient_directory = (
        tmp_path / "app-data" / "patients" / "patient-000007" / "videos"
    )
    assert list(patient_directory.iterdir()) == []


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
        managed_path=str(tmp_path / "patients/patient-000001/videos/demo.mp4"),
        camera_view="床尾",
        body_side="双侧",
        capture_protocol="静息观察",
        video_notes="测试视频",
        quality_rule_version="m2b-basic-v1",
        quality_warnings_json="[]",
    )

    created = repository.create_video_asset(video)
    assert created.id is not None
    stored = repository.list_video_assets(patient.id)[0]
    assert stored.file_sha256 == "a" * 64
    assert stored.source_path == str(tmp_path / "demo.mp4")
    assert stored.managed_path == str(
        tmp_path / "patients/patient-000001/videos/demo.mp4"
    )
    assert stored.camera_view == "床尾"
    assert stored.body_side == "双侧"
    assert stored.capture_protocol == "静息观察"
    assert stored.video_notes == "测试视频"
    assert stored.quality_rule_version == "m2b-basic-v1"

    with pytest.raises(DuplicateVideoError):
        repository.create_video_asset(video)

    assert repository.audit_events()[-1]["action"] == "IMPORT_VIDEO"

    deleted = repository.delete_video_asset(created.id)
    assert deleted.file_sha256 == video.file_sha256
    assert repository.list_video_assets(patient.id) == []
    assert repository.audit_events()[-1]["action"] == "DELETE_VIDEO"

    recreated = repository.create_video_asset(video)
    assert recreated.id is not None


def test_video_capture_metadata_update_is_limited_and_audited(tmp_path: Path) -> None:
    repository = PatientRepository(tmp_path / "corporidoc.sqlite3")
    patient = repository.create_patient(Patient(patient_code="DEMO-EDIT-001"))
    assert patient.id is not None
    created = repository.create_video_asset(
        VideoAsset(
            patient_id=patient.id,
            source_path="/source/demo.mp4",
            filename="demo.mp4",
            file_sha256="b" * 64,
            file_size_bytes=2048,
            extension=".mp4",
            duration_seconds=20.0,
            fps=30.0,
            frame_count=600,
            width=1280,
            height=720,
            managed_path="/managed/demo.mp4",
            quality_rule_version="m2b-basic-v1",
            quality_warnings_json="[]",
        )
    )
    assert created.id is not None

    updated = repository.update_video_metadata(
        created.id,
        camera_view=" 俯视 ",
        body_side=" 双侧 ",
        capture_protocol=" 指令任务 ",
        video_notes=" 复核后修改 ",
    )

    assert updated.camera_view == "俯视"
    assert updated.body_side == "双侧"
    assert updated.capture_protocol == "指令任务"
    assert updated.video_notes == "复核后修改"
    assert updated.patient_id == created.patient_id
    assert updated.file_sha256 == created.file_sha256
    assert updated.source_path == created.source_path
    assert updated.managed_path == created.managed_path
    assert updated.imported_at == created.imported_at
    assert updated.quality_rule_version == created.quality_rule_version
    assert repository.audit_events()[-1]["action"] == "UPDATE_VIDEO_METADATA"

    audit_count = len(repository.audit_events())
    unchanged = repository.update_video_metadata(
        created.id,
        camera_view="俯视",
        body_side="双侧",
        capture_protocol="指令任务",
        video_notes="复核后修改",
    )
    assert unchanged == updated
    assert len(repository.audit_events()) == audit_count


def test_missing_video_cannot_be_updated(tmp_path: Path) -> None:
    repository = PatientRepository(tmp_path / "corporidoc.sqlite3")

    with pytest.raises(KeyError, match="视频登记不存在"):
        repository.update_video_metadata(
            999,
            camera_view="正面",
            body_side="双侧",
            capture_protocol="静息观察",
            video_notes="",
        )


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
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(video_assets)")}
    assert {
        "managed_path",
        "camera_view",
        "body_side",
        "capture_protocol",
        "video_notes",
        "quality_rule_version",
        "quality_warnings_json",
    }.issubset(columns)
