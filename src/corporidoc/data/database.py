from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from corporidoc.domain import Patient, VideoAsset


class DuplicatePatientCodeError(ValueError):
    pass


class DuplicateVideoError(ValueError):
    pass


class PatientRepository:
    """Small SQLite repository with an append-only audit trail."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
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

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER,
                    summary TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS video_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE RESTRICT,
                    source_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL UNIQUE,
                    file_size_bytes INTEGER NOT NULL,
                    extension TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    fps REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    managed_path TEXT NOT NULL DEFAULT '',
                    camera_view TEXT NOT NULL DEFAULT '未记录',
                    body_side TEXT NOT NULL DEFAULT '未记录',
                    capture_protocol TEXT NOT NULL DEFAULT '',
                    video_notes TEXT NOT NULL DEFAULT '',
                    quality_rule_version TEXT NOT NULL DEFAULT '',
                    quality_warnings_json TEXT NOT NULL DEFAULT '[]',
                    imported_at TEXT NOT NULL
                );
                """
            )
            video_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(video_assets)")
            }
            if "managed_path" not in video_columns:
                connection.execute(
                    "ALTER TABLE video_assets ADD COLUMN managed_path TEXT NOT NULL DEFAULT ''"
                )
            migrations = {
                "camera_view": "TEXT NOT NULL DEFAULT '未记录'",
                "body_side": "TEXT NOT NULL DEFAULT '未记录'",
                "capture_protocol": "TEXT NOT NULL DEFAULT ''",
                "video_notes": "TEXT NOT NULL DEFAULT ''",
                "quality_rule_version": "TEXT NOT NULL DEFAULT ''",
                "quality_warnings_json": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column, definition in migrations.items():
                if column not in video_columns:
                    connection.execute(
                        f"ALTER TABLE video_assets ADD COLUMN {column} {definition}"
                    )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Patient:
        return Patient(**dict(row))

    def list_patients(self) -> list[Patient]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM patients ORDER BY updated_at DESC, patient_code ASC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get_patient(self, patient_id: int) -> Patient | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def create_patient(self, patient: Patient) -> Patient:
        patient = patient.normalized()
        if not patient.patient_code:
            raise ValueError("患者研究编号不能为空")
        now = self._now()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO patients (
                        patient_code, display_name, sex, date_of_birth, etiology,
                        injury_date, current_diagnosis, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient.patient_code,
                        patient.display_name,
                        patient.sex,
                        patient.date_of_birth,
                        patient.etiology,
                        patient.injury_date,
                        patient.current_diagnosis,
                        patient.notes,
                        now,
                        now,
                    ),
                )
                patient_id = int(cursor.lastrowid)
                self._audit(
                    connection,
                    action="CREATE",
                    entity_type="patient",
                    entity_id=patient_id,
                    summary=f"patient_code={patient.patient_code}",
                )
        except sqlite3.IntegrityError as error:
            raise DuplicatePatientCodeError("患者研究编号已存在") from error
        created = self.get_patient(patient_id)
        assert created is not None
        return created

    def update_patient(self, patient: Patient) -> Patient:
        patient = patient.normalized()
        if patient.id is None:
            raise ValueError("更新患者资料时缺少患者 ID")
        if not patient.patient_code:
            raise ValueError("患者研究编号不能为空")
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE patients SET
                        patient_code = ?, display_name = ?, sex = ?, date_of_birth = ?,
                        etiology = ?, injury_date = ?, current_diagnosis = ?, notes = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        patient.patient_code,
                        patient.display_name,
                        patient.sex,
                        patient.date_of_birth,
                        patient.etiology,
                        patient.injury_date,
                        patient.current_diagnosis,
                        patient.notes,
                        self._now(),
                        patient.id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"患者不存在: {patient.id}")
                self._audit(
                    connection,
                    action="UPDATE",
                    entity_type="patient",
                    entity_id=patient.id,
                    summary=f"patient_code={patient.patient_code}",
                )
        except sqlite3.IntegrityError as error:
            raise DuplicatePatientCodeError("患者研究编号已存在") from error
        updated = self.get_patient(patient.id)
        assert updated is not None
        return updated

    def audit_events(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM audit_events ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]

    def _audit(
        self,
        connection: sqlite3.Connection,
        action: str,
        entity_type: str,
        entity_id: int,
        summary: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (occurred_at, action, entity_type, entity_id, summary)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self._now(), action, entity_type, entity_id, summary),
        )

    def export_patient_dict(self, patient_id: int) -> dict[str, object] | None:
        patient = self.get_patient(patient_id)
        return asdict(patient) if patient else None

    @staticmethod
    def _video_from_row(row: sqlite3.Row) -> VideoAsset:
        return VideoAsset(**dict(row))

    def list_video_assets(self, patient_id: int) -> list[VideoAsset]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM video_assets
                WHERE patient_id = ?
                ORDER BY imported_at DESC, id DESC
                """,
                (patient_id,),
            ).fetchall()
        return [self._video_from_row(row) for row in rows]

    def find_video_by_sha256(self, file_sha256: str) -> VideoAsset | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM video_assets WHERE file_sha256 = ?", (file_sha256,)
            ).fetchone()
        return self._video_from_row(row) if row else None

    def delete_video_asset(self, video_id: int) -> VideoAsset:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM video_assets WHERE id = ?", (video_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"视频登记不存在: {video_id}")
            video = self._video_from_row(row)
            connection.execute("DELETE FROM video_assets WHERE id = ?", (video_id,))
            self._audit(
                connection,
                action="DELETE_VIDEO",
                entity_type="video",
                entity_id=video_id,
                summary=f"filename={video.filename};sha256={video.file_sha256[:12]}",
            )
        return video

    def create_video_asset(self, video: VideoAsset) -> VideoAsset:
        if video.patient_id <= 0:
            raise ValueError("导入视频时缺少有效患者")
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO video_assets (
                        patient_id, source_path, filename, file_sha256, file_size_bytes,
                        extension, duration_seconds, fps, frame_count, width, height,
                        managed_path, camera_view, body_side, capture_protocol, video_notes,
                        quality_rule_version, quality_warnings_json, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        video.patient_id,
                        video.source_path,
                        video.filename,
                        video.file_sha256,
                        video.file_size_bytes,
                        video.extension,
                        video.duration_seconds,
                        video.fps,
                        video.frame_count,
                        video.width,
                        video.height,
                        video.managed_path,
                        video.camera_view,
                        video.body_side,
                        video.capture_protocol,
                        video.video_notes,
                        video.quality_rule_version,
                        video.quality_warnings_json,
                        self._now(),
                    ),
                )
                video_id = int(cursor.lastrowid)
                self._audit(
                    connection,
                    action="IMPORT_VIDEO",
                    entity_type="video",
                    entity_id=video_id,
                    summary=f"filename={video.filename};sha256={video.file_sha256[:12]}",
                )
        except sqlite3.IntegrityError as error:
            message = str(error).lower()
            if "file_sha256" in message or "unique" in message:
                raise DuplicateVideoError("该视频内容已经登记，未重复导入") from error
            if "foreign key" in message:
                raise ValueError("所选患者不存在，无法登记视频") from error
            raise

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM video_assets WHERE id = ?", (video_id,)
            ).fetchone()
        assert row is not None
        return self._video_from_row(row)
