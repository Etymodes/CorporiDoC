from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from corporidoc.domain import Patient


class DuplicatePatientCodeError(ValueError):
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
                """
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
                self._audit(connection, "CREATE", patient_id, patient.patient_code)
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
                self._audit(connection, "UPDATE", patient.id, patient.patient_code)
        except sqlite3.IntegrityError as error:
            raise DuplicatePatientCodeError("患者研究编号已存在") from error
        updated = self.get_patient(patient.id)
        assert updated is not None
        return updated

    def audit_events(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def _audit(
        self,
        connection: sqlite3.Connection,
        action: str,
        patient_id: int,
        patient_code: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (occurred_at, action, entity_type, entity_id, summary)
            VALUES (?, ?, 'patient', ?, ?)
            """,
            (self._now(), action, patient_id, f"patient_code={patient_code}"),
        )

    def export_patient_dict(self, patient_id: int) -> dict[str, object] | None:
        patient = self.get_patient(patient_id)
        return asdict(patient) if patient else None

