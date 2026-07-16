from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from corporidoc.domain import (
    InferenceArtifactRecord,
    InferenceRunRecord,
    ModelAsset,
    Patient,
    VideoAsset,
)
from corporidoc.pose import InferenceRequest, InferenceResult, InferenceStatus


class DuplicatePatientCodeError(ValueError):
    pass


class DuplicateVideoError(ValueError):
    pass


class DuplicateModelError(ValueError):
    pass


class PatientRepository:
    """Small SQLite repository with an append-only audit trail."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()
        self.recover_interrupted_inference_runs()

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

                CREATE TABLE IF NOT EXISTS inference_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE RESTRICT,
                    video_asset_id INTEGER NOT NULL REFERENCES video_assets(id) ON DELETE RESTRICT,
                    status TEXT NOT NULL,
                    backend_name TEXT NOT NULL,
                    backend_version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    weights_sha256 TEXT NOT NULL DEFAULT '',
                    keypoint_schema_version TEXT NOT NULL DEFAULT '',
                    video_sha256 TEXT NOT NULL,
                    requested_artifacts_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    processed_frames INTEGER NOT NULL DEFAULT 0,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    error_message TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS inference_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_run_id INTEGER NOT NULL
                        REFERENCES inference_runs(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    UNIQUE(inference_run_id, kind, path)
                );

                CREATE TABLE IF NOT EXISTS model_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    backend_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_sha256 TEXT NOT NULL UNIQUE,
                    file_size_bytes INTEGER NOT NULL,
                    license_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
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

    def get_video_asset(self, video_id: int) -> VideoAsset | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM video_assets WHERE id = ?", (video_id,)
            ).fetchone()
        return self._video_from_row(row) if row else None

    def update_video_metadata(
        self,
        video_id: int,
        *,
        camera_view: str,
        body_side: str,
        capture_protocol: str,
        video_notes: str,
    ) -> VideoAsset:
        current = self.get_video_asset(video_id)
        if current is None:
            raise KeyError(f"视频登记不存在: {video_id}")

        values = {
            "camera_view": camera_view.strip() or "未记录",
            "body_side": body_side.strip() or "未记录",
            "capture_protocol": capture_protocol.strip(),
            "video_notes": video_notes.strip(),
        }
        changed_fields = [
            field for field, value in values.items() if getattr(current, field) != value
        ]
        if not changed_fields:
            return current

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE video_assets SET
                    camera_view = ?, body_side = ?, capture_protocol = ?, video_notes = ?
                WHERE id = ?
                """,
                (
                    values["camera_view"],
                    values["body_side"],
                    values["capture_protocol"],
                    values["video_notes"],
                    video_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"视频登记不存在: {video_id}")
            self._audit(
                connection,
                action="UPDATE_VIDEO_METADATA",
                entity_type="video",
                entity_id=video_id,
                summary=f"filename={current.filename};fields={','.join(changed_fields)}",
            )

        updated = self.get_video_asset(video_id)
        assert updated is not None
        return updated

    def delete_video_asset(self, video_id: int) -> VideoAsset:
        try:
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
        except sqlite3.IntegrityError as error:
            raise ValueError("视频已有推理运行记录，为保留证据链不能删除") from error
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

    @staticmethod
    def _model_from_row(row: sqlite3.Row) -> ModelAsset:
        return ModelAsset(**dict(row))

    def list_model_assets(self) -> list[ModelAsset]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM model_assets ORDER BY imported_at DESC, id DESC"
            ).fetchall()
        return [self._model_from_row(row) for row in rows]

    def create_model_asset(self, model: ModelAsset) -> ModelAsset:
        required = {
            "模型名称": model.name,
            "后端名称": model.backend_name,
            "模型版本": model.model_version,
            "许可证": model.license_name,
            "来源网址": model.source_url,
        }
        missing = [label for label, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"模型登记缺少：{','.join(missing)}")
        if len(model.file_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in model.file_sha256
        ):
            raise ValueError("模型 SHA-256 格式无效")

        path = Path(model.file_path).expanduser().resolve()
        try:
            path.relative_to(self.database_path.parent.resolve())
        except ValueError as error:
            raise ValueError("模型文件必须位于应用数据目录内") from error
        if not path.is_file():
            raise ValueError("受管模型文件不存在或不是普通文件")
        if path.stat().st_size != model.file_size_bytes:
            raise ValueError("受管模型大小与登记记录不一致")
        sha256 = model.file_sha256.lower()
        if self._sha256_file(path) != sha256:
            raise ValueError("受管模型 SHA-256 校验失败")

        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO model_assets (
                        name, backend_name, model_version, file_path, file_sha256,
                        file_size_bytes, license_name, source_url, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model.name.strip(),
                        model.backend_name.strip(),
                        model.model_version.strip(),
                        str(path),
                        sha256,
                        model.file_size_bytes,
                        model.license_name.strip(),
                        model.source_url.strip(),
                        self._now(),
                    ),
                )
                model_id = int(cursor.lastrowid)
                self._audit(
                    connection,
                    action="IMPORT_MODEL",
                    entity_type="model_asset",
                    entity_id=model_id,
                    summary=f"backend={model.backend_name};sha256={sha256[:12]}",
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateModelError("该模型文件已经登记，未重复导入") from error

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM model_assets WHERE id = ?", (model_id,)
            ).fetchone()
        assert row is not None
        return self._model_from_row(row)

    def create_inference_run(self, request: InferenceRequest) -> InferenceRunRecord:
        errors = request.validation_errors()
        if errors:
            raise ValueError("；".join(errors))
        try:
            parameters_json = json.dumps(
                request.parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except TypeError as error:
            raise ValueError("推理参数必须可以保存为 JSON") from error
        requested_artifacts_json = json.dumps(
            [artifact.value for artifact in request.requested_artifacts],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._connection() as connection:
            video = connection.execute(
                "SELECT patient_id, file_sha256 FROM video_assets WHERE id = ?",
                (request.video_asset_id,),
            ).fetchone()
            if video is None:
                raise ValueError("姿态任务引用的视频登记不存在")
            if video["patient_id"] != request.patient_id:
                raise ValueError("姿态任务患者与视频登记不一致")
            if video["file_sha256"].lower() != request.video_sha256.lower():
                raise ValueError("姿态任务视频哈希与登记记录不一致")

            cursor = connection.execute(
                """
                INSERT INTO inference_runs (
                    request_id, patient_id, video_asset_id, status,
                    backend_name, backend_version, model_name, model_version,
                    weights_sha256, keypoint_schema_version, video_sha256,
                    requested_artifacts_json, parameters_json, requested_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.patient_id,
                    request.video_asset_id,
                    InferenceStatus.RUNNING.value,
                    request.backend.name,
                    request.backend.version,
                    request.backend.model_name,
                    request.backend.model_version,
                    request.backend.weights_sha256,
                    request.backend.keypoint_schema_version,
                    request.video_sha256.lower(),
                    requested_artifacts_json,
                    parameters_json,
                    request.requested_at.isoformat(),
                    self._now(),
                ),
            )
            run_id = int(cursor.lastrowid)
            self._audit(
                connection,
                action="START_INFERENCE",
                entity_type="inference_run",
                entity_id=run_id,
                summary=f"backend={request.backend.name};video_id={request.video_asset_id}",
            )
        run = self.get_inference_run(request.request_id)
        assert run is not None
        return run

    def finish_inference_run(self, result: InferenceResult) -> InferenceRunRecord:
        if result.status not in {
            InferenceStatus.SUCCEEDED,
            InferenceStatus.FAILED,
            InferenceStatus.CANCELLED,
        }:
            raise ValueError("只能保存推理终态")
        if result.status is not InferenceStatus.SUCCEEDED and result.artifacts:
            raise ValueError("失败或取消的任务不能登记完成产物")

        artifact_rows: list[tuple[str, str, str]] = []
        for artifact in result.artifacts:
            path = artifact.path.expanduser().resolve()
            try:
                path.relative_to(self.database_path.parent.resolve())
            except ValueError as error:
                raise ValueError("推理产物必须位于应用数据目录内") from error
            if not path.is_file():
                raise ValueError("推理产物不存在或不是文件")
            if self._sha256_file(path) != artifact.sha256.lower():
                raise ValueError("推理产物 SHA-256 校验失败")
            artifact_rows.append((artifact.kind.value, str(path), artifact.sha256.lower()))

        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, status FROM inference_runs WHERE request_id = ?",
                (result.request_id,),
            ).fetchone()
            if row is None:
                raise ValueError("推理运行记录不存在")
            if row["status"] != InferenceStatus.RUNNING.value:
                raise ValueError("推理运行记录已经处于终态")
            run_id = int(row["id"])
            connection.execute(
                """
                UPDATE inference_runs SET
                    status = ?, started_at = ?, finished_at = ?, processed_frames = ?,
                    warnings_json = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    result.status.value,
                    result.started_at.isoformat(),
                    result.finished_at.isoformat(),
                    result.processed_frames,
                    json.dumps(result.warnings, ensure_ascii=False, separators=(",", ":")),
                    result.error_message,
                    run_id,
                ),
            )
            connection.executemany(
                """
                INSERT INTO inference_artifacts (inference_run_id, kind, path, sha256)
                VALUES (?, ?, ?, ?)
                """,
                [(run_id, kind, path, sha256) for kind, path, sha256 in artifact_rows],
            )
            self._audit(
                connection,
                action="FINISH_INFERENCE",
                entity_type="inference_run",
                entity_id=run_id,
                summary=f"status={result.status.value};artifacts={len(artifact_rows)}",
            )
        run = self.get_inference_run(result.request_id)
        assert run is not None
        return run

    def get_inference_run(self, request_id: str) -> InferenceRunRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM inference_runs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            artifacts = connection.execute(
                """
                SELECT kind, path, sha256 FROM inference_artifacts
                WHERE inference_run_id = ? ORDER BY id
                """,
                (row["id"],),
            ).fetchall()
        return self._inference_run_from_rows(row, artifacts)

    def list_inference_runs(self, patient_id: int) -> list[InferenceRunRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM inference_runs
                WHERE patient_id = ? ORDER BY id DESC
                """,
                (patient_id,),
            ).fetchall()
            runs: list[InferenceRunRecord] = []
            for row in rows:
                artifacts = connection.execute(
                    """
                    SELECT kind, path, sha256 FROM inference_artifacts
                    WHERE inference_run_id = ? ORDER BY id
                    """,
                    (row["id"],),
                ).fetchall()
                runs.append(self._inference_run_from_rows(row, artifacts))
        return runs

    def recover_interrupted_inference_runs(self) -> int:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM inference_runs WHERE status = ? ORDER BY id",
                (InferenceStatus.RUNNING.value,),
            ).fetchall()
            if not rows:
                return 0
            finished_at = self._now()
            connection.execute(
                """
                UPDATE inference_runs SET
                    status = ?, finished_at = ?,
                    error_message = '应用上次退出前任务未完成'
                WHERE status = ?
                """,
                (
                    InferenceStatus.FAILED.value,
                    finished_at,
                    InferenceStatus.RUNNING.value,
                ),
            )
            for row in rows:
                self._audit(
                    connection,
                    action="RECOVER_INFERENCE",
                    entity_type="inference_run",
                    entity_id=int(row["id"]),
                    summary="status=failed;reason=interrupted",
                )
        return len(rows)

    @staticmethod
    def _inference_run_from_rows(
        row: sqlite3.Row,
        artifacts: list[sqlite3.Row],
    ) -> InferenceRunRecord:
        values = dict(row)
        values["artifacts"] = tuple(
            InferenceArtifactRecord(**dict(artifact)) for artifact in artifacts
        )
        return InferenceRunRecord(**values)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
