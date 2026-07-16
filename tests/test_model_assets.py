import hashlib
import importlib.util
from pathlib import Path

import pytest

from corporidoc.data import (
    DuplicateModelError,
    ManagedModelStore,
    ModelStorageError,
    PatientRepository,
)
from corporidoc.domain import ModelAsset
from corporidoc.pose import check_mediapipe_preflight


def model_asset(path: Path, sha256: str, size_bytes: int) -> ModelAsset:
    return ModelAsset(
        name="MediaPipe Pose Landmarker Full",
        backend_name="mediapipe-pose-landmarker",
        model_version="full-2023-04-17",
        file_path=str(path),
        file_sha256=sha256,
        file_size_bytes=size_bytes,
        license_name="Apache-2.0",
        source_url="https://storage.googleapis.com/mediapipe-models/example.task",
    )


def test_model_is_copied_and_registered_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / "pose_landmarker_full.task"
    source.write_bytes(b"fake model for storage test")
    data_root = tmp_path / "app-data"
    managed = ManagedModelStore(data_root).archive(source)
    source.unlink()

    expected_hash = hashlib.sha256(b"fake model for storage test").hexdigest()
    assert managed.path == data_root / "models" / f"{expected_hash}.task"
    assert managed.sha256 == expected_hash
    assert managed.path.is_file()

    repository = PatientRepository(data_root / "corporidoc.sqlite3")
    created = repository.create_model_asset(
        model_asset(managed.path, managed.sha256, managed.size_bytes)
    )

    assert repository.list_model_assets() == [created]
    assert created.id is not None
    assert created.imported_at
    assert repository.audit_events()[-1]["action"] == "IMPORT_MODEL"
    with pytest.raises(DuplicateModelError):
        repository.create_model_asset(
            model_asset(managed.path, managed.sha256, managed.size_bytes)
        )


def test_model_store_rejects_empty_or_non_task_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.task"
    empty.touch()
    wrong_extension = tmp_path / "model.bin"
    wrong_extension.write_bytes(b"model")
    store = ManagedModelStore(tmp_path / "app-data")

    with pytest.raises(ModelStorageError, match="为空"):
        store.archive(empty)
    with pytest.raises(ModelStorageError, match=".task"):
        store.archive(wrong_extension)


def test_mediapipe_preflight_checks_package_and_model_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.task"
    path.write_bytes(b"model")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    asset = model_asset(path, sha256, path.stat().st_size)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)

    missing_package = check_mediapipe_preflight(asset)
    assert missing_package.ready is False
    assert "尚未安装 MediaPipe" in missing_package.errors[0]
    assert "临床验证" in missing_package.warnings[0]

    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    path.write_bytes(b"changed")
    changed_model = check_mediapipe_preflight(asset)
    assert changed_model.ready is False
    assert "SHA-256" in changed_model.errors[0]
