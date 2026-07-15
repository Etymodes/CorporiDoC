from pathlib import Path

import pytest

from corporidoc.data import DuplicatePatientCodeError, PatientRepository
from corporidoc.domain import Patient


def repository(tmp_path: Path) -> PatientRepository:
    return PatientRepository(tmp_path / "corporidoc.sqlite3")


def test_create_list_and_update_patient(tmp_path: Path) -> None:
    patients = repository(tmp_path)
    created = patients.create_patient(
        Patient(patient_code="DEMO-001", display_name="示例患者", etiology="TBI")
    )

    assert created.id is not None
    assert patients.list_patients()[0].patient_code == "DEMO-001"

    created.current_diagnosis = "MCS-"
    updated = patients.update_patient(created)
    assert updated.current_diagnosis == "MCS-"
    assert [event["action"] for event in patients.audit_events()] == ["CREATE", "UPDATE"]


def test_patient_code_must_be_unique(tmp_path: Path) -> None:
    patients = repository(tmp_path)
    patients.create_patient(Patient(patient_code="DEMO-001"))

    with pytest.raises(DuplicatePatientCodeError):
        patients.create_patient(Patient(patient_code="DEMO-001"))


def test_patient_code_is_required(tmp_path: Path) -> None:
    patients = repository(tmp_path)

    with pytest.raises(ValueError, match="不能为空"):
        patients.create_patient(Patient(patient_code="  "))
