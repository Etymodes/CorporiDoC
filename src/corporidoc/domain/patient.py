from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Patient:
    """De-identified patient metadata used by the research application."""

    patient_code: str
    display_name: str = ""
    sex: str = "未知"
    date_of_birth: str = ""
    etiology: str = ""
    injury_date: str = ""
    current_diagnosis: str = "待评估"
    notes: str = ""
    id: int | None = None
    created_at: str = ""
    updated_at: str = ""

    def normalized(self) -> Patient:
        return Patient(
            id=self.id,
            patient_code=self.patient_code.strip(),
            display_name=self.display_name.strip(),
            sex=self.sex.strip() or "未知",
            date_of_birth=self.date_of_birth.strip(),
            etiology=self.etiology.strip(),
            injury_date=self.injury_date.strip(),
            current_diagnosis=self.current_diagnosis.strip() or "待评估",
            notes=self.notes.strip(),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
