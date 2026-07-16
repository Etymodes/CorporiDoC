from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelAsset:
    name: str
    backend_name: str
    model_version: str
    file_path: str
    file_sha256: str
    file_size_bytes: int
    license_name: str
    source_url: str
    imported_at: str = ""
    id: int | None = None
