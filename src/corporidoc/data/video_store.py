from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from corporidoc.data.video_probe import VideoMetadata, sha256_file


class VideoStorageError(OSError):
    pass


class ManagedVideoStore:
    """Copy source videos into patient-scoped, content-addressed storage."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()

    def archive(self, metadata: VideoMetadata, patient_id: int) -> Path:
        if patient_id <= 0:
            raise VideoStorageError("缺少有效患者，无法建立应用副本")
        extension = self._safe_extension(metadata.extension)
        patient_directory = (
            self.data_root / "patients" / f"patient-{patient_id:06d}" / "videos"
        )
        patient_directory.mkdir(parents=True, exist_ok=True)
        destination = patient_directory / f"{metadata.file_sha256}{extension}"

        if destination.exists():
            if sha256_file(destination) != metadata.file_sha256:
                raise VideoStorageError("应用目录中存在同名但内容不一致的文件，请停止导入并检查存储")
            return destination

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{metadata.file_sha256[:12]}-",
            suffix=".partial",
            dir=patient_directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(metadata.source_path, temporary)
            if sha256_file(temporary) != metadata.file_sha256:
                raise VideoStorageError("复制后哈希不一致；源视频可能在导入期间发生变化")
            os.replace(temporary, destination)
        except VideoStorageError:
            temporary.unlink(missing_ok=True)
            raise
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise VideoStorageError(f"无法复制视频到应用目录：{error}") from error
        return destination

    @staticmethod
    def _safe_extension(extension: str) -> str:
        suffix = extension.lower()
        if suffix.startswith(".") and suffix[1:].isalnum() and len(suffix) <= 10:
            return suffix
        return ".video"
