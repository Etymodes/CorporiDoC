from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from corporidoc.data.video_probe import sha256_file


class ModelStorageError(OSError):
    pass


@dataclass(frozen=True, slots=True)
class ManagedModelFile:
    path: Path
    sha256: str
    size_bytes: int


class ManagedModelStore:
    """Copy a MediaPipe task file into content-addressed local storage."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()

    def archive(self, source_path: Path) -> ManagedModelFile:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise ModelStorageError("模型文件不存在或不是普通文件")
        if source.suffix.lower() != ".task":
            raise ModelStorageError("MediaPipe 模型必须是 .task 文件")
        if source.stat().st_size <= 0:
            raise ModelStorageError("模型文件为空")

        sha256 = sha256_file(source)
        model_directory = self.data_root / "models"
        model_directory.mkdir(parents=True, exist_ok=True)
        destination = model_directory / f"{sha256}.task"
        if destination.exists():
            if sha256_file(destination) != sha256:
                raise ModelStorageError("受管模型与文件名哈希不一致，请停止使用并检查存储")
            return ManagedModelFile(destination, sha256, destination.stat().st_size)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{sha256[:12]}-",
            suffix=".partial",
            dir=model_directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != sha256:
                raise ModelStorageError("复制后哈希不一致；模型文件可能在导入期间发生变化")
            os.replace(temporary, destination)
        except ModelStorageError:
            temporary.unlink(missing_ok=True)
            raise
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ModelStorageError(f"无法复制模型到应用目录：{error}") from error
        return ManagedModelFile(destination, sha256, destination.stat().st_size)
