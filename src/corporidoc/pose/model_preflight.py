from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from corporidoc.data.video_probe import sha256_file
from corporidoc.domain import ModelAsset


@dataclass(frozen=True, slots=True)
class ModelPreflightResult:
    ready: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def check_mediapipe_preflight(asset: ModelAsset) -> ModelPreflightResult:
    errors: list[str] = []
    path = Path(asset.file_path).expanduser()
    if asset.backend_name != "mediapipe-pose-landmarker":
        errors.append("模型登记的后端不是 MediaPipe Pose Landmarker")
    if not path.is_file():
        errors.append("受管模型文件不存在")
    elif sha256_file(path) != asset.file_sha256.lower():
        errors.append("受管模型 SHA-256 与登记记录不一致")
    if importlib.util.find_spec("mediapipe") is None:
        errors.append("soma 环境尚未安装 MediaPipe 人体姿态依赖")

    warnings = (
        "MediaPipe 人体姿态仅作为工程基线，尚未针对意识障碍患者完成临床验证",
    )
    return ModelPreflightResult(not errors, tuple(errors), warnings)
