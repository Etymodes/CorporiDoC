from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Protocol, runtime_checkable
from uuid import uuid4


class InferenceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(str, Enum):
    KEYPOINTS = "keypoints"
    LABELED_VIDEO = "labeled_video"
    TRAJECTORY_VIDEO = "trajectory_video"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    version: str
    model_name: str
    model_version: str
    weights_sha256: str = ""
    keypoint_schema_version: str = ""

    def __post_init__(self) -> None:
        required = (self.name, self.version, self.model_name, self.model_version)
        if any(not value.strip() for value in required):
            raise ValueError("后端、代码和模型名称及版本不能为空")
        if self.weights_sha256 and not _is_sha256(self.weights_sha256):
            raise ValueError("模型权重 SHA-256 必须是 64 位十六进制字符串")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    patient_id: int
    video_asset_id: int
    video_path: Path
    video_sha256: str
    output_directory: Path
    backend: BackendInfo
    requested_artifacts: tuple[ArtifactKind, ...] = (ArtifactKind.KEYPOINTS,)
    parameters: Mapping[str, object] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        *,
        patient_id: int,
        video_asset_id: int,
        video_path: Path,
        video_sha256: str,
        output_directory: Path,
        backend: BackendInfo,
        requested_artifacts: tuple[ArtifactKind, ...] = (ArtifactKind.KEYPOINTS,),
        parameters: Mapping[str, object] | None = None,
    ) -> InferenceRequest:
        return cls(
            request_id=uuid4().hex,
            patient_id=patient_id,
            video_asset_id=video_asset_id,
            video_path=video_path,
            video_sha256=video_sha256,
            output_directory=output_directory,
            backend=backend,
            requested_artifacts=requested_artifacts,
            parameters=dict(parameters or {}),
        )

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.request_id.strip():
            errors.append("任务编号不能为空")
        if self.patient_id <= 0:
            errors.append("患者数据库 ID 必须大于 0")
        if self.video_asset_id <= 0:
            errors.append("视频数据库 ID 必须大于 0")
        if not self.video_path.is_file():
            errors.append("待分析视频不存在或不是文件")
        if not _is_sha256(self.video_sha256):
            errors.append("视频 SHA-256 必须是 64 位十六进制字符串")
        if self.output_directory.exists() and not self.output_directory.is_dir():
            errors.append("输出路径已存在但不是目录")
        if not self.requested_artifacts:
            errors.append("至少需要一种输出产物")
        if len(set(self.requested_artifacts)) != len(self.requested_artifacts):
            errors.append("输出产物不能重复")
        if self.requested_at.tzinfo is None:
            errors.append("任务时间必须包含时区")
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    completed_frames: int
    total_frames: int
    message: str = ""

    def __post_init__(self) -> None:
        if self.completed_frames < 0 or self.total_frames < 0:
            raise ValueError("帧数不能为负数")

    @property
    def fraction(self) -> float | None:
        if self.total_frames == 0:
            return None
        return min(self.completed_frames / self.total_frames, 1.0)


@dataclass(frozen=True, slots=True)
class InferenceArtifact:
    kind: ArtifactKind
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.sha256):
            raise ValueError("产物 SHA-256 必须是 64 位十六进制字符串")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    request_id: str
    status: InferenceStatus
    started_at: datetime
    finished_at: datetime
    processed_frames: int = 0
    artifacts: tuple[InferenceArtifact, ...] = ()
    warnings: tuple[str, ...] = ()
    error_message: str = ""

    def __post_init__(self) -> None:
        terminal = {
            InferenceStatus.SUCCEEDED,
            InferenceStatus.FAILED,
            InferenceStatus.CANCELLED,
        }
        if self.status not in terminal:
            raise ValueError("推理结果必须使用终态")
        if self.processed_frames < 0:
            raise ValueError("已处理帧数不能为负数")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("推理开始和结束时间必须包含时区")
        if self.finished_at < self.started_at:
            raise ValueError("结束时间不能早于开始时间")
        if self.status is InferenceStatus.FAILED and not self.error_message.strip():
            raise ValueError("失败结果必须包含错误信息")
        if self.status is InferenceStatus.SUCCEEDED and self.error_message:
            raise ValueError("成功结果不能包含错误信息")


class InferenceCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise InferenceCancelled("姿态推理已取消")


ProgressCallback = Callable[[ProgressUpdate], None]


@runtime_checkable
class PoseBackend(Protocol):
    @property
    def info(self) -> BackendInfo: ...

    def analyze(
        self,
        request: InferenceRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> InferenceResult: ...


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)
