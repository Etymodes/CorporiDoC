from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

QUALITY_RULE_VERSION = "m2b-basic-v1"


class QualityMetadata(Protocol):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class VideoQualityAssessment:
    rule_version: str
    warnings: tuple[str, ...]

    @property
    def summary(self) -> str:
        if not self.warnings:
            return "基础检查通过"
        return f"需复核（{len(self.warnings)}）"

    def to_json(self) -> str:
        return json.dumps(self.warnings, ensure_ascii=False)


def assess_video_quality(metadata: QualityMetadata) -> VideoQualityAssessment:
    """Apply versioned engineering checks, not clinical suitability criteria."""

    warnings: list[str] = []
    if metadata.width < 640 or metadata.height < 480:
        warnings.append("分辨率低于 640×480，细微运动可能难以辨认")
    if metadata.fps <= 0:
        warnings.append("未能读取帧率，时间相关指标需复核")
    elif metadata.fps < 15:
        warnings.append("帧率低于 15 FPS，快速或短暂运动可能漏检")
    if metadata.frame_count <= 0:
        warnings.append("未能读取总帧数")
    if metadata.duration_seconds <= 0:
        warnings.append("未能估算视频时长")

    return VideoQualityAssessment(QUALITY_RULE_VERSION, tuple(warnings))


def decode_quality_warnings(value: str) -> tuple[str, ...]:
    try:
        decoded = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return ("质控记录无法解析",)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        return ("质控记录格式无效",)
    return tuple(decoded)
