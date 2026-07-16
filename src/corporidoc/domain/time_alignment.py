from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite, sqrt


class SyncMethod(str, Enum):
    TTL = "ttl"
    PHOTODIODE = "photodiode"
    AUDIO_VISUAL = "audio_visual"
    LSL_MARKER = "lsl_marker"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class SyncAnchor:
    video_seconds: float
    eeg_seconds: float
    method: SyncMethod
    label: str

    def __post_init__(self) -> None:
        if not isfinite(self.video_seconds) or self.video_seconds < 0:
            raise ValueError("视频时间必须是非负有限秒数")
        if not isfinite(self.eeg_seconds) or self.eeg_seconds < 0:
            raise ValueError("EEG 时间必须是非负有限秒数")
        if not self.label.strip():
            raise ValueError("同步锚点标签不能为空")


@dataclass(frozen=True, slots=True)
class TimeAlignment:
    slope: float
    offset_seconds: float
    rms_error_seconds: float
    max_error_seconds: float
    anchor_count: int
    method_version: str = "affine-v1"

    def __post_init__(self) -> None:
        values = (
            self.slope,
            self.offset_seconds,
            self.rms_error_seconds,
            self.max_error_seconds,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("时间对齐参数必须是有限数值")
        if self.slope <= 0:
            raise ValueError("时间对齐斜率必须大于零")
        if self.rms_error_seconds < 0 or self.max_error_seconds < 0:
            raise ValueError("时间对齐误差不能为负数")
        if self.anchor_count < 2:
            raise ValueError("时间对齐至少需要两个锚点")

    def video_to_eeg(self, video_seconds: float) -> float:
        if not isfinite(video_seconds):
            raise ValueError("视频时间必须是有限数值")
        return self.slope * video_seconds + self.offset_seconds

    def eeg_to_video(self, eeg_seconds: float) -> float:
        if not isfinite(eeg_seconds):
            raise ValueError("EEG 时间必须是有限数值")
        return (eeg_seconds - self.offset_seconds) / self.slope


def fit_time_alignment(anchors: tuple[SyncAnchor, ...]) -> TimeAlignment:
    if len(anchors) < 2:
        raise ValueError("时间对齐至少需要两个同步锚点")

    ordered = tuple(sorted(anchors, key=lambda anchor: anchor.video_seconds))
    for previous, current in zip(ordered[:-1], ordered[1:], strict=True):
        if current.video_seconds <= previous.video_seconds:
            raise ValueError("视频同步锚点时间必须严格递增")
        if current.eeg_seconds <= previous.eeg_seconds:
            raise ValueError("EEG 同步锚点时间必须严格递增")

    video_mean = fsum(anchor.video_seconds for anchor in ordered) / len(ordered)
    eeg_mean = fsum(anchor.eeg_seconds for anchor in ordered) / len(ordered)
    denominator = fsum((anchor.video_seconds - video_mean) ** 2 for anchor in ordered)
    if denominator == 0:
        raise ValueError("视频同步锚点不能全部位于同一时间")

    numerator = fsum(
        (anchor.video_seconds - video_mean) * (anchor.eeg_seconds - eeg_mean)
        for anchor in ordered
    )
    slope = numerator / denominator
    offset = eeg_mean - slope * video_mean
    residuals = tuple(
        anchor.eeg_seconds - (slope * anchor.video_seconds + offset) for anchor in ordered
    )
    rms_error = sqrt(fsum(residual**2 for residual in residuals) / len(residuals))

    return TimeAlignment(
        slope=slope,
        offset_seconds=offset,
        rms_error_seconds=rms_error,
        max_error_seconds=max(abs(residual) for residual in residuals),
        anchor_count=len(ordered),
    )
