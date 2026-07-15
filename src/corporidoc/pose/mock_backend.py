from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

from corporidoc.pose.contracts import (
    ArtifactKind,
    BackendInfo,
    CancellationToken,
    InferenceArtifact,
    InferenceCancelled,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    ProgressCallback,
    ProgressUpdate,
)


class MockPoseBackend:
    """Generate deterministic fake keypoints for testing the task lifecycle."""

    info = BackendInfo(
        name="corporidoc-mock",
        version="1.0",
        model_name="deterministic-demo-points",
        model_version="1",
        keypoint_schema_version="mock-v1",
    )

    def analyze(
        self,
        request: InferenceRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> InferenceResult:
        started_at = datetime.now(timezone.utc)
        output_path = request.output_directory / f"{request.request_id}-mock-keypoints.csv"
        partial_path = output_path.with_suffix(".csv.part")
        capture = None
        processed_frames = 0

        try:
            errors = request.validation_errors()
            if errors:
                raise ValueError("；".join(errors))
            if request.backend != self.info:
                raise ValueError(
                    "任务指定的姿态后端或模型版本与当前 Mock 后端不一致"
                )
            if set(request.requested_artifacts) != {ArtifactKind.KEYPOINTS}:
                raise ValueError("Mock 后端目前只生成关键点 CSV")
            if _sha256_file(request.video_path) != request.video_sha256.lower():
                raise ValueError("视频内容与登记的 SHA-256 不一致，已停止推理")

            token = cancellation or CancellationToken()
            token.raise_if_cancelled()

            import cv2

            capture = cv2.VideoCapture(str(request.video_path))
            if not capture.isOpened():
                raise ValueError("无法打开待分析视频")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            warnings: list[str] = []
            if fps <= 0:
                fps = 25.0
                warnings.append("视频未提供有效 FPS；Mock 时间轴暂按 25 FPS 生成")

            request.output_directory.mkdir(parents=True, exist_ok=True)
            with partial_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(
                    ("frame_index", "time_seconds", "keypoint", "x", "y", "confidence", "source")
                )

                while True:
                    token.raise_if_cancelled()
                    readable, frame = capture.read()
                    if not readable:
                        break
                    height, width = frame.shape[:2]
                    for keypoint, x, y in _mock_points(processed_frames, width, height):
                        writer.writerow(
                            (
                                processed_frames,
                                f"{processed_frames / fps:.6f}",
                                keypoint,
                                f"{x:.3f}",
                                f"{y:.3f}",
                                "1.000",
                                "mock-not-clinical",
                            )
                        )
                    processed_frames += 1
                    if progress and (
                        processed_frames == total_frames or processed_frames % 30 == 0
                    ):
                        progress(
                            ProgressUpdate(
                                processed_frames,
                                total_frames,
                                "正在生成 Mock 关键点",
                            )
                        )

            if processed_frames == 0:
                raise ValueError("视频未解码出任何帧")

            partial_path.replace(output_path)
            artifact = InferenceArtifact(
                ArtifactKind.KEYPOINTS,
                output_path,
                _sha256_file(output_path),
            )
            if progress and processed_frames != total_frames:
                progress(ProgressUpdate(processed_frames, processed_frames, "Mock 推理完成"))
            return InferenceResult(
                request.request_id,
                InferenceStatus.SUCCEEDED,
                started_at,
                datetime.now(timezone.utc),
                processed_frames=processed_frames,
                artifacts=(artifact,),
                warnings=tuple(warnings),
            )
        except InferenceCancelled:
            partial_path.unlink(missing_ok=True)
            return InferenceResult(
                request.request_id,
                InferenceStatus.CANCELLED,
                started_at,
                datetime.now(timezone.utc),
                processed_frames=processed_frames,
            )
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            return InferenceResult(
                request.request_id,
                InferenceStatus.FAILED,
                started_at,
                datetime.now(timezone.utc),
                processed_frames=processed_frames,
                error_message=str(error),
            )
        finally:
            if capture is not None:
                capture.release()


def _mock_points(frame_index: int, width: int, height: int) -> tuple[tuple[str, float, float], ...]:
    phase = 2 * math.pi * (frame_index % 60) / 60
    movement = math.sin(phase) * width * 0.05
    return (
        ("nose", width * 0.5, height * 0.2),
        ("left_wrist", width * 0.3 + movement, height * 0.6),
        ("right_wrist", width * 0.7 - movement, height * 0.6),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
