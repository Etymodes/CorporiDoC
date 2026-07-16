from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from corporidoc.domain import ModelAsset
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
from corporidoc.pose.model_preflight import check_mediapipe_preflight


MEDIAPIPE_POSE_33 = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

CSV_HEADER = (
    "frame_index",
    "timestamp_ms",
    "time_seconds",
    "pose_index",
    "keypoint_index",
    "keypoint",
    "x_normalized",
    "y_normalized",
    "z_normalized",
    "x_pixels",
    "y_pixels",
    "visibility",
    "presence",
    "x_world_m",
    "y_world_m",
    "z_world_m",
    "detected",
    "source",
)

RuntimeLoader = Callable[[], tuple[object, object, object]]


class MediaPipePoseBackend:
    """Run the official single-person Pose Landmarker in video mode."""

    def __init__(
        self,
        model: ModelAsset,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        package_version: str | None = None,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        for value in (
            min_detection_confidence,
            min_presence_confidence,
            min_tracking_confidence,
        ):
            if not 0 <= value <= 1:
                raise ValueError("MediaPipe 置信度阈值必须位于 0 到 1")
        self.model = model
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._runtime_loader = runtime_loader or _load_mediapipe_runtime
        self.info = BackendInfo(
            name="mediapipe-pose-landmarker",
            version=package_version or _mediapipe_version(),
            model_name=model.name,
            model_version=model.model_version,
            weights_sha256=model.file_sha256,
            keypoint_schema_version="mediapipe-pose-33-v1",
        )

    @property
    def parameters(self) -> dict[str, float | int]:
        return {
            "num_poses": 1,
            "min_pose_detection_confidence": self.min_detection_confidence,
            "min_pose_presence_confidence": self.min_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }

    def analyze(
        self,
        request: InferenceRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> InferenceResult:
        started_at = datetime.now(timezone.utc)
        output_path = (
            request.output_directory / f"{request.request_id}-mediapipe-keypoints.csv"
        )
        partial_path = output_path.with_suffix(".csv.part")
        capture = None
        processed_frames = 0

        try:
            errors = request.validation_errors()
            if errors:
                raise ValueError("；".join(errors))
            if request.backend != self.info:
                raise ValueError("任务指定的后端或模型版本与当前 MediaPipe 后端不一致")
            if set(request.requested_artifacts) != {ArtifactKind.KEYPOINTS}:
                raise ValueError("当前 MediaPipe 后端只生成关键点 CSV")
            if _sha256_file(request.video_path) != request.video_sha256.lower():
                raise ValueError("视频内容与登记的 SHA-256 不一致，已停止推理")
            preflight = check_mediapipe_preflight(self.model)
            if not preflight.ready:
                raise ValueError("；".join(preflight.errors))

            token = cancellation or CancellationToken()
            token.raise_if_cancelled()
            mp, task_python, vision = self._runtime_loader()

            import cv2

            capture = cv2.VideoCapture(str(request.video_path))
            if not capture.isOpened():
                raise ValueError("无法打开待分析视频")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            warnings = list(preflight.warnings)
            if fps <= 0:
                fps = 25.0
                warnings.append("视频未提供有效 FPS；时间戳暂按 25 FPS 生成")

            options = vision.PoseLandmarkerOptions(
                base_options=task_python.BaseOptions(
                    model_asset_path=str(Path(self.model.file_path).resolve())
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            request.output_directory.mkdir(parents=True, exist_ok=True)
            failed_frames = 0
            low_visibility_points = 0
            world_missing_frames = 0
            last_timestamp_ms = -1

            with (
                vision.PoseLandmarker.create_from_options(options) as landmarker,
                partial_path.open("w", encoding="utf-8", newline="") as output,
            ):
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(CSV_HEADER)
                while True:
                    token.raise_if_cancelled()
                    readable, frame = capture.read()
                    if not readable:
                        break
                    height, width = frame.shape[:2]
                    timestamp_ms = max(
                        round(processed_frames * 1000 / fps),
                        last_timestamp_ms + 1,
                    )
                    last_timestamp_ms = timestamp_ms
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect_for_video(image, timestamp_ms)

                    if not result.pose_landmarks:
                        failed_frames += 1
                        _write_missing_pose(writer, processed_frames, timestamp_ms, fps)
                    else:
                        image_landmarks = result.pose_landmarks[0]
                        if len(image_landmarks) != len(MEDIAPIPE_POSE_33):
                            raise ValueError("MediaPipe 返回的关键点数量与 33 点规范不一致")
                        world_landmarks = (
                            result.pose_world_landmarks[0]
                            if result.pose_world_landmarks
                            else ()
                        )
                        if len(world_landmarks) != len(MEDIAPIPE_POSE_33):
                            world_landmarks = ()
                            world_missing_frames += 1
                        low_visibility_points += _write_detected_pose(
                            writer,
                            processed_frames,
                            timestamp_ms,
                            fps,
                            width,
                            height,
                            image_landmarks,
                            world_landmarks,
                        )

                    processed_frames += 1
                    if progress and (
                        processed_frames == total_frames or processed_frames % 10 == 0
                    ):
                        progress(
                            ProgressUpdate(
                                processed_frames,
                                total_frames,
                                "正在运行 MediaPipe 人体姿态",
                            )
                        )

            if processed_frames == 0:
                raise ValueError("视频未解码出任何帧")
            if failed_frames:
                warnings.append(
                    f"{failed_frames}/{processed_frames} 帧未检出人体姿态；CSV 已保留空记录"
                )
            if low_visibility_points:
                warnings.append(f"{low_visibility_points} 个关键点 visibility 低于 0.5")
            if world_missing_frames:
                warnings.append(f"{world_missing_frames} 帧缺少世界坐标")
            if total_frames and processed_frames < total_frames:
                warnings.append(
                    f"视频登记为 {total_frames} 帧，实际只解码 {processed_frames} 帧"
                )

            partial_path.replace(output_path)
            artifact = InferenceArtifact(
                ArtifactKind.KEYPOINTS,
                output_path,
                _sha256_file(output_path),
            )
            if progress and processed_frames != total_frames:
                progress(
                    ProgressUpdate(processed_frames, processed_frames, "MediaPipe 推理完成")
                )
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


def _write_missing_pose(
    writer: object,
    frame_index: int,
    timestamp_ms: int,
    fps: float,
) -> None:
    for keypoint_index, keypoint in enumerate(MEDIAPIPE_POSE_33):
        writer.writerow(
            (
                frame_index,
                timestamp_ms,
                f"{frame_index / fps:.6f}",
                0,
                keypoint_index,
                keypoint,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                0,
                "mediapipe-pose-landmarker",
            )
        )


def _write_detected_pose(
    writer: object,
    frame_index: int,
    timestamp_ms: int,
    fps: float,
    width: int,
    height: int,
    image_landmarks: object,
    world_landmarks: object,
) -> int:
    low_visibility_points = 0
    for keypoint_index, (keypoint, landmark) in enumerate(
        zip(MEDIAPIPE_POSE_33, image_landmarks, strict=True)
    ):
        visibility = float(getattr(landmark, "visibility", 0.0))
        presence = float(getattr(landmark, "presence", 0.0))
        if visibility < 0.5:
            low_visibility_points += 1
        world = world_landmarks[keypoint_index] if world_landmarks else None
        writer.writerow(
            (
                frame_index,
                timestamp_ms,
                f"{frame_index / fps:.6f}",
                0,
                keypoint_index,
                keypoint,
                f"{landmark.x:.8f}",
                f"{landmark.y:.8f}",
                f"{landmark.z:.8f}",
                f"{landmark.x * width:.3f}",
                f"{landmark.y * height:.3f}",
                f"{visibility:.8f}",
                f"{presence:.8f}",
                f"{world.x:.8f}" if world else "",
                f"{world.y:.8f}" if world else "",
                f"{world.z:.8f}" if world else "",
                1,
                "mediapipe-pose-landmarker",
            )
        )
    return low_visibility_points


def _load_mediapipe_runtime() -> tuple[object, object, object]:
    import mediapipe as mp
    from mediapipe.tasks import python as task_python
    from mediapipe.tasks.python import vision

    return mp, task_python, vision


def _mediapipe_version() -> str:
    try:
        return version("mediapipe")
    except PackageNotFoundError as error:
        raise ValueError("soma 环境尚未安装 MediaPipe 人体姿态依赖") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
