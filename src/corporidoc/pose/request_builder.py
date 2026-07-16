from collections.abc import Mapping
from pathlib import Path

from corporidoc.data import resolve_video_playback_source
from corporidoc.domain import VideoAsset
from corporidoc.pose.contracts import ArtifactKind, BackendInfo, InferenceRequest


def build_inference_request(
    video: VideoAsset,
    data_directory: Path,
    backend: BackendInfo,
    requested_artifacts: tuple[ArtifactKind, ...] = (ArtifactKind.KEYPOINTS,),
    parameters: Mapping[str, object] | None = None,
) -> InferenceRequest:
    if video.id is None:
        raise ValueError("视频登记缺少数据库 ID")
    if video.patient_id <= 0:
        raise ValueError("视频登记缺少有效患者 ID")

    playback_source = resolve_video_playback_source(video)
    output_directory = (
        Path(data_directory).expanduser().resolve()
        / "patients"
        / f"patient-{video.patient_id:06d}"
        / "inference"
        / f"video-{video.id:06d}"
    )
    return InferenceRequest.create(
        patient_id=video.patient_id,
        video_asset_id=video.id,
        video_path=playback_source.path,
        video_sha256=video.file_sha256,
        output_directory=output_directory,
        backend=backend,
        requested_artifacts=requested_artifacts,
        parameters=parameters,
    )
