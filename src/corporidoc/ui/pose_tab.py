from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from corporidoc.data import PatientRepository, VideoPlaybackError
from corporidoc.domain import Patient, VideoAsset
from corporidoc.pose import (
    CancellationToken,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    MockPoseBackend,
    ProgressUpdate,
)
from corporidoc.pose.request_builder import build_inference_request


class PoseWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)

    def __init__(self, request: InferenceRequest, cancellation: CancellationToken) -> None:
        super().__init__()
        self.request = request
        self.backend = MockPoseBackend()
        self.cancellation = cancellation

    def run(self) -> None:
        result = self.backend.analyze(
            self.request,
            progress=self.progress.emit,
            cancellation=self.cancellation,
        )
        self.finished.emit(result)


class PoseTab(QWidget):
    task_idle = Signal()

    def __init__(self, repository: PatientRepository) -> None:
        super().__init__()
        self.repository = repository
        self.active_patient: Patient | None = None
        self._videos: list[VideoAsset] = []
        self._thread: QThread | None = None
        self._worker: PoseWorker | None = None
        self._cancellation: CancellationToken | None = None

        self.patient_label = QLabel("请先在“患者”页切换当前患者")
        self.patient_label.setObjectName("activePatient")
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(360)
        self.refresh_button = QPushButton("刷新视频")
        self.refresh_button.clicked.connect(self.refresh)
        self.start_button = QPushButton("运行 Mock 姿态 Demo")
        self.start_button.clicked.connect(self.start_mock_inference)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("视频"))
        controls.addWidget(self.video_combo, 1)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)

        notice = QLabel(
            "当前仅运行确定性 Mock：输出是软件测试用虚拟轨迹，不代表患者真实姿态，"
            "不得进入临床判断或报告。"
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #9a5b00;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("等待选择患者和视频")
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("任务结果与产物路径将在这里显示")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 30, 32, 30)
        layout.addWidget(self.patient_label)
        layout.addLayout(controls)
        layout.addWidget(notice)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_text, 1)
        self._set_controls_enabled(False)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def set_active_patient(self, patient: Patient) -> None:
        self.active_patient = patient
        if self.is_running:
            self.patient_label.setText(
                f"当前患者已切换为 {patient.patient_code}；运行中的任务仍使用原视频"
            )
            return
        self.patient_label.setText(f"当前患者：{patient.patient_code}")
        self.refresh()

    def refresh(self) -> None:
        if self.is_running:
            return
        self.video_combo.clear()
        if self.active_patient is None or self.active_patient.id is None:
            self._videos = []
            self._set_controls_enabled(False)
            return

        self._videos = self.repository.list_video_assets(self.active_patient.id)
        for video in self._videos:
            summary = (
                f"{video.filename} · {video.camera_view}/{video.body_side} · "
                f"{video.file_sha256[:12]}"
            )
            self.video_combo.addItem(
                summary
            )
        self._set_controls_enabled(bool(self._videos))
        self.status_label.setText("请选择视频并运行 Mock 姿态 Demo" if self._videos else "当前患者无视频")

    def start_mock_inference(self) -> None:
        if self.is_running:
            return
        row = self.video_combo.currentIndex()
        if not 0 <= row < len(self._videos):
            QMessageBox.information(self, "未选择视频", "请先选择一条视频登记记录。")
            return

        backend = MockPoseBackend()
        try:
            request = build_inference_request(
                self._videos[row],
                self.repository.database_path.parent,
                backend.info,
            )
        except (VideoPlaybackError, ValueError) as error:
            QMessageBox.warning(self, "无法创建姿态任务", str(error))
            return

        thread = QThread(self)
        cancellation = CancellationToken()
        worker = PoseWorker(request, cancellation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_progress)
        worker.finished.connect(self._handle_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self._cancellation = cancellation

        self.video_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.result_text.clear()
        self.status_label.setText(f"Mock 任务运行中 · {request.request_id}")
        thread.start()

    def cancel_task(self) -> None:
        if self._cancellation is None:
            return
        self._cancellation.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("正在取消；当前帧处理结束后停止")

    def _update_progress(self, update: ProgressUpdate) -> None:
        if update.fraction is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(round(update.fraction * 100))
        self.status_label.setText(
            f"{update.message} · {update.completed_frames}/{update.total_frames or '?'} 帧"
        )

    def _handle_result(self, result: InferenceResult) -> None:
        self.cancel_button.setEnabled(False)
        if result.status is InferenceStatus.SUCCEEDED:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            artifact_lines = [
                f"- {artifact.kind.value}: {artifact.path}\n  SHA-256: {artifact.sha256}"
                for artifact in result.artifacts
            ]
            self.status_label.setText("Mock 任务成功（非临床结果）")
            self.result_text.setPlainText(
                f"任务：{result.request_id}\n"
                f"终态：{result.status.value}\n"
                f"处理帧数：{result.processed_frames}\n"
                "产物：\n"
                + "\n".join(artifact_lines)
            )
        elif result.status is InferenceStatus.CANCELLED:
            self.progress_bar.setRange(0, 100)
            self.status_label.setText("任务已取消；没有完成产物")
            self.result_text.setPlainText(f"任务：{result.request_id}\n终态：cancelled")
        else:
            self.progress_bar.setRange(0, 100)
            self.status_label.setText("任务失败；没有完成产物")
            self.result_text.setPlainText(
                f"任务：{result.request_id}\n终态：failed\n错误：{result.error_message}"
            )
            QMessageBox.warning(self, "Mock 姿态任务失败", result.error_message)

    def _thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        self._cancellation = None
        if self.active_patient is not None:
            self.patient_label.setText(f"当前患者：{self.active_patient.patient_code}")
        self.refresh()
        self.task_idle.emit()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.video_combo.setEnabled(enabled)
        self.refresh_button.setEnabled(self.active_patient is not None)
        self.start_button.setEnabled(enabled)
        self.cancel_button.setEnabled(False)
