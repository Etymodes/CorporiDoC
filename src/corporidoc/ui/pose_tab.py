from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from corporidoc.data import PatientRepository, VideoPlaybackError
from corporidoc.domain import ModelAsset, Patient, VideoAsset
from corporidoc.pose import (
    CancellationToken,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    MediaPipePoseBackend,
    MockPoseBackend,
    PoseBackend,
    ProgressUpdate,
    check_mediapipe_preflight,
)
from corporidoc.pose.request_builder import build_inference_request


class PoseWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)

    def __init__(
        self,
        request: InferenceRequest,
        backend: PoseBackend,
        cancellation: CancellationToken,
    ) -> None:
        super().__init__()
        self.request = request
        self.backend = backend
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
        self._models: list[ModelAsset] = []
        self._thread: QThread | None = None
        self._worker: PoseWorker | None = None
        self._cancellation: CancellationToken | None = None

        self.patient_label = QLabel("请先在“患者”页切换当前患者")
        self.patient_label.setObjectName("activePatient")
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(360)
        self.backend_combo = QComboBox()
        self.backend_combo.currentIndexChanged.connect(self._update_notice)
        self.refresh_button = QPushButton("刷新视频")
        self.refresh_button.clicked.connect(self.refresh)
        self.start_button = QPushButton("运行姿态分析")
        self.start_button.clicked.connect(self.start_inference)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("视频"))
        controls.addWidget(self.video_combo, 1)
        controls.addWidget(QLabel("后端"))
        controls.addWidget(self.backend_combo)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)

        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #9a5b00;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("等待选择患者和视频")
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("任务结果与产物路径将在这里显示")

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(
            ["终态", "视频 ID", "后端/模型", "帧数", "开始时间", "产物数"]
        )
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 30, 32, 30)
        layout.addWidget(self.patient_label)
        layout.addLayout(controls)
        layout.addWidget(self.notice)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_text, 1)
        layout.addWidget(QLabel("当前患者的近期任务"))
        layout.addWidget(self.history_table, 1)
        self._refresh_backends()
        self._update_notice()
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
        self._refresh_backends()
        self.video_combo.clear()
        if self.active_patient is None or self.active_patient.id is None:
            self._videos = []
            self.history_table.setRowCount(0)
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
        self.status_label.setText("请选择视频和姿态后端" if self._videos else "当前患者无视频")
        self._refresh_history()

    def start_inference(self) -> None:
        if self.is_running:
            return
        row = self.video_combo.currentIndex()
        if not 0 <= row < len(self._videos):
            QMessageBox.information(self, "未选择视频", "请先选择一条视频登记记录。")
            return

        try:
            backend = self._selected_backend()
            request = build_inference_request(
                self._videos[row],
                self.repository.database_path.parent,
                backend.info,
                parameters=getattr(backend, "parameters", None),
            )
        except (VideoPlaybackError, ValueError) as error:
            QMessageBox.warning(self, "无法创建姿态任务", str(error))
            return

        try:
            self.repository.create_inference_run(request)
        except Exception as error:
            QMessageBox.warning(self, "无法登记姿态任务", str(error))
            return
        self._refresh_history()

        thread = QThread(self)
        cancellation = CancellationToken()
        worker = PoseWorker(request, backend, cancellation)
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
        self.backend_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.result_text.clear()
        self.status_label.setText(f"姿态任务运行中 · {backend.info.name} · {request.request_id}")
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
        try:
            self.repository.finish_inference_run(result)
        except Exception as persistence_error:
            result = self._record_persistence_failure(result, persistence_error)
        if result.status is InferenceStatus.SUCCEEDED:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            artifact_lines = [
                f"- {artifact.kind.value}: {artifact.path}\n  SHA-256: {artifact.sha256}"
                for artifact in result.artifacts
            ]
            backend_label = (
                "Mock 软件测试任务完成"
                if self._worker and self._worker.backend.info.name == "corporidoc-mock"
                else "MediaPipe 任务完成（必须人工复核）"
            )
            warning_lines = [f"- {warning}" for warning in result.warnings]
            self.status_label.setText(backend_label)
            self.result_text.setPlainText(
                f"任务：{result.request_id}\n"
                f"终态：{result.status.value}\n"
                f"处理帧数：{result.processed_frames}\n"
                "警告：\n"
                + ("\n".join(warning_lines) if warning_lines else "- 无")
                + "\n"
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
            QMessageBox.warning(self, "姿态任务失败", result.error_message)

    def _record_persistence_failure(
        self,
        result: InferenceResult,
        persistence_error: Exception,
    ) -> InferenceResult:
        message = f"任务结果未能安全写入数据库：{persistence_error}"
        failed = InferenceResult(
            result.request_id,
            InferenceStatus.FAILED,
            result.started_at,
            result.finished_at,
            processed_frames=result.processed_frames,
            warnings=result.warnings,
            error_message=message,
        )
        try:
            self.repository.finish_inference_run(failed)
        except Exception as recovery_error:
            return InferenceResult(
                result.request_id,
                InferenceStatus.FAILED,
                result.started_at,
                result.finished_at,
                processed_frames=result.processed_frames,
                warnings=result.warnings,
                error_message=f"{message}；失败终态也未能落库：{recovery_error}",
            )
        return failed

    def _thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        self._cancellation = None
        if self.active_patient is not None:
            self.patient_label.setText(f"当前患者：{self.active_patient.patient_code}")
        self.refresh()
        self.task_idle.emit()

    def _refresh_history(self) -> None:
        if self.active_patient is None or self.active_patient.id is None:
            self.history_table.setRowCount(0)
            return
        runs = self.repository.list_inference_runs(self.active_patient.id)
        self.history_table.setRowCount(len(runs))
        status_labels = {
            "running": "运行中",
            "succeeded": "成功",
            "failed": "失败",
            "cancelled": "已取消",
        }
        for row, run in enumerate(runs):
            values = (
                status_labels.get(run.status, run.status),
                str(run.video_asset_id),
                f"{run.backend_name}/{run.model_name}",
                str(run.processed_frames),
                run.started_at,
                str(len(run.artifacts)),
            )
            for column, value in enumerate(values):
                self.history_table.setItem(row, column, QTableWidgetItem(value))

    def _refresh_backends(self) -> None:
        selected_model_id = self.backend_combo.currentData()
        self._models = self.repository.list_model_assets()
        self.backend_combo.clear()
        self.backend_combo.addItem("Mock（软件测试）", None)
        for model in self._models:
            self.backend_combo.addItem(f"{model.name} · {model.model_version}", model.id)
        if selected_model_id is not None:
            index = self.backend_combo.findData(selected_model_id)
            if index >= 0:
                self.backend_combo.setCurrentIndex(index)

    def _selected_backend(self) -> PoseBackend:
        model_id = self.backend_combo.currentData()
        if model_id is None:
            return MockPoseBackend()
        model = next((item for item in self._models if item.id == model_id), None)
        if model is None:
            raise ValueError("所选模型登记已不存在，请刷新后重试")
        preflight = check_mediapipe_preflight(model)
        if not preflight.ready:
            raise ValueError("；".join(preflight.errors))
        return MediaPipePoseBackend(model)

    def _update_notice(self) -> None:
        if self.backend_combo.currentData() is None:
            self.notice.setText(
                "Mock 输出是软件测试用虚拟轨迹，不代表患者真实姿态，不得进入临床判断。"
            )
            return
        self.notice.setText(
            "MediaPipe 是通用单人人体姿态工程基线，并非 DoC 临床模型。"
            "未检出、低可见度和遮挡结果必须结合源视频人工复核。"
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.video_combo.setEnabled(enabled)
        self.backend_combo.setEnabled(enabled)
        self.refresh_button.setEnabled(self.active_patient is not None)
        self.start_button.setEnabled(enabled)
        self.cancel_button.setEnabled(False)
