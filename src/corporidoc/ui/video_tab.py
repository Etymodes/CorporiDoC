from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from corporidoc.data import (
    DuplicateVideoError,
    ManagedVideoStore,
    PatientRepository,
    VideoProbe,
    VideoProbeError,
    VideoStorageError,
)
from corporidoc.domain import Patient, VideoAsset


class VideoTab(QWidget):
    def __init__(self, repository: PatientRepository) -> None:
        super().__init__()
        self.repository = repository
        self.probe = VideoProbe()
        self.video_store = ManagedVideoStore(repository.database_path.parent)
        self.active_patient: Patient | None = None
        self._videos: list[VideoAsset] = []

        self.patient_label = QLabel("请先在“患者”页切换当前患者")
        self.patient_label.setObjectName("activePatient")
        self.import_button = QPushButton("导入视频副本")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_video)
        self.delete_button = QPushButton("删除所选登记")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_selected_video)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh)

        controls = QHBoxLayout()
        controls.addWidget(self.patient_label)
        controls.addStretch()
        controls.addWidget(self.import_button)
        controls.addWidget(self.delete_button)
        controls.addWidget(refresh_button)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "文件",
                "时长",
                "分辨率",
                "FPS",
                "帧数",
                "大小",
                "SHA-256",
                "应用副本",
                "原路径",
            ]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._update_delete_button)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        notice = QLabel(
            "导入时会在对应患者的应用目录中建立经 SHA-256 校验的副本。"
            "原文件不会被修改；原路径移动后仍使用应用副本。"
        )
        notice.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(notice)
        layout.addWidget(self.table)

    def set_active_patient(self, patient: Patient) -> None:
        self.active_patient = patient
        self.patient_label.setText(f"当前患者：{patient.patient_code}")
        self.import_button.setEnabled(True)
        self.refresh()

    def refresh(self) -> None:
        if self.active_patient is None or self.active_patient.id is None:
            self._videos = []
            self.table.setRowCount(0)
            self._update_delete_button()
            return
        videos = self.repository.list_video_assets(self.active_patient.id)
        self._videos = videos
        self.table.setRowCount(len(videos))
        for row, video in enumerate(videos):
            values = [
                video.filename,
                self._duration(video.duration_seconds),
                f"{video.width}×{video.height}",
                f"{video.fps:.2f}" if video.fps else "未知",
                str(video.frame_count) if video.frame_count else "未知",
                self._file_size(video.file_size_bytes),
                video.file_sha256[:12],
                "可用" if video.managed_path and Path(video.managed_path).is_file() else "缺失",
                "可用" if Path(video.source_path).is_file() else "缺失",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (7, 8) and value == "缺失":
                    item.setForeground(Qt.red)
                self.table.setItem(row, column, item)
        self._update_delete_button()

    def _update_delete_button(self) -> None:
        row = self.table.currentRow()
        self.delete_button.setEnabled(0 <= row < len(self._videos))

    def delete_selected_video(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._videos):
            QMessageBox.information(self, "未选择视频", "请先选择一条视频登记记录。")
            return
        video = self._videos[row]
        if video.id is None:
            QMessageBox.warning(self, "无法删除", "视频登记缺少数据库 ID。")
            return

        answer = QMessageBox.question(
            self,
            "确认删除视频登记",
            f"确定删除“{video.filename}”的登记记录和应用副本吗？\n\n"
            "用户原始视频不会被删除；删除操作会保留审计记录。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            deleted = self.repository.delete_video_asset(video.id)
        except (KeyError, ValueError) as error:
            QMessageBox.warning(self, "无法删除", str(error))
            return

        cleanup_error: VideoStorageError | None = None
        try:
            self.video_store.remove_managed_copy(deleted.managed_path)
        except VideoStorageError as error:
            cleanup_error = error

        self.refresh()
        if cleanup_error is not None:
            QMessageBox.warning(
                self,
                "登记已删除",
                f"登记和去重限制已删除，但应用副本清理失败：{cleanup_error}",
            )
            return
        QMessageBox.information(
            self,
            "删除完成",
            "视频登记和应用副本已删除；用户原始视频未被修改。",
        )

    def import_video(self) -> None:
        if self.active_patient is None or self.active_patient.id is None:
            QMessageBox.information(self, "未选择患者", "请先切换当前患者。")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择源视频",
            str(Path.home()),
            "视频文件 (*.mp4 *.mov *.m4v *.avi *.mkv);;所有文件 (*)",
        )
        if not filename:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            metadata = self.probe.inspect(Path(filename))
            if self.repository.find_video_by_sha256(metadata.file_sha256):
                raise DuplicateVideoError("该视频内容已经登记，未重复导入")
            managed_path = self.video_store.archive(metadata, self.active_patient.id)
            self.repository.create_video_asset(
                VideoAsset(
                    patient_id=self.active_patient.id,
                    source_path=str(metadata.source_path),
                    filename=metadata.filename,
                    file_sha256=metadata.file_sha256,
                    file_size_bytes=metadata.file_size_bytes,
                    extension=metadata.extension,
                    duration_seconds=metadata.duration_seconds,
                    fps=metadata.fps,
                    frame_count=metadata.frame_count,
                    width=metadata.width,
                    height=metadata.height,
                    managed_path=str(managed_path),
                )
            )
        except (
            VideoProbeError,
            VideoStorageError,
            DuplicateVideoError,
            ValueError,
        ) as error:
            QMessageBox.warning(self, "无法登记视频", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.refresh()
        QMessageBox.information(
            self,
            "导入完成",
            "视频已复制到患者应用目录并通过 SHA-256 校验；原文件未被修改。",
        )

    @staticmethod
    def _duration(seconds: float) -> str:
        if seconds <= 0:
            return "未知"
        minutes, remainder = divmod(round(seconds), 60)
        return f"{minutes:02d}:{remainder:02d}"

    @staticmethod
    def _file_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024 or unit == "GB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{size} B"
