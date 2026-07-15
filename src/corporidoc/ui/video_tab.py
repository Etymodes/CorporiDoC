from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    PatientRepository,
    VideoProbe,
    VideoProbeError,
)
from corporidoc.domain import Patient, VideoAsset


class VideoTab(QWidget):
    def __init__(self, repository: PatientRepository) -> None:
        super().__init__()
        self.repository = repository
        self.probe = VideoProbe()
        self.active_patient: Patient | None = None

        self.patient_label = QLabel("请先在“患者”页切换当前患者")
        self.patient_label.setObjectName("activePatient")
        self.import_button = QPushButton("登记源视频")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_video)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh)

        controls = QHBoxLayout()
        controls.addWidget(self.patient_label)
        controls.addStretch()
        controls.addWidget(self.import_button)
        controls.addWidget(refresh_button)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["文件", "时长", "分辨率", "FPS", "帧数", "大小", "SHA-256", "源文件"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        notice = QLabel(
            "当前步骤只登记源文件路径和元数据，不复制、不改写视频。"
            "移动或删除原文件后，此记录会显示为“缺失”。"
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
            self.table.setRowCount(0)
            return
        videos = self.repository.list_video_assets(self.active_patient.id)
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
                "可用" if Path(video.source_path).is_file() else "缺失",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 7 and value == "缺失":
                    item.setForeground(Qt.red)
                self.table.setItem(row, column, item)

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
                )
            )
        except (VideoProbeError, DuplicateVideoError, ValueError) as error:
            QMessageBox.warning(self, "无法登记视频", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.refresh()
        QMessageBox.information(self, "登记完成", "源视频已登记，文件内容未被修改。")

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
