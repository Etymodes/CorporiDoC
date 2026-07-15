from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from corporidoc.data import VideoMetadata
from corporidoc.domain import VideoQualityAssessment


@dataclass(frozen=True, slots=True)
class VideoIntakeDetails:
    camera_view: str
    body_side: str
    capture_protocol: str
    video_notes: str


class VideoIntakeDialog(QDialog):
    def __init__(
        self,
        metadata: VideoMetadata,
        assessment: VideoQualityAssessment,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("确认视频采集信息")
        self.setMinimumWidth(560)

        metadata_label = QLabel(
            f"{metadata.filename}\n"
            f"{metadata.width}×{metadata.height} · {metadata.fps:.2f} FPS · "
            f"{metadata.duration_seconds:.1f} 秒"
        )
        metadata_label.setWordWrap(True)

        self.camera_view = QComboBox()
        self.camera_view.addItems(["未记录", "正面", "左侧", "右侧", "床尾", "俯视", "其他"])
        self.body_side = QComboBox()
        self.body_side.addItems(["未记录", "双侧", "左侧", "右侧", "不适用"])
        self.capture_protocol = QLineEdit()
        self.capture_protocol.setPlaceholderText("例如：静息观察、指令任务、CRS-R 评估片段")
        self.video_notes = QPlainTextEdit()
        self.video_notes.setMaximumHeight(80)

        quality_label = QLabel(assessment.summary)
        quality_label.setWordWrap(True)
        if assessment.warnings:
            quality_label.setText(
                assessment.summary + "\n• " + "\n• ".join(assessment.warnings)
            )
            quality_label.setStyleSheet("color: #a15c00;")
        else:
            quality_label.setStyleSheet("color: #207245;")

        form = QFormLayout()
        form.addRow("视频", metadata_label)
        form.addRow("机位", self.camera_view)
        form.addRow("观察侧别", self.body_side)
        form.addRow("采集协议", self.capture_protocol)
        form.addRow("备注", self.video_notes)
        form.addRow("基础质控", quality_label)

        notice = QLabel(
            "基础质控仅提示元数据风险，不代表视频适合临床分析；请按实际采集情况填写。"
        )
        notice.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(notice)
        layout.addWidget(buttons)

    def value(self) -> VideoIntakeDetails:
        return VideoIntakeDetails(
            camera_view=self.camera_view.currentText(),
            body_side=self.body_side.currentText(),
            capture_protocol=self.capture_protocol.text().strip(),
            video_notes=self.video_notes.toPlainText().strip(),
        )
