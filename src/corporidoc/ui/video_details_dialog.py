from __future__ import annotations

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

from corporidoc.domain import VideoAsset, decode_quality_warnings
from corporidoc.ui.video_intake_dialog import VideoIntakeDetails


class VideoDetailsDialog(QDialog):
    def __init__(self, video: VideoAsset, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("视频详情与采集信息")
        self.setMinimumSize(680, 620)

        warnings = decode_quality_warnings(video.quality_warnings_json)
        if not video.quality_rule_version:
            quality = "未评估"
        elif warnings:
            quality = "需复核\n• " + "\n• ".join(warnings)
        else:
            quality = "基础检查通过"

        immutable = QPlainTextEdit(
            "\n".join(
                [
                    f"文件名：{video.filename}",
                    f"患者内部 ID：{video.patient_id}",
                    f"导入时间：{video.imported_at}",
                    f"时长：{video.duration_seconds:.2f} 秒",
                    f"分辨率：{video.width}×{video.height}",
                    f"帧率/帧数：{video.fps:.2f} FPS / {video.frame_count}",
                    f"文件大小：{video.file_size_bytes} bytes",
                    f"SHA-256：{video.file_sha256}",
                    f"原路径：{video.source_path}",
                    f"应用副本：{video.managed_path or '未建立'}",
                    f"质控规则：{video.quality_rule_version or '未评估'}",
                    f"质控结果：{quality}",
                ]
            )
        )
        immutable.setReadOnly(True)
        immutable.setMinimumHeight(270)

        self.camera_view = QComboBox()
        self.camera_view.addItems(["未记录", "正面", "左侧", "右侧", "床尾", "俯视", "其他"])
        self.camera_view.setCurrentText(video.camera_view)
        self.body_side = QComboBox()
        self.body_side.addItems(["未记录", "双侧", "左侧", "右侧", "不适用"])
        self.body_side.setCurrentText(video.body_side)
        self.capture_protocol = QLineEdit(video.capture_protocol)
        self.capture_protocol.setPlaceholderText("例如：静息观察、指令任务、CRS-R 评估片段")
        self.video_notes = QPlainTextEdit(video.video_notes)
        self.video_notes.setMaximumHeight(90)

        form = QFormLayout()
        form.addRow("只读登记信息", immutable)
        form.addRow("机位", self.camera_view)
        form.addRow("观察侧别", self.body_side)
        form.addRow("采集协议", self.capture_protocol)
        form.addRow("备注", self.video_notes)

        notice = QLabel(
            "保存只会修改上述四项采集信息；患者归属、路径、哈希、导入时间和质控结果不会改变。"
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
