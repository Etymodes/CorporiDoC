from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from corporidoc.data import (
    DuplicateModelError,
    ManagedModelFile,
    ManagedModelStore,
    ModelStorageError,
    PatientRepository,
)
from corporidoc.domain import ModelAsset
from corporidoc.pose import check_mediapipe_preflight


FULL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


class ModelImportDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导入 MediaPipe 人体姿态模型")
        self.setMinimumWidth(680)

        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        browse_button = QPushButton("选择 .task 文件")
        browse_button.clicked.connect(self._browse)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path, 1)
        file_row.addWidget(browse_button)

        self.name = QLineEdit("MediaPipe Pose Landmarker Full")
        self.version = QLineEdit("full-float16-latest")
        self.license_name = QLineEdit("Apache-2.0")
        self.source_url = QLineEdit(FULL_MODEL_URL)

        form = QFormLayout()
        form.addRow("模型文件 *", file_row)
        form.addRow("模型名称 *", self.name)
        form.addRow("模型版本 *", self.version)
        form.addRow("许可证 *", self.license_name)
        form.addRow("来源网址 *", self.source_url)

        notice = QLabel(
            "请只导入从可信来源获得的模型。CorporiDoC 会复制文件并记录哈希，"
            "但不会替你确认下载来源真实性。"
        )
        notice.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(notice)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 MediaPipe Pose Landmarker 模型",
            str(Path.home()),
            "MediaPipe Task (*.task)",
        )
        if selected:
            self.file_path.setText(selected)

    def _validate(self) -> None:
        values = (
            self.file_path.text(),
            self.name.text(),
            self.version.text(),
            self.license_name.text(),
            self.source_url.text(),
        )
        if any(not value.strip() for value in values):
            QMessageBox.warning(self, "资料不完整", "模型文件和溯源字段均不能为空。")
            return
        self.accept()

    def model_asset(self, managed: ManagedModelFile) -> ModelAsset:
        return ModelAsset(
            name=self.name.text(),
            backend_name="mediapipe-pose-landmarker",
            model_version=self.version.text(),
            file_path=str(managed.path),
            file_sha256=managed.sha256,
            file_size_bytes=managed.size_bytes,
            license_name=self.license_name.text(),
            source_url=self.source_url.text(),
        )


class SettingsTab(QWidget):
    def __init__(self, repository: PatientRepository) -> None:
        super().__init__()
        self.repository = repository
        self.model_store = ManagedModelStore(repository.database_path.parent)

        heading = QLabel("模型与系统设置")
        heading.setObjectName("sectionTitle")
        boundary = QLabel(
            "MediaPipe 模型卡注明其面向单人全身与健身场景，不用于攸关生命的决策。"
            "在 DoC 视频中必须保留失败帧、遮挡和人工复核。"
        )
        boundary.setWordWrap(True)

        import_button = QPushButton("导入 MediaPipe 模型")
        import_button.clicked.connect(self.import_model)
        refresh_button = QPushButton("重新预检")
        refresh_button.clicked.connect(self.refresh)
        controls = QHBoxLayout()
        controls.addWidget(import_button)
        controls.addWidget(refresh_button)
        controls.addStretch()

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["模型", "版本", "后端", "SHA-256", "导入时间", "预检"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 30, 32, 30)
        layout.addWidget(heading)
        layout.addWidget(boundary)
        layout.addLayout(controls)
        layout.addWidget(self.table, 1)
        self.refresh()

    def import_model(self) -> None:
        dialog = ModelImportDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            managed = self.model_store.archive(Path(dialog.file_path.text()))
            created = self.repository.create_model_asset(dialog.model_asset(managed))
        except (DuplicateModelError, ModelStorageError, ValueError) as error:
            QMessageBox.warning(self, "无法导入模型", str(error))
            return
        self.refresh()
        QMessageBox.information(
            self,
            "模型已登记",
            f"{created.name}\nSHA-256: {created.file_sha256}",
        )

    def refresh(self) -> None:
        models = self.repository.list_model_assets()
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            preflight = check_mediapipe_preflight(model)
            status = "可运行" if preflight.ready else "未就绪：" + "；".join(preflight.errors)
            values = (
                model.name,
                model.model_version,
                model.backend_name,
                model.file_sha256[:12],
                model.imported_at,
                status,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
