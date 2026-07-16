from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from corporidoc.data import DuplicatePatientCodeError, PatientRepository
from corporidoc.domain import Patient
from corporidoc.ui.pose_tab import PoseTab
from corporidoc.ui.settings_tab import SettingsTab
from corporidoc.ui.video_tab import VideoTab


class PatientDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, patient: Patient | None = None) -> None:
        super().__init__(parent)
        self.patient = patient
        self.setWindowTitle("修改患者资料" if patient else "注册患者")
        self.setMinimumWidth(480)

        self.patient_code = QLineEdit(patient.patient_code if patient else "")
        self.display_name = QLineEdit(patient.display_name if patient else "")
        self.sex = QComboBox()
        self.sex.addItems(["未知", "女", "男", "其他"])
        self.sex.setCurrentText(patient.sex if patient else "未知")
        self.date_of_birth = QLineEdit(patient.date_of_birth if patient else "")
        self.date_of_birth.setPlaceholderText("YYYY-MM-DD；不确定可留空")
        self.etiology = QLineEdit(patient.etiology if patient else "")
        self.injury_date = QLineEdit(patient.injury_date if patient else "")
        self.injury_date.setPlaceholderText("YYYY-MM-DD；不确定可留空")
        self.diagnosis = QComboBox()
        self.diagnosis.setEditable(True)
        self.diagnosis.addItems(["待评估", "昏迷", "UWS/VS", "MCS-", "MCS+", "EMCS"])
        self.diagnosis.setCurrentText(patient.current_diagnosis if patient else "待评估")
        self.notes = QPlainTextEdit(patient.notes if patient else "")
        self.notes.setMaximumHeight(90)

        form = QFormLayout()
        form.addRow("研究编号 *", self.patient_code)
        form.addRow("显示名/别名", self.display_name)
        form.addRow("性别", self.sex)
        form.addRow("出生日期", self.date_of_birth)
        form.addRow("病因", self.etiology)
        form.addRow("损伤日期", self.injury_date)
        form.addRow("当前临床分类", self.diagnosis)
        form.addRow("备注", self.notes)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        if not self.patient_code.text().strip():
            QMessageBox.warning(self, "资料不完整", "患者研究编号不能为空。")
            return
        self.accept()

    def value(self) -> Patient:
        return Patient(
            id=self.patient.id if self.patient else None,
            patient_code=self.patient_code.text(),
            display_name=self.display_name.text(),
            sex=self.sex.currentText(),
            date_of_birth=self.date_of_birth.text(),
            etiology=self.etiology.text(),
            injury_date=self.injury_date.text(),
            current_diagnosis=self.diagnosis.currentText(),
            notes=self.notes.toPlainText(),
            created_at=self.patient.created_at if self.patient else "",
            updated_at=self.patient.updated_at if self.patient else "",
        )


class StartTab(QWidget):
    register_requested = Signal()
    patients_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        title = QLabel("CorporiDoC")
        title.setObjectName("heroTitle")
        subtitle = QLabel("TiantanDoC · 意识障碍患者视频行为研究平台")
        subtitle.setObjectName("heroSubtitle")
        self.active_patient = QLabel("当前未选择患者")
        self.active_patient.setObjectName("activePatient")
        notice = QLabel(
            "研究原型：所有自动结果必须回看源视频并由临床人员确认，不得单独用于诊断或治疗决策。"
        )
        notice.setWordWrap(True)

        register_button = QPushButton("注册新患者")
        register_button.clicked.connect(self.register_requested)
        switch_button = QPushButton("进入患者管理")
        switch_button.clicked.connect(self.patients_requested)

        buttons = QHBoxLayout()
        buttons.addWidget(register_button)
        buttons.addWidget(switch_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 42, 48, 42)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(28)
        layout.addWidget(self.active_patient)
        layout.addSpacing(14)
        layout.addLayout(buttons)
        layout.addStretch()
        layout.addWidget(notice)

    def set_active_patient(self, patient: Patient | None) -> None:
        if patient is None:
            self.active_patient.setText("当前未选择患者")
            return
        name = f" · {patient.display_name}" if patient.display_name else ""
        self.active_patient.setText(
            f"当前患者：{patient.patient_code}{name} · {patient.current_diagnosis}"
        )


class PatientTab(QWidget):
    active_patient_changed = Signal(object)

    def __init__(self, repository: PatientRepository) -> None:
        super().__init__()
        self.repository = repository
        self.patients: list[Patient] = []
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["研究编号", "显示名", "性别", "病因", "损伤日期", "临床分类", "最近修改"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.doubleClicked.connect(self.edit_selected)

        add_button = QPushButton("注册")
        add_button.clicked.connect(self.add_patient)
        edit_button = QPushButton("修改")
        edit_button.clicked.connect(self.edit_selected)
        activate_button = QPushButton("切换为当前患者")
        activate_button.clicked.connect(self.activate_selected)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh)

        controls = QHBoxLayout()
        controls.addWidget(add_button)
        controls.addWidget(edit_button)
        controls.addWidget(activate_button)
        controls.addStretch()
        controls.addWidget(refresh_button)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.table)
        self.refresh()

    def selected_patient(self) -> Patient | None:
        row = self.table.currentRow()
        return self.patients[row] if 0 <= row < len(self.patients) else None

    def refresh(self) -> None:
        self.patients = self.repository.list_patients()
        self.table.setRowCount(len(self.patients))
        for row, patient in enumerate(self.patients):
            values = [
                patient.patient_code,
                patient.display_name,
                patient.sex,
                patient.etiology,
                patient.injury_date,
                patient.current_diagnosis,
                patient.updated_at,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, patient.id)
                self.table.setItem(row, column, item)

    def add_patient(self) -> None:
        dialog = PatientDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        self._save(lambda: self.repository.create_patient(dialog.value()))

    def edit_selected(self) -> None:
        patient = self.selected_patient()
        if patient is None:
            QMessageBox.information(self, "未选择患者", "请先选择一行患者资料。")
            return
        dialog = PatientDialog(self, patient)
        if dialog.exec() != QDialog.Accepted:
            return
        self._save(lambda: self.repository.update_patient(dialog.value()))

    def _save(self, operation: Callable[[], Patient]) -> None:
        try:
            patient = operation()
        except (DuplicatePatientCodeError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "无法保存", str(error))
            return
        self.refresh()
        self.active_patient_changed.emit(patient)

    def activate_selected(self) -> None:
        patient = self.selected_patient()
        if patient is None:
            QMessageBox.information(self, "未选择患者", "请先选择一行患者资料。")
            return
        self.active_patient_changed.emit(patient)


class PlaceholderTab(QWidget):
    def __init__(self, title: str, scope: str) -> None:
        super().__init__()
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        text = QLabel(scope)
        text.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 30, 32, 30)
        layout.addWidget(heading)
        layout.addWidget(text)
        layout.addStretch()


class MainWindow(QMainWindow):
    def __init__(self, repository: PatientRepository) -> None:
        super().__init__()
        self.setWindowTitle("CorporiDoC · TiantanDoC")
        self.resize(1180, 760)
        self.active_patient: Patient | None = None

        self.tabs = QTabWidget()
        self.start_tab = StartTab()
        self.patient_tab = PatientTab(repository)
        self.video_tab = VideoTab(repository)
        self.pose_tab = PoseTab(repository)
        self.tabs.addTab(self.start_tab, "开始")
        self.tabs.addTab(self.patient_tab, "患者")
        self.tabs.addTab(self.video_tab, "视频")
        self.tabs.addTab(self.pose_tab, "姿态")
        self.tabs.addTab(
            PlaceholderTab(
                "刺激—反应与行为评估", "对齐指令和刺激事件，量化时锁反应；由临床人员确认评分证据。"
            ),
            "评估",
        )
        self.tabs.addTab(
            PlaceholderTab(
                "轨迹与报告", "勾选模块后导出标记视频、运动轨迹、结构化数据和可追溯报告。"
            ),
            "报告",
        )
        self.settings_tab = SettingsTab(repository)
        self.tabs.addTab(self.settings_tab, "设置")
        self.setCentralWidget(self.tabs)

        self.start_tab.register_requested.connect(self.patient_tab.add_patient)
        self.start_tab.patients_requested.connect(
            lambda: self.tabs.setCurrentWidget(self.patient_tab)
        )
        self.patient_tab.active_patient_changed.connect(self.set_active_patient)
        self.pose_tab.task_idle.connect(self._finish_pending_close)
        self._close_when_pose_idle = False
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f7fa; }
            QTabWidget::pane { border: 1px solid #d8dee7; background: white; }
            QTabBar::tab { padding: 10px 18px; }
            QTabBar::tab:selected { color: #0d5c63; font-weight: 600; }
            QPushButton { padding: 7px 14px; }
            QLabel#heroTitle { font-size: 34px; font-weight: 700; color: #17324d; }
            QLabel#heroSubtitle { font-size: 16px; color: #537188; }
            QLabel#activePatient { font-size: 20px; font-weight: 600; color: #0d5c63; }
            QLabel#sectionTitle { font-size: 22px; font-weight: 650; color: #17324d; }
            """
        )

    def set_active_patient(self, patient: Patient) -> None:
        self.active_patient = patient
        self.start_tab.set_active_patient(patient)
        self.video_tab.set_active_patient(patient)
        self.pose_tab.set_active_patient(patient)
        self.statusBar().showMessage(f"当前患者：{patient.patient_code}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.pose_tab.is_running:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "姿态任务仍在运行",
            "是否取消当前 Mock 姿态任务，并在清理完成后关闭？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.No:
            event.ignore()
            return
        self._close_when_pose_idle = True
        self.pose_tab.cancel_task()
        event.ignore()

    def _finish_pending_close(self) -> None:
        if not self._close_when_pose_idle:
            return
        self._close_when_pose_idle = False
        self.close()
