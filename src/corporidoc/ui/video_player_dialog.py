from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QCloseEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class VideoPlayerDialog(QDialog):
    def __init__(
        self,
        video_path: Path,
        source_label: str,
        filename: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"视频播放 · {filename}")
        self.setMinimumSize(900, 650)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.5)
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumHeight(480)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        source = QLabel(f"播放来源：{source_label}\n{video_path}")
        source.setWordWrap(True)
        source.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self._toggle_playback)
        stop_button = QPushButton("停止")
        stop_button.clicked.connect(self.player.stop)
        self.mute_button = QPushButton("静音")
        self.mute_button.setCheckable(True)
        self.mute_button.toggled.connect(self.audio_output.setMuted)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)
        self.time_label = QLabel("00:00 / 00:00")

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(stop_button)
        controls.addWidget(self.mute_button)
        controls.addWidget(self.position_slider, 1)
        controls.addWidget(self.time_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(source)
        layout.addWidget(self.video_widget, 1)
        layout.addLayout(controls)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

        self.player.durationChanged.connect(self._duration_changed)
        self.player.positionChanged.connect(self._position_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.player.errorOccurred.connect(self._playback_error)
        self.player.setSource(QUrl.fromLocalFile(str(video_path)))

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, max(0, duration))
        self._update_time(self.player.position(), duration)

    def _position_changed(self, position: int) -> None:
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        self._update_time(position, self.player.duration())

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("暂停")
        else:
            self.play_button.setText("播放")

    def _playback_error(self, _: QMediaPlayer.Error, message: str) -> None:
        detail = message or self.player.errorString() or "未知多媒体错误"
        self.error_label.setText(f"无法解码或播放该视频：{detail}")

    def _update_time(self, position: int, duration: int) -> None:
        self.time_label.setText(
            f"{self._format_milliseconds(position)} / "
            f"{self._format_milliseconds(duration)}"
        )

    @staticmethod
    def _format_milliseconds(milliseconds: int) -> str:
        total_seconds = max(0, milliseconds) // 1000
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def closeEvent(self, event: QCloseEvent) -> None:
        self.player.stop()
        super().closeEvent(event)
