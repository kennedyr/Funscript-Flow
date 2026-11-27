import glob
import json
import os
import threading
import time
from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QPainter,
    QPen,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from funscript_flow.funscript_flow import (
    SUPPORTED_VIDEO_EXTENSIONS,
    get_available_backends,
    process_video,
)
from funscript_flow.localization import STRINGS

# ---------- GUI Code ----------
# Old tkinter helper functions removed - no longer needed with PySide6


class WorkerThread(QThread):
    """Thread for processing videos in background."""

    progressChanged = Signal(int)
    videoProgressChanged = Signal(int)
    finished = Signal(
        bool, str, list, list
    )  # error_occurred, time_str, log_messages, generated_files
    logMessage = Signal(str)

    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings
        self.cancel_event = threading.Event()
        self.log_messages = []
        self.log_file = None
        self.generated_files = []  # Track (video_path, funscript_path) pairs

    def log(self, msg):
        self.log_messages.append(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()
        self.logMessage.emit(msg)

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        error_occurred = False

        try:
            # Create logs folder
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(log_path, exist_ok=True)

            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(log_path, f"{timestamp}.log")
            self.log_file = open(log_filename, "w", encoding="utf-8")
        except Exception as e:
            self.finished.emit(True, "0s", [f"Cannot open log file: {e}"], [])
            return

        batch_start_time = time.time()
        total_files = len(self.files)

        for idx, video in enumerate(self.files):
            if self.cancel_event.is_set():
                self.log(STRINGS["cancelled_by_user"])
                break

            self.videoProgressChanged.emit(0)

            # Generate the expected output path
            base = os.path.splitext(video)[0]
            funscript_path = base + ".funscript"

            err = process_video(
                video,
                self.settings,
                self.log,
                progress_callback=lambda prog: self.videoProgressChanged.emit(prog),
                cancel_flag=lambda: self.cancel_event.is_set(),
            )
            if err:
                error_occurred = True
            else:
                # If processing succeeded, track the generated file
                self.generated_files.append((video, funscript_path))

            overall = int(100 * (idx + 1) / total_files)
            self.progressChanged.emit(overall)

        # Calculate and format total time
        total_time = time.time() - batch_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        self.log(f"{STRINGS['batch_processing_complete']} Total time: {time_str}")
        if self.log_file:
            self.log_file.close()

        self.finished.emit(
            error_occurred, time_str, self.log_messages, self.generated_files
        )


class MotionIndicatorWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Motion Indicator")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(50, 300)

        # Make window resizable
        self.setMinimumSize(20, 50)

        # Current position value (0-100)
        self.current_position = 0

        # Set up the widget
        self.setStyleSheet("background-color: black;")

    def set_position(self, position):
        """Set the position of the motion indicator (0-100)"""
        self.current_position = max(0, min(100, position))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        # Draw the vertical bar
        height = self.height()

        # Calculate bar height based on position (0-100)
        bar_height = int((self.current_position / 100.0) * height)

        # Draw from bottom up
        bar_rect = self.rect()
        bar_rect.setTop(height - bar_height)

        # Use red color for the bar
        painter.fillRect(bar_rect, QColor(255, 0, 0))

        # Draw border
        painter.setPen(QPen(QColor(128, 128, 128), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(STRINGS["app_title"])
        self.setGeometry(100, 100, 800, 600)

        # Application icon is set at QApplication level

        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Initialize variables first
        self.files = []
        self.worker_thread = None
        self.params = {}
        self.backends = get_available_backends()
        self.available_backends = []

        # Initialize motion indicator window
        self.motion_indicator = MotionIndicatorWindow(self)
        self.motion_indicator_visible = False

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Funscript Generation Tab
        self.generation_tab = QWidget()
        self.generation_tab.setAcceptDrops(True)
        self.tab_widget.addTab(self.generation_tab, "Funscript Generation")
        self.setup_generation_tab()

        # Script Preview Tab
        self.preview_tab = QWidget()
        self.preview_tab.setAcceptDrops(True)
        self.tab_widget.addTab(self.preview_tab, "Script Preview")
        self.setup_preview_tab()

        # Load configuration
        self.load_config()

    def setup_generation_tab(self):
        """Setup the funscript generation tab."""
        layout = QVBoxLayout(self.generation_tab)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout(file_group)

        self.btn_select_files = QPushButton(STRINGS["select_videos"])
        self.btn_select_files.clicked.connect(self.select_files)
        file_layout.addWidget(self.btn_select_files)

        self.btn_select_folder = QPushButton(STRINGS["select_folder"])
        self.btn_select_folder.clicked.connect(self.select_folder)
        file_layout.addWidget(self.btn_select_folder)

        self.lbl_files = QLabel(STRINGS["no_files_selected"])
        file_layout.addWidget(self.lbl_files)

        file_layout.addStretch()

        self.btn_readme = QPushButton(STRINGS["readme"])
        self.btn_readme.clicked.connect(self.show_readme)
        file_layout.addWidget(self.btn_readme)

        layout.addWidget(file_group)

        # Mode selection section
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout(mode_group)

        self.chk_vr = QCheckBox(STRINGS["vr_mode"])
        ToolTip(self.chk_vr, STRINGS["vr_mode_tooltip"])
        mode_layout.addWidget(self.chk_vr)

        self.chk_pov = QCheckBox("POV Mode")
        ToolTip(self.chk_pov, STRINGS["pov_mode_tooltip"])
        mode_layout.addWidget(self.chk_pov)

        self.chk_balance_global = QCheckBox("Balance Global Motion")
        ToolTip(self.chk_balance_global, STRINGS["balance_global_tooltip"])
        mode_layout.addWidget(self.chk_balance_global)

        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        progress_layout.addWidget(QLabel(STRINGS["overall_progress"]))
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        progress_layout.addWidget(self.overall_progress)

        progress_layout.addWidget(QLabel(STRINGS["current_video_progress"]))
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        progress_layout.addWidget(self.video_progress)

        layout.addWidget(progress_group)

        # Live Log section (collapsible)
        self.log_group = QGroupBox(STRINGS["live_log"])
        self.log_group.setCheckable(True)
        self.log_group.setChecked(True)  # Expanded by default
        log_layout = QVBoxLayout(self.log_group)

        # Log display text area
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(200)
        self.log_display.setMinimumHeight(150)
        log_layout.addWidget(self.log_display)

        # Clear log button
        log_button_layout = QHBoxLayout()
        self.btn_clear_log = QPushButton(STRINGS["clear_log"])
        self.btn_clear_log.clicked.connect(self.clear_log_display)
        log_button_layout.addWidget(self.btn_clear_log)
        log_button_layout.addStretch()
        log_layout.addLayout(log_button_layout)

        layout.addWidget(self.log_group)

        # Advanced settings section (collapsible)
        self.adv_group = QGroupBox(STRINGS["advanced_settings"])
        self.adv_group.setCheckable(True)
        self.adv_group.setChecked(False)
        self.setup_advanced_settings()
        layout.addWidget(self.adv_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.btn_run = QPushButton(STRINGS["run"])
        self.btn_run.clicked.connect(self.run_batch)
        button_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(STRINGS["cancel"])
        self.btn_cancel.clicked.connect(self.cancel_run)
        self.btn_cancel.setEnabled(False)
        button_layout.addWidget(self.btn_cancel)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        layout.addStretch()

    def setup_preview_tab(self):
        """Setup the script preview tab."""
        layout = QVBoxLayout(self.preview_tab)

        # Video player
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setAcceptDrops(
            False
        )  # Allow drag events to bubble up to parent
        layout.addWidget(self.video_widget, 1)  # Give it stretch factor of 1

        # Media player setup
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Initialize volume state
        self.muted = False
        self.previous_volume = 50
        self.audio_output.setVolume(0.5)  # Set initial volume to 50%

        # Video controls
        controls_layout = QHBoxLayout()

        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setFixedWidth(60)  # Fixed width to prevent resizing
        controls_layout.addWidget(self.btn_play_pause)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(1000)  # Higher resolution for smoother seeking
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.valueChanged.connect(self.on_slider_value_changed)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.setAcceptDrops(
            False
        )  # Allow drag events to bubble up to parent
        controls_layout.addWidget(self.position_slider)

        self.lbl_time = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.lbl_time)

        # Volume controls
        volume_label = QLabel("Volume:")
        controls_layout.addWidget(volume_label)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)  # Default to 50% volume
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        controls_layout.addWidget(self.volume_slider)

        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(30)
        self.btn_mute.clicked.connect(self.toggle_mute)
        controls_layout.addWidget(self.btn_mute)

        layout.addLayout(controls_layout)

        # FunScript visualizer
        visualizer_group = QGroupBox("FunScript Visualizer")
        visualizer_layout = QVBoxLayout(visualizer_group)
        visualizer_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        visualizer_layout.setSpacing(3)  # Reduce spacing between elements

        self.funscript_visualizer = FunScriptVisualizer()
        self.funscript_visualizer.positionChanged.connect(self.seek_to_position)
        self.funscript_visualizer.set_reference_slider(self.position_slider)
        self.funscript_visualizer.setAcceptDrops(
            False
        )  # Allow drag events to bubble up to parent
        visualizer_layout.addWidget(self.funscript_visualizer)

        # Visualizer controls
        viz_controls_layout = QHBoxLayout()
        viz_controls_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(self.funscript_visualizer.reset_view)
        viz_controls_layout.addWidget(self.btn_reset_view)

        self.btn_toggle_indicator = QPushButton("Show Indicator")
        self.btn_toggle_indicator.clicked.connect(self.toggle_motion_indicator)
        viz_controls_layout.addWidget(self.btn_toggle_indicator)

        self.lbl_zoom = QLabel("Zoom: 1.0x")
        viz_controls_layout.addWidget(self.lbl_zoom)

        viz_controls_layout.addStretch()
        visualizer_layout.addLayout(viz_controls_layout)

        layout.addWidget(visualizer_group)

        # File loading buttons at bottom right
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()  # Push buttons to the right

        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.clicked.connect(self.load_video)
        bottom_layout.addWidget(self.btn_load_video)

        self.btn_load_funscript = QPushButton("Load FunScript")
        self.btn_load_funscript.clicked.connect(self.load_funscript)
        bottom_layout.addWidget(self.btn_load_funscript)

        layout.addLayout(bottom_layout)

        # Initialize media player connections
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)

        # Timer for updating visualizer position
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_visualizer_position)
        self.position_timer.start(50)  # Update every 50ms for smooth visualization

        # State variables
        self.loaded_video_path = None
        self.loaded_funscript_data = None
        self.slider_being_dragged = False

    def setup_advanced_settings(self):
        """Setup advanced settings form."""
        layout = QFormLayout(self.adv_group)

        # Get the number of cores available
        num_cores = os.cpu_count()

        # Create parameter inputs
        self.params = {}

        self.params["threads"] = QLineEdit(str(num_cores))
        ToolTip(
            self.params["threads"],
            "Number of threads used for optical flow computation.",
        )
        layout.addRow(STRINGS["threads"], self.params["threads"])

        self.params["detrend_window"] = QLineEdit("1.5")
        ToolTip(
            self.params["detrend_window"],
            "Controls the aggressiveness of drift removal. See readme for detail. Recommended: 1-10, higher values for more stable cameras.",
        )
        layout.addRow(STRINGS["detrend_window"], self.params["detrend_window"])

        self.params["norm_window"] = QLineEdit("4")
        ToolTip(
            self.params["norm_window"],
            "Time window to calibrate motion range (seconds). Shorter values amplify motion, but also cause artifacts in long thrusts.",
        )
        layout.addRow(STRINGS["norm_window"], self.params["norm_window"])

        self.params["batch_size"] = QLineEdit("3000")
        ToolTip(
            self.params["batch_size"],
            "Number of frames to process per batch (Higher values will be faster, but also take more RAM).",
        )
        layout.addRow(STRINGS["batch_size"], self.params["batch_size"])

        # GPU Backend selection
        self.backend_combo = QComboBox()

        # Format backend options to show availability
        backend_display = []
        self.available_backends = []
        for backend, available in [
            ("CPU", True),
            ("CUDA", self.backends["CUDA"]),
            ("OpenCL", self.backends["OpenCL"]),
            ("DNN", self.backends["DNN"]),
        ]:
            if available:
                backend_display.append(backend)
                self.available_backends.append(backend)
            else:
                backend_display.append(f"{backend} (unavailable)")

        self.backend_combo.addItems(backend_display)
        self.backend_combo.setCurrentText("CPU")
        ToolTip(
            self.backend_combo,
            "Select processing backend. GPU acceleration can significantly speed up processing.\nUnavailable options require specific hardware or OpenCV build configuration.",
        )
        layout.addRow("Processing Backend:", self.backend_combo)

        # Checkboxes
        self.chk_keyframe = QCheckBox("Enable keyframe reduction")
        self.chk_keyframe.setChecked(True)
        layout.addRow(self.chk_keyframe)

        self.chk_overwrite = QCheckBox(STRINGS["overwrite_files"])
        layout.addRow(self.chk_overwrite)

    def select_files(self):
        """Select video files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            STRINGS["select_videos"],
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.wmv *.flv *.mpg *.mpeg *.ts);;All Files (*)",
        )
        if files:
            self.files = files
            self.lbl_files.setText(f"{len(self.files)} file(s) selected")
        else:
            self.files = []
            self.lbl_files.setText(STRINGS["no_files_selected"])

    def select_folder(self):
        """Select folder containing videos."""
        folder = QFileDialog.getExistingDirectory(self, STRINGS["select_folder"])
        if folder:
            found = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_VIDEO_EXTENSIONS:
                        found.append(os.path.join(root, f))
            self.files = found
            self.lbl_files.setText(f"{len(self.files)} file(s) found in folder")

    def show_readme(self):
        """Show readme in a dialog."""
        try:
            with open("readme.txt", "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading readme.txt: {e}"

        dialog = QDialog(self)
        dialog.setWindowTitle("Readme")
        dialog.setModal(False)
        dialog.resize(600, 400)
        dialog.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setPlainText(content)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.show()

    def load_video(self):
        """Load a video file for preview."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.wmv *.flv *.mpg *.mpeg *.ts);;All Files (*)",
        )
        if file_path:
            self.loaded_video_path = file_path
            self.media_player.setSource(QUrl.fromLocalFile(file_path))

    def load_funscript(self):
        """Load a FunScript file for visualization."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load FunScript",
            "",
            "FunScript Files (*.funscript);;JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.loaded_funscript_data = json.load(f)

                self.funscript_visualizer.load_funscript(self.loaded_funscript_data)

                # If we have a video duration, update the visualizer
                if self.media_player.duration() > 0:
                    self.funscript_visualizer.set_duration(self.media_player.duration())

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load FunScript: {e}")

    def load_video_and_funscript_in_preview(self, video_path, funscript_path):
        """Load video and funscript in the preview tab and switch to it."""
        try:
            # Switch to preview tab
            self.tab_widget.setCurrentIndex(1)  # Preview tab is index 1

            # Load video
            self.loaded_video_path = video_path
            self.media_player.setSource(QUrl.fromLocalFile(video_path))

            # Load funscript if it exists
            if os.path.exists(funscript_path):
                with open(funscript_path, "r", encoding="utf-8") as f:
                    self.loaded_funscript_data = json.load(f)

                self.funscript_visualizer.load_funscript(self.loaded_funscript_data)

                # If we have a video duration, update the visualizer
                if self.media_player.duration() > 0:
                    self.funscript_visualizer.set_duration(self.media_player.duration())

            return True
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load video and funscript in preview: {e}"
            )
            return False

    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def on_slider_pressed(self):
        """Handle when user starts interacting with the slider."""
        self.slider_being_dragged = True
        # Immediately seek to the clicked position for instant feedback
        self.set_position(self.position_slider.value())

    def on_slider_released(self):
        """Handle when user releases the slider."""
        self.slider_being_dragged = False

    def on_slider_value_changed(self, position):
        """Handle slider value changes from user interaction."""
        # Only respond to user-initiated changes when user is interacting with slider
        if self.slider_being_dragged:
            self.set_position(position)

    def set_position(self, position):
        """Set the video position from slider."""
        duration = self.media_player.duration()
        if duration > 0:
            new_position = (position * duration) // 1000
            self.media_player.setPosition(new_position)

    def seek_to_position(self, position_ms):
        """Seek to a specific position from the visualizer."""
        self.media_player.setPosition(position_ms)

    def on_position_changed(self, position):
        """Handle video position changes."""
        duration = self.media_player.duration()
        if duration > 0:
            # Only update slider if user is not dragging it
            if not self.slider_being_dragged:
                slider_position = (position * 1000) // duration
                # Block signals temporarily to prevent feedback loop
                self.position_slider.blockSignals(True)
                self.position_slider.setValue(slider_position)
                self.position_slider.blockSignals(False)

            # Always update time label
            current_time = self.format_time(position)
            total_time = self.format_time(duration)
            self.lbl_time.setText(f"{current_time} / {total_time}")

    def on_duration_changed(self, duration):
        """Handle video duration changes."""
        if duration > 0 and self.loaded_funscript_data:
            self.funscript_visualizer.set_duration(duration)

    def on_playback_state_changed(self, state):
        """Handle playback state changes."""
        if state == QMediaPlayer.PlayingState:
            self.btn_play_pause.setText("Pause")
        else:
            self.btn_play_pause.setText("Play")

    def on_volume_changed(self, volume):
        """Handle volume slider changes."""
        # Convert from 0-100 to 0.0-1.0 range
        volume_float = volume / 100.0
        self.audio_output.setVolume(volume_float)

        # Update mute button icon based on volume
        if volume == 0:
            self.btn_mute.setText("🔇")
        elif volume < 50:
            self.btn_mute.setText("🔉")
        else:
            self.btn_mute.setText("🔊")

    def toggle_mute(self):
        """Toggle mute/unmute functionality."""
        if hasattr(self, "muted") and self.muted:
            # Unmute: restore previous volume
            self.volume_slider.setValue(getattr(self, "previous_volume", 50))
            self.muted = False
        else:
            # Mute: save current volume and set to 0
            self.previous_volume = self.volume_slider.value()
            self.volume_slider.setValue(0)
            self.muted = True

    def update_visualizer_position(self):
        """Update the visualizer with current video position."""
        if self.media_player.duration() > 0:
            current_position = self.media_player.position()
            self.funscript_visualizer.set_position(current_position)

            # Update zoom label
            zoom_level = self.funscript_visualizer.zoom_level
            self.lbl_zoom.setText(f"Zoom: {zoom_level:.1f}x")

            # Update motion indicator if visible
            if self.motion_indicator_visible:
                current_value = self.get_current_funscript_value()
                self.motion_indicator.set_position(current_value)

    def format_time(self, ms):
        """Format time in milliseconds to MM:SS format."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def save_config(self):
        """Save configuration to file."""
        config = {key: widget.text() for key, widget in self.params.items()}
        config["overwrite"] = self.chk_overwrite.isChecked()
        config["vr_mode"] = self.chk_vr.isChecked()
        config["pov_mode"] = self.chk_pov.isChecked()
        config["balance_global"] = self.chk_balance_global.isChecked()
        config["backend"] = self.backend_combo.currentText()
        config["keyframe_reduction"] = self.chk_keyframe.isChecked()

        config_path = "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(
                self,
                "Config Saved",
                STRINGS["config_saved"].format(config_path=config_path),
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config: {e}")

    def load_config(self):
        """Load configuration from file."""
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Load after UI is set up
                QTimer.singleShot(100, lambda: self._load_config_values(config))

            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Config Load",
                    STRINGS["config_load_error"].format(error=str(e)),
                )

    def _load_config_values(self, config):
        """Helper to load config values after UI setup."""
        for key, widget in self.params.items():
            if key in config:
                widget.setText(str(config[key]))

        if "overwrite" in config:
            self.chk_overwrite.setChecked(config["overwrite"])
        if "vr_mode" in config:
            self.chk_vr.setChecked(config["vr_mode"])
        if "pov_mode" in config:
            self.chk_pov.setChecked(config["pov_mode"])
        if "balance_global" in config:
            self.chk_balance_global.setChecked(config["balance_global"])
        if "keyframe_reduction" in config:
            self.chk_keyframe.setChecked(config["keyframe_reduction"])
        if "backend" in config and config["backend"] in self.available_backends:
            self.backend_combo.setCurrentText(config["backend"])

    def clear_log_display(self):
        """Clear the log display."""
        self.log_display.clear()

    def append_log_message(self, message):
        """Append a log message to the display and auto-scroll."""
        self.log_display.append(message)
        # Auto-scroll to the bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def run_batch(self):
        """Start batch processing."""
        if not self.files:
            QMessageBox.warning(self, "No files", STRINGS["no_files_warning"])
            return

        try:
            settings = {
                "threads": int(self.params["threads"].text()),
                "detrend_window": float(self.params["detrend_window"].text()),
                "norm_window": float(self.params["norm_window"].text()),
                "batch_size": int(self.params["batch_size"].text()),
                "overwrite": self.chk_overwrite.isChecked(),
                "keyframe_reduction": self.chk_keyframe.isChecked(),
                "vr_mode": self.chk_vr.isChecked(),
                "pov_mode": self.chk_pov.isChecked(),
                "balance_global": self.chk_balance_global.isChecked(),
                "backend": self.backend_combo.currentText(),
                "cut_threshold": 7,
            }
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", f"Invalid parameters: {e}")
            return

        # Reset progress and clear log
        self.overall_progress.setValue(0)
        self.video_progress.setValue(0)
        self.clear_log_display()

        # Disable controls
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # Start worker thread
        self.worker_thread = WorkerThread(self.files, settings)
        self.worker_thread.progressChanged.connect(self.overall_progress.setValue)
        self.worker_thread.videoProgressChanged.connect(self.video_progress.setValue)
        self.worker_thread.logMessage.connect(self.append_log_message)
        self.worker_thread.finished.connect(self.on_batch_finished)
        self.worker_thread.start()

    def cancel_run(self):
        """Cancel the running batch process."""
        if self.worker_thread:
            self.worker_thread.cancel()

    def on_batch_finished(
        self, error_occurred, time_str, log_messages, generated_files
    ):
        """Handle batch processing completion."""
        # Re-enable controls
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        # Show completion message with preview option
        if error_occurred:
            message_box = QMessageBox(self)
            message_box.setWindowTitle("Run Finished")
            message_box.setText(
                f"{STRINGS['processing_completed_with_errors']}\nCompleted in {time_str}"
            )
            message_box.setStandardButtons(QMessageBox.NoButton)

            # Add custom buttons
            log_button = message_box.addButton("View Log", QMessageBox.ActionRole)

            if generated_files:
                preview_button = message_box.addButton(
                    "Open in Preview", QMessageBox.ActionRole
                )

            close_button = message_box.addButton("Close", QMessageBox.RejectRole)
            message_box.setDefaultButton(close_button)

            clicked_button = message_box.clickedButton()

            if clicked_button == log_button:
                self.show_log_dialog(log_messages)
            elif generated_files and clicked_button == preview_button:
                # Open the first successfully generated file in preview
                video_path, funscript_path = generated_files[0]
                self.load_video_and_funscript_in_preview(video_path, funscript_path)

        else:
            message_box = QMessageBox(self)
            message_box.setWindowTitle("Run Finished")
            message_box.setText(f"Batch processing complete.\nCompleted in {time_str}")
            message_box.setStandardButtons(QMessageBox.NoButton)

            # Add custom buttons
            log_button = message_box.addButton("View Log", QMessageBox.ActionRole)

            if generated_files:
                preview_button = message_box.addButton(
                    "Open in Preview", QMessageBox.ActionRole
                )

            close_button = message_box.addButton("Close", QMessageBox.RejectRole)
            message_box.setDefaultButton(close_button)

            clicked_button = message_box.clickedButton()

            if clicked_button == log_button:
                self.show_log_dialog(log_messages)
            elif generated_files and clicked_button == preview_button:
                # Open the first successfully generated file in preview
                video_path, funscript_path = generated_files[0]
                self.load_video_and_funscript_in_preview(video_path, funscript_path)

    def show_log_dialog(self, log_messages):
        """Show log messages in a dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing Log")
        dialog.setModal(True)
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setPlainText("\n".join(log_messages))
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for both tabs."""
        if event.mimeData().hasUrls():
            # Check which tab is currently active
            current_tab = self.tab_widget.currentWidget()

            if current_tab == self.generation_tab:
                # Generation tab: accept videos and folders
                accepted = False
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if os.path.isdir(path):
                        accepted = True
                        break
                    elif os.path.isfile(path):
                        ext = os.path.splitext(path)[1].lower()
                        if ext in SUPPORTED_VIDEO_EXTENSIONS:
                            accepted = True
                            break

                if accepted:
                    event.acceptProposedAction()
                else:
                    event.ignore()

            elif current_tab == self.preview_tab:
                # Preview tab: accept videos and funscript files
                accepted = False
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if os.path.isfile(path):
                        ext = os.path.splitext(path)[1].lower()
                        if ext in SUPPORTED_VIDEO_EXTENSIONS or ext == ".funscript":
                            accepted = True
                            break

                if accepted:
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events for both tabs."""
        if event.mimeData().hasUrls():
            current_tab = self.tab_widget.currentWidget()

            if current_tab == self.generation_tab:
                # Generation tab: handle video files and folders
                video_files = []
                folders = []

                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if os.path.isdir(path):
                        folders.append(path)
                    elif os.path.isfile(path):
                        ext = os.path.splitext(path)[1].lower()
                        if ext in SUPPORTED_VIDEO_EXTENSIONS:
                            video_files.append(path)

                # Process dropped files/folders
                if folders:
                    # If folders were dropped, scan them for video files
                    for folder in folders:
                        for ext in SUPPORTED_VIDEO_EXTENSIONS:
                            pattern = os.path.join(folder, f"*{ext}")
                            video_files.extend(glob.glob(pattern))

                if video_files:
                    self.files = video_files
                    file_count = len(self.files)
                    self.lbl_files.setText(f"{file_count} file(s) selected")

                event.acceptProposedAction()

            elif current_tab == self.preview_tab:
                # Preview tab: handle video and funscript files
                video_file = None
                funscript_file = None

                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if os.path.isfile(path):
                        ext = os.path.splitext(path)[1].lower()
                        if ext in SUPPORTED_VIDEO_EXTENSIONS:
                            video_file = path
                        elif ext == ".funscript":
                            funscript_file = path

                # Load video if dropped
                if video_file:
                    video_url = QUrl.fromLocalFile(video_file)
                    self.media_player.setSource(video_url)
                    self.current_video_path = video_file

                # Load funscript if dropped
                if funscript_file:
                    try:
                        with open(funscript_file, "r", encoding="utf-8") as f:
                            self.loaded_funscript_data = json.load(f)

                        self.funscript_visualizer.load_funscript(
                            self.loaded_funscript_data
                        )

                        # If we have a video duration, update the visualizer
                        if self.media_player.duration() > 0:
                            self.funscript_visualizer.set_duration(
                                self.media_player.duration()
                            )
                    except Exception as e:
                        QMessageBox.warning(
                            self, "Error", f"Failed to load funscript: {e}"
                        )

                event.acceptProposedAction()
        else:
            event.ignore()

    def get_current_funscript_value(self):
        """Get the current funscript position value at current playback time."""
        if (
            not hasattr(self, "loaded_funscript_data")
            or not self.loaded_funscript_data
            or not self.funscript_visualizer.actions
        ):
            return 0

        current_time = self.media_player.position()
        actions: list[Any] = self.funscript_visualizer.actions

        if not actions:
            return 0

        # Find surrounding actions for interpolation
        for i in range(len(actions) - 1):
            if actions[i]["at"] <= current_time <= actions[i + 1]["at"]:
                # Interpolate between the two points
                t1, pos1 = actions[i]["at"], actions[i]["pos"]
                t2, pos2 = actions[i + 1]["at"], actions[i + 1]["pos"]

                if t2 == t1:
                    return pos1

                # Linear interpolation
                ratio = (current_time - t1) / (t2 - t1)
                return pos1 + (pos2 - pos1) * ratio

        # If before first action or after last action
        if current_time < actions[0]["at"]:
            return actions[0]["pos"]
        else:
            return actions[-1]["pos"]

    def toggle_motion_indicator(self):
        """Toggle the visibility of the motion indicator window"""
        if self.motion_indicator_visible:
            self.motion_indicator.hide()
            self.motion_indicator_visible = False
            self.btn_toggle_indicator.setText("Show Indicator")
        else:
            self.motion_indicator.show()
            self.motion_indicator_visible = True
            self.btn_toggle_indicator.setText("Hide Indicator")


# ---------- Tooltip Implementation ----------
class ToolTip:
    """Simple tooltip for Qt widgets using setToolTip."""

    def __init__(self, widget, text="widget info"):
        if widget and text:
            widget.setToolTip(str(text))


# ---------- FunScript Visualizer Widget ----------
class FunScriptVisualizer(QWidget):
    """Custom widget for visualizing FunScript data with pan and zoom functionality."""

    positionChanged = Signal(int)  # Emitted when user clicks to seek

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)

        # Data
        self.actions: list[Any] = []
        self.duration_ms = 0
        self.current_position_ms = 0

        # View state
        self.zoom_level = 1.0
        self.pan_offset = 0.0
        self.min_zoom = 1.0
        self.max_zoom = 50.0

        # Interaction state
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.dragging = False

        # Slider synchronization
        self.reference_slider = None
        self.slider_margins = 0
        self.groove_width = 0

        # Visual settings
        self.background_color = QColor(40, 40, 40)
        self.grid_color = QColor(60, 60, 60)
        self.line_color = QColor(100, 150, 255)
        self.point_color = QColor(255, 255, 255)
        self.current_pos_color = QColor(255, 100, 100)

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

    def load_funscript(self, funscript_data):
        """Load FunScript data."""
        self.actions = funscript_data.get("actions", [])
        if self.actions:
            self.duration_ms = max(action["at"] for action in self.actions)
        else:
            self.duration_ms = 0
        self.reset_view()
        self.update()

    def set_duration(self, duration_ms):
        """Set the video duration."""
        self.duration_ms = duration_ms
        self.reset_view()
        self.update()

    def set_position(self, position_ms):
        """Set the current playback position."""
        self.current_position_ms = position_ms
        self.update()

    def set_reference_slider(self, slider):
        """Set the reference slider for width synchronization."""
        self.reference_slider = slider
        self.update_slider_geometry()

    def update_slider_geometry(self):
        """Update geometry to match the slider's groove area."""
        if not self.reference_slider:
            self.slider_margins = 10
            return

        # Get the slider's groove rectangle
        slider = self.reference_slider
        style = slider.style()

        # Create style option for the slider
        from PySide6.QtWidgets import QStyleOptionSlider

        option = QStyleOptionSlider()
        option.initFrom(slider)
        option.minimum = slider.minimum()
        option.maximum = slider.maximum()
        option.sliderPosition = slider.value()
        option.orientation = slider.orientation()

        # Get the groove rectangle
        groove_rect = style.subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderGroove, slider
        )

        # Calculate margins from widget edge to groove
        self.slider_margins = groove_rect.left()
        self.groove_width = groove_rect.width()

    def reset_view(self):
        """Reset pan and zoom to show full timeline."""
        self.zoom_level = 1.0
        self.pan_offset = 0.0
        self.update()

    def time_to_x(self, time_ms):
        """Convert time to x coordinate, matching slider groove area."""
        if self.duration_ms == 0:
            return 0
        # Apply zoom and pan
        normalized_time = time_ms / self.duration_ms
        zoomed_time = (normalized_time - self.pan_offset) * self.zoom_level

        # Use actual groove dimensions
        return self.slider_margins + (zoomed_time * self.groove_width)

    def x_to_time(self, x):
        """Convert x coordinate to time, matching slider groove area."""
        if self.duration_ms == 0:
            return 0

        # Use actual groove dimensions
        normalized_x = (x - self.slider_margins) / self.groove_width
        time_normalized = (normalized_x / self.zoom_level) + self.pan_offset
        return int(time_normalized * self.duration_ms)

    def position_to_y(self, position):
        """Convert position value (0-100) to y coordinate."""
        # Invert Y axis so 0 is at bottom, 100 at top
        return self.height() - (position / 100.0) * self.height()

    def paintEvent(self, event):
        """Paint the visualizer."""
        # Update slider geometry on each paint to ensure sync
        self.update_slider_geometry()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), self.background_color)

        if not self.actions or self.duration_ms == 0:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No FunScript data loaded")
            return

        # Draw grid lines
        self.draw_grid(painter)

        # Draw FunScript line
        self.draw_funscript_line(painter)

        # Draw current position indicator
        self.draw_current_position(painter)

    def draw_grid(self, painter):
        """Draw background grid."""
        painter.setPen(QPen(self.grid_color, 1))

        # Horizontal lines (position levels)
        for i in range(0, 101, 25):
            y = self.position_to_y(i)
            painter.drawLine(0, y, self.width(), y)

        # Vertical lines (time markers)
        visible_duration = self.duration_ms / self.zoom_level
        time_step = max(1000, visible_duration / 10)  # At least 1 second steps

        start_time = self.pan_offset * self.duration_ms
        end_time = start_time + visible_duration

        current_time = int(start_time / time_step) * time_step
        while current_time <= end_time:
            x = self.time_to_x(current_time)
            if 0 <= x <= self.width():
                painter.drawLine(x, 0, x, self.height())
            current_time += time_step

    def draw_funscript_line(self, painter):
        """Draw the FunScript data as a connected line."""
        if len(self.actions) < 2:
            return

        # Filter actions that are visible in current view
        visible_actions = []
        for action in self.actions:
            x = self.time_to_x(action["at"])
            if -10 <= x <= self.width() + 10:  # Small margin for smooth edges
                visible_actions.append(action)

        if len(visible_actions) < 2:
            return

        # Draw lines between points
        painter.setPen(QPen(self.line_color, 2))
        for i in range(len(visible_actions) - 1):
            action1 = visible_actions[i]
            action2 = visible_actions[i + 1]

            x1 = self.time_to_x(action1["at"])
            y1 = self.position_to_y(action1["pos"])
            x2 = self.time_to_x(action2["at"])
            y2 = self.position_to_y(action2["pos"])

            painter.drawLine(x1, y1, x2, y2)

        # Draw points
        painter.setPen(QPen(self.point_color, 1))
        painter.setBrush(QBrush(self.point_color))
        for action in visible_actions:
            x = self.time_to_x(action["at"])
            y = self.position_to_y(action["pos"])
            painter.drawEllipse(x - 2, y - 2, 4, 4)

    def draw_current_position(self, painter):
        """Draw the current playback position indicator."""
        x = self.time_to_x(self.current_position_ms)
        if 0 <= x <= self.width():
            painter.setPen(QPen(self.current_pos_color, 3))
            painter.drawLine(x, 0, x, self.height())

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_x = event.position().x()
            self.dragging = False

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.mouse_pressed:
            dx = event.position().x() - self.last_mouse_x
            if abs(dx) > 3:  # Start dragging only after significant movement
                self.dragging = True
                # Pan the view
                pan_delta = -dx / self.width() / self.zoom_level
                self.pan_offset = max(
                    0, min(1 - 1 / self.zoom_level, self.pan_offset + pan_delta)
                )
                self.update()
            self.last_mouse_x = event.position().x()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            if self.mouse_pressed and not self.dragging:
                # Single click - seek to position
                time_ms = self.x_to_time(event.position().x())
                time_ms = max(0, min(self.duration_ms, time_ms))
                self.positionChanged.emit(time_ms)
            self.mouse_pressed = False
            self.dragging = False

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        # Get mouse position as a fraction of widget width
        mouse_fraction = event.position().x() / self.width()

        # Calculate the time at mouse position before zoom
        time_at_mouse = self.x_to_time(event.position().x())

        # Apply zoom
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        new_zoom = self.zoom_level * zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom

            # Adjust pan to keep the same time under the mouse
            new_mouse_fraction = self.time_to_x(time_at_mouse) / self.width()
            pan_adjustment = (mouse_fraction - new_mouse_fraction) / self.zoom_level
            self.pan_offset = max(
                0, min(1 - 1 / self.zoom_level, self.pan_offset + pan_adjustment)
            )

            self.update()
