# caliscope/gui/vizualize/playback_triangulation_widget.py

import subprocess
from pathlib import Path
from time import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pyqtgraph.opengl as gl
import rtoml
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColorConstants, QImage
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calipod.annotation_management.annotations_config_manager import (
    STRUCTURE_GEOMETRY_FLAT,
    STRUCTURE_GEOMETRY_SEMI_SPHERE,
)
from calipod.cameras.camera_array import CameraArray
from calipod.core import logger as calipod_logger
from calipod.export import (
    CompareVideoExportWorker,
    FrameCompositor,
    GenericWorker,
    VideoExporter,
    VideoExportProgressDialog,
    VideoExportWorker,
)
from calipod.gui.utils.grids import adaptive_grid_spacing, build_complete_grid_label_specs, build_plane_grid_lines
from calipod.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from calipod.gui.vizualize.filter_parameter_manager import FilterParameterManager
from calipod.gui.vizualize.interactive_3d_graph_window import Interactive3DGraphWindow
from calipod.gui.vizualize.metrics_computer import MetricsComputer
from calipod.motion_trial import MotionTrial
from calipod.trackers.motion_models import ConstantVelocity3DModel

logger = calipod_logger.get(__name__)


class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None, config=None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.config = config
        self.visualizer = TriangulationVisualizer(self.camera_array)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.export_button = QPushButton("Export Triang'd-Sim Video")
        self.export_button.setCheckable(True)

        self.export_compare_button = QPushButton("Export Real Video Compare")

        self.measurement_view_button = QPushButton("Measurement View")

        self.generate_graph_button = QPushButton("Generate Graph")

        self.toggle_frustums_button = QPushButton("Toggle Camera Frustums")
        self.toggle_frustums_button.setCheckable(True)
        self.toggle_frustums_button.setChecked(True)  # Start with frustums visible

        self.compute_metrics_button = QPushButton("Compute Performance Metrics")
        self.toggle_filtered_button = QPushButton("Use Filtered Track")
        self.toggle_filtered_button.setCheckable(True)

        # Reference to hybrid checkbox (will be set from post_processing_widget)
        self.use_hybrid_bgs = None  # Will be assigned the checkbox from post_processing_widget

        # Defaults for video and filtering - will be set from video when loaded
        self.video_framerate = 100  # Initial default, updated when video is loaded
        # Tracking/filter params
        self.filtered_predictions_path: Optional[Path] = None
        self.kalman_process_noise_scale = 0.0490
        self.kalman_measurement_noise_std = 0.00039  # meters (0.390 mm)
        self.kalman_fps_override: Optional[int] = None
        self.gap_fill_only = False  # If True, only interpolate missing frames; if False, smooth all
        self.gate_distance_sigma = 3.1  # Mahalanobis distance gating threshold in sigma (standard deviations)
        self.max_distance_threshold = 10.0  # Fixed distance gating threshold in mm (hybrid gating)
        self.filter_start_frame: Optional[int] = None  # Optional start frame for filtering
        self.filter_end_frame: Optional[int] = None  # Optional end frame for filtering

        # BGS-specific filter parameters (for extended frames with only BGS measurements)
        self.bgs_kalman_process_noise_scale = 0.2
        self.bgs_kalman_measurement_noise_std = 0.0002  # meters (0.2 mm)
        self.bgs_gate_distance_sigma = 9.2
        self.bgs_max_distance_threshold = 28.0

        # Create filter parameter manager early (before UI widget signal connections)
        self.filter_manager = FilterParameterManager(
            kalman_process_noise_scale=self.kalman_process_noise_scale,
            kalman_measurement_noise_std=self.kalman_measurement_noise_std,
            gate_distance_sigma=self.gate_distance_sigma,
            max_distance_threshold=self.max_distance_threshold,
            bgs_kalman_process_noise_scale=self.bgs_kalman_process_noise_scale,
            bgs_kalman_measurement_noise_std=self.bgs_kalman_measurement_noise_std,
            bgs_gate_distance_sigma=self.bgs_gate_distance_sigma,
            bgs_max_distance_threshold=self.bgs_max_distance_threshold,
            gap_fill_only=self.gap_fill_only,
            extend_filtered_track=False,
        )

        # Create metrics computer for handling metrics computation and display
        self.metrics_computer = MetricsComputer(self)

        # UI controls for filter params
        self.process_noise_spin = QDoubleSpinBox()
        self.process_noise_spin.setRange(0.0001, 10.0)
        self.process_noise_spin.setSingleStep(0.01)
        self.process_noise_spin.setDecimals(4)
        self.process_noise_spin.setValue(self.kalman_process_noise_scale)

        self.meas_noise_spin = QDoubleSpinBox()
        self.meas_noise_spin.setRange(0.01, 100.0)
        self.meas_noise_spin.setSingleStep(0.1)
        self.meas_noise_spin.setDecimals(3)
        self.meas_noise_spin.setSuffix(" mm")
        self.meas_noise_spin.setValue(self.kalman_measurement_noise_std * 1000.0)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(self.video_framerate)

        self.gate_distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.gate_distance_slider.setRange(0, 100)  # 0 to 10.0 sigma (divide by 10)
        self.gate_distance_slider.setValue(int(self.gate_distance_sigma * 10))
        self.gate_distance_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gate_distance_slider.setTickInterval(10)  # Tick every 1.0 sigma
        self.gate_distance_slider.setMaximumWidth(150)

        self.gate_distance_label = QLabel(f"{self.gate_distance_sigma:.1f}σ")
        self.gate_distance_slider.valueChanged.connect(self.filter_manager.update_gate_label)

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.1, 5000.0)
        self.max_distance_spin.setSingleStep(10.0)
        self.max_distance_spin.setDecimals(1)
        self.max_distance_spin.setSuffix(" mm")
        self.max_distance_spin.setValue(self.max_distance_threshold)

        # Checkbox for gap-fill only mode
        from PySide6.QtWidgets import QCheckBox

        self.gap_fill_only_checkbox = QCheckBox("Gap-fill only (don't smooth existing)")

        # Checkbox for extending filtered track using available measurements (BGS or continuity)
        self.extend_filtered_track_checkbox = QCheckBox("Extend filter past predictions (using BGS/continuity)")
        self.extend_filtered_track_checkbox.setToolTip(
            "If enabled, extends Kalman filter predictions beyond where YOLO data ends, "
            "using BGS measurements (if available) or filter continuity to fill remaining ground truth frames."
        )
        self.extend_filtered_track = False

        # Spinboxes for filter start/end frames
        self.filter_start_spin = QSpinBox()
        self.filter_start_spin.setRange(0, 999999)
        self.filter_start_spin.setMinimumWidth(80)
        self.filter_start_spin.setMaximumWidth(150)
        self.filter_start_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.filter_end_spin = QSpinBox()
        self.filter_end_spin.setRange(0, 999999)
        self.filter_end_spin.setMinimumWidth(80)
        self.filter_end_spin.setMaximumWidth(150)
        self.filter_end_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        # Checkbox to cut exported video to filter frame range
        self.cut_video_to_filter_frames_checkbox = QCheckBox("Cut video to start & end frames")
        self.cut_video_to_filter_frames_checkbox.setEnabled(False)
        self.cut_video_to_filter_frames_checkbox.setToolTip(
            "When enabled, export will only include frames between the specified start and end frame values"
        )

        # BGS-specific filter parameter controls (disabled until hybrid mode is enabled)
        self.bgs_process_noise_spin = QDoubleSpinBox()
        self.bgs_process_noise_spin.setRange(0.0001, 10.0)
        self.bgs_process_noise_spin.setSingleStep(0.01)
        self.bgs_process_noise_spin.setDecimals(4)
        self.bgs_process_noise_spin.setValue(self.bgs_kalman_process_noise_scale)
        self.bgs_process_noise_spin.setEnabled(False)

        self.bgs_meas_noise_spin = QDoubleSpinBox()
        self.bgs_meas_noise_spin.setRange(0.01, 100.0)
        self.bgs_meas_noise_spin.setSingleStep(0.1)
        self.bgs_meas_noise_spin.setDecimals(3)
        self.bgs_meas_noise_spin.setSuffix(" mm")
        self.bgs_meas_noise_spin.setValue(self.bgs_kalman_measurement_noise_std * 1000.0)
        self.bgs_meas_noise_spin.setEnabled(False)

        self.bgs_gate_distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.bgs_gate_distance_slider.setRange(0, 100)  # 0 to 10.0 sigma (divide by 10)
        self.bgs_gate_distance_slider.setValue(int(self.bgs_gate_distance_sigma * 10))
        self.bgs_gate_distance_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bgs_gate_distance_slider.setTickInterval(10)  # Tick every 1.0 sigma
        self.bgs_gate_distance_slider.setMaximumWidth(150)
        self.bgs_gate_distance_slider.setEnabled(False)

        self.bgs_gate_distance_label = QLabel(f"{self.bgs_gate_distance_sigma:.1f}σ")
        self.bgs_gate_distance_slider.valueChanged.connect(self.filter_manager.update_bgs_gate_label)

        self.bgs_max_distance_spin = QDoubleSpinBox()
        self.bgs_max_distance_spin.setRange(0.1, 5000.0)
        self.bgs_max_distance_spin.setSingleStep(10.0)
        self.bgs_max_distance_spin.setDecimals(1)
        self.bgs_max_distance_spin.setSuffix(" mm")
        self.bgs_max_distance_spin.setValue(self.bgs_max_distance_threshold)
        self.bgs_max_distance_spin.setEnabled(False)

        # Label to display detected video FPS
        self.video_fps_label = QLabel("Video FPS: --")

        self.export_video_mode = False
        self.last_exported_video_path: Optional[Path] = None
        self.interactive_graph_window: Optional[Interactive3DGraphWindow] = None
        # Export and processing threads
        self.export_thread: Optional[QThread] = None
        self.export_worker: Optional[VideoExportWorker] = None
        self.compare_export_thread: Optional[QThread] = None
        self.compare_export_worker: Optional[CompareVideoExportWorker] = None
        self.graph_thread: Optional[QThread] = None
        self.graph_worker: Optional[GenericWorker] = None

        self._session_start_frame: Optional[int] = None
        self._session_end_frame: Optional[int] = None
        self.xyz_history_path: Optional[Path] = None

        self.setMinimumSize(500, 500)

        # Set up filter manager's UI widget references (for enabling/disabling BGS controls)
        self.filter_manager.set_ui_widgets(
            gate_distance_label=self.gate_distance_label,
            bgs_gate_distance_label=self.bgs_gate_distance_label,
            use_hybrid_bgs=self.use_hybrid_bgs,
            bgs_process_noise_spin=self.bgs_process_noise_spin,
            bgs_meas_noise_spin=self.bgs_meas_noise_spin,
            bgs_gate_distance_slider=self.bgs_gate_distance_slider,
            bgs_max_distance_spin=self.bgs_max_distance_spin,
            process_noise_spin=self.process_noise_spin,
            meas_noise_spin=self.meas_noise_spin,
            gate_distance_slider=self.gate_distance_slider,
            max_distance_spin=self.max_distance_spin,
            filter_start_spin=self.filter_start_spin,
            filter_end_spin=self.filter_end_spin,
            gap_fill_only_checkbox=self.gap_fill_only_checkbox,
            extend_filtered_track_checkbox=self.extend_filtered_track_checkbox,
            cut_video_to_filter_frames_checkbox=self.cut_video_to_filter_frames_checkbox,
        )

        self.place_widgets()
        self.connect_widgets()
        if xyz_history_path is not None:
            self.update_motion_trial(xyz_history_path)
        else:
            self.motion_trial = None
            self.slider.setMinimum(0)
            self.slider.setMaximum(0)
            self.slider.setValue(0)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)
        button_row = QHBoxLayout()
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.export_compare_button)
        self.layout().addLayout(button_row)
        view_button_row = QHBoxLayout()
        view_button_row.addWidget(self.measurement_view_button)
        view_button_row.addWidget(self.generate_graph_button)
        self.layout().addLayout(view_button_row)
        metrics_frustum_row = QHBoxLayout()
        metrics_frustum_row.addWidget(self.compute_metrics_button)
        metrics_frustum_row.addWidget(self.toggle_frustums_button)
        self.layout().addLayout(metrics_frustum_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(self.toggle_filtered_button)
        filter_row.addWidget(self.video_fps_label)
        filter_row.addWidget(QLabel("Proc noise"))
        filter_row.addWidget(self.process_noise_spin)
        filter_row.addWidget(QLabel("Meas noise"))
        filter_row.addWidget(self.meas_noise_spin)
        filter_row.addWidget(QLabel("FPS"))
        filter_row.addWidget(self.fps_spin)
        filter_row.addWidget(QLabel("Gate (σ)"))
        filter_row.addWidget(self.gate_distance_slider)
        filter_row.addWidget(self.gate_distance_label)
        filter_row.addWidget(QLabel("Max dist"))
        filter_row.addWidget(self.max_distance_spin)
        filter_row.addStretch()
        self.layout().addLayout(filter_row)

        # Additional filter options row (gap-fill, frame range, extend, and hybrid mode)
        filter_options_row = QHBoxLayout()
        filter_options_row.addWidget(self.gap_fill_only_checkbox)
        filter_options_row.addWidget(QLabel("Start frame"))
        filter_options_row.addWidget(self.filter_start_spin)
        filter_options_row.addWidget(QLabel("End frame"))
        filter_options_row.addWidget(self.filter_end_spin)
        filter_options_row.addWidget(self.cut_video_to_filter_frames_checkbox)
        filter_options_row.addWidget(self.extend_filtered_track_checkbox)

        # Add hybrid checkbox if available (will be set from post_processing_widget)
        if self.use_hybrid_bgs is not None:
            filter_options_row.addWidget(self.use_hybrid_bgs)

        filter_options_row.addStretch()
        self.filter_options_layout = filter_options_row  # Store reference for updating later
        self.layout().addLayout(filter_options_row)

        # BGS-specific filter parameters row (only enabled when hybrid mode is active)
        bgs_filter_row = QHBoxLayout()
        bgs_filter_row.addWidget(QLabel("BGS Filter Params (YOLO+BGS only):"))
        bgs_filter_row.addWidget(QLabel("Proc noise"))
        bgs_filter_row.addWidget(self.bgs_process_noise_spin)
        bgs_filter_row.addWidget(QLabel("Meas noise"))
        bgs_filter_row.addWidget(self.bgs_meas_noise_spin)
        bgs_filter_row.addWidget(QLabel("Gate (σ)"))
        bgs_filter_row.addWidget(self.bgs_gate_distance_slider)
        bgs_filter_row.addWidget(self.bgs_gate_distance_label)
        bgs_filter_row.addWidget(QLabel("Max dist"))
        bgs_filter_row.addWidget(self.bgs_max_distance_spin)
        bgs_filter_row.addStretch()
        self.bgs_filter_row_layout = bgs_filter_row  # Store reference for enabling/disabling
        self.layout().addLayout(bgs_filter_row)

    def update_filter_options_row(self):
        """Update filter options row to include hybrid checkbox after it's been assigned."""
        if self.use_hybrid_bgs is not None and self.filter_options_layout.count() <= 8:
            # Insert hybrid checkbox before the stretch
            self.filter_options_layout.insertWidget(self.filter_options_layout.count() - 1, self.use_hybrid_bgs)
            # Connect hybrid checkbox to toggle BGS filter row via filter manager
            self.use_hybrid_bgs.stateChanged.connect(self.filter_manager.toggle_bgs_filter_row)
            # Update filter manager's reference to use_hybrid_bgs now that it's been assigned
            self.filter_manager.set_ui_widgets(use_hybrid_bgs=self.use_hybrid_bgs)
            # Apply the hybrid state (in case it was already checked)
            self.filter_manager.toggle_bgs_filter_row()

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.slider.valueChanged.connect(self.visualizer.update_segment_lines)
        self.export_button.toggled.connect(self.toggle_export_mode)
        self.export_compare_button.clicked.connect(self.export_real_video_compare)
        self.measurement_view_button.clicked.connect(self.visualizer.toggle_measurement_mode)
        self.generate_graph_button.clicked.connect(self.generate_3d_graph)
        self.compute_metrics_button.clicked.connect(self.compute_and_show_metrics)
        self.toggle_frustums_button.toggled.connect(self.visualizer.toggle_camera_frustums)
        self.toggle_filtered_button.toggled.connect(self.toggle_filtered_track)

    def set_environment_structure_settings(
            self, frame_roi_structures: list[dict], arena_vertices: list[dict], enabled: bool
            ):
        """Pass environment-structure settings from post-processing side panel into visualizers."""
        self.visualizer.set_environment_structure_settings(
            frame_roi_structures=frame_roi_structures,
            arena_vertices=arena_vertices,
            enabled=enabled,
        )

    def toggle_export_mode(self, checked):
        """Toggles video export mode and launches export on main thread with progress dialog."""
        if not checked:
            logger.info("Video export cancelled.")
            self.export_button.setChecked(False)
            return

        # Start export on main thread (GL context required)
        self._start_motion_video_export()

    def _start_motion_video_export(self):
        """Launch motion video export on main thread with modal progress dialog."""
        export_video_path, _, _ = self._derive_export_paths()

        # Ensure output directory exists
        export_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Capture viewport dimensions at export start
        viewport_width = self.visualizer.scene.width()
        viewport_height = self.visualizer.scene.height()
        logger.info(f"Export starting with locked viewport dimensions: {viewport_width}x{viewport_height}")

        self.visualizer.locked_export_width = viewport_width
        self.visualizer.locked_export_height = viewport_height
        self.visualizer.set_export_mode(True)
        self.visualizer.clear_collected_frames()

        # Determine frame range
        exporter = VideoExporter(self.video_framerate)
        export_start_frame = 0
        export_end_frame = 0

        if self._session_start_frame is not None and self._session_end_frame is not None:
            export_start_frame = self._session_start_frame
            export_end_frame = self._session_end_frame
            logger.info(
                f"Exporting video based on full session frame range: {export_start_frame} to {export_end_frame}."
            )
            if self.visualizer.motion_trial is None:
                self.visualizer.motion_trial = MotionTrial()

        elif self.motion_trial and not self.motion_trial.is_empty:
            export_start_frame = self.motion_trial.start_index
            export_end_frame = self.motion_trial.end_index
            logger.info(
                f"Exporting video based on loaded motion trial range: {export_start_frame} to {export_end_frame}."
            )
            self.visualizer.motion_trial = self.motion_trial
        else:
            export_start_frame = 0
            export_end_frame = self.video_framerate * 3
            logger.warning(
                "No valid motion trial or session range available. "
                f"Exporting {export_end_frame - export_start_frame + 1} frames of empty scene."
            )
            if self.visualizer.motion_trial is None:
                self.visualizer.motion_trial = MotionTrial()

        # Override frame range if "cut video to filter frames" is checked
        if self.cut_video_to_filter_frames_checkbox.isChecked():
            filter_start = self.filter_start_spin.value()
            filter_end = self.filter_end_spin.value()
            export_start_frame = filter_start
            export_end_frame = filter_end
            logger.info(f"Overriding export frame range with filter frames: {export_start_frame} to {export_end_frame}")

        # Create modal progress dialog
        self.export_progress_dialog = VideoExportProgressDialog(
            parent=self, title="Exporting Triangulated Video", allow_cancel=True
        )

        # Disable export button during process
        self.export_button.setEnabled(False)

        try:
            logger.info("Starting motion video export on main thread (GL context required)...")
            self.export_progress_dialog.show()

            # Run export on main thread with progress updates
            self._run_motion_video_export_main_thread(exporter, export_start_frame, export_end_frame, export_video_path)

            # Only show "completed" message if not cancelled and dialog still exists
            if not self.export_progress_dialog.cancelled and self.export_progress_dialog.isVisible():
                logger.info("Motion video export completed successfully")
                self.export_progress_dialog.set_status("Export completed!")
        except Exception as e:
            # Only show error if dialog wasn't cancelled and closed
            if not self.export_progress_dialog.cancelled:
                logger.error(f"Motion video export failed: {e}")
                if self.export_progress_dialog.isVisible():
                    self.export_progress_dialog.set_status(f"ERROR: {e}")
                QMessageBox.critical(self, "Export Failed", f"Motion video export failed:\n{e}")
        finally:
            self.visualizer.locked_export_width = None
            self.visualizer.locked_export_height = None
            self.visualizer.set_export_mode(False)
            self.export_button.setEnabled(True)
            self.export_button.setChecked(False)

    def _run_motion_video_export_main_thread(self, exporter, export_start_frame, export_end_frame, output_path):
        """Execute motion video export on main thread with frequent UI updates."""
        total_frames = export_end_frame - export_start_frame + 1

        # Collect frames by iterating through frame range
        frame_count = 0
        for i in range(export_start_frame, export_end_frame + 1):
            # Check if user clicked cancel
            if self.export_progress_dialog.cancelled:
                logger.info("Motion video export cancelled by user")
                self.export_progress_dialog.set_status("Cancelled by user")
                return

            self.visualizer.display_points(i)
            self.visualizer.update_segment_lines(i)

            frame_count += 1

            # Update progress dialog
            self.export_progress_dialog.set_progress(frame_count, total_frames)

            # Keep UI responsive - process events every 5 frames
            if frame_count % 5 == 0:
                QApplication.processEvents()

        logger.info(f"Collected {frame_count} frames for video encoding")

        # Render collected frames to video via FFmpeg
        collected_frames = self.visualizer.get_collected_frames()
        if not collected_frames:
            logger.error("No frames were collected")
            raise RuntimeError("No frames were collected for video export")

        # Update dialog status while encoding
        self.export_progress_dialog.set_status("Encoding video (this may take a moment)...")
        QApplication.processEvents()

        # Render to video
        exporter._render_frames_to_video(collected_frames, output_path, self.video_framerate)

    def _derive_export_paths(self):
        if self.xyz_history_path:
            tracker_suffix = self.xyz_history_path.stem.replace("xyz_", "")
            export_video_path = self.xyz_history_path.parent / f"exported_{tracker_suffix}.mp4"
            compare_video_path = self.xyz_history_path.parent / f"exported_compare_{tracker_suffix}.mp4"
            recording_dir = self.xyz_history_path.parent.parent
        else:
            export_video_path = Path.cwd() / "exported_motion_video.mp4"
            compare_video_path = Path.cwd() / "exported_real_compare.mp4"
            recording_dir = None

        return export_video_path, compare_video_path, recording_dir

    def create_quad_split_video_streaming(
        self, port_video_paths: list[Path], start_frame: int, end_frame: int, output_path: Path, progress_callback=None
    ):
        """Create quad-split video using port videos and on-demand sim frame rendering (no memory pre-collection)."""
        if len(port_video_paths) != 3:
            raise ValueError("Please provide exactly 3 port video paths.")

        caps = []
        opened_meta = []
        for video_path in port_video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                for c in caps:
                    c.release()
                raise IOError(f"Failed to open video: {video_path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            opened_meta.append((video_path, width, height, fps))
            caps.append(cap)

        # Choose a base size from the real videos
        valid_sizes = [(w, h) for (_, w, h, _) in opened_meta if w > 0 and h > 0]
        if valid_sizes:
            quad_width, quad_height = min(valid_sizes, key=lambda wh: wh[0] * wh[1])
        else:
            quad_width, quad_height = 640, 480

        output_resolution = (quad_width * 2, quad_height * 2)
        if output_resolution[0] <= 0 or output_resolution[1] <= 0:
            for cap in caps:
                cap.release()
            raise IOError(f"Invalid output resolution derived for {output_path}: {output_resolution}")

        target_fps = int(opened_meta[0][3]) if opened_meta and opened_meta[0][3] > 0 else (self.video_framerate or 30)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use FFmpeg for compressed output via pipe
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{output_resolution[0]}x{output_resolution[1]}",
            "-r",
            str(target_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-b:v",
            "8000k",  # 8 Mbps bitrate (adjust as needed)
            "-preset",
            "medium",  # medium speed/quality tradeoff
            "-pix_fmt",
            "yuv420p",  # Ensure compatibility
            "-movflags",
            "+faststart",  # Make mp4 streamable
            "-y",  # Overwrite output
            str(output_path),
        ]

        try:
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install FFmpeg or add it to PATH.")
            for cap in caps:
                cap.release()
            raise IOError("FFmpeg is required for video encoding but was not found.")

        frame_count = 0
        total_frames = end_frame - start_frame + 1
        try:
            for sync_idx in range(start_frame, end_frame + 1):
                # Check if user clicked cancel
                if progress_callback and hasattr(self, "compare_progress_dialog"):
                    if self.compare_progress_dialog.cancelled:
                        logger.info("Compare video export cancelled by user")
                        return

                port_frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        port_frames = []
                        break
                    port_frames.append(frame)

                if len(port_frames) < 3:
                    break

                # Render sim frame on-demand (no memory pre-collection)
                self.visualizer.display_points(sync_idx)
                self.visualizer.update_segment_lines(sync_idx)

                # Grab framebuffer
                qimage = self.visualizer.scene.grabFramebuffer()
                sim_frame = self.visualizer.qimage_to_cv2(qimage) if not qimage.isNull() else None

                if sim_frame is None or sim_frame.size == 0:
                    logger.warning(f"Failed to grab sim frame at sync_idx {sync_idx}, skipping")
                    continue

                # Stitch: resize real videos without crop, crop only sim
                resized_frames = [
                    cv2.resize(port_frames[i], (quad_width, quad_height))
                    if i < 3
                    else FrameCompositor.resize_and_center_crop(sim_frame, quad_width, quad_height)
                    for i in range(3)
                ]
                resized_frames.append(FrameCompositor.resize_and_center_crop(sim_frame, quad_width, quad_height))

                top_row = np.hstack((resized_frames[0], resized_frames[1]))
                bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
                combined_frame = np.vstack((top_row, bottom_row))

                # Send to FFmpeg via pipe
                try:
                    proc.stdin.write(combined_frame.tobytes())
                except BrokenPipeError:
                    logger.error("FFmpeg pipe closed unexpectedly")
                    break

                frame_count += 1

                # Emit progress updates if callback provided
                if progress_callback:
                    progress_callback(frame_count, total_frames)

                # Keep UI responsive during long video processing - call on every frame for compare export
                # (compare export runs on main thread, so processEvents is essential)
                if frame_count % 5 == 0:
                    QApplication.processEvents()

            # Close FFmpeg pipe
            logger.info(f"Finished processing {frame_count} frames, finalizing video file...")
            proc.stdin.close()
            QApplication.processEvents()

            # Wait for FFmpeg to finish
            stdout, stderr = proc.communicate(timeout=300)
            if proc.returncode != 0:
                logger.error(f"FFmpeg encoding failed: {stderr.decode() if stderr else 'Unknown error'}")
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
                return

        except Exception as e:
            logger.error(f"Error during video encoding: {e}")
            try:
                proc.stdin.close()
                proc.terminate()
            except Exception:
                pass
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            raise
        finally:
            for cap in caps:
                cap.release()

        QApplication.processEvents()

        if frame_count == 0:
            logger.error(f"No frames written to combined video: {output_path}")
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            return

        logger.info(
            f"Combined video saved to: {output_path} "
            f"({frame_count} frames at {target_fps} fps, base quad {quad_width}x{quad_height})"
        )

    def export_real_video_compare(self):
        """Export compare video (real + sim quad split). Runs on main thread due to GL context requirement."""
        export_video_path, compare_video_path, recording_dir = self._derive_export_paths()

        if recording_dir is None or not recording_dir.exists():
            logger.error("Cannot export compare video without a valid recording directory.")
            return

        port_videos = FrameCompositor.collect_port_videos(recording_dir, expected_count=3)
        if len(port_videos) < 3:
            return

        self.export_compare_button.setEnabled(False)

        # Determine frame range
        export_start_frame = 0
        export_end_frame = 0

        if self._session_start_frame is not None and self._session_end_frame is not None:
            export_start_frame = self._session_start_frame
            export_end_frame = self._session_end_frame
            logger.info(f"Compare export using session frame range: {export_start_frame} to {export_end_frame}")
        elif self.motion_trial and not self.motion_trial.is_empty:
            export_start_frame = self.motion_trial.start_index
            export_end_frame = self.motion_trial.end_index
            logger.info(f"Compare export using motion trial range: {export_start_frame} to {export_end_frame}")
        else:
            export_start_frame = 0
            export_end_frame = self.video_framerate * 3
            logger.warning(f"Compare export using default 3-second range: {export_start_frame} to {export_end_frame}")

        # Override frame range if "cut video to filter frames" is checked
        if self.cut_video_to_filter_frames_checkbox.isChecked():
            filter_start = self.filter_start_spin.value()
            filter_end = self.filter_end_spin.value()
            export_start_frame = filter_start
            export_end_frame = filter_end
            logger.info(
                f"Overriding compare export frame range with filter frames: {export_start_frame} to {export_end_frame}"
            )

        # CRITICAL: Capture viewport dimensions at export start to ensure consistent frame sizes
        viewport_width = self.visualizer.scene.width()
        viewport_height = self.visualizer.scene.height()
        logger.info(f"Compare export starting with locked viewport dimensions: {viewport_width}x{viewport_height}")

        # Store locked dimensions in visualizer to enforce consistent frame sizes
        self.visualizer.locked_export_width = viewport_width
        self.visualizer.locked_export_height = viewport_height

        # Create progress dialog (non-modal, updates as export progresses)
        self.compare_progress_dialog = VideoExportProgressDialog(
            parent=self,
            title="Exporting Compare Video",
            allow_cancel=False,  # Compare export can't be cancelled mid-GL operation
        )
        self.compare_progress_dialog.show()

        try:
            logger.info("Starting compare video export (main thread required for GL context)...")
            # NOTE: Must run on main thread because create_quad_split_video_streaming needs Qt/GL context
            # We show progress dialog and call periodically during rendering
            self.create_quad_split_video_streaming(
                port_videos[:3],
                export_start_frame,
                export_end_frame,
                compare_video_path,
                progress_callback=self._on_compare_export_progress,
            )

            # Only show "completed" message if not cancelled and dialog still exists
            if not self.compare_progress_dialog.cancelled and self.compare_progress_dialog.isVisible():
                logger.info("Compare video export completed successfully")
                self.compare_progress_dialog.set_status("Comparison export completed!")
        except Exception as exc:
            # Only show error if dialog wasn't cancelled and closed
            if not self.compare_progress_dialog.cancelled:
                logger.error(f"Failed to export real video compare: {exc}")
                if self.compare_progress_dialog.isVisible():
                    self.compare_progress_dialog.set_status(f"ERROR: {exc}")
                QMessageBox.critical(self, "Export Failed", f"Compare video export failed:\n{exc}")
        finally:
            # Clear locked dimensions and re-enable button
            self.visualizer.locked_export_width = None
            self.visualizer.locked_export_height = None
            self.export_compare_button.setEnabled(True)

    def _on_compare_export_progress(self, current_frame, total_frames):
        """Callback for compare video export progress updates."""
        self.compare_progress_dialog.set_progress(current_frame, total_frames)

    def update_motion_trial(self, xyz_history_path):
        tic = time()
        logger.info(f"Beginning to load in motion trial: {time()}")

        self.xyz_history_path = xyz_history_path
        self.motion_trial = MotionTrial(xyz_history_path)
        logger.info(f"Motion trial loading complete: {time()} ")
        toc = time()
        logger.info(f"Elapsed time to load: {toc - tic}")

        self.visualizer.update_motion_trial(self.motion_trial)

        if self.xyz_history_path:
            project_config_path = self.xyz_history_path.parent / "config.toml"
            fps_detected = None

            # Try to detect FPS from actual video files in the recording directory
            recording_dir = self.xyz_history_path.parent
            video_files = list(recording_dir.glob("port_*.mp4"))

            if video_files:
                try:
                    cap = cv2.VideoCapture(str(video_files[0]))
                    fps_detected = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    if fps_detected > 0:
                        self.video_framerate = fps_detected
                        # Save the detected FPS to config
                        try:
                            if project_config_path.exists():
                                project_config_data = rtoml.load(project_config_path)
                                project_config_data["fps_recording"] = int(fps_detected)
                                rtoml.dump(project_config_data, project_config_path)
                                logger.info(f"Saved detected video FPS ({fps_detected}) to config.toml")
                        except Exception as e:
                            logger.warning(f"Could not save FPS to config: {e}")
                        logger.info(f"Video framerate detected from video file: {self.video_framerate}")
                except Exception as e:
                    logger.warning(f"Could not detect FPS from video file: {e}")

            # If no video FPS detected, try to read from config
            if not fps_detected:
                try:
                    if project_config_path.exists():
                        project_config_data = rtoml.load(project_config_path)
                        # Try fps_recording first, then fall back to fps_sync_stream_processing
                        self.video_framerate = project_config_data.get("fps_recording") or project_config_data.get(
                            "fps_sync_stream_processing", 60
                        )
                        logger.info(
                            "Video export framerate set from project config "
                            f"({project_config_path}) to: {self.video_framerate}"
                        )
                    else:
                        logger.info(
                            f"Project config.toml not found at {project_config_path}. "
                            f"Using default video framerate: {self.video_framerate}"
                        )
                except Exception as e:
                    logger.error(
                        "Error reading project config.toml for framerate: "
                        f"{e}. Using default video framerate: {self.video_framerate}"
                    )

            # Update the FPS label display
            self.video_fps_label.setText(f"Video FPS: {self.video_framerate:.1f}")
        else:
            logger.warning(
                "Motion trial path not available, cannot load project-specific "
                "config.toml for framerate. Using default."
            )

        self._session_start_frame = None
        self._session_end_frame = None

        if self.xyz_history_path:
            frame_time_history_path = self.xyz_history_path.parent / "frame_time_history.csv"
            try:
                if frame_time_history_path.exists():
                    frame_time_df = pd.read_csv(frame_time_history_path, usecols=["sync_index"], engine="pyarrow")
                    if not frame_time_df.empty and "sync_index" in frame_time_df.columns:
                        self._session_start_frame = int(frame_time_df["sync_index"].min())
                        self._session_end_frame = int(frame_time_df["sync_index"].max())
                        logger.info(
                            f"Session frame range loaded from {frame_time_history_path}: "
                            f"{self._session_start_frame} to {self._session_end_frame}"
                        )
                    else:
                        logger.warning(
                            f"frame_time_history.csv at {frame_time_history_path} is empty "
                            "or missing 'sync_index' column. Cannot determine full "
                            "session frame range."
                        )
                else:
                    logger.warning(
                        f"frame_time_history.csv not found at {frame_time_history_path}. "
                        "Cannot determine full session frame range."
                    )
            except Exception as e:
                logger.error(f"Error reading frame_time_history.csv: {e}. Cannot determine full session frame range.")

        if self._session_start_frame is not None and self._session_end_frame is not None:
            self.slider.setMinimum(self._session_start_frame)
            self.slider.setMaximum(self._session_end_frame)
            self.slider.setValue(self._session_start_frame)
        elif self.motion_trial.is_empty:
            self.slider.setMinimum(0)
            self.slider.setMaximum(0)
            self.slider.setValue(0)
        else:
            self.slider.setMinimum(self.motion_trial.start_index)
            self.slider.setMaximum(self.motion_trial.end_index)
            self.slider.setValue(self.motion_trial.start_index)

        # Try to load filter metadata if available
        # MotionTrial auto-detects and prefers filtered predictions if they exist
        metadata_loaded = self.filter_manager.load_metadata(
            Path(self.motion_trial.predictions_csv) if self.motion_trial.predictions_csv else None
        )

        # If filter metadata was loaded, automatically enable filtering in the GUI
        # and capture the loaded parameters so we don't unnecessarily recompute
        if metadata_loaded:
            logger.info("Filter metadata was loaded; automatically enabling 'Use Filtered Track' toggle")
            # Capture the loaded parameters so we know they match the cached filtered CSV
            # IMPORTANT: kalman_fps_override is loaded from metadata file (not hardcoded to None)
            fps_override_from_meta = self.filter_manager.kalman_fps_override
            self._last_computed_params = {
                "kalman_process_noise_scale": self.filter_manager.kalman_process_noise_scale,
                "kalman_measurement_noise_std": self.filter_manager.kalman_measurement_noise_std,
                "kalman_fps_override": fps_override_from_meta,
                "gap_fill_only": self.filter_manager.gap_fill_only,
                "extend_filtered_track": self.filter_manager.extend_filtered_track,
                "gate_distance_sigma": self.filter_manager.gate_distance_sigma,
                "max_distance_threshold": self.filter_manager.max_distance_threshold,
                "bgs_kalman_process_noise_scale": self.filter_manager.bgs_kalman_process_noise_scale,
                "bgs_kalman_measurement_noise_std": self.filter_manager.bgs_kalman_measurement_noise_std,
                "bgs_gate_distance_sigma": self.filter_manager.bgs_gate_distance_sigma,
                "bgs_max_distance_threshold": self.filter_manager.bgs_max_distance_threshold,
            }
            logger.info(
                f"Loaded _last_computed_params from metadata file with kalman_fps_override={fps_override_from_meta}"
            )
            # Enable the toggle, which will trigger apply the extend logic and update metadata
            self.toggle_filtered_button.setChecked(True)

    def update_camera_array(self, camera_array: CameraArray):
        self.visualizer.update_camera_array(camera_array)

    def compute_and_show_metrics(self):
        """Display metrics from MotionTrial.performance_metrics in a scrollable dialog, and save metadata."""
        if self.motion_trial is None:
            return
        use_filtered = self.toggle_filtered_button.isChecked()
        # Compute metrics and get them for reuse in metadata saving
        metrics, metrics_by_source = self.metrics_computer.compute_and_display(self.motion_trial, use_filtered)

        # Save metadata after computing metrics; pass pre-computed metrics to avoid redundant recomputation
        if use_filtered and self.motion_trial.predictions_csv:
            try:
                filtered_path = Path(self.motion_trial.predictions_csv)
                filtered_df = pd.read_csv(filtered_path, engine="pyarrow") if filtered_path.exists() else None

                # Get fps_override: try from current session, fallback to metadata file to preserve value
                fps_override = getattr(self, "kalman_fps_override", None)
                if fps_override is None:
                    try:
                        metadata_path = filtered_path.with_suffix(".json").with_stem(filtered_path.stem + "_metadata")
                        if metadata_path.exists():
                            import json

                            with open(metadata_path, "r") as f:
                                existing_metadata = json.load(f)
                                fps_override = existing_metadata.get("kalman_fps_override")
                    except Exception as e:
                        logger.debug(f"Could not load metadata for fps_override: {e}")

                self.filter_manager.save_metadata(
                    filtered_path,
                    self.motion_trial,
                    filtered_df=filtered_df,
                    kalman_fps_override=fps_override,
                    precomputed_metrics=metrics,
                    precomputed_metrics_by_source=metrics_by_source,
                )
            except Exception as e:
                logger.warning(f"Failed to save metadata after metrics computation: {e}")

    def toggle_filtered_track(self, checked: bool):
        """Toggle use of filtered predictions cached alongside the predictions CSV."""
        if self.motion_trial is None or self.motion_trial.is_empty:
            QMessageBox.warning(self, "No Data", "Load a motion trial before enabling filtering.")
            self.toggle_filtered_button.setChecked(False)
            return

        if not hasattr(self.motion_trial, "predictions_df") or self.motion_trial.predictions_df.empty:
            QMessageBox.warning(self, "No Predictions", "Predictions are empty; load predictions before filtering.")
            self.toggle_filtered_button.setChecked(False)
            return

        if not hasattr(self.motion_trial, "predictions_csv") or self.motion_trial.predictions_csv is None:
            QMessageBox.warning(self, "Missing Path", "Predictions path is unknown; cannot cache filtered output.")
            self.toggle_filtered_button.setChecked(False)
            return

        base_pred_path = Path(self.motion_trial.predictions_csv)
        # Strip any existing "_filtered" suffix to avoid chaining
        base_stem = base_pred_path.stem
        while base_stem.endswith("_filtered"):
            base_stem = base_stem[:-9]
        raw_pred_path = base_pred_path.with_name(f"{base_stem}{base_pred_path.suffix}")
        filtered_path = raw_pred_path.with_name(f"{base_stem}_filtered{base_pred_path.suffix}")

        if checked:
            self.toggle_filtered_button.setEnabled(False)
            # Enable the "cut video to filter frames" checkbox when filtering is enabled
            self.cut_video_to_filter_frames_checkbox.setEnabled(True)
            try:
                # Capture current UI parameter values to check if recomputation is needed
                current_params = {
                    "kalman_process_noise_scale": float(self.process_noise_spin.value()),
                    "kalman_measurement_noise_std": float(self.meas_noise_spin.value()) / 1000.0,
                    "kalman_fps_override": int(self.fps_spin.value()) if self.fps_spin.value() > 0 else None,
                    "gap_fill_only": self.gap_fill_only_checkbox.isChecked(),
                    "extend_filtered_track": self.extend_filtered_track_checkbox.isChecked(),
                    "gate_distance_sigma": float(self.gate_distance_slider.value()) / 10.0,
                    "max_distance_threshold": float(self.max_distance_spin.value()),
                    "bgs_kalman_process_noise_scale": float(self.bgs_process_noise_spin.value()),
                    "bgs_kalman_measurement_noise_std": float(self.bgs_meas_noise_spin.value()) / 1000.0,
                    "bgs_gate_distance_sigma": float(self.bgs_gate_distance_slider.value()) / 10.0,
                    "bgs_max_distance_threshold": float(self.bgs_max_distance_spin.value()),
                }

                # Check if we should recompute or load cached filtered CSV
                force_recompute = True

                if filtered_path.exists() and hasattr(self, "_last_computed_params") and self._last_computed_params:
                    # Check if current params match the loaded params
                    params_match = all(
                        current_params.get(key) == self._last_computed_params.get(key) for key in current_params.keys()
                    )

                    if params_match:
                        force_recompute = False

                if force_recompute:
                    logger.info("\n" + "=" * 80)
                    logger.info("COMPUTING FILTERED PREDICTIONS")
                    logger.info("=" * 80 + "\n")
                    # Set spinbox ranges based on actual prediction data
                    pred_df = self.motion_trial.predictions_df
                    if pred_df is not None and not pred_df.empty:
                        int(pred_df["sync_index"].min())
                        max_frame = int(pred_df["sync_index"].max())

                        # Allow spinbox range to extend to ground truth end if available (for extend filter logic)
                        spinbox_max = max_frame
                        if self.motion_trial is not None and hasattr(self.motion_trial, "end_index"):
                            spinbox_max = max(max_frame, self.motion_trial.end_index)

                        self.filter_start_spin.setRange(0, spinbox_max)
                        self.filter_end_spin.setRange(0, spinbox_max)

                    # Capture UI parameter values
                    self.kalman_process_noise_scale = float(self.process_noise_spin.value())
                    self.kalman_measurement_noise_std = float(self.meas_noise_spin.value()) / 1000.0  # mm -> m
                    self.kalman_fps_override = int(self.fps_spin.value()) if self.fps_spin.value() > 0 else None
                    self.gap_fill_only = self.gap_fill_only_checkbox.isChecked()
                    self.extend_filtered_track = self.extend_filtered_track_checkbox.isChecked()
                    logger.debug(
                        f"In toggle_filtered_track: read extend_filtered_track_checkbox = {self.extend_filtered_track}"
                    )
                    self.gate_distance_sigma = float(self.gate_distance_slider.value()) / 10.0
                    self.max_distance_threshold = float(self.max_distance_spin.value())  # mm

                    # Capture BGS-specific parameters
                    self.bgs_kalman_process_noise_scale = float(self.bgs_process_noise_spin.value())
                    self.bgs_kalman_measurement_noise_std = float(self.bgs_meas_noise_spin.value()) / 1000.0  # mm -> m
                    self.bgs_gate_distance_sigma = float(self.bgs_gate_distance_slider.value()) / 10.0
                    self.bgs_max_distance_threshold = float(self.bgs_max_distance_spin.value())  # mm

                    # Capture filter start/end frame spinbox values
                    self.filter_start_frame = self.filter_start_spin.value()
                    self.filter_end_frame = self.filter_end_spin.value()
                    logger.debug(
                        "Captured filter frame range from spinboxes: "
                        f"start={self.filter_start_frame}, end={self.filter_end_frame}"
                    )

                    mode_label = "gap-fill-only" if self.gap_fill_only else "full-smooth"
                    extend_label = "yes" if self.extend_filtered_track else "no"
                    fps_value = self.kalman_fps_override or self.video_framerate or 60
                    msg = (
                        "Beginning Kalman filter computation ("
                        f"process_noise={self.kalman_process_noise_scale}, "
                        f"meas_noise={self.kalman_measurement_noise_std * 1000:.2f}mm, "
                        f"fps={fps_value}, gate={self.gate_distance_sigma:.1f}σ, "
                        f"max_dist={self.max_distance_threshold:.1f}mm, "
                        f"mode={mode_label}, extend_past_pred={extend_label})"
                    )
                    logger.info(msg)

                    # Check if hybrid YOLO+BGS mode is enabled
                    use_hybrid = False
                    if hasattr(self, "use_hybrid_bgs") and self.use_hybrid_bgs:
                        use_hybrid = self.use_hybrid_bgs.isChecked()
                        if use_hybrid:
                            logger.info("Using hybrid YOLO+BGS measurement selection")

                    filtered_df = self._compute_filtered_predictions(use_hybrid=use_hybrid)

                    if filtered_df is None or filtered_df.empty:
                        err_msg = "Kalman filter computation failed: empty result"
                        logger.error(err_msg)
                        QMessageBox.warning(
                            self, "Filter Error", "Filtered predictions are empty; keeping raw predictions."
                        )
                        self.toggle_filtered_button.setChecked(False)
                        self.toggle_filtered_button.setEnabled(True)
                        return

                    # Log measurement source distribution
                    if "measurement_source" in filtered_df.columns:
                        src_counts = filtered_df["measurement_source"].value_counts().to_dict()
                        logger.info(
                            "Measurement source distribution: "
                            + ", ".join(f"{src}: {cnt}" for src, cnt in sorted(src_counts.items()))
                        )

                    logger.info(f"Computed {len(filtered_df)} frames")

                    filtered_path.parent.mkdir(parents=True, exist_ok=True)
                    filtered_df.to_csv(filtered_path, index=False)
                    logger.info(f"Saved filtered predictions to {filtered_path}")

                    # Save metadata immediately after computing to preserve parameters
                    try:
                        self.filter_manager.save_metadata(
                            filtered_path,
                            self.motion_trial,
                            filtered_df=filtered_df,
                            kalman_fps_override=self.kalman_fps_override,
                        )
                    except Exception as e:
                        logger.warning(f"Could not save metadata after filter computation: {e}")

                    # Store parameters for cache detection on next toggle
                    self._last_computed_params = current_params

                    self.motion_trial.predictions_csv = filtered_path
                    self.motion_trial.predictions_df = pd.read_csv(filtered_path, engine="pyarrow")
                    self.filtered_predictions_path = filtered_path

                else:
                    logger.info("\n" + "=" * 80)
                    logger.info("LOADING CACHED FILTERED PREDICTIONS")
                    logger.info("=" * 80 + "\n")

                    # Restore UI spinbox values to match cached parameters
                    if self._last_computed_params:
                        cached_fps = self._last_computed_params.get("kalman_fps_override")
                        if cached_fps is None:
                            self.fps_spin.setValue(0)
                        else:
                            self.fps_spin.setValue(int(cached_fps))
                        self.process_noise_spin.setValue(
                            self._last_computed_params.get("kalman_process_noise_scale", 1.0)
                        )
                        meas_noise_mm = self._last_computed_params.get("kalman_measurement_noise_std", 0.01) * 1000.0
                        self.meas_noise_spin.setValue(int(meas_noise_mm))

                    self.motion_trial.predictions_csv = filtered_path
                    self.motion_trial.predictions_df = pd.read_csv(filtered_path, engine="pyarrow")
                    self.filtered_predictions_path = filtered_path

                    # Log measurement source distribution
                    if "measurement_source" in self.motion_trial.predictions_df.columns:
                        src_counts = self.motion_trial.predictions_df["measurement_source"].value_counts().to_dict()
                        logger.info(
                            "Measurement source distribution (cached): "
                            + ", ".join(f"{src}: {cnt}" for src, cnt in sorted(src_counts.items()))
                        )

                # Refresh display with filtered predictions
                if hasattr(self.visualizer, "update_motion_trial"):
                    self.visualizer.update_motion_trial(self.motion_trial)

                logger.info("=" * 80)
            except Exception as exc:
                logger.error(f"Failed to enable filtered track: {exc}", exc_info=True)
                QMessageBox.critical(self, "Filter Error", f"Failed to filter predictions:\n{exc}")
                self.toggle_filtered_button.setChecked(False)
                self.toggle_filtered_button.setEnabled(True)
                return
            finally:
                self.toggle_filtered_button.setEnabled(True)
        else:
            # revert to raw predictions
            self.toggle_filtered_button.setEnabled(False)
            # Disable the "cut video to filter frames" checkbox when filtering is disabled
            self.cut_video_to_filter_frames_checkbox.setEnabled(False)
            try:
                if raw_pred_path.exists():
                    self.motion_trial.predictions_df = pd.read_csv(raw_pred_path, engine="pyarrow")
                    self.motion_trial.predictions_csv = raw_pred_path
                    logger.info(f"Reverted to raw predictions at {raw_pred_path}")

                    # Refresh display with raw predictions
                    if hasattr(self.visualizer, "update_motion_trial"):
                        self.visualizer.update_motion_trial(self.motion_trial)
                    logger.info("Filter toggle complete; UI refreshed with raw predictions")
            except Exception as exc:
                logger.error(f"Failed to reload raw predictions: {exc}", exc_info=True)
                QMessageBox.critical(self, "Filter Error", f"Failed to reload raw predictions:\n{exc}")
                self.toggle_filtered_button.setChecked(True)
                self.toggle_filtered_button.setEnabled(True)
                return
            finally:
                self.toggle_filtered_button.setEnabled(True)

        # Note: visualization refresh already happened above for both checked and unchecked cases

    def _compute_filtered_predictions(self, use_hybrid: bool = False) -> Optional[pd.DataFrame]:
        """Apply an RTS (Forward-Backward) CV Kalman filter to point_id==0 predictions, filling gaps and smoothing.

        If use_hybrid=True and BGS predictions available, select best measurement (YOLO or BGS)
        at each frame based on which is closest to the Kalman predicted position.

        If self.extend_filtered_track=True, extends filtering beyond the YOLO prediction range
        using available BGS measurements or filter continuity to reach the ground truth end frame.

        Other point_ids from the original predictions are preserved unchanged.
        Returns a DataFrame with the same schema as predictions:
        sync_index, point_id, x_coord, y_coord, z_coord.
        """
        pred_df = self.motion_trial.predictions_df
        if pred_df is None or pred_df.empty:
            return None

        # Ensure point_id is numeric to avoid string comparison errors
        pred_df = pred_df.copy()
        pred_df["point_id"] = pd.to_numeric(pred_df["point_id"], errors="coerce")

        fly_df = pred_df[pred_df["point_id"] == 0].copy()
        if fly_df.empty:
            logger.warning("No point_id==0 rows in predictions; skipping filter.")
            return None

        fly_df = fly_df.sort_values("sync_index")
        # Use prediction frame range, potentially extended if option is enabled
        pred_fly_frames = fly_df["sync_index"].unique()
        frames = sorted(pred_fly_frames)

        # Determine the ground truth end frame from available sources
        ground_truth_end = None
        if self._session_end_frame is not None:
            ground_truth_end = self._session_end_frame
        elif self.motion_trial is not None and hasattr(self.motion_trial, "end_index"):
            ground_truth_end = self.motion_trial.end_index

        # If extend option enabled, expand frame range to include ground truth frames beyond predictions
        if self.extend_filtered_track and ground_truth_end is not None:
            pred_max = frames[-1] if frames else 0
            if ground_truth_end > pred_max:
                logger.info(
                    f"Extending filter from prediction end frame {pred_max} "
                    f"to ground truth end frame {ground_truth_end}"
                )
                frames = list(range(frames[0], ground_truth_end + 1))
            else:
                logger.info(f"Ground truth ends at {ground_truth_end}, already within prediction range {pred_max}")
        elif self.extend_filtered_track:
            logger.warning(
                "Extend filter option enabled but ground truth end frame not available; using prediction range only"
            )

        # Apply optional start/end frame filtering only if explicitly set
        start_frame = self.filter_start_spin.value()
        end_frame = self.filter_end_spin.value()
        logger.info(
            f"Filter frame range being applied: start={start_frame}, end={end_frame}, "
            f"before filtering had {len(frames)} frames"
        )

        if start_frame > 0:
            frames = [f for f in frames if f >= start_frame]
        if end_frame > 0:
            frames = [f for f in frames if f <= end_frame]

        logger.info(f"After applying frame range filters (start={start_frame}, end={end_frame}): {len(frames)} frames")

        fps_used = self.kalman_fps_override or self.video_framerate or 60
        base_dt = 1.0 / fps_used
        model = ConstantVelocity3DModel(self.kalman_process_noise_scale)

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * (self.kalman_measurement_noise_std**2)

        # INITIALIZATION: Start with high uncertainty to handle bad initial points
        first_row = fly_df.iloc[0]
        state = np.zeros(6)
        state[:3] = [first_row.x_coord, first_row.y_coord, first_row.z_coord]
        P = np.eye(6) * 10.0  # Increased from 1e-3 to allow the gate to find the object

        measurements = {int(r.sync_index): np.array([r.x_coord, r.y_coord, r.z_coord]) for r in fly_df.itertuples()}

        # If hybrid mode enabled, try to load BGS predictions for backup measurements
        bgs_measurements = {}
        if use_hybrid:
            try:
                # xyz_history_path is like: /path/to/recording/FLY/xyz_FLY_predictions.csv
                # So parent is: /path/to/recording/FLY
                # We want: /path/to/recording/FLY/bgs/xyz_FLY_bgs_predictions.csv
                if hasattr(self, "xyz_history_path") and self.xyz_history_path:
                    bgs_path = self.xyz_history_path.parent / "bgs" / "xyz_FLY_bgs_predictions.csv"
                    if bgs_path.exists():
                        bgs_df = pd.read_csv(bgs_path, engine="pyarrow")
                        # Ensure point_id is numeric
                        bgs_df["point_id"] = pd.to_numeric(bgs_df["point_id"], errors="coerce")
                        bgs_fly = bgs_df[bgs_df["point_id"] == 0]
                        bgs_measurements = {
                            int(r.sync_index): np.array([r.x_coord, r.y_coord, r.z_coord])
                            for r in bgs_fly.itertuples()
                            if pd.notna(r.x_coord) and pd.notna(r.y_coord) and pd.notna(r.z_coord)
                        }
                        logger.info(f"Loaded {len(bgs_measurements)} BGS measurements for hybrid selection")
                    else:
                        logger.info(f"BGS predictions file not found at: {bgs_path}")
                else:
                    logger.info("xyz_history_path not available; cannot load BGS predictions for hybrid mode")
            except Exception as e:
                logger.warning(f"Could not load BGS predictions for hybrid mode: {e}")

        # Buffers for RTS Backward Pass
        states_pred = []  # x_{k|k-1}
        covs_pred = []  # P_{k|k-1}
        states_filt = []  # x_{k|k}
        covs_filt = []  # P_{k|k}
        transitions = []  # A matrices (since dt varies)
        measurement_sources = []  # Track which source was used for each frame (YOLO, BGS, or filter_only)

        # --- FORWARD PASS ---
        prev_frame = frames[0]
        consecutive_rejections = 0

        for frame in frames:
            gap = max(frame - prev_frame, 1)
            dt = gap * base_dt
            mats = model.calc_for_dt(dt)
            A, Q = mats["transition_model"], mats["transition_noise_covariance"]

            # Predict
            state_p = A @ state
            P_p = A @ P @ A.T + Q

            # Store prediction
            states_pred.append(state_p.copy())
            covs_pred.append(P_p.copy())
            transitions.append(A.copy())

            updated = False

            # Hybrid measurement selection: pick best available measurement
            # (YOLO or BGS) based on distance to prediction.
            z = None
            z_source = None

            if use_hybrid:
                # Collect available measurements for this frame
                available = []
                if frame in measurements and pd.notna(measurements[frame][0]):
                    available.append(("YOLO", measurements[frame]))
                if frame in bgs_measurements and pd.notna(bgs_measurements[frame][0]):
                    available.append(("BGS", bgs_measurements[frame]))

                # If both available, pick the one closest to Kalman prediction
                if available:
                    if len(available) == 2:
                        # Both YOLO and BGS available - pick closest to prediction
                        yolo_z = available[0][1]
                        bgs_z = available[1][1]
                        yolo_dist = np.linalg.norm(yolo_z - H @ state_p)
                        bgs_dist = np.linalg.norm(bgs_z - H @ state_p)

                        if bgs_dist < yolo_dist:
                            z = bgs_z
                            z_source = "BGS"
                        else:
                            z = yolo_z
                            z_source = "YOLO"
                    else:
                        # Only one available
                        z_source = available[0][0]
                        z = available[0][1]
            else:
                # Normal mode: use YOLO only
                if frame in measurements:
                    z = measurements[frame]
                    z_source = "YOLO"

            if z is not None:
                y_res = z - H @ state_p

                # Use appropriate measurement noise covariance based on measurement source
                if z_source == "BGS" and frame not in measurements:
                    # BGS-only measurement: use BGS-specific noise covariance
                    R_meas = np.eye(3) * (self.bgs_kalman_measurement_noise_std**2)
                else:
                    # YOLO measurement: use standard noise covariance
                    R_meas = R

                S = H @ P_p @ H.T + R_meas

                try:
                    S_inv = np.linalg.inv(S)
                    mahal_dist = np.sqrt(y_res @ S_inv @ y_res)
                    euclidean_dist_mm = np.linalg.norm(y_res) * 1000.0

                    # Use BGS-specific gating thresholds if this is a BGS-only measurement
                    if z_source == "BGS" and frame not in measurements:
                        # BGS-only frame: use BGS-specific thresholds
                        mahal_threshold = self.bgs_gate_distance_sigma
                        dist_threshold = self.bgs_max_distance_threshold
                    else:
                        # YOLO frame (or hybrid with YOLO): use standard thresholds
                        mahal_threshold = self.gate_distance_sigma
                        dist_threshold = self.max_distance_threshold

                    mahal_pass = mahal_dist <= mahal_threshold
                    dist_pass = euclidean_dist_mm <= dist_threshold

                    if mahal_pass and dist_pass:
                        K = P_p @ H.T @ S_inv
                        state = state_p + K @ y_res
                        P = (np.eye(6) - K @ H) @ P_p
                        updated = True
                        consecutive_rejections = 0
                    else:
                        consecutive_rejections += 1
                        # Logic: If we miss too many points, the filter is likely anchored to noise.
                        # For BGS-only measurements (extended frames), use prediction instead of resetting
                        if consecutive_rejections > 5:
                            if z_source == "BGS" and frame not in measurements:
                                # For extended BGS frames, continue with prediction
                                # instead of resetting to noisy measurement.
                                logger.debug(
                                    f"Frame {frame}: BGS measurement rejected too many times; "
                                    "using filter prediction instead of resetting"
                                )
                                consecutive_rejections = 0
                            else:
                                # For YOLO frames, use rescue logic (reset to measurement)
                                state = np.zeros(6)
                                state[:3] = z
                                P = np.eye(6) * 10.0
                                consecutive_rejections = 0
                                updated = True
                except np.linalg.LinAlgError:
                    pass

            if not updated:
                state, P = state_p, P_p
                z_source = "filter_only"  # No measurement was used

            states_filt.append(state.copy())
            covs_filt.append(P.copy())
            measurement_sources.append(z_source)  # Track which source was used
            prev_frame = frame

        # --- BACKWARD PASS (RTS Smoothing) ---
        smoothed_states = [None] * len(frames)
        smoothed_states[-1] = states_filt[-1]

        # Iterate backwards from second-to-last frame
        for k in range(len(frames) - 2, -1, -1):
            # We need the prediction for k+1 that was made FROM k
            A_next = transitions[k + 1]
            x_filt_k = states_filt[k]
            P_filt_k = covs_filt[k]
            x_pred_next = states_pred[k + 1]
            P_pred_next = covs_pred[k + 1]

            # Smoother Gain
            C = P_filt_k @ A_next.T @ np.linalg.inv(P_pred_next)

            # Smooth the state
            smoothed_states[k] = x_filt_k + C @ (smoothed_states[k + 1] - x_pred_next)

        # --- RECONSTRUCT DATAFRAME ---
        results = []
        for i, frame in enumerate(frames):
            s = smoothed_states[i]
            z_src = measurement_sources[i]

            # Compute error against ground truth if available
            error_mm = np.nan
            if (
                self.motion_trial
                and hasattr(self.motion_trial, "ground_truth")
                and self.motion_trial.ground_truth is not None
            ):
                gt_df = self.motion_trial.ground_truth[self.motion_trial.ground_truth["sync_index"] == frame]
                if not gt_df.empty:
                    gt_row = gt_df.iloc[0]
                    gt_pos = np.array([gt_row.x_coord, gt_row.y_coord, gt_row.z_coord])
                    error_mm = np.linalg.norm(s[:3] - gt_pos) * 1000.0

            # If gap_fill_only is True, we only use smoothed values where measurements were missing
            if self.gap_fill_only and frame in measurements:
                z = measurements[frame]
                results.append((frame, 0, z[0], z[1], z[2], "YOLO", error_mm))
            else:
                results.append((frame, 0, s[0], s[1], s[2], z_src, error_mm))

        filtered_fly_df = pd.DataFrame(
            results,
            columns=["sync_index", "point_id", "x_coord", "y_coord", "z_coord", "measurement_source", "error_mm"],
        )
        other_points_df = pred_df[pred_df["point_id"] != 0].copy()
        # Add missing columns to other_points_df to match schema
        other_points_df["measurement_source"] = "other_points"
        other_points_df["error_mm"] = np.nan
        combined_df = pd.concat([filtered_fly_df, other_points_df], ignore_index=True)
        return combined_df.sort_values(["sync_index", "point_id"]).reset_index(drop=True)

    def generate_3d_graph(self):
        """Generate an interactive 3D matplotlib window for trajectory visualization."""
        if self.motion_trial is None or self.motion_trial.is_empty:
            logger.warning("No motion trial loaded; cannot generate graph.")
            return

        try:
            import importlib.util

            if importlib.util.find_spec("matplotlib") is None:
                raise ImportError("matplotlib is not installed")
            if importlib.util.find_spec("mpl_toolkits.mplot3d.art3d") is None:
                raise ImportError("mpl_toolkits.mplot3d is not installed")
        except ImportError:
            logger.error("Matplotlib not available; cannot generate graph.")
            return

        # Close old window if it exists
        if self.interactive_graph_window is not None:
            self.interactive_graph_window.close()

        # Create new interactive window
        is_filtered = self.toggle_filtered_button.isChecked() if hasattr(self, "toggle_filtered_button") else False
        self.interactive_graph_window = Interactive3DGraphWindow(
            self.motion_trial,
            self.camera_array,
            self.xyz_history_path,
            is_filtered,
            frame_roi_structures=self.visualizer.frame_roi_structures,
            arena_vertices=self.visualizer.arena_vertices,
            show_environment_structures=self.visualizer.environment_structures_enabled,
        )
        self.interactive_graph_window.show()
        logger.info("Interactive 3D graph window opened.")


class TriangulationVisualizer:
    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.default_scatter_color = (1, 1, 1, 1)  # White
        self.default_mesh_color = (1, 1, 1, 1)  # White
        self.point_size = 0.005  # Shared size for track markers

        # Measurement-grid state must exist before first build_scene call.
        self.grid_labels = []
        self.axis_labels = []
        self.is_measurement_mode_active = False
        self.measurement_grid_items = []
        self.measurement_grid_extent_m = 10.0

        self.build_scene()
        self.export_video_mode = False
        self.collected_frames = []
        self.motion_trial: Optional[MotionTrial] = None

        # Locked viewport dimensions during export to prevent frame size variation
        self.locked_export_width: Optional[int] = None
        self.locked_export_height: Optional[int] = None

        # Storage for custom mesh items for special labels
        self.custom_mesh_items = []  # Store references to added mesh items
        self.environment_structures_enabled = True
        self.frame_roi_structures = []
        self.arena_vertices = []

    def set_environment_structure_settings(
        self,
        frame_roi_structures: list[dict],
        arena_vertices: list[dict],
        enabled: bool,
    ):
        """Update runtime environment rendering settings from UI."""
        self.frame_roi_structures = frame_roi_structures or []
        self.arena_vertices = arena_vertices or []
        self.environment_structures_enabled = bool(enabled)

    def build_scene(self):
        if hasattr(self, "scene"):
            logger.info("Clearing scene in capture volume visualizer")
            self.scene.clear()
        else:
            logger.info("Creating initial scene in capture volume visualizer")
            self.scene = gl.GLViewWidget()
            self.scene.setCameraPosition(distance=4)

        # Default background color
        self.scene.setBackgroundColor(QColorConstants.Black)

        axis = gl.GLAxisItem()

        self.scene.addItem(axis)

        self._clear_measurement_grid_items()
        self.clear_grid_labels()
        if self.is_measurement_mode_active:
            self._build_measurement_grid_items()
            self.add_grid_labels(
                grid_total_extent_m=self.measurement_grid_extent_m,
                label_interval_m=self._get_measurement_grid_major_spacing_m(),
                text_color="white",
            )

        if self.camera_array.all_extrinsics_calibrated():
            self.meshes = {}
            self.origin_points = {}
            for port, cam in self.camera_array.cameras.items():
                mesh, origin_point = mesh_from_camera(cam)
                mesh: CameraMesh = mesh
                origin_point: CameraMesh = origin_point
                origin_point.setColor(cam.color)
                self.meshes[port] = mesh
                self.origin_points[port] = origin_point
                self.scene.addItem(mesh)
                self.scene.addItem(origin_point)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3)),  # Start with empty array instead of None
            color=self.default_scatter_color,  # Set initial scatter color
            size=self.point_size,
            pxMode=False,
        )
        self.scatter.setGLOptions("opaque")  # keep points solid
        self.scatter.setVisible(False)  # Hide until we have data

        # Dedicated overlay for fly center points to stay visible over translucent meshes
        self.fly_overlay = gl.GLScatterPlotItem(
            pos=np.empty((0, 3)),
            color=(1, 1, 1, 1),
            size=self.point_size,
            pxMode=False,
        )
        self.fly_overlay.setGLOptions("additive")
        self.fly_overlay.setVisible(False)

        # Overlay for prediction fly points (orange)
        self.pred_overlay = gl.GLScatterPlotItem(
            pos=np.empty((0, 3)),
            color=(1, 0.5, 0, 1),
            size=self.point_size * 1.3,  # Slightly larger to render over ground truth
            pxMode=False,
        )
        self.pred_overlay.setGLOptions("translucent")  # Render on top with depth write disabled
        self.pred_overlay.setVisible(False)

        self.segments = {}
        self.scene.addItem(self.scatter)
        self.scene.addItem(self.fly_overlay)
        self.scene.addItem(self.pred_overlay)

    def update_camera_array(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()

    def toggle_camera_frustums(self, checked: bool):
        """Toggle visibility of camera frustum meshes."""
        if hasattr(self, "meshes") and hasattr(self, "origin_points"):
            for mesh in self.meshes.values():
                mesh.setVisible(checked)
            for origin_point in self.origin_points.values():
                origin_point.setVisible(checked)
            logger.info(f"Camera frustums visibility toggled to: {checked}")
        else:
            logger.warning("Camera meshes not available to toggle")

    def update_motion_trial(self, motion_trial: MotionTrial):
        logger.info("Updating xyz history in playback widget")
        self.motion_trial: MotionTrial = motion_trial

        if hasattr(self.motion_trial.tracker, "wireframe"):
            for segment_line in self.motion_trial.tracker.wireframe.line_plots.values():
                self.scene.addItem(segment_line)

        self.display_points(self.motion_trial.start_index)

    def set_export_mode(self, enabled: bool):
        self.export_video_mode = enabled

    def clear_collected_frames(self):
        self.collected_frames = []
        logger.debug("Cleared collected frames list.")

    def get_collected_frames(self):
        return self.collected_frames

    def qimage_to_cv2(self, qimage: QImage):
        """
        Converts a QImage to a NumPy array suitable for OpenCV (BGR format).
        Handles ARGB32 format, which is common for QImage grabbed from GL.
        """
        if qimage.isNull():
            logger.warning("qimage_to_cv2 received a null QImage.")
            return None

        if qimage.format() != QImage.Format_ARGB32:
            qimage = qimage.convertToFormat(QImage.Format_ARGB32)

        ptr = qimage.constBits()
        arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)
        bgr_frame = arr[:, :, :3].copy()

        return bgr_frame

    def display_points(self, sync_index: int):
        logger.debug(f"display_points called for sync_index: {sync_index}")

        # Clear previous custom meshes (safely handle items that may not be in scene)
        for mesh_item in self.custom_mesh_items:
            try:
                if mesh_item in self.scene.items:
                    self.scene.removeItem(mesh_item)
            except Exception as e:
                logger.debug(f"Could not remove mesh item: {e}")
        self.custom_mesh_items = []

        if self.motion_trial is None or self.motion_trial.is_empty:
            logger.debug(
                f"Motion trial is not loaded or is empty for sync_index: {sync_index}. Skipping point display."
            )
            self.scatter.setVisible(False)  # Hide scatter when no data
            self.scatter.setData(pos=np.empty((0, 3)))  # Use empty array instead of None
            self.fly_overlay.setVisible(False)
            self.fly_overlay.setData(pos=np.empty((0, 3)))
        else:
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_packet = self.motion_trial.get_xyz(sync_index)
            xyz_coords = xyz_packet.point_xyz
            point_ids = xyz_packet.point_ids

            # Check if we're using the insect tracker path (FLY legacy + INSECT alias)
            has_insect_tracker = (
                hasattr(self.motion_trial, "tracker")
                and hasattr(self.motion_trial.tracker, "name")
                and self.motion_trial.tracker.name in {"FLY", "INSECT"}
            )
            has_environment_structure_labels = bool(self.frame_roi_structures or self.arena_vertices)

            logger.info(
                f"has_insect_tracker: {has_insect_tracker}, tracker: "
                f"{self.motion_trial.tracker if hasattr(self.motion_trial, 'tracker') else 'None'}, "
                f"point_ids: {point_ids}, num_points: {len(xyz_coords)}"
            )

            if (has_insect_tracker or has_environment_structure_labels) and len(xyz_coords) > 0:
                # Separate environment structures from regular points.
                regular_mask = np.ones(len(point_ids), dtype=bool)
                structure_geometry_by_id = {
                    int(item["id"]): item.get("geometry", STRUCTURE_GEOMETRY_FLAT)
                    for item in self.frame_roi_structures
                    if item.get("enabled", True)
                }

                for i, (point_id, xyz) in enumerate(zip(point_ids, xyz_coords)):
                    logger.info(f"Processing point {i}: point_id={point_id} (type: {type(point_id)}), xyz={xyz}")
                    # Convert point_id to int for dictionary lookup
                    point_id_int = int(point_id)

                    # Check if this is a center point for a configured frame-ROI structure
                    if self.environment_structures_enabled and point_id_int in structure_geometry_by_id:
                        regular_mask[i] = False
                        structure_geometry = structure_geometry_by_id.get(point_id_int, STRUCTURE_GEOMETRY_FLAT)

                        # Find the 4 corner points for this object
                        # Corner IDs are: point_id * 1000 + [0, 1, 2, 3]
                        corner_ids = [point_id_int * 1000 + j for j in range(4)]
                        corner_xyzs = []

                        for corner_id in corner_ids:
                            corner_mask = point_ids == corner_id
                            if np.any(corner_mask):
                                corner_xyz = xyz_coords[corner_mask][0]
                                corner_xyzs.append(corner_xyz)
                                # Mark corners as not regular points
                                corner_idx = np.where(point_ids == corner_id)[0]
                                if len(corner_idx) > 0:
                                    regular_mask[corner_idx[0]] = False

                        # If we found all 4 corners, use them to create the mesh
                        if len(corner_xyzs) == 4:
                            logger.info(f"Found all 4 triangulated corners for point_id={point_id_int}")

                            if structure_geometry == STRUCTURE_GEOMETRY_FLAT:
                                ordered = self._order_corners_on_plane(np.array(corner_xyzs, dtype=np.float32))

                                # Skip degenerate quads that could render as crosses
                                v0, v1, v2 = ordered[0], ordered[1], ordered[2]
                                tri_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
                                if tri_area < 1e-6:
                                    logger.debug("Leaf quad degenerate; skipping mesh to avoid artefacts")
                                else:
                                    # Two triangles to form the square
                                    faces = np.array(
                                        [
                                            [0, 1, 2],  # First triangle: TL, TR, BR
                                            [0, 2, 3],  # Second triangle: TL, BR, BL
                                        ],
                                        dtype=np.uint32,
                                    )

                                    colors = np.array([(0, 1, 0, 0.35), (0, 1, 0, 0.35)], dtype=np.float32)

                                    mesh_item = gl.GLMeshItem(
                                        vertexes=ordered,
                                        faces=faces,
                                        faceColors=colors,
                                        smooth=False,
                                        drawEdges=True,
                                        edgeColor=(0, 0.5, 0, 1),
                                    )
                                    mesh_item.setGLOptions("translucent")  # Draw both faces to avoid backface culling
                                    self.scene.addItem(mesh_item)
                                    self.custom_mesh_items.append(mesh_item)
                                    logger.info(
                                        "Created flat square mesh from triangulated corners "
                                        f"for structure class {point_id_int}"
                                    )

                            elif structure_geometry == STRUCTURE_GEOMETRY_SEMI_SPHERE:
                                corners_array = self._order_corners_on_plane(np.array(corner_xyzs, dtype=np.float32))

                                width_3d = np.linalg.norm(corners_array[1] - corners_array[0])
                                height_3d = np.linalg.norm(corners_array[3] - corners_array[0])
                                radius_3d = min(width_3d, height_3d) * 0.5 * 1.05  # slight inflate for visibility

                                # Derive plane axes from corners; snap to world axes if nearly horizontal
                                centered = corners_array - corners_array.mean(axis=0)
                                _, _, vh = np.linalg.svd(centered)
                                plane_normal = vh[2]
                                axis_x = vh[0]
                                axis_y = np.cross(plane_normal, axis_x)

                                # Normalize axes
                                for vec in (plane_normal, axis_x, axis_y):
                                    norm = np.linalg.norm(vec)
                                    if norm > 1e-8:
                                        vec /= norm

                                vertices, faces, colors = self.create_oriented_hemisphere(
                                    center_xyz=xyz,
                                    radius_3d=radius_3d,
                                    normal=plane_normal,
                                    axis_x=axis_x,
                                    axis_y=axis_y,
                                    color=(1, 0, 0, 0.9),
                                    segments=16,
                                )

                                mesh_item = gl.GLMeshItem(
                                    vertexes=vertices, faces=faces, faceColors=colors, smooth=True, drawEdges=False
                                )
                                mesh_item.setGLOptions("translucent")  # brighter
                                self.scene.addItem(mesh_item)
                                self.custom_mesh_items.append(mesh_item)
                                logger.info(
                                    f"Created hemisphere mesh for class {point_id_int} with radius={radius_3d:.4f}"
                                )
                        else:
                            logger.warning(
                                f"Could not find all 4 corners for point_id={point_id_int}, "
                                f"found {len(corner_xyzs)} corners"
                            )
                            # Fall back to showing just the center point

                    # Skip corner points (IDs >= 1000) - they're already handled above
                    elif point_id_int >= 1000:
                        regular_mask[i] = False

                enabled_arena_vertex_ids = {
                    int(item["id"])
                    for item in self.arena_vertices
                    if item.get("enabled", True)
                }
                all_arena_vertex_ids = {int(item["id"]) for item in self.arena_vertices}
                floor_point_ids = [
                    int(item["id"])
                    for item in self.arena_vertices
                    if item.get("enabled", True) and item.get("add_to_floor", False)
                ]

                # Arena vertices are controlled by the Environment Structures panel.
                if self.environment_structures_enabled and len(all_arena_vertex_ids) > 0:
                    for idx, point_id in enumerate(point_ids):
                        if int(point_id) in all_arena_vertex_ids:
                            regular_mask[idx] = False

                floor_coords = []

                for floor_id in floor_point_ids:
                    floor_coord = self._get_point_coord_for_id(point_ids, xyz_coords, floor_id)
                    if floor_coord is not None:
                        floor_coords.append(floor_coord)

                # If we have 4+ selected floor points, create the floor mesh.
                if self.environment_structures_enabled and len(floor_coords) >= 4:
                    ordered_floor = self._order_corners_on_plane(np.array(floor_coords, dtype=np.float32))
                    vertices = ordered_floor

                    # Check for degenerate floor mesh
                    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
                    tri_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5

                    if tri_area < 1e-6:
                        logger.debug(f"Floor mesh degenerate (area={tri_area}); skipping to avoid artefacts")
                    else:
                        # Triangulate polygon with a simple fan from vertex 0.
                        faces = []
                        for tri_idx in range(1, len(vertices) - 1):
                            faces.append([0, tri_idx, tri_idx + 1])
                        faces = np.array(faces, dtype=np.uint32)

                        colors = np.array([(1, 1, 1, 0.3)] * len(faces), dtype=np.float32)

                        floor_mesh = gl.GLMeshItem(
                            vertexes=vertices,
                            faces=faces,
                            faceColors=colors,
                            smooth=False,
                            drawEdges=True,
                            edgeColor=(1, 1, 1, 0.8),  # White edges
                        )
                        floor_mesh.setGLOptions("translucent")
                        self.scene.addItem(floor_mesh)
                        self.custom_mesh_items.append(floor_mesh)
                        logger.info(f"Created floor mesh from {len(vertices)} selected arena vertices")
                else:
                    logger.debug(f"Need >=4 floor vertices; found {len(floor_coords)} points")

                # Draw perimeter edges for enabled arena vertices using planar ordering.
                edge_pairs = []
                if self.environment_structures_enabled and len(enabled_arena_vertex_ids) >= 3:
                    arena_coords = []
                    for arena_id in sorted(enabled_arena_vertex_ids):
                        arena_coord = self._get_point_coord_for_id(point_ids, xyz_coords, arena_id)
                        if arena_coord is not None:
                            arena_coords.append((arena_id, arena_coord))
                    if len(arena_coords) >= 3:
                        edge_pairs = self._build_arena_edge_pairs(arena_coords)

                edges_found = 0
                for point_id_a, point_id_b in edge_pairs:
                    coord_a = self._get_point_coord_for_id(point_ids, xyz_coords, point_id_a)
                    coord_b = self._get_point_coord_for_id(point_ids, xyz_coords, point_id_b)

                    if coord_a is not None and coord_b is not None:

                        # Skip edges with NaN or infinite values
                        if (
                            np.any(np.isnan(coord_a))
                            or np.any(np.isinf(coord_a))
                            or np.any(np.isnan(coord_b))
                            or np.any(np.isinf(coord_b))
                        ):
                            logger.debug(f"Skipping edge ({point_id_a}, {point_id_b}): contains NaN or Inf values")
                            continue

                        # Skip degenerate edges (zero length)
                        edge_length = np.linalg.norm(coord_b - coord_a)
                        if edge_length < 1e-8:
                            logger.debug(
                                f"Skipping edge ({point_id_a}, {point_id_b}): degenerate (length={edge_length})"
                            )
                            continue

                        # Create line segment
                        line_pos = np.array([coord_a, coord_b], dtype=np.float32)
                        line = gl.GLLinePlotItem(
                            pos=line_pos,
                            color=(1, 1, 1, 0.6),  # White, semi-transparent
                            width=1.5,
                            antialias=True,
                        )
                        self.scene.addItem(line)
                        self.custom_mesh_items.append(line)
                        edges_found += 1

                logger.info(f"Created {edges_found} edge lines for capture volume box")

                # Display regular points (excluding special labels)
                regular_coords = xyz_coords[regular_mask]

                # Filter out NaN and Inf values from regular coordinates
                valid_mask = np.all(np.isfinite(regular_coords), axis=1)
                valid_coords = regular_coords[valid_mask]

                if len(valid_coords) > 0:
                    self.scatter.setVisible(True)
                    self.scatter.setData(pos=valid_coords)
                else:
                    self.scatter.setVisible(False)
                    self.scatter.setData(pos=np.empty((0, 3)))

                # Ensure fly point(s) are always visible over translucent meshes (ground truth)
                fly_mask = point_ids == 0
                if np.any(fly_mask):
                    fly_coords = xyz_coords[fly_mask]
                    self.fly_overlay.setVisible(True)
                    self.fly_overlay.setData(pos=fly_coords, color=(1, 1, 1, 1))
                else:
                    self.fly_overlay.setVisible(False)
                    self.fly_overlay.setData(pos=np.empty((0, 3)))

                # Overlay predictions for the same sync_index (green dot for point_id==0)
                if (
                    hasattr(self.motion_trial, "predictions_df")
                    and isinstance(self.motion_trial.predictions_df, pd.DataFrame)
                    and not self.motion_trial.predictions_df.empty
                ):
                    logger.debug(f"Checking predictions overlay for sync_index={sync_index}")
                    pred_rows = self.motion_trial.predictions_df[
                        self.motion_trial.predictions_df["sync_index"] == sync_index
                    ]
                    logger.debug(f"Predictions rows for sync_index={sync_index}: {len(pred_rows)}")
                    if not pred_rows.empty:
                        pred_fly = pred_rows[pred_rows["point_id"] == 0]
                        logger.debug(f"Pred fly rows for sync_index={sync_index}: {len(pred_fly)}")
                        if not pred_fly.empty:
                            pred_xyz = pred_fly[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
                            self.pred_overlay.setVisible(True)
                            self.pred_overlay.setData(pos=pred_xyz, color=(1, 0.5, 0, 1))
                            logger.info(
                                "Plotted prediction fly overlay at sync_index="
                                f"{sync_index}: {pred_xyz.shape[0]} point(s)"
                            )
                        else:
                            self.pred_overlay.setVisible(False)
                            self.pred_overlay.setData(pos=np.empty((0, 3)))
                            logger.debug(f"No prediction fly points for sync_index={sync_index}")
                    else:
                        self.pred_overlay.setVisible(False)
                        self.pred_overlay.setData(pos=np.empty((0, 3)))
                        logger.debug(f"No predictions rows for sync_index={sync_index}")
                else:
                    self.pred_overlay.setVisible(False)
                    self.pred_overlay.setData(pos=np.empty((0, 3)))
                    logger.debug("Predictions DataFrame not available or empty; overlay disabled")
            else:
                # Default behavior for non-FlyTracker or when no special handling needed
                self.scatter.setVisible(True)  # Make visible when we have data
                self.scatter.setData(pos=xyz_coords)
                # Fly overlay for default path
                fly_mask = point_ids == 0
                if np.any(fly_mask):
                    fly_coords = xyz_coords[fly_mask]
                    self.fly_overlay.setVisible(True)
                    self.fly_overlay.setData(pos=fly_coords, color=(1, 1, 1, 1))
                else:
                    self.fly_overlay.setVisible(False)
                    self.fly_overlay.setData(pos=np.empty((0, 3)))

            if self.export_video_mode:
                logger.debug(f"Export mode is active for sync_index: {sync_index}. Attempting to grab framebuffer.")
                image = self.scene.grabFramebuffer()

                if image.isNull():
                    logger.warning(f"grabFramebuffer returned a null image for sync_index: {sync_index}!")
                else:
                    cv2_frame = self.qimage_to_cv2(image)
                    if cv2_frame is not None and cv2_frame.size > 0:
                        # Enforce consistent frame dimensions if locked (prevents resizing during export)
                        if self.locked_export_width is not None and self.locked_export_height is not None:
                            current_height, current_width = cv2_frame.shape[:2]
                            if current_width != self.locked_export_width or current_height != self.locked_export_height:
                                logger.debug(
                                    f"Frame {sync_index} size mismatch: "
                                    f"{current_width}x{current_height}, expected "
                                    f"{self.locked_export_width}x{self.locked_export_height}. Resizing..."
                                )
                                import cv2

                                cv2_frame = cv2.resize(cv2_frame, (self.locked_export_width, self.locked_export_height))

                        self.collected_frames.append(cv2_frame)
                        logger.debug(
                            f"Successfully collected frame {sync_index} to memory. "
                            f"Total frames: {len(self.collected_frames)}"
                        )
                    else:
                        logger.warning(
                            f"qimage_to_cv2 returned an empty or invalid frame for sync_index: {sync_index}!"
                        )
            else:
                logger.debug(f"Export mode is INACTIVE for sync_index: {sync_index}.")

    def update_segment_lines(self, sync_index: int):
        if (
            self.motion_trial
            and hasattr(self.motion_trial, "tracker")
            and hasattr(self.motion_trial.tracker, "wireframe")
            and self.motion_trial.tracker.wireframe is not None
        ):
            self.motion_trial.tracker.update_wireframe_data(sync_index)
        else:
            logger.debug(f"No wireframe to update from PlaybackTriangulationWidget for sync index {sync_index}.")

    def create_oriented_hemisphere(
        self, center_xyz, radius_3d, normal, axis_x, axis_y, color=(1, 0, 0, 0.6), segments=16
    ):
        """Create a hemisphere oriented along a plane normal using provided axes.

        Args:
            center_xyz: (x, y, z) center position (base center)
            radius_3d: radius in 3D world coordinates
            normal: unit normal vector pointing the bulge direction
            axis_x: unit vector along one base edge
            axis_y: unit vector perpendicular to axis_x in the plane
            color: RGBA color tuple
            segments: number of segments for hemisphere smoothness

        Returns:
            tuple: (vertices, faces, colors) for GLMeshItem
        """
        cx, cy, cz = center_xyz

        vertices = []
        faces = []

        # Peak of the dome along +normal
        peak = np.array(center_xyz) + normal * radius_3d
        vertices.append(peak.tolist())

        # Generate hemisphere vertices relative to center
        for i in range(segments // 2 + 1):  # From equator (0) to pole (pi/2)
            lat = i * (np.pi / 2) / (segments // 2)
            height = radius_3d * np.cos(lat)  # along normal
            ring_r = radius_3d * np.sin(lat)  # in plane

            ring_center = np.array(center_xyz) + normal * height
            for j in range(segments):
                lon = j * (2 * np.pi) / segments
                offset = axis_x * (ring_r * np.cos(lon)) + axis_y * (ring_r * np.sin(lon))
                vertex = ring_center + offset
                vertices.append(vertex.tolist())

        vertices = np.array(vertices, dtype=np.float32)

        # Faces: connect peak to first ring
        for j in range(segments):
            next_j = (j + 1) % segments
            faces.append([0, j + 1, next_j + 1])

        # Connect rings
        for i in range(segments // 2):
            for j in range(segments):
                next_j = (j + 1) % segments

                current_base = 1 + i * segments
                next_base = 1 + (i + 1) * segments

                v1 = current_base + j
                v2 = current_base + next_j
                v3 = next_base + next_j
                v4 = next_base + j

                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        faces = np.array(faces, dtype=np.uint32)
        colors = np.array([color] * len(faces), dtype=np.float32)

        return vertices, faces, colors

    def _order_corners_on_plane(self, corners: np.ndarray) -> np.ndarray:
        """Return corners ordered consistently around their plane.
        Uses PCA to find the best-fit plane, projects to 2D, and sorts by angle.
        """
        # Center the points
        centroid = corners.mean(axis=0)
        centered = corners - centroid

        # PCA to find plane axes
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        axis_x = vh[0]
        axis_y = np.cross(normal, axis_x)

        # Project to 2D plane coordinates
        proj_x = centered @ axis_x
        proj_y = centered @ axis_y
        angles = np.arctan2(proj_y, proj_x)

        order = np.argsort(angles)
        ordered = corners[order]
        return ordered

    def _order_ids_on_plane(self, id_coord_pairs: list[tuple[int, np.ndarray]]) -> list[int]:
        """Order point IDs around the best-fit plane of their coordinates."""
        ids = [pair[0] for pair in id_coord_pairs]
        coords = np.array([pair[1] for pair in id_coord_pairs], dtype=np.float32)
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        axis_x = vh[0]
        axis_y = np.cross(normal, axis_x)
        proj_x = centered @ axis_x
        proj_y = centered @ axis_y
        angles = np.arctan2(proj_y, proj_x)
        order = np.argsort(angles)
        return [ids[idx] for idx in order]

    def _build_arena_edge_pairs(self, id_coord_pairs: list[tuple[int, np.ndarray]]) -> list[tuple[int, int]]:
        """Build robust arena edges from selected vertices.

        - Coplanar selections: connect as a closed ordered loop.
        - Non-coplanar selections: connect each point to its nearest neighbors.
        """
        if len(id_coord_pairs) < 2:
            return []

        ids = [pair[0] for pair in id_coord_pairs]
        coords = np.array([pair[1] for pair in id_coord_pairs], dtype=np.float32)

        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        plane_distances = np.abs(centered @ normal)
        max_plane_dist = float(np.max(plane_distances)) if len(plane_distances) else 0.0

        # If points are close to one plane, use perimeter loop.
        if max_plane_dist < 0.01 or len(ids) <= 4:
            ordered_ids = self._order_ids_on_plane(id_coord_pairs)
            return [
                (ordered_ids[i], ordered_ids[(i + 1) % len(ordered_ids)])
                for i in range(len(ordered_ids))
            ]

        # For non-coplanar sets (e.g., full 3D arena corners), use nearest-neighbor graph.
        neighbor_count = min(3, len(ids) - 1)
        edge_set = set()
        for i, point_id in enumerate(ids):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            nearest_indices = np.argsort(dists)[1 : neighbor_count + 1]
            for j in nearest_indices:
                edge_set.add(tuple(sorted((point_id, ids[j]))))

        return list(edge_set)

    def _get_point_coord_for_id(
            self, point_ids: np.ndarray, xyz_coords: np.ndarray, point_id: int
            ) -> Optional[np.ndarray]:
        """Get point coordinates from current frame, with mean-position fallback from trial data."""
        current_mask = point_ids == point_id
        if np.any(current_mask):
            coord = xyz_coords[current_mask][0]
            if np.all(np.isfinite(coord)):
                return coord

        if self.motion_trial is None or not hasattr(self.motion_trial, "xyz_df"):
            return None

        df = self.motion_trial.xyz_df
        if df is None or df.empty:
            return None

        pid_df = df[df["point_id"] == point_id]
        if pid_df.empty:
            return None

        coord = np.array(
            [
                pid_df["x_coord"].mean(),
                pid_df["y_coord"].mean(),
                pid_df["z_coord"].mean(),
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(coord)):
            return None
        return coord

    def clear_grid_labels(self):  # <--- ADD THIS ENTIRE METHOD
        for label in self.grid_labels:
            self.scene.removeItem(label)
        self.grid_labels = []
        logger.info("Cleared existing grid labels.")

    def _clear_measurement_grid_items(self):
        """Remove explicit measurement grid line items from the scene."""
        for grid_item in self.measurement_grid_items:
            try:
                self.scene.removeItem(grid_item)
            except Exception:
                pass
        self.measurement_grid_items = []

    def _scene_extent_m(self) -> float:
        """Estimate a reasonable scene extent from the loaded data."""
        points = []
        if (
            self.motion_trial is not None
            and hasattr(self.motion_trial, "xyz_df")
            and not self.motion_trial.xyz_df.empty
        ):
            xyz = self.motion_trial.xyz_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float)
            if xyz.size:
                points.append(xyz)
        if self.camera_array is not None and hasattr(self.camera_array, "get_world_origins"):
            origins = self.camera_array.get_world_origins()
            if origins is not None and np.size(origins):
                points.append(np.asarray(origins, dtype=float))

        if not points:
            return self.measurement_grid_extent_m

        stacked = np.vstack(points)
        span = np.max(stacked, axis=0) - np.min(stacked, axis=0)
        extent = float(max(span.max() * 1.5, 2.0))
        return min(max(extent, 2.0), 50.0)

    def _get_measurement_grid_major_spacing_m(self) -> float:
        """Pick a readable major spacing for the current measurement grid extent."""
        extent_m = self._scene_extent_m()
        _, major_spacing_m = adaptive_grid_spacing(extent_m, target_major_intervals=20.0)
        return major_spacing_m

    def _get_measurement_grid_minor_spacing_m(self) -> float:
        """Pick a readable minor spacing for the current measurement grid extent."""
        major_spacing = self._get_measurement_grid_major_spacing_m()
        minor_spacing = major_spacing / 5.0
        return max(minor_spacing, 0.05)

    def _add_grid_line_item(self, plane: str, spacing_m: float, half_extent_m: float, color, width: float):
        """Create and add a grid line item for a plane."""
        lines = build_plane_grid_lines(plane, spacing_m, half_extent_m)
        if not lines:
            return None

        grid_item = gl.GLLinePlotItem(
            pos=np.asarray(lines, dtype=np.float32).reshape(-1, 3),
            color=color,
            width=width,
            mode="lines",
            antialias=True,
        )
        grid_item.setGLOptions("opaque")
        self.scene.addItem(grid_item)
        self.measurement_grid_items.append(grid_item)
        return grid_item

    def _build_measurement_grid_items(self):
        """Render the measurement grid as explicit white line grids."""
        self._clear_measurement_grid_items()

        self.measurement_grid_extent_m = self._scene_extent_m()
        half_extent_m = self.measurement_grid_extent_m / 2.0
        minor_spacing_m = self._get_measurement_grid_minor_spacing_m()
        major_spacing_m = self._get_measurement_grid_major_spacing_m()

        minor_color = (0.82, 0.82, 0.82, 0.35)
        major_color = (1.0, 1.0, 1.0, 0.75)

        self._add_grid_line_item("xy", minor_spacing_m, half_extent_m, minor_color, width=0.7)
        self._add_grid_line_item("xz", minor_spacing_m, half_extent_m, minor_color, width=0.7)
        self._add_grid_line_item("xy", major_spacing_m, half_extent_m, major_color, width=1.3)
        self._add_grid_line_item("xz", major_spacing_m, half_extent_m, major_color, width=1.3)

    def add_grid_labels(self, grid_total_extent_m=10, label_interval_m=0.5, text_color="white"):
        self.clear_grid_labels()  # Clear existing labels before adding new ones

        # Determine the range for labels based on half the grid extent
        half_extent_m = grid_total_extent_m / 2

        for pos, text in build_complete_grid_label_specs(
            half_extent=half_extent_m,
            label_interval=label_interval_m,
            label_formatter=lambda value: f"{value:.1f} m",
        ):
            tick = gl.GLTextItem(pos=pos, text=text, color=text_color)
            self.scene.addItem(tick)
            self.grid_labels.append(tick)

        logger.info(f"Added {len(self.grid_labels)} grid labels.")

    def toggle_measurement_mode(self):
        self.is_measurement_mode_active = not self.is_measurement_mode_active
        logger.info(f"Toggling measurement mode. New state: {self.is_measurement_mode_active}")

        self.measurement_grid_extent_m = 10.0  # 10 meters extent
        major_spacing_m = self._get_measurement_grid_major_spacing_m()

        if self.is_measurement_mode_active:
            self.scene.setBackgroundColor(QColorConstants.Black)
            self.scatter.setData(color=self.default_scatter_color)
            self._build_measurement_grid_items()
            self.add_grid_labels(
                grid_total_extent_m=self.measurement_grid_extent_m, label_interval_m=major_spacing_m, text_color="white"
            )
        else:
            self.scene.setBackgroundColor(QColorConstants.Black)
            self.scatter.setData(color=self.default_scatter_color)
            self._clear_measurement_grid_items()
            self.clear_grid_labels()
