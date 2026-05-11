"""Focus testing tab for browsing a video, annotating a focus region, and logging results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.path_url_entry import create_path_url_entry
from calipod.gui.utils.styles import create_styled_groupbox
from calipod.lens_focus.analyze_lens_focus import LensFocusAnalyzer

logger = calipod_logger.get(__name__)


class FocusVideoCanvas(QLabel):
    roi_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setMouseTracking(True)
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None
        self._frame_pixmap = QPixmap()

    def set_frame(self, pixmap: QPixmap):
        self._frame_pixmap = pixmap
        self.setPixmap(pixmap)
        if not pixmap.isNull():
            self.setFixedSize(pixmap.size())
        self._drag_start = None
        self._drag_end = None
        self.update()

    def clear_roi(self):
        self._drag_start = None
        self._drag_end = None
        self.update()

    def roi_rect(self) -> QRect | None:
        if self._drag_start is None or self._drag_end is None:
            return None
        rect = QRect(self._drag_start, self._drag_end).normalized()
        if rect.width() < 2 or rect.height() < 2:
            return None
        return rect

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.pixmap().isNull():
            self._drag_start = event.position().toPoint()
            self._drag_end = event.position().toPoint()
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_end = event.position().toPoint()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._drag_end = event.position().toPoint()
            self.update()
            self.roi_changed.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self.roi_rect()
        if rect is None:
            return

        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.green)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.end()


class FocusWidget(QWidget):
    def __init__(self, controller: Controller):
        super(FocusWidget, self).__init__()
        self.controller = controller
        self.analyzer = LensFocusAnalyzer(self.controller.workspace_guide.focus_dir)
        self.current_video_path: Path | None = None
        self.current_capture: cv2.VideoCapture | None = None
        self.current_frame_bgr: np.ndarray | None = None
        self.current_frame_index = 0
        self.frame_count = 0
        self.results_rows: list[dict] = self.analyzer.load_results()

        self.place_widgets()
        self.connect_widgets()
        self.refresh_results_table()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.top_vbox = QVBoxLayout()
        self.middle_vbox = QVBoxLayout()
        self.bottom_vbox = QVBoxLayout()

        self._build_browse_section()
        self._build_video_section()
        self._build_results_section()

        self.layout().addLayout(self.top_vbox, stretch=0)
        self.layout().addLayout(self.middle_vbox, stretch=3)
        self.layout().addLayout(self.bottom_vbox, stretch=2)

    def _build_browse_section(self):
        browse_group, browse_layout = create_styled_groupbox("1. Browse Focus Video")
        browse_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.video_path_row = create_path_url_entry(
            initial_path="",
            parent=self,
            dialog_caption="Select Focus Video",
            select_directory=False,
            file_filter="Video Files (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.wmv *.flv *.webm);;All Files (*)",
            on_path_selected=self._load_selected_video,
        )

        self.load_video_btn = QPushButton("Load Video", self)
        self.load_video_btn.clicked.connect(self.load_video_from_entry)

        browse_layout.addWidget(self.video_path_row.container)
        browse_layout.addWidget(self.load_video_btn)

        self.top_vbox.addWidget(browse_group)

    def _build_video_section(self):
        video_group, video_layout = create_styled_groupbox("2. Annotate the Focus Region")
        video_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.video_status_label = QLabel("Select a video to begin.", self)
        self.video_frame_label = QLabel("No frame loaded", self)

        self.video_canvas = FocusVideoCanvas(self)
        self.video_canvas.setMinimumSize(640, 360)
        self.video_canvas.setText("Video preview will appear here")

        self.video_scroll = QScrollArea(self)
        self.video_scroll.setWidgetResizable(False)
        self.video_scroll.setWidget(self.video_canvas)
        self.video_scroll.setMinimumHeight(360)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.frame_slider.setVisible(False)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)

        controls_row = QHBoxLayout()
        self.prev_frame_btn = QPushButton("Prev", self)
        self.next_frame_btn = QPushButton("Next", self)
        self.analyze_btn = QPushButton("Analyze Selection", self)
        self.clear_roi_btn = QPushButton("Clear ROI", self)
        self.prev_frame_btn.clicked.connect(self.show_previous_frame)
        self.next_frame_btn.clicked.connect(self.show_next_frame)
        self.analyze_btn.clicked.connect(self.analyze_current_selection)
        self.clear_roi_btn.clicked.connect(self.video_canvas.clear_roi)

        controls_row.addWidget(self.prev_frame_btn)
        controls_row.addWidget(self.next_frame_btn)
        controls_row.addWidget(self.analyze_btn)
        controls_row.addWidget(self.clear_roi_btn)
        controls_row.addStretch(1)

        video_layout.addWidget(self.video_status_label)
        video_layout.addWidget(self.video_frame_label)
        video_layout.addWidget(self.video_scroll)
        video_layout.addWidget(self.frame_slider)
        video_layout.addLayout(controls_row)

        self.middle_vbox.addWidget(video_group)

    def _build_results_section(self):
        results_group, results_layout = create_styled_groupbox("3. Results Table")
        results_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.results_table = QTableWidget(self)
        self.results_table.setColumnCount(12)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Video",
                "Frame",
                "ROI",
                "Width",
                "Height",
                "Mean Bright",
                "Variance",
                "Entropy",
                "Edges",
                "Sobel",
                "Laplacian",
                "Timestamp",
            ]
        )
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        table_buttons = QHBoxLayout()
        self.clear_selected_btn = QPushButton("Clear Selected Row(s)", self)
        self.clear_all_btn = QPushButton("Clear Table", self)
        self.clear_selected_btn.clicked.connect(self.clear_selected_rows)
        self.clear_all_btn.clicked.connect(self.clear_all_results)
        table_buttons.addWidget(self.clear_selected_btn)
        table_buttons.addWidget(self.clear_all_btn)
        table_buttons.addStretch(1)

        results_layout.addLayout(table_buttons)
        results_layout.addWidget(self.results_table)

        self.bottom_vbox.addWidget(results_group)

    def connect_widgets(self):
        self.video_path_row.line_edit.returnPressed.connect(self.load_video_from_entry)

    def _load_selected_video(self, selected_path: str) -> str | None:
        self.load_video(selected_path)
        return selected_path

    def load_video_from_entry(self):
        self.load_video(self.video_path_row.line_edit.text().strip())

    def load_video(self, video_path: str):
        if not video_path:
            return

        path = Path(video_path)
        if not path.exists():
            QMessageBox.warning(self, "Focus Video", f"Video not found:\n{path}")
            return

        self._release_capture()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            QMessageBox.warning(self, "Focus Video", f"Could not open video:\n{path}")
            return

        self.current_video_path = path
        self.current_capture = capture
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        if self.frame_count <= 0:
            QMessageBox.warning(self, "Focus Video", f"Video has no readable frames:\n{path}")
            self._release_capture()
            return

        self.frame_slider.setVisible(True)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.video_status_label.setText(f"Loaded: {path.name}")
        self._show_frame(0)

    def _release_capture(self):
        if self.current_capture is not None:
            self.current_capture.release()
            self.current_capture = None

    def _show_frame(self, frame_index: int):
        if self.current_capture is None:
            return

        frame_index = max(0, min(frame_index, max(0, self.frame_count - 1)))
        self.current_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame_bgr = self.current_capture.read()
        if not success or frame_bgr is None:
            self.video_status_label.setText("Unable to read frame.")
            return

        self.current_frame_index = frame_index
        self.current_frame_bgr = frame_bgr
        self.video_frame_label.setText(f"Frame {frame_index + 1} / {self.frame_count}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel_count = frame_rgb.shape
        bytes_per_line = channel_count * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_canvas.set_frame(QPixmap.fromImage(image.copy()))
        self.video_canvas.clear_roi()

    def on_frame_changed(self, value: int):
        self._show_frame(value)

    def show_previous_frame(self):
        if self.current_capture is None or self.current_frame_index <= 0:
            return
        self.frame_slider.setValue(self.current_frame_index - 1)

    def show_next_frame(self):
        if self.current_capture is None or self.current_frame_index >= self.frame_count - 1:
            return
        self.frame_slider.setValue(self.current_frame_index + 1)

    def _current_roi(self) -> tuple[int, int, int, int] | None:
        rect = self.video_canvas.roi_rect()
        if rect is None:
            return None
        return rect.left(), rect.top(), rect.right() + 1, rect.bottom() + 1

    def analyze_current_selection(self):
        if self.current_frame_bgr is None or self.current_video_path is None:
            QMessageBox.information(self, "Focus Analysis", "Load a video before analyzing.")
            return

        roi = self._current_roi()
        if roi is None:
            QMessageBox.information(self, "Focus Analysis", "Drag a rectangle on the frame before analyzing.")
            return

        try:
            result = self.analyzer.analyze_frame(
                frame_bgr=self.current_frame_bgr,
                roi=roi,
                source_path=self.current_video_path,
                frame_index=self.current_frame_index,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Focus Analysis", str(exc))
            return

        self.results_rows.append(result.as_dict())
        self.analyzer.write_results(self.results_rows)
        self.refresh_results_table()
        self.video_status_label.setText(f"Analyzed {self.current_video_path.name} at frame {self.current_frame_index + 1}")

    def refresh_results_table(self):
        self.results_table.setRowCount(0)
        for row_index, result in enumerate(self.results_rows):
            self.results_table.insertRow(row_index)
            values = [
                result.get("video_name", ""),
                str(result.get("frame_index", "")),
                result.get("roi", ""),
                str(result.get("width", "")),
                str(result.get("height", "")),
                f"{self._float_value(result.get('mean_brightness', 0.0)):.4f}",
                f"{self._float_value(result.get('variance', 0.0)):.4f}",
                f"{self._float_value(result.get('entropy', 0.0)):.4f}",
                str(result.get("canny_edges_count", "")),
                f"{self._float_value(result.get('average_sobel_magnitude', 0.0)):.4f}",
                str(result.get("laplacian_zero_crossings", "")),
                result.get("timestamp", ""),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(row_index, column_index, item)

    def _float_value(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def clear_selected_rows(self):
        selected_rows = sorted({index.row() for index in self.results_table.selectionModel().selectedRows()}, reverse=True)
        if not selected_rows:
            return

        for row_index in selected_rows:
            if 0 <= row_index < len(self.results_rows):
                del self.results_rows[row_index]

        self.analyzer.write_results(self.results_rows)
        self.refresh_results_table()

    def clear_all_results(self):
        if not self.results_rows:
            return

        self.results_rows = []
        self.analyzer.write_results(self.results_rows)
        self.refresh_results_table()

    def closeEvent(self, event):
        self._release_capture()
        super().closeEvent(event)
