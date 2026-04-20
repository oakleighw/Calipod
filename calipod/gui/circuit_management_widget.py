"""This widget acts as a guide for creating a camera trigger circuit and wiring guidelines"""

from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase, QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.styles import create_styled_groupbox, resolve_camera_title_color
from circuit_management.config_manager import CircuitManagementConfigManager
from circuit_management.connector_preview_renderer import render_connector_preview_frame
from circuit_management.timing_sync_planner import build_timing_sync_plan, render_timing_sync_table
from circuit_management.wire_labeling_rules import (
    CONNECTOR_PIN_COUNTS,
    WIRE_COLOUR_OPTIONS,
    WIRE_TYPE_OPTIONS,
    available_wire_type_options,
    normalize_wire_type_selections,
)

logger = calipod_logger.get(__name__)

CONNECTOR_IMAGE_PATHS = {
    "Hirose 6 Pin": Path(__file__).parent / "icons" / "connectors" / "6_pin_hirose_female.png",
    "Hirose 4 Pin": Path(__file__).parent / "icons" / "connectors" / "4_pin_hirose_female.png",
}

class CircuitManagementWidget(QWidget):
    def __init__(self, controller: Controller):
        super(CircuitManagementWidget, self).__init__()
        self.controller = controller
        self.config_manager = CircuitManagementConfigManager(controller.workspace)
        self.current_camera_port: int | None = None
        self._is_loading_camera_data = False
        self.place_widgets()
        self._load_initial_camera_data()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.top_vbox = QVBoxLayout()

        self.bottom_vbox = QHBoxLayout()
        self.bottom_left_vbox = QVBoxLayout()
        self.bottom_right_vbox = QVBoxLayout()

        self.bottom_container = QWidget()
        self.bottom_container.setLayout(self.bottom_vbox)
        self.bottom_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.wire_labeling_widget()
        self.triggerbox_test_widget()
        self.timing_test_widget()

        self.bottom_vbox.addLayout(self.bottom_left_vbox)
        self.bottom_vbox.addLayout(self.bottom_right_vbox)

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addWidget(self.bottom_container, stretch=1)

    def wire_labeling_widget(self):
        wire_label_group, wire_label_layout = create_styled_groupbox("Wire Labeling")
        self.wire_camera_selector = QComboBox()
        for port, color in self._camera_selection_entries():
            self.wire_camera_selector.addItem(f"Camera {port}", userData=port)
            combo_index = self.wire_camera_selector.count() - 1
            self.wire_camera_selector.setItemData(combo_index, QColor(color), Qt.ItemDataRole.ForegroundRole)

        self.wire_camera_selector.currentIndexChanged.connect(self._on_camera_selection_changed)

        self.connector_selector = QComboBox()
        for connector_name in CONNECTOR_IMAGE_PATHS:
            self.connector_selector.addItem(connector_name, userData=connector_name)
        self.connector_selector.currentIndexChanged.connect(self._update_connector_preview)
        self.connector_selector.currentIndexChanged.connect(self._on_connector_changed)

        self.connector_preview = QLabel()
        self.connector_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connector_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.connector_preview.setMinimumHeight(220)

        selector_row = QHBoxLayout()
        selector_row.addWidget(self.wire_camera_selector)
        selector_row.addWidget(self.connector_selector)

        self.connector_table_widget = QWidget()
        self.connector_table_layout = QGridLayout()
        self.connector_table_widget.setLayout(self.connector_table_layout)
        self._build_connector_table()

        wire_label_layout.addLayout(selector_row)
        content_row = QHBoxLayout()
        content_row.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_row.addWidget(self.connector_table_widget, stretch=3)
        content_row.addWidget(self.connector_preview, stretch=2)
        wire_label_layout.addLayout(content_row)

        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("Delay"))

        self.delay_none_radio = QRadioButton("None")
        self.delay_custom_radio = QRadioButton()
        self.delay_none_radio.setChecked(True)

        self.delay_mode_group = QButtonGroup(self)
        self.delay_mode_group.addButton(self.delay_none_radio)
        self.delay_mode_group.addButton(self.delay_custom_radio)

        self.delay_ms_spin = QSpinBox()
        self.delay_ms_spin.setRange(0, 10_000_000)
        self.delay_ms_spin.setEnabled(False)

        self.delay_none_radio.toggled.connect(self._on_delay_mode_changed)
        self.delay_custom_radio.toggled.connect(self._on_delay_mode_changed)
        self.delay_ms_spin.valueChanged.connect(self._on_delay_value_changed)

        delay_row.addWidget(self.delay_none_radio)
        delay_row.addWidget(self.delay_custom_radio)
        delay_row.addWidget(self.delay_ms_spin)
        delay_row.addWidget(QLabel("ms"))
        delay_row.addStretch(1)
        wire_label_layout.addLayout(delay_row)

        self._update_connector_preview(self.connector_selector.currentIndex())
        self.top_vbox.addWidget(wire_label_group)

    def _update_connector_preview(self, _index: int):
        connector_name = self.connector_selector.currentData()
        row_count = CONNECTOR_PIN_COUNTS.get(connector_name, 0)
        self._set_connector_table_visible_rows(row_count)
        self._update_preview_image()
        self._refresh_wire_type_dropdowns()

    def _update_preview_image(self) -> None:
        """Update the preview image for the current connector."""
        connector_name = self.connector_selector.currentData()
        image_path = CONNECTOR_IMAGE_PATHS.get(connector_name)
        if image_path is None or not image_path.exists():
            self.connector_preview.setText("Connector image not found")
            self.connector_preview.setPixmap(QPixmap())
            return

        row_count = CONNECTOR_PIN_COUNTS.get(connector_name, 0)
        wire_colours = {
            row_index: self.connector_row_widgets[row_index - 1][2].currentData()
            for row_index in range(1, row_count + 1)
        }
        try:
            rendered_frame = render_connector_preview_frame(image_path, connector_name, wire_colours)
        except FileNotFoundError:
            self.connector_preview.setText("Connector image not found")
            self.connector_preview.setPixmap(QPixmap())
            return

        rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGRA2RGBA)
        qimage = QImage(
            rendered_frame.data,
            rendered_frame.shape[1],
            rendered_frame.shape[0],
            rendered_frame.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()

        self.connector_preview.setText("")
        self.connector_preview.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                320,
                220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _build_connector_table(self):
        self.connector_row_widgets: list[tuple[QLabel, QComboBox, QComboBox]] = []
        headers = ("Connector Port", "Wire Type", "Colour")
        for col, header in enumerate(headers):
            header_label = QLabel(header)
            self.connector_table_layout.addWidget(header_label, 0, col)

        for row in range(1, 7):
            port_label = QLabel(str(row))
            wire_type_combo = QComboBox()
            colour_combo = QComboBox()
            wire_type_combo.currentIndexChanged.connect(self._on_wire_type_changed)
            wire_type_combo.currentIndexChanged.connect(
                lambda _index, row_idx=row: self._on_table_data_changed(row_idx)
            )
            colour_combo.currentIndexChanged.connect(
                lambda _index, combo=colour_combo: self._on_colour_selection_changed(combo)
            )
            colour_combo.currentIndexChanged.connect(
                lambda _index, row_idx=row: self._on_table_data_changed(row_idx)
            )

            self._populate_colour_dropdown(colour_combo)

            self.connector_table_layout.addWidget(port_label, row, 0)
            self.connector_table_layout.addWidget(wire_type_combo, row, 1)
            self.connector_table_layout.addWidget(colour_combo, row, 2)

            self.connector_row_widgets.append((port_label, wire_type_combo, colour_combo))

        for col in range(3):
            self.connector_table_layout.setColumnStretch(col, 1)

    def _populate_colour_dropdown(self, combo: QComboBox):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")

        for colour_name, colour_hex in WIRE_COLOUR_OPTIONS:
            combo.addItem(colour_name, userData=colour_hex)
            combo_index = combo.count() - 1
            combo.setItemData(combo_index, QColor(colour_hex), Qt.ItemDataRole.ForegroundRole)

        combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self._update_colour_combo_style(combo)

    def _on_colour_selection_changed(self, combo: QComboBox):
        self._update_colour_combo_style(combo)

    def _update_colour_combo_style(self, combo: QComboBox):
        colour_hex = combo.currentData()
        if not colour_hex:
            combo.setStyleSheet("")
            return

        combo.setStyleSheet(f"QComboBox {{ color: {colour_hex}; }}")

    def _set_connector_table_visible_rows(self, row_count: int):
        for row_index, row_widgets in enumerate(self.connector_row_widgets, start=1):
            is_visible = row_index <= row_count
            for widget in row_widgets:
                widget.setVisible(is_visible)

    def _on_wire_type_changed(self, _index: int):
        self._refresh_wire_type_dropdowns()

    def _on_delay_mode_changed(self, _checked: bool) -> None:
        self.delay_ms_spin.setEnabled(self.delay_custom_radio.isChecked())

        if self._is_loading_camera_data or self.current_camera_port is None:
            return

        if self.delay_none_radio.isChecked():
            self.config_manager.set_delay_ms(self.current_camera_port, None)
        else:
            self.config_manager.set_delay_ms(self.current_camera_port, self.delay_ms_spin.value())

    def _on_delay_value_changed(self, _value: int) -> None:
        if self._is_loading_camera_data or self.current_camera_port is None:
            return
        if self.delay_custom_radio.isChecked():
            self.config_manager.set_delay_ms(self.current_camera_port, self.delay_ms_spin.value())

    def _refresh_wire_type_dropdowns(self):
        connector_name = self.connector_selector.currentData()
        row_count = CONNECTOR_PIN_COUNTS.get(connector_name, 0)

        wire_combos = [row_widgets[1] for row_widgets in self.connector_row_widgets[:row_count]]
        selected_values = [combo.currentText() for combo in wire_combos]
        canonical_values, used_counts = normalize_wire_type_selections(connector_name, selected_values)

        for combo, current_value in zip(wire_combos, canonical_values):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")

            for option in available_wire_type_options(connector_name, current_value, used_counts):
                combo.addItem(option)

            if current_value and combo.findText(current_value) >= 0:
                combo.setCurrentText(current_value)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def triggerbox_test_widget(self):
        triggerbox_test_group, triggerbox_test_layout = create_styled_groupbox("Trigger Box Testing")
        triggerbox_test_layout.addWidget(QLabel("Coming Soon -" \
        " This widget will ping the Pi or other triggering device for a connection."))
        self.bottom_left_vbox.addWidget(triggerbox_test_group)


    def timing_test_widget(self):
        timing_test_group, timing_test_layout = create_styled_groupbox("Timing Testing")

        intro_label = QLabel(
            "Enter camera FPS and phone refresh Hz to generate clear timestamp checkpoints."
        )
        intro_label.setWordWrap(True)

        input_grid = QGridLayout()

        self.video_fps_spin = QDoubleSpinBox()
        self.video_fps_spin.setRange(0.1, 1000.0)
        self.video_fps_spin.setDecimals(3)
        self.video_fps_spin.setValue(100.0)
        self.video_fps_spin.setSingleStep(1.0)

        self.phone_hz_spin = QDoubleSpinBox()
        self.phone_hz_spin.setRange(0.1, 1000.0)
        self.phone_hz_spin.setDecimals(3)
        self.phone_hz_spin.setValue(120.0)
        self.phone_hz_spin.setSingleStep(1.0)

        self.checkpoint_count_spin = QSpinBox()
        self.checkpoint_count_spin.setRange(1, 5000)
        self.checkpoint_count_spin.setValue(20)

        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 10_000_000)
        self.start_frame_spin.setValue(0)

        self.first_frame_time_minutes_spin = QSpinBox()
        self.first_frame_time_minutes_spin.setRange(0, 999)
        self.first_frame_time_minutes_spin.setValue(0)

        self.first_frame_time_seconds_spin = QSpinBox()
        self.first_frame_time_seconds_spin.setRange(0, 59)
        self.first_frame_time_seconds_spin.setValue(0)

        self.first_frame_time_milliseconds_spin = QSpinBox()
        self.first_frame_time_milliseconds_spin.setRange(0, 999)
        self.first_frame_time_milliseconds_spin.setValue(0)

        first_frame_time_layout = QHBoxLayout()
        first_frame_time_layout.addWidget(self.first_frame_time_minutes_spin)
        first_frame_time_layout.addWidget(QLabel(":"))
        first_frame_time_layout.addWidget(self.first_frame_time_seconds_spin)
        first_frame_time_layout.addWidget(QLabel(":"))
        first_frame_time_layout.addWidget(self.first_frame_time_milliseconds_spin)
        first_frame_time_layout.addStretch(1)

        timer_video_placeholder_label = QLabel(
            "Timer-frame video slice display is planned for an upcoming release."
        )
        timer_video_placeholder_label.setWordWrap(True)

        input_grid.addWidget(QLabel("Video FPS"), 0, 0)
        input_grid.addWidget(self.video_fps_spin, 0, 1)
        input_grid.addWidget(QLabel("Phone Screen Hz"), 1, 0)
        input_grid.addWidget(self.phone_hz_spin, 1, 1)
        input_grid.addWidget(QLabel("Checkpoints"), 2, 0)
        input_grid.addWidget(self.checkpoint_count_spin, 2, 1)
        input_grid.addWidget(QLabel("Start Frame"), 3, 0)
        input_grid.addWidget(self.start_frame_spin, 3, 1)
        input_grid.addWidget(QLabel("Time at Frame 0 (mm:ss:ms)"), 4, 0)
        input_grid.addLayout(first_frame_time_layout, 4, 1)
        input_grid.addWidget(QLabel("Timer video path:"), 5, 0)
        input_grid.addWidget(timer_video_placeholder_label, 5, 1)

        self.generate_timing_plan_btn = QPushButton("Generate Timing Checkpoints")
        self.generate_timing_plan_btn.clicked.connect(self._generate_timing_plan)

        display_mode_row = QHBoxLayout()
        display_mode_row.addWidget(QLabel("Display Format"))

        self.display_ms_radio = QRadioButton("ms")
        self.display_ms_radio.setChecked(True)
        self.display_clock_radio = QRadioButton("mm:ss:ms")

        self.display_mode_group = QButtonGroup(self)
        self.display_mode_group.addButton(self.display_ms_radio)
        self.display_mode_group.addButton(self.display_clock_radio)

        self.display_ms_radio.toggled.connect(self._generate_timing_plan)
        self.display_clock_radio.toggled.connect(self._generate_timing_plan)

        display_mode_row.addWidget(self.display_ms_radio)
        display_mode_row.addWidget(self.display_clock_radio)
        display_mode_row.addStretch(1)

        self.timing_plan_summary_label = QLabel("")
        self.timing_plan_summary_label.setWordWrap(True)

        self.timing_plan_output = QPlainTextEdit()
        self.timing_plan_output.setReadOnly(True)
        self.timing_plan_output.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.timing_plan_output.setPlaceholderText(
            "Frame checkpoints and expected phone timer values will appear here."
        )
        self.timing_plan_output.setMinimumHeight(180)

        timing_test_layout.addWidget(intro_label)
        timing_test_layout.addLayout(input_grid)
        timing_test_layout.addWidget(self.generate_timing_plan_btn)
        timing_test_layout.addLayout(display_mode_row)
        timing_test_layout.addWidget(self.timing_plan_summary_label)
        timing_test_layout.addWidget(self.timing_plan_output)

        self._generate_timing_plan()
        self.bottom_right_vbox.addWidget(timing_test_group)

    def _generate_timing_plan(self) -> None:
        """Generate frame checkpoints and expected phone timer readings."""
        fps = self.video_fps_spin.value()
        phone_hz = self.phone_hz_spin.value()
        checkpoint_count = self.checkpoint_count_spin.value()
        start_frame = self.start_frame_spin.value()
        first_frame_time_ms = self._get_first_frame_time_ms()

        try:
            plan = build_timing_sync_plan(
                video_fps=fps,
                phone_refresh_hz=phone_hz,
                checkpoint_count=checkpoint_count,
                start_frame=start_frame,
            )
        except ValueError as exc:
            self.timing_plan_summary_label.setText(f"Invalid input: {exc}")
            self.timing_plan_output.setPlainText("")
            return

        self.timing_plan_summary_label.setText(
            f"Clear-read interval: every {plan.frame_interval} frame(s), "
            f"approximately every {plan.interval_ms:.3f} ms."
        )

        use_clock_format = self.display_clock_radio.isChecked()
        table_text = render_timing_sync_table(
            plan,
            first_frame_time_ms=first_frame_time_ms,
            use_clock_format=use_clock_format,
        )
        self.timing_plan_output.setPlainText(table_text)

    def _get_first_frame_time_ms(self) -> int:
        """Convert mm:ss:ms input fields into a total millisecond offset."""
        minutes = self.first_frame_time_minutes_spin.value()
        seconds = self.first_frame_time_seconds_spin.value()
        milliseconds = self.first_frame_time_milliseconds_spin.value()
        return (minutes * 60_000) + (seconds * 1_000) + milliseconds

    def _camera_selection_entries(self) -> list[tuple[int, str]]:
        camera_array = getattr(self.controller, "camera_array", None)
        cameras = getattr(camera_array, "cameras", None)
        if not cameras:
            return [(1, resolve_camera_title_color(1, 1))]

        ports = sorted(cameras.keys())
        camera_count = len(ports)
        entries = []
        for position, port in enumerate(ports, start=1):
            entries.append(
                (
                    port,
                    resolve_camera_title_color(
                        position,
                        camera_count,
                        camera_data=cameras.get(port),
                    ),
                )
            )
        return entries

    def _load_initial_camera_data(self) -> None:
        """Load data for the first camera when widget initializes."""
        if self.wire_camera_selector.count() > 0:
            self.wire_camera_selector.setCurrentIndex(0)
            self._on_camera_selection_changed(0)

    def _on_camera_selection_changed(self, index: int) -> None:
        """Handle camera selection change by loading saved data."""
        if index < 0:
            return

        self.current_camera_port = self.wire_camera_selector.itemData(index)
        if self.current_camera_port is None:
            return

        self._load_camera_data()

    def _load_camera_data(self) -> None:
        """Load all saved data for the current camera."""
        if self.current_camera_port is None:
            return

        self._is_loading_camera_data = True
        # Block signals to prevent triggering save operations during load
        self.connector_selector.blockSignals(True)

        try:
            # Load connector type - use current selection if none is saved
            connector_type, wire_rows = self.config_manager.get_connector_configuration(self.current_camera_port)
            if connector_type:
                index = self.connector_selector.findData(connector_type)
                if index >= 0:
                    self.connector_selector.setCurrentIndex(index)
            else:
                # No saved connector type - save the currently selected one as default
                current_connector = self.connector_selector.currentData()
                if current_connector:
                    self.config_manager.set_connector_type(self.current_camera_port, current_connector)

            # Update visible rows and preview based on current connector
            connector_name = self.connector_selector.currentData()
            row_count = CONNECTOR_PIN_COUNTS.get(connector_name, 0)
            self._set_connector_table_visible_rows(row_count)

            # Load wire row data
            for row_index, row_widgets in enumerate(self.connector_row_widgets, start=1):
                row_key = str(row_index)
                row_data = wire_rows.get(row_key, {})
                wire_type = row_data.get("wire_type", "")
                colour = row_data.get("colour", "")

                port_label, wire_type_combo, colour_combo = row_widgets

                # Set wire type
                wire_type_combo.blockSignals(True)
                wire_type_combo.clear()
                # Add all available options for this connector
                connector_name = self.connector_selector.currentData()
                options = WIRE_TYPE_OPTIONS.get(connector_name, [])
                wire_type_combo.addItem("")
                for option in options:
                    wire_type_combo.addItem(option)
                # Set to saved value or empty
                if wire_type and wire_type_combo.findText(wire_type) >= 0:
                    wire_type_combo.setCurrentText(wire_type)
                else:
                    wire_type_combo.setCurrentIndex(0)
                wire_type_combo.blockSignals(False)

                # Set colour
                colour_combo.blockSignals(True)
                if colour:
                    idx = colour_combo.findData(colour)
                    if idx >= 0:
                        colour_combo.setCurrentIndex(idx)
                else:
                    colour_combo.setCurrentIndex(0)
                colour_combo.blockSignals(False)
                # Apply styling to show the selected color
                self._update_colour_combo_style(colour_combo)

            # Re-render preview after table values for this camera are loaded.
            self._update_preview_image()

            delay_ms = self.config_manager.get_delay_ms(self.current_camera_port)
            self.delay_none_radio.blockSignals(True)
            self.delay_custom_radio.blockSignals(True)
            self.delay_ms_spin.blockSignals(True)

            if delay_ms is None:
                self.delay_none_radio.setChecked(True)
            else:
                self.delay_custom_radio.setChecked(True)
                self.delay_ms_spin.setValue(delay_ms)

            self.delay_ms_spin.setEnabled(self.delay_custom_radio.isChecked())

            self.delay_ms_spin.blockSignals(False)
            self.delay_custom_radio.blockSignals(False)
            self.delay_none_radio.blockSignals(False)
        finally:
            self.connector_selector.blockSignals(False)
            self._is_loading_camera_data = False

    def _on_connector_changed(self, index: int) -> None:
        """Save connector type when changed and preserve applicable wire row values."""
        if self._is_loading_camera_data:
            return

        if self.current_camera_port is None:
            return

        connector_type = self.connector_selector.itemData(index)
        if connector_type:
            pin_count = CONNECTOR_PIN_COUNTS.get(connector_type, 0)

            applicable_rows: dict[str, dict[str, str]] = {}
            for row_index in range(1, pin_count + 1):
                if row_index <= len(self.connector_row_widgets):
                    port_label, wire_type_combo, colour_combo = self.connector_row_widgets[row_index - 1]
                    wire_type = wire_type_combo.currentText()
                    colour_hex = colour_combo.currentData()
                    applicable_rows[str(row_index)] = {
                        "wire_type": wire_type or "",
                        "colour": colour_hex or "",
                    }

            self.config_manager.save_connector_configuration(
                self.current_camera_port,
                connector_type,
                applicable_rows,
            )

    def _on_table_data_changed(self, row_index: int) -> None:
        """Save wire row data when changed."""
        if self._is_loading_camera_data:
            return

        if self.current_camera_port is None:
            return

        if row_index < 1 or row_index > len(self.connector_row_widgets):
            return

        port_label, wire_type_combo, colour_combo = self.connector_row_widgets[row_index - 1]
        wire_type = wire_type_combo.currentText()
        colour_hex = colour_combo.currentData()

        self.config_manager.set_wire_row(
            self.current_camera_port,
            row_index,
            wire_type or "",
            colour_hex or "",
        )
        self._update_preview_image()
