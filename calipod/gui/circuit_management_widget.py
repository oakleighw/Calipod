"""This widget acts as a guide for creating a camera trigger circuit and wiring guidelines"""

from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.styles import create_styled_groupbox, resolve_camera_title_color
from circuit_management.connector_preview_renderer import render_connector_preview_frame
from circuit_management.config_manager import CircuitManagementConfigManager

logger = calipod_logger.get(__name__)

CONNECTOR_IMAGE_PATHS = {
    "Hirose 6 Pin": Path(__file__).parent / "icons" / "connectors" / "6_pin_hirose_female.png",
    "Hirose 4 Pin": Path(__file__).parent / "icons" / "connectors" / "4_pin_hirose_female.png",
}

CONNECTOR_PIN_COUNTS = {
    "Hirose 6 Pin": 6,
    "Hirose 4 Pin": 4,
}

WIRE_TYPE_OPTIONS = {
    "Hirose 4 Pin": [
        "External Ground",
        "Octo-coupled Output",
        "Octo-coupled Ground",
        "Octo-coupled Input",
    ],
    "Hirose 6 Pin": [
        "General purpose I/O (GPIO) line",
        "Octo-coupled Output",
        "Octo-coupled Input",
        "GPIO Ground",
        "Octo-coupled Ground",
    ],
}

WIRE_TYPE_MAX_COUNTS = {
    "Hirose 4 Pin": {
        "External Ground": 1,
        "Octo-coupled Output": 1,
        "Octo-coupled Ground": 1,
        "Octo-coupled Input": 1,
    },
    "Hirose 6 Pin": {
        "General purpose I/O (GPIO) line": 2,
        "Octo-coupled Output": 1,
        "Octo-coupled Input": 1,
        "GPIO Ground": 1,
        "Octo-coupled Ground": 1,
    },
}

WIRE_COLOUR_OPTIONS = [
    ("black", "#000000"),
    ("white", "#6E6E6E"),
    ("red", "#D32F2F"),
    ("green", "#2E7D32"),
    ("brown", "#795548"),
    ("blue", "#1976D2"),
    ("orange", "#EF6C00"),
    ("yellow", "#C9A200"),
    ("violet", "#7B1FA2"),
    ("grey", "#616161"),
    ("pink", "#C2185B"),
    ("light blue", "#2A9DDF"),
]


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
        self.bottom_vbox = QVBoxLayout()

        self.bottom_container = QWidget()
        self.bottom_container.setLayout(self.bottom_vbox)
        self.bottom_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.wire_labeling_widget()
        self.triggerbox_test_widget()

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addLayout(self.bottom_vbox, stretch=1)

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

    def _refresh_wire_type_dropdowns(self):
        connector_name = self.connector_selector.currentData()
        row_count = CONNECTOR_PIN_COUNTS.get(connector_name, 0)
        options = WIRE_TYPE_OPTIONS.get(connector_name, [])
        max_counts = WIRE_TYPE_MAX_COUNTS.get(connector_name, {})

        wire_combos = [row_widgets[1] for row_widgets in self.connector_row_widgets[:row_count]]
        canonical_values: list[str] = []
        used_counts: dict[str, int] = {}

        for combo in wire_combos:
            value = combo.currentText().strip()
            max_allowed = max_counts.get(value, 1)
            if value in options and used_counts.get(value, 0) < max_allowed:
                canonical_values.append(value)
                used_counts[value] = used_counts.get(value, 0) + 1
            else:
                canonical_values.append("")

        for combo, current_value in zip(wire_combos, canonical_values):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")

            for option in options:
                used_elsewhere = used_counts.get(option, 0) - (1 if current_value == option else 0)
                max_allowed = max_counts.get(option, 1)
                if used_elsewhere < max_allowed:
                    combo.addItem(option)

            if current_value and combo.findText(current_value) >= 0:
                combo.setCurrentText(current_value)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def triggerbox_test_widget(self):
        triggerbox_test_group, triggerbox_test_layout = create_styled_groupbox("Trigger Box Testing")
        triggerbox_test_layout.addWidget(QLabel("Coming Soon!"))
        self.top_vbox.addWidget(triggerbox_test_group)

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
            connector_type = self.config_manager.get_connector_type(self.current_camera_port)
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
            wire_rows = self.config_manager.get_all_wire_rows(self.current_camera_port)
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

            # First, clear all existing wire rows
            self.config_manager.clear_all_wire_rows(self.current_camera_port)

            # Then save only the rows that apply to the new connector
            for row_index in range(1, pin_count + 1):
                if row_index <= len(self.connector_row_widgets):
                    port_label, wire_type_combo, colour_combo = self.connector_row_widgets[row_index - 1]
                    wire_type = wire_type_combo.currentText()
                    colour_hex = colour_combo.currentData()
                    self.config_manager.set_wire_row(
                        self.current_camera_port,
                        row_index,
                        wire_type or "",
                        colour_hex or "",
                    )

            # Now update the connector type
            self.config_manager.set_connector_type(self.current_camera_port, connector_type)

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
