"""This widget acts as a guide for creating a camera trigger circuit and wiring guidelines"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.styles import create_styled_groupbox, resolve_camera_title_color

logger = calipod_logger.get(__name__)

CONNECTOR_IMAGE_PATHS = {
    "Hirose 6 Pin": Path(__file__).parent / "icons" / "connectors" / "6_pin_hirose_female.png",
    "Hirose 4 Pin": Path(__file__).parent / "icons" / "connectors" / "4_pin_hirose_female.png",
}


class CircuitManagementWidget(QWidget):
    def __init__(self, controller: Controller):
        super(CircuitManagementWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

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

        self.connector_selector = QComboBox()
        for connector_name in CONNECTOR_IMAGE_PATHS:
            self.connector_selector.addItem(connector_name, userData=connector_name)
        self.connector_selector.currentIndexChanged.connect(self._update_connector_preview)

        self.connector_preview = QLabel()
        self.connector_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connector_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        selector_row = QHBoxLayout()
        selector_row.addWidget(self.wire_camera_selector)
        selector_row.addWidget(self.connector_selector)

        wire_label_layout.addLayout(selector_row)
        wire_label_layout.addWidget(self.connector_preview)
        self._update_connector_preview(self.connector_selector.currentIndex())
        self.top_vbox.addWidget(wire_label_group)

    def _update_connector_preview(self, _index: int):
        connector_name = self.connector_selector.currentData()
        image_path = CONNECTOR_IMAGE_PATHS.get(connector_name)
        if image_path is None or not image_path.exists():
            self.connector_preview.setText("Connector image not found")
            self.connector_preview.setPixmap(QPixmap())
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.connector_preview.setText("Connector image not found")
            self.connector_preview.setPixmap(QPixmap())
            return

        self.connector_preview.setText("")
        self.connector_preview.setPixmap(
            pixmap.scaled(
                320,
                220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

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
