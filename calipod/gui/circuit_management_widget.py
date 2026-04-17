"""This widget acts as a guide for creating a camera trigger circuit and wiring guidelines"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.styles import create_styled_groupbox, resolve_camera_title_color

logger = calipod_logger.get(__name__)


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
        wire_label_layout.addWidget(self.wire_camera_selector)
        self.top_vbox.addWidget(wire_label_group)

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
