"""This widget acts as a guide for creating a camera trigger circuit and wiring guidelines"""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller

logger = calipod_logger.get(__name__)



class CircuitManagementWidget(QWidget):
    def __init__(self, controller: Controller):
        super(CircuitManagementWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        temp_to_do_widget = QLabel("To do: circuit management widget")
        temp_to_do_widget.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(temp_to_do_widget)
        self.setLayout(layout)
