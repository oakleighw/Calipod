""" This widget will link to camera capture software such as Mokap"""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller

logger = calipod_logger.get(__name__)



class CameraCaptureWidget(QWidget):
    def __init__(self, controller: Controller):
        super(CameraCaptureWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        temp_to_do_widget = QLabel("To do: camera capture widget")
        temp_to_do_widget.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(temp_to_do_widget)
        self.setLayout(layout)

