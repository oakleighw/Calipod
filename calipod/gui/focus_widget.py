"""This widget will accept videos or images,
allow the user to crop-annotate a bounding box,
and will perform focus calculations on the bounding box"""


from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller

logger = calipod_logger.get(__name__)


class FocusWidget(QWidget):
    def __init__(self, controller: Controller):
        super(FocusWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        temp_to_do_widget = QLabel("To do: focus widget")
        temp_to_do_widget.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(temp_to_do_widget)
        self.setLayout(layout)
