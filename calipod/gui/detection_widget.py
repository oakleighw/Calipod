"""This widget will supply a basic detection tool and also point to Jupyter Notebook / Wandb for extended functionality."""


from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
import calipod.logger
from calipod.controller import Controller

logger = calipod.logger.get(__name__)



class DetectionWidget(QWidget):
    def __init__(self, controller: Controller):
        super(DetectionWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        temp_to_do_widget = QLabel("To do: detection widget")
        temp_to_do_widget.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(temp_to_do_widget)
        self.setLayout(layout)