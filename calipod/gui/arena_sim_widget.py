from pathlib import Path
import re
import cv2
import numpy as np


from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTreeWidget,
    QListWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QMessageBox,
    QSlider,
    QCheckBox,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

from calipod.gui.vizualize.calibration.capture_volume_visualizer import CaptureVolumeVisualizer
import calipod.logger
from calipod.controller import Controller

logger = calipod.logger.get(__name__)

# Arena sim widget - this is a widget that simulates the camera arrangement to check for frustum overlap (triangulatable) and ensures the arena is within this overlap.
class ArenaSimWidget(QWidget):
    def __init__(self, controller: Controller):
        super(ArenaSimWidget, self).__init__()
        self.controller = controller

        #arena designer window placeholder (capture volume vizualiser for now)
        self.visualizer = CaptureVolumeVisualizer(self.controller.capture_volume)
        # self.visualizer.scene.show()
        self.place_widgets()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()

        self.sim_parameters = QListWidget()
        self.parameter_title = QLabel("Simulation Parameters")

        self.left_vbox.addWidget(self.parameter_title)
        self.left_vbox.addWidget(self.sim_parameters)


        self.layout().addLayout(self.right_vbox, stretch=2)
        self.layout().addLayout(self.left_vbox, stretch=2)

        self.right_vbox.addWidget(self.visualizer.scene, stretch=2)

        
