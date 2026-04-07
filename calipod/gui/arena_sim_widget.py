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
        self.layout().addWidget(self.visualizer.scene, stretch=2)

        
