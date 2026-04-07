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
from PySide6.QtGui import QImage, QPixmap, QFont
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
        self.cameras = self.controller.get_camera_count()


        #arena designer window placeholder (capture volume vizualiser for now)
        self.visualizer = CaptureVolumeVisualizer(self.controller.capture_volume)
        # self.visualizer.scene.show()
        self.place_widgets()

    def _create_section_title(self, text: str) -> QLabel:
        """Create a styled section title label."""
        title = QLabel(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        title.setFont(font)
        return title

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()

        # Simulation parameters
        self.parameter_title = self._create_section_title("Simulation Parameters")
        self.camera_count_label = QLabel(f"Camera count: {self.cameras}")


        self.left_vbox.addWidget(self.parameter_title)
        self.left_vbox.addWidget(self.camera_count_label)

        # Lens angles entry
        self.lens_angles_title = self._create_section_title("Lens Angles")

        self.left_vbox.addWidget(self.lens_angles_title)

        # Create lens angle entries based on camera count
        for i in range(self.cameras):
            item_hlabel = QLabel(f"Camera {i+1} Horizontal Lens Angle:")
            item_hangle_spinbox = QSpinBox()
            item_vlabel = QLabel(f"Camera {i+1} Vertical Lens Angle:")
            item_vlabel_spinbox = QSpinBox()

            self.left_vbox.addWidget(item_hlabel)
            self.left_vbox.addWidget(item_hangle_spinbox)
            self.left_vbox.addWidget(item_vlabel)
            self.left_vbox.addWidget(item_vlabel_spinbox)

        # Pixel-To-Animal Calculation
        self.pixel_to_animal_title = self._create_section_title("Pixel-To-Animal Calculation")
        self.pixel_to_animal = QListWidget()

        self.left_vbox.addWidget(self.pixel_to_animal_title)
        self.left_vbox.addWidget(self.pixel_to_animal)

        # Simulation visualizer
        self.right_vbox.addWidget(self.visualizer.scene, stretch=2)

        # Camera Movement Controls
        self.cam_controls = self._create_section_title("Camera Placement Controls")
        self.cam_placement = QListWidget()

        self.right_vbox.addWidget(self.cam_controls, stretch=0)
        self.right_vbox.addWidget(self.cam_placement, stretch=1)

        # Add left and right vboxes to main layout
        self.layout().addLayout(self.left_vbox, stretch=1)
        self.layout().addLayout(self.right_vbox, stretch=2)



        
