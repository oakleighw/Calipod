from pathlib import Path
import re
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
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
    QRadioButton,
    QButtonGroup,
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer

from calipod.gui.utils.spinbox_utils import create_labeled_spinbox_row
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

    # Helper function to create section titles
    def _create_section_title(self, text: str) -> QLabel:
        """Create a styled section title label."""
        title = QLabel(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        title.setFont(font)
        return title
    
    # Helper function to create subsection titles
    def _create_subsection_title(self, text: str) -> QLabel:
        """Create a styled subsection title label."""
        title = QLabel(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        title.setFont(font)
        return title

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()

        # Simulation Parameters
        self.simulation_parameters_widget()

        # Lens Parameters
        self.lens_widget(self.cameras)

        # Pixel-To-Animal Calculation
        self.pixel_to_animal_title = self._create_section_title("Pixel-To-Animal Calculation")
        self.pixel_to_animal = QListWidget()

        self.left_vbox.addWidget(self.pixel_to_animal_title, stretch=0)
        self.left_vbox.addWidget(self.pixel_to_animal, stretch=0)
        self.left_vbox.addStretch()

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

    # Widgets for simulation parameters
    def simulation_parameters_widget(self):

        self.parameter_title = self._create_section_title("Simulation Parameters")

        # Camera count entry
        self.camera_count_label = QLabel(f"Camera count: {self.cameras}")

        # Overlap visualisation toggle
        self.overlap_visualization_layout = QVBoxLayout()
        self.overlap_visualization_group = QButtonGroup()
        
        overlap_min = QRadioButton("Min Coverage (2 cameras)")
        overlap_max = QRadioButton("Max Coverage (All cameras)")
        self.overlap_visualization_group.addButton(overlap_min, 0)
        self.overlap_visualization_group.addButton(overlap_max, 1)
        
        self.overlap_visualization_layout.addWidget(overlap_min)
        self.overlap_visualization_layout.addWidget(overlap_max)
        self.overlap_visualization_layout.addStretch()


        self.left_vbox.addWidget(self.parameter_title)
        self.left_vbox.addWidget(self.camera_count_label)
                # Arena scale
        create_labeled_spinbox_row(self.left_vbox, "Arena Scale/ Depth (cm):", 0.1, 10000.0, decimals=1)

        # Visualised frustum depth (How far the frustum extends in the visualizer - does not affect calculations)
        create_labeled_spinbox_row(self.left_vbox, "Visualised Frustum Depth:", 0.1, 10000.0, decimals=2)

        # Overlap visualisation toggle
        self.left_vbox.addLayout(self.overlap_visualization_layout)

    # Widgets for lens parameters
    def lens_widget(self, cam_num):

         # Lens angles entry
        self.lens_angles_title = self._create_section_title("Lens Angles")

        self.left_vbox.addWidget(self.lens_angles_title)

        # Create lens angle entries based on camera count
        for i in range(cam_num):
            cam_label = self._create_subsection_title(f"Camera {i+1}")
            self.left_vbox.addWidget(cam_label)
            create_labeled_spinbox_row(self.left_vbox, "Min Working Distance (mm):", 1, 100000.00)
            create_labeled_spinbox_row(self.left_vbox, "Horizontal Angle (deg):", 1, 360.00)
            create_labeled_spinbox_row(self.left_vbox, "Vertical Angle (deg):", 1, 360.00)




        
