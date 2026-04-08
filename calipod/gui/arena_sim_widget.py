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
    QListWidgetItem,
    QDoubleSpinBox,
    QGroupBox,
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer

from calipod.gui.utils.spinbox_utils import create_labeled_spinbox_row
from calipod.arena_simulation.arena_designer import ArenaDesignerVisualizer
import calipod.logger
from calipod.controller import Controller

logger = calipod.logger.get(__name__)

# Arena sim widget - this is a widget that simulates the camera arrangement to check for frustum overlap (triangulatable) and ensures the arena is within this overlap.
class ArenaSimWidget(QWidget):
    def __init__(self, controller: Controller):
        super(ArenaSimWidget, self).__init__()
        self.controller = controller
        self.cameras = self.controller.get_camera_count()


        # Arena designer starts from an empty scene; controls can populate it later.
        self.visualizer = ArenaDesignerVisualizer()
        
        #calculated pixel-to-animal values
        self.furthest_distance_mm = None
        self.insect_pixel_count = None

        self.place_widgets()

    # Helper function to create section titles
    def _create_section_font(self, point_size: int = 11) -> QFont:
        """Create a styled font for section headers."""
        font = QFont()
        font.setBold(True)
        font.setPointSize(point_size)
        return font

    def _create_section_title(self, text: str) -> QLabel:
        """Create a styled section title label."""
        title = QLabel(text)
        title.setFont(self._create_section_font(11))
        return title
    
    # Helper function to create subsection titles
    def _create_subsection_title(self, text: str) -> QLabel:
        """Create a styled subsection title label."""
        title = QLabel(text)
        title.setFont(self._create_section_font(10))
        return title

    def _create_styled_groupbox(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """Create a QGroupBox with styled section title (returns group and layout)."""
        group = QGroupBox()
        layout = QVBoxLayout()
        
        # Add styled title as a label, not as QGroupBox title
        title_label = self._create_section_title(title)
        layout.addWidget(title_label)
        
        group.setLayout(layout)
        return group, layout

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()

        # Simulation Parameters
        self.simulation_parameters_widget()

        # Lens Parameters
        self.lens_widget(self.cameras)

        # Pixel-To-Animal Calculation
        self.pixel_to_animal_widget()

        self.left_vbox.addStretch()

        # Simulation visualizer
        self.right_vbox.addWidget(self.visualizer.scene, stretch=2)

        # Camera Movement Controls
        self.cam_controls_widget()

        # Add left and right vboxes to main layout
        self.layout().addLayout(self.left_vbox, stretch=1)
        self.layout().addLayout(self.right_vbox, stretch=2)

    # Widgets for simulation parameters
    def simulation_parameters_widget(self):
        params_group, params_layout = self._create_styled_groupbox("Simulation Parameters")

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

        params_layout.addWidget(self.camera_count_label)
        create_labeled_spinbox_row(params_layout, "Arena Scale/ Depth (cm):", 0.1, 10000.0, decimals=1)
        create_labeled_spinbox_row(params_layout, "Visualised Frustum Depth:", 0.1, 10000.0, decimals=2)
        params_layout.addLayout(self.overlap_visualization_layout)
        params_layout.addStretch()
        
        self.left_vbox.addWidget(params_group)

    # Widgets for lens parameters
    def lens_widget(self, cam_num):
        lens_group, lens_layout = self._create_styled_groupbox("Lens Angles")

        # Create lens angle entries based on camera count
        for i in range(cam_num):
            cam_label = self._create_subsection_title(f"Camera {i+1}")
            lens_layout.addWidget(cam_label)
            create_labeled_spinbox_row(lens_layout, "Min Working Distance (mm):", 1, 100000.00)
            create_labeled_spinbox_row(lens_layout, "Horizontal Angle (deg):", 1, 360.00)
            create_labeled_spinbox_row(lens_layout, "Vertical Angle (deg):", 1, 360.00)
        
        lens_layout.addStretch()
        self.left_vbox.addWidget(lens_group)

    def pixel_to_animal_widget(self):
        # Widgets for pixel to animal calculation parameters
        pixel_group, pixel_layout = self._create_styled_groupbox("Pixel-To-Animal Calculation")
        
        create_labeled_spinbox_row(pixel_layout, "Min Insect Size (mm):", 0.1, 1000.0)
        create_labeled_spinbox_row(pixel_layout, "Pixel Size on Sensor (μm):", 0.001, 10.0)
        create_labeled_spinbox_row(pixel_layout, "Furthest Distance (mm):", 0.1, 10000.0)
        create_labeled_spinbox_row(pixel_layout, "Focal Length (mm):", 0.1, 1000.0)

        #results
        insect_pixel_count_result = QHBoxLayout()
        self.insect_pixel_count_label = self._create_subsection_title("Insect size:") # to be updated with actual calculation
        self.insect_pixel_count_value =  QLabel(f"{self.insect_pixel_count} pixels @ furthest distance")
        insect_pixel_count_result.addWidget(self.insect_pixel_count_label)
        insect_pixel_count_result.addWidget(self.insect_pixel_count_value)
        pixel_layout.addLayout(insect_pixel_count_result)


        furthest_distance_mm_result = QHBoxLayout()
        self.furthest_distance_mm_label = self._create_subsection_title("Furthest Distance:")
        self.furthest_distance_mm_value = QLabel(f"{self.furthest_distance_mm} mm") # to be updated with actual value, given if entered or calculated from lens parameters
        
        furthest_distance_mm_result.addWidget(self.furthest_distance_mm_label)
        furthest_distance_mm_result.addWidget(self.furthest_distance_mm_value)
        pixel_layout.addLayout(furthest_distance_mm_result)

        pixel_layout.addStretch()
        
        self.left_vbox.addWidget(pixel_group)



    # Camera placement controls - sliders to adjust camera position and orientation within the visualizer, with the option to sync these to the extrinsic calibration values for each camera once calibrated.
    def cam_controls_widget(self):
        controls_group, controls_layout = self._create_styled_groupbox("Camera Placement Controls")
        self.cam_placement = QListWidget()

        # For each camera, create control sliders to adjust position and orientation, displayed left to right with scrollbar
        for i in range(self.cameras):
            cam_label = self._create_subsection_title(f"Camera {i+1}")
            item = QListWidgetItem()
            self.cam_placement.addItem(item)
            self.cam_placement.setItemWidget(item, cam_label)

            # Create sliders for X, Y, Z position and Pan, Tilt, Roll orientation
            for param in ["X Position (mm)", "Y Position (mm)", "Z Position (mm)", "Pan (deg)", "Tilt (deg)", "Roll (deg)"]:
                slider_layout = QHBoxLayout()
                slider_label = QLabel(param)
                
                # Spinbox to display/edit the value
                value_spinbox = QDoubleSpinBox()
                value_spinbox.setMinimum(-1000)
                value_spinbox.setMaximum(1000)
                value_spinbox.setValue(0)
                value_spinbox.setMaximumWidth(80)
                value_spinbox.setMinimumHeight(24)
                
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(-1000)
                slider.setMaximum(1000)
                slider.setValue(0)
                
                # Sync spinbox and slider
                value_spinbox.valueChanged.connect(slider.setValue)
                slider.valueChanged.connect(value_spinbox.setValue)
                
                slider_layout.addWidget(slider_label)
                slider_layout.addWidget(value_spinbox)
                slider_layout.addWidget(slider)
                self.cam_placement.addItem(QListWidgetItem())
                self.cam_placement.setItemWidget(self.cam_placement.item(self.cam_placement.count()-1), QWidget())
                self.cam_placement.itemWidget(self.cam_placement.item(self.cam_placement.count()-1)).setLayout(slider_layout)

        controls_layout.addWidget(self.cam_placement, stretch=1)
        self.right_vbox.addWidget(controls_group, stretch=1)


    def connect_widgets(self):
        pass
        
