from pathlib import Path
import re
from itertools import combinations
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
    QSizePolicy,
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer

from calipod.gui.utils.spinbox_utils import create_labeled_spinbox_row
from calipod.arena_simulation.arena_designer import ArenaDesignerVisualizer
from calipod.arena_simulation.arena_config_manager import ArenaConfigManager
from calipod.arena_simulation.arena_matplotlib_graph import ArenaMatplotlibGraphWindow
import calipod.logger
from calipod.controller import Controller

logger = calipod.logger.get(__name__)

# Arena sim widget - this is a widget that simulates the camera arrangement to check for frustum overlap (triangulatable) and ensures the arena is within this overlap.
class ArenaSimWidget(QWidget):
    def __init__(self, controller: Controller):
        super(ArenaSimWidget, self).__init__()
        self.controller = controller
        self.cameras = self.controller.get_camera_count()


        # Arena designer starts with camera cubes and can be updated by controls later.
        self.visualizer = ArenaDesignerVisualizer(camera_count=self.cameras)
        
        #calculated pixel-to-animal values
        self.furthest_distance_mm = None
        self.insect_pixel_count = None

        self.arena_config_manager = ArenaConfigManager(self.controller.workspace_guide.arena_sim_dir)

        self.place_widgets()
        self.load_arena_config()
        self.connect_widgets()

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
    def _create_subsection_title(self, text: str, color: str | None = None) -> QLabel:
        """Create a styled subsection title label."""
        title = QLabel(text)
        title.setFont(self._create_section_font(10))
        if color is not None:
            title.setStyleSheet(f"color: {color};")
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

        self.save_arena_config_button = QPushButton("Save Arena Config")
        self.generate_matplotl_graph_button = QPushButton("Generate Matplotlib Graph")
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_arena_config_button)
        button_row.addWidget(self.generate_matplotl_graph_button)
        self.right_vbox.addLayout(button_row, stretch=0)

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
        
        self.overlap_min_radio = QRadioButton("Min Coverage (2 cameras)")
        self.overlap_max_radio = QRadioButton("Max Coverage (All cameras)")
        self.overlap_visualization_group.addButton(self.overlap_min_radio, 0)
        self.overlap_visualization_group.addButton(self.overlap_max_radio, 1)
        self.overlap_min_radio.setChecked(True)
        
        self.overlap_visualization_layout.addWidget(self.overlap_min_radio)
        self.overlap_visualization_layout.addWidget(self.overlap_max_radio)
        self.overlap_visualization_layout.addStretch()

        params_layout.addWidget(self.camera_count_label)
        self.arena_scale_depth_cm = create_labeled_spinbox_row(
            params_layout,
            "Arena Scale/ Depth (cm):",
            0.1,
            10000.0,
            decimals=1,
        )
        self.arena_scale_depth_cm.setValue(100.0)
        self.visualised_frustum_depth = create_labeled_spinbox_row(
            params_layout,
            "Visualised Frustum Depth (cm):",
            0.1,
            10000.0,
            decimals=2,
        )
        self.visualised_frustum_depth.setValue(100.0)
        params_layout.addLayout(self.overlap_visualization_layout)
        params_layout.addStretch()
        
        self.left_vbox.addWidget(params_group)

    # Widgets for lens parameters
    def lens_widget(self, cam_num):
        lens_group, lens_layout = self._create_styled_groupbox("Lens Angles")
        self.lens_angle_spinboxes = {}

        # Create lens angle entries based on camera count
        for i in range(cam_num):
            cam_label = self._create_subsection_title(f"Camera {i+1}")
            lens_layout.addWidget(cam_label)
            min_working_distance_spinbox = create_labeled_spinbox_row(lens_layout, "Min Working Distance (mm):", 1, 100000.00)
            horizontal_spinbox = create_labeled_spinbox_row(lens_layout, "Horizontal Angle (deg):", 1, 360.00)
            vertical_spinbox = create_labeled_spinbox_row(lens_layout, "Vertical Angle (deg):", 1, 360.00)

            self.lens_angle_spinboxes[i] = {
                "min_working_distance": min_working_distance_spinbox,
                "horizontal": horizontal_spinbox,
                "vertical": vertical_spinbox,
            }

            min_working_distance_spinbox.valueChanged.connect(lambda _, cam_index=i: self._update_camera_min_working_distance(cam_index))
            horizontal_spinbox.valueChanged.connect(lambda _, cam_index=i: self._update_camera_frustum(cam_index))
            vertical_spinbox.valueChanged.connect(lambda _, cam_index=i: self._update_camera_frustum(cam_index))

            self._update_camera_min_working_distance(i)
            self._update_camera_frustum(i)
        
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
        controls_row = QHBoxLayout()
        self.cam_placement = QListWidget()
        self.cam_placement.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.camera_position_spinboxes = {}
        self.camera_rotation_spinboxes = {}

        # For each camera, create control sliders to adjust position and orientation, displayed left to right with scrollbar
        for i in range(self.cameras):
            position_spinboxes = {}
            rotation_spinboxes = {}
            camera_color = self.visualizer.color_to_css(self.visualizer.get_camera_color(i))
            cam_label = self._create_subsection_title(f"Camera {i+1}", color=camera_color)
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
                value_spinbox.setDecimals(0)
                value_spinbox.setSingleStep(1)
                value_spinbox.setMaximumWidth(80)
                value_spinbox.setMinimumHeight(24)
                
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(-1000)
                slider.setMaximum(1000)
                slider.setValue(0)
                
                # Sync spinbox and slider
                value_spinbox.valueChanged.connect(lambda value, s=slider: s.setValue(int(value)))
                slider.valueChanged.connect(value_spinbox.setValue)

                if param == "X Position (mm)":
                    position_spinboxes["x"] = value_spinbox
                elif param == "Y Position (mm)":
                    position_spinboxes["y"] = value_spinbox
                elif param == "Z Position (mm)":
                    position_spinboxes["z"] = value_spinbox
                elif param == "Pan (deg)":
                    rotation_spinboxes["pan"] = value_spinbox
                elif param == "Tilt (deg)":
                    rotation_spinboxes["tilt"] = value_spinbox
                elif param == "Roll (deg)":
                    rotation_spinboxes["roll"] = value_spinbox
                
                slider_layout.addWidget(slider_label)
                slider_layout.addWidget(value_spinbox)
                slider_layout.addWidget(slider)
                self.cam_placement.addItem(QListWidgetItem())
                self.cam_placement.setItemWidget(self.cam_placement.item(self.cam_placement.count()-1), QWidget())
                self.cam_placement.itemWidget(self.cam_placement.item(self.cam_placement.count()-1)).setLayout(slider_layout)

            self.camera_position_spinboxes[i] = position_spinboxes
            self.camera_rotation_spinboxes[i] = rotation_spinboxes
            for axis in ["x", "y", "z"]:
                position_spinboxes[axis].valueChanged.connect(lambda _, cam_index=i: self._update_camera_translation(cam_index))

            for axis in ["pan", "tilt", "roll"]:
                rotation_spinboxes[axis].valueChanged.connect(lambda _, cam_index=i: self._update_camera_rotation(cam_index))

            self._update_camera_translation(i)
            self._update_camera_rotation(i)

        controls_row.addWidget(self.cam_placement, stretch=3)

        self.optical_centres_group = self._create_optical_centres_distance_widget()
        controls_row.addWidget(self.optical_centres_group, stretch=2)

        controls_layout.addLayout(controls_row)
        self.right_vbox.addWidget(controls_group, stretch=1)

    def _create_optical_centres_distance_widget(self) -> QGroupBox:
        """Create a scrollable box showing pairwise distances between camera optical centres."""
        group, layout = self._create_styled_groupbox("Optical Centre Distances")
        self.optical_centres_distance_layout = QVBoxLayout()

        self.optical_centres_distance_scroll = QScrollArea()
        self.optical_centres_distance_scroll.setWidgetResizable(True)
        self.optical_centres_distance_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self.optical_centres_distance_content = QWidget()
        self.optical_centres_distance_content_layout = QVBoxLayout(self.optical_centres_distance_content)
        self.optical_centres_distance_content_layout.setContentsMargins(4, 4, 4, 4)
        self.optical_centres_distance_content_layout.setSpacing(2)
        self.optical_centres_distance_content_layout.addStretch()

        self.optical_centres_distance_scroll.setWidget(self.optical_centres_distance_content)
        layout.addWidget(self.optical_centres_distance_scroll)

        self.optical_centres_distance_labels = []
        self._refresh_optical_centres_distance_labels()
        return group

    def _format_distance_mm(self, distance_mm: float) -> str:
        """Format distances in a readable unit based on magnitude."""
        distance_mm = float(distance_mm)
        abs_mm = abs(distance_mm)
        if abs_mm >= 1000.0:
            return f"{distance_mm / 1000.0:.2f} m".rstrip("0").rstrip(".")
        if abs_mm >= 10.0:
            return f"{distance_mm / 10.0:.1f} cm".rstrip("0").rstrip(".")
        return f"{distance_mm:.0f} mm"

    def _pair_distance_block(self, camera_a: int, camera_b: int, distance_3d_mm: float, distance_xy_mm: float, distance_z_mm: float) -> str:
        """Build a capture-volume-style rich-text block for one camera pair."""
        color_a = self.visualizer.color_to_css(self.visualizer.get_camera_color(camera_a))
        color_b = self.visualizer.color_to_css(self.visualizer.get_camera_color(camera_b))
        total_text = self._format_distance_mm(distance_3d_mm)
        xy_text = self._format_distance_mm(distance_xy_mm)
        z_text = self._format_distance_mm(distance_z_mm)
        return (
            f"<pre><font color='{color_a}'>Camera {camera_a + 1}</font>-"
            f"<font color='{color_b}'>Camera {camera_b + 1}</font>: {total_text}\n"
            f"    (xy): {xy_text}\n"
            f"    (z): {z_text}</pre>"
        )

    def _clear_optical_centres_distance_labels(self):
        """Remove old distance labels from the scroll area."""
        for label in self.optical_centres_distance_labels:
            self.optical_centres_distance_content_layout.removeWidget(label)
            label.deleteLater()
        self.optical_centres_distance_labels = []

    def _refresh_optical_centres_distance_labels(self):
        """Rebuild pairwise optical-centre distances from current camera translations."""
        if not hasattr(self, "optical_centres_distance_content_layout"):
            return

        self._clear_optical_centres_distance_labels()

        title = QLabel("Pairwise distances between optical centres")
        title.setWordWrap(True)
        title.setStyleSheet("font-weight: bold;")
        self.optical_centres_distance_content_layout.insertWidget(0, title)
        self.optical_centres_distance_labels.append(title)

        if self.cameras < 2:
            empty_label = QLabel("Need at least two cameras.")
            empty_label.setWordWrap(True)
            self.optical_centres_distance_content_layout.insertWidget(1, empty_label)
            self.optical_centres_distance_labels.append(empty_label)
            return

        pair_count = 0
        for camera_a, camera_b in combinations(range(self.cameras), 2):
            # Use the simulated camera positions from the visualizer, not the raw UI fields.
            translation_a = np.array(self.visualizer.camera_translations_mm.get(camera_a, (0.0, 0.0, 0.0)), dtype=float)
            translation_b = np.array(self.visualizer.camera_translations_mm.get(camera_b, (0.0, 0.0, 0.0)), dtype=float)
            distance_mm = float(np.linalg.norm(translation_a - translation_b))
            distance_xy_mm = float(np.linalg.norm(translation_a[:2] - translation_b[:2]))
            distance_z_mm = float(abs(translation_a[2] - translation_b[2]))

            line = QLabel(self._pair_distance_block(camera_a, camera_b, distance_mm, distance_xy_mm, distance_z_mm))
            line.setTextFormat(Qt.TextFormat.RichText)
            line.setWordWrap(True)
            self.optical_centres_distance_content_layout.insertWidget(self.optical_centres_distance_content_layout.count() - 1, line)
            self.optical_centres_distance_labels.append(line)
            pair_count += 1

        if pair_count == 0:
            empty_label = QLabel("No pairwise distances available.")
            self.optical_centres_distance_content_layout.insertWidget(1, empty_label)
            self.optical_centres_distance_labels.append(empty_label)

    def _update_optical_centres_distances(self):
        """Refresh the optical-centre distance panel after camera movement changes."""
        self._refresh_optical_centres_distance_labels()

    def _update_camera_translation(self, camera_index: int):
        """Push camera translation controls into the arena visualizer."""
        position_spinboxes = self.camera_position_spinboxes.get(camera_index, {})
        x_mm = position_spinboxes.get("x").value()
        y_mm = position_spinboxes.get("y").value()
        z_mm = position_spinboxes.get("z").value()
        self.visualizer.set_camera_translation(camera_index, x_mm, y_mm, z_mm)
        self._update_optical_centres_distances()

    def _update_camera_rotation(self, camera_index: int):
        """Push camera rotation controls into the arena visualizer."""
        rotation_spinboxes = self.camera_rotation_spinboxes.get(camera_index, {})
        pan_deg = rotation_spinboxes.get("pan").value()
        tilt_deg = rotation_spinboxes.get("tilt").value()
        roll_deg = rotation_spinboxes.get("roll").value()
        self.visualizer.set_camera_rotation(camera_index, pan_deg, tilt_deg, roll_deg)
        self._update_optical_centres_distances()


    def connect_widgets(self):
        self.arena_scale_depth_cm.valueChanged.connect(self._update_arena_scale)
        self.visualised_frustum_depth.valueChanged.connect(self._update_frustum_depth)
        self.overlap_visualization_group.idToggled.connect(self._update_overlap_mode)
        self.save_arena_config_button.clicked.connect(self.save_arena_config)
        self.generate_matplotl_graph_button.clicked.connect(self.generate_matplotl_graph)
        self._update_arena_scale(self.arena_scale_depth_cm.value())
        self._update_frustum_depth(self.visualised_frustum_depth.value())
        self._update_overlap_mode(self.overlap_visualization_group.checkedId(), True)

    def _update_arena_scale(self, depth_cm: float):
        """Update mm-to-scene scaling when arena depth changes."""
        self.visualizer.set_arena_depth_cm(depth_cm)

    def _update_frustum_depth(self, depth_cm: float):
        """Update the visualised frustum depth in the arena visualizer."""
        self.visualizer.set_visualised_frustum_depth_cm(depth_cm)

    def _update_camera_frustum(self, camera_index: int):
        """Push lens angle controls into the arena visualizer."""
        angle_spinboxes = self.lens_angle_spinboxes.get(camera_index, {})
        horizontal_angle_deg = angle_spinboxes.get("horizontal").value()
        vertical_angle_deg = angle_spinboxes.get("vertical").value()
        self.visualizer.set_camera_frustum_angles(camera_index, horizontal_angle_deg, vertical_angle_deg)

    def _update_camera_min_working_distance(self, camera_index: int):
        """Push minimum working distance control into the arena visualizer."""
        angle_spinboxes = self.lens_angle_spinboxes.get(camera_index, {})
        min_working_distance_mm = angle_spinboxes.get("min_working_distance").value()
        self.visualizer.set_camera_min_working_distance(camera_index, min_working_distance_mm)

    def _update_overlap_mode(self, mode_id: int, checked: bool):
        """Push overlap mode selection into the arena visualizer."""
        if not checked:
            return
        if mode_id == 1:
            self.visualizer.set_overlap_mode("max_all")
        else:
            self.visualizer.set_overlap_mode("min_two")

    def _get_arena_config_payload(self) -> dict:
        """Collect current arena-sim widget values into a JSON-serializable payload."""
        cameras = []
        for camera_index in range(self.cameras):
            lens_spinboxes = self.lens_angle_spinboxes.get(camera_index, {})
            position_spinboxes = self.camera_position_spinboxes.get(camera_index, {})
            rotation_spinboxes = self.camera_rotation_spinboxes.get(camera_index, {})
            cameras.append(
                {
                    "camera_index": camera_index,
                    "min_working_distance_mm": lens_spinboxes.get("min_working_distance").value(),
                    "horizontal_angle_deg": lens_spinboxes.get("horizontal").value(),
                    "vertical_angle_deg": lens_spinboxes.get("vertical").value(),
                    "translation_mm": {
                        "x": position_spinboxes.get("x").value(),
                        "y": position_spinboxes.get("y").value(),
                        "z": position_spinboxes.get("z").value(),
                    },
                    "rotation_deg": {
                        "pan": rotation_spinboxes.get("pan").value(),
                        "tilt": rotation_spinboxes.get("tilt").value(),
                        "roll": rotation_spinboxes.get("roll").value(),
                    },
                }
            )

        return {
            "arena_scale_depth_cm": self.arena_scale_depth_cm.value(),
            "visualised_frustum_depth_cm": self.visualised_frustum_depth.value(),
            "overlap_mode": "max_all" if self.overlap_max_radio.isChecked() else "min_two",
            "cameras": cameras,
        }

    def save_arena_config(self):
        """Persist current arena simulation controls to workspace metadata."""
        self.arena_config_manager.save(self._get_arena_config_payload())

    def generate_matplotl_graph(self):
        """Open a matplotlib popup rendering the current arena simulation geometry."""
        if hasattr(self, "arena_graph_window") and self.arena_graph_window is not None:
            try:
                self.arena_graph_window.close()
            except Exception:
                pass

        camera_colors = {
            i: self.visualizer.get_camera_color(i)
            for i in range(self.cameras)
        }
        arena_state = {
            "camera_count": self.cameras,
            "overlap_mode": "max_all" if self.overlap_max_radio.isChecked() else "min_two",
            "camera_frustum_angles_deg": dict(self.visualizer.camera_frustum_angles_deg),
            "camera_translations_mm": dict(self.visualizer.camera_translations_mm),
            "camera_rotations_deg": dict(self.visualizer.camera_rotations_deg),
            "visualised_frustum_depth_cm": float(self.visualizer.visualised_frustum_depth_cm),
            "mm_to_scene_scale": float(self.visualizer.mm_to_scene_scale),
            "camera_colors": camera_colors,
        }
        self.arena_graph_window = ArenaMatplotlibGraphWindow(
            arena_state=arena_state,
            arena_sim_dir=self.controller.workspace_guide.arena_sim_dir,
            parent=None,
        )
        self.arena_graph_window.show()

    def load_arena_config(self):
        """Load previously saved arena simulation controls from workspace metadata."""
        config = self.arena_config_manager.load()
        if not config:
            return

        self.arena_scale_depth_cm.setValue(float(config.get("arena_scale_depth_cm", self.arena_scale_depth_cm.value())))
        self.visualised_frustum_depth.setValue(float(config.get("visualised_frustum_depth_cm", self.visualised_frustum_depth.value())))
        if config.get("overlap_mode", "min_two") == "max_all":
            self.overlap_max_radio.setChecked(True)
        else:
            self.overlap_min_radio.setChecked(True)

        for camera_config in config.get("cameras", []):
            camera_index = int(camera_config.get("camera_index", -1))
            if camera_index < 0 or camera_index >= self.cameras:
                continue

            lens_spinboxes = self.lens_angle_spinboxes.get(camera_index, {})
            position_spinboxes = self.camera_position_spinboxes.get(camera_index, {})
            rotation_spinboxes = self.camera_rotation_spinboxes.get(camera_index, {})

            if "min_working_distance_mm" in camera_config:
                lens_spinboxes.get("min_working_distance").setValue(float(camera_config["min_working_distance_mm"]))
            if "horizontal_angle_deg" in camera_config:
                lens_spinboxes.get("horizontal").setValue(float(camera_config["horizontal_angle_deg"]))
            if "vertical_angle_deg" in camera_config:
                lens_spinboxes.get("vertical").setValue(float(camera_config["vertical_angle_deg"]))

            translation = camera_config.get("translation_mm", {})
            if "x" in translation:
                position_spinboxes.get("x").setValue(float(translation["x"]))
            if "y" in translation:
                position_spinboxes.get("y").setValue(float(translation["y"]))
            if "z" in translation:
                position_spinboxes.get("z").setValue(float(translation["z"]))

            rotation = camera_config.get("rotation_deg", {})
            if "pan" in rotation:
                rotation_spinboxes.get("pan").setValue(float(rotation["pan"]))
            if "tilt" in rotation:
                rotation_spinboxes.get("tilt").setValue(float(rotation["tilt"]))
            if "roll" in rotation:
                rotation_spinboxes.get("roll").setValue(float(rotation["roll"]))

        self._update_optical_centres_distances()
        
