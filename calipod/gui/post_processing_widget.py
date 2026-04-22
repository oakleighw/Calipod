from pathlib import Path

import pandas as pd
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calipod.annotation_management import AnnotationsConfigManager
from calipod.annotation_management.annotations_config_manager import (
    STRUCTURE_GEOMETRY_FLAT,
    STRUCTURE_GEOMETRY_SEMI_SPHERE,
)
from calipod.core import logger as calipod_logger
from calipod.core.configurator import Configurator
from calipod.core.controller import Controller
from calipod.gui.vizualize.playback_triangulation_widget import PlaybackTriangulationWidget
from calipod.post_processing.environment_structures import load_environment_structure_settings
from calipod.post_processing.metarig_config import generate_metarig_config
from calipod.trackers.tracker_enum import TrackerEnum

logger = calipod_logger.get(__name__)


class PostProcessingWidget(QWidget):
    processing_complete = Signal()

    def __init__(self, controller: Controller):
        super(PostProcessingWidget, self).__init__()
        self.controller = controller
        self.config = self.controller.config

        self.sync_index_cursors = {}  # track where the slider is for each playback...
        self.recording_folders = QListWidget()
        self.update_recording_folders()

        self.vis_widget = PlaybackTriangulationWidget(self.controller.camera_array, config=self.config)

        # Hybrid YOLO+BGS filtering checkbox
        # When enabled, Kalman filtering will select the best measurement (YOLO or BGS) at each frame
        # based on which is closest to the Kalman-predicted position. This helps:
        # - Fill gaps where YOLO has no detection but BGS does
        # - Replace bad YOLO detections with better BGS detections in ROI regions
        # - Prevent outliers at ends of track from affecting smoothing via RTS
        self.use_hybrid_filtering_checkbox = QCheckBox("Use Hybrid YOLO+BGS Filtering")
        self.use_hybrid_filtering_checkbox.setEnabled(False)
        self.use_hybrid_filtering_checkbox.setChecked(False)
        self.use_hybrid_filtering_checkbox.setToolTip(
            "When enabled with Kalman filtering active, selects the best measurement (YOLO or BGS) "
            "at each frame based on distance to Kalman prediction. BGS triangulations must be pre-computed."
        )

        self.tracker_combo = QComboBox()
        self.vizualizer_title = QLabel()
        self.annotations_config_manager = AnnotationsConfigManager(self.controller.workspace_guide.workspace_dir)

        self.environment_group = QGroupBox("Environment Structures")
        self.environment_group_layout = QVBoxLayout()
        self.environment_master_checkbox = QCheckBox("Show Environment Structures")
        self.environment_master_checkbox.setChecked(True)
        self.environment_group_layout.addWidget(self.environment_master_checkbox)
        self.environment_group.setLayout(self.environment_group_layout)
        self.environment_structure_widgets: dict[int, dict] = {}
        self.arena_vertex_widgets: dict[int, dict] = {}
        self.environment_structure_rows = []
        self.current_arena_vertices: list[dict] = []

        self.geometry_label_to_key = {
            "Flat Plane": STRUCTURE_GEOMETRY_FLAT,
            "Semi-sphere": STRUCTURE_GEOMETRY_SEMI_SPHERE,
        }
        self.geometry_key_to_label = {
            STRUCTURE_GEOMETRY_FLAT: "Flat Plane",
            STRUCTURE_GEOMETRY_SEMI_SPHERE: "Semi-sphere",
        }

        # Add items to the combo box using the name attribute of the TrackerEnum
        for tracker in TrackerEnum:
            if tracker.name != "CHARUCO":
                display_name = "INSECT" if tracker.name == "FLY" else tracker.name
                self.tracker_combo.addItem(display_name, tracker)

        self.open_folder_btn = QPushButton("&Open Folder")
        self.process_current_btn = QPushButton("&Process")
        self.generate_metarig_config_btn = QPushButton("Generate Metarig Config")

        # Pass reference to hybrid mode checkbox to visualizer
        self.vis_widget.use_hybrid_bgs = self.use_hybrid_filtering_checkbox
        self.vis_widget.update_filter_options_row()  # Rebuild filter options row with hybrid checkbox

        self.refresh_visualizer()  # must happen before placement to create vis_widget and vizualizer_title
        self.place_widgets()
        self.connect_widgets()

    def set_current_xyz(self):
        if self.xyz_processed_path.exists():
            self.vis_widget.update_motion_trial(self.xyz_processed_path)
        else:
            logger.info(f"No points displayed; Nothing stored in {self.xyz_processed_path}")
            # self.xyz = None

            # check if there aren't any points to track and warn about that
            if self.xy_base_path.exists():
                xy = pd.read_csv(self.xy_base_path)
                if xy.shape[0] == 0:
                    logger.info("No points tracked")
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"The {self.active_tracker_enum.name} tracker did not identify any points to track in recordings stored in:\n{self.active_recording_path}.",  # noqa 501
                    )  # show a warning dialog

    def update_recording_folders(self):
        # this check here is an artifact of the way that the main widget handles refresh
        self.recording_folders.clear()
        # create list of recording directories
        dir_list = self.controller.workspace_guide.valid_recording_dirs()

        # add each folder to the QListWidget
        for folder in dir_list:
            self.recording_folders.addItem(folder)

        if len(dir_list) > 0:
            self.recording_folders.setCurrentRow(0)

    @property
    def processed_subfolder(self):
        subfolder = Path(
            self.controller.workspace_guide.recording_dir,
            self.recording_folders.currentItem().text(),
            self.tracker_combo.currentData().name,
        )
        return subfolder

    @property
    def xyz_processed_path(self):
        # Check if hybrid filtering is enabled and BGS predictions are available
        if self.use_hybrid_filtering_checkbox.isChecked():
            bgs_xyz_path = Path(
                self.processed_subfolder.parent,  # Go up to FLY folder
                "bgs",
                "xyz_FLY_bgs_predictions.csv",
            )
            if bgs_xyz_path.exists():
                logger.info(f"Using hybrid filtering with BGS predictions: {bgs_xyz_path}")
                return bgs_xyz_path
            else:
                logger.warning(f"BGS predictions not found, using original YOLO: {bgs_xyz_path}")

        # Use original predictions
        file_name = f"xyz_{self.tracker_combo.currentData().name}.csv"
        result = Path(self.processed_subfolder, file_name)
        return result

    @property
    def archived_config_path(self):
        return Path(self.processed_subfolder, "config.toml")

    @property
    def xy_base_path(self):
        file_name = f"xy_{self.tracker_combo.currentData().name}.csv"
        result = Path(self.processed_subfolder, file_name)
        return result

    @property
    def active_tracker_enum(self):
        return self.tracker_combo.currentData()

    @property
    def metarig_config_path(self):
        file_name = f"metarig_config_{self.tracker_combo.currentData().name}.json"
        result = Path(self.processed_subfolder, file_name)
        return result

    @property
    def active_folder(self):
        if self.recording_folders.count() == 0:
            active_folder = None
        elif self.recording_folders.currentItem() is None:
            self.recording_folders.setCurrentRow(0)
            active_folder: str = self.recording_folders.currentItem().text()
        else:
            active_folder: str = self.recording_folders.currentItem().text()

        return active_folder

    @property
    def active_recording_path(self) -> Path:
        p = Path(self.controller.workspace_guide.recording_dir, self.active_folder)
        logger.info(f"Active recording path is {p}")
        return p

    @property
    def viz_title_html(self):
        if self.xyz_processed_path.exists():
            suffix = "(x,y,z) estimates"
        else:
            suffix = "(no processed data)"

        tracker_label = self.tracker_combo.currentText().title()
        title = f"<div align='center'><b>{tracker_label} Tracker: {self.active_folder} {suffix} </b></div>"  # noqa E501

        return title

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()
        self.button_hbox = QHBoxLayout()

        self.layout().addLayout(self.left_vbox)

        self.left_vbox.addWidget(self.recording_folders)
        self.left_vbox.addWidget(self.open_folder_btn)
        self.left_vbox.addWidget(self.tracker_combo)
        self.left_vbox.addWidget(self.environment_group)
        self.button_hbox.addWidget(self.process_current_btn)
        self.button_hbox.addWidget(self.generate_metarig_config_btn)
        self.left_vbox.addLayout(self.button_hbox)

        self.layout().addLayout(self.right_vbox, stretch=2)
        self.right_vbox.addWidget(self.vizualizer_title)
        self.right_vbox.addWidget(self.vis_widget, stretch=2)

    def connect_widgets(self):
        self.recording_folders.currentItemChanged.connect(self.refresh_visualizer)
        self.tracker_combo.currentIndexChanged.connect(self.refresh_visualizer)
        self.vis_widget.slider.valueChanged.connect(self.store_sync_index_cursor)
        self.use_hybrid_filtering_checkbox.stateChanged.connect(self.on_hybrid_filtering_toggled)
        self.environment_master_checkbox.toggled.connect(self.on_environment_master_toggled)
        self.process_current_btn.clicked.connect(self.process_current)
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.generate_metarig_config_btn.clicked.connect(self.create_metarig_config)

        self.controller.post_processing_complete.connect(self.enable_all_inputs)
        self.controller.post_processing_complete.connect(self.refresh_visualizer)

    def on_hybrid_filtering_toggled(self):
        """Handle hybrid YOLO+BGS filtering toggle"""
        # Store current filtering state before switching
        was_filtering_enabled = False
        if hasattr(self.vis_widget, "toggle_filtered_button"):
            was_filtering_enabled = self.vis_widget.toggle_filtered_button.isChecked()

        if self.use_hybrid_filtering_checkbox.isChecked():
            logger.info("Enabled hybrid YOLO+BGS measurement selection during Kalman filtering")
        else:
            logger.info("Disabled hybrid mode - using YOLO predictions only")

        # Reload the motion trial with the current prediction source
        self.set_current_xyz()

        # If filtering was enabled, reapply it to use the new hybrid setting
        if was_filtering_enabled and hasattr(self.vis_widget, "toggle_filtered_button"):
            logger.info("Reapplying Kalman filtering with updated measurement selection...")
            # Reset the button state first to avoid double-toggle
            self.vis_widget.toggle_filtered_button.setChecked(False)
            # Now enable filtering again which will apply it with the new hybrid setting
            self.vis_widget.toggle_filtered_button.setChecked(True)

    def store_sync_index_cursor(self, cursor_value):
        if self.xyz_processed_path.exists():
            self.sync_index_cursors[self.xyz_processed_path] = cursor_value

    def open_folder(self):
        """Opens the currently active folder in a system file browser"""
        if self.active_folder is not None:
            folder_path = Path(self.controller.workspace_guide.recording_dir, self.active_folder)
            url = QUrl.fromLocalFile(str(folder_path))
            QDesktopServices.openUrl(url)
        else:
            logger.warning("No folder selected")

    def process_current(self):
        """

        This needs to get pushed into the controller layer
        """
        recording_path = Path(self.controller.workspace_guide.recording_dir, self.active_folder)
        logger.info(f"Beginning processing of recordings at {recording_path}")
        tracker_enum = self.tracker_combo.currentData()
        logger.info(f"(x,y) tracking will be applied using {tracker_enum.name}")
        recording_config_toml = Path(recording_path, "config.toml")
        logger.info(f"Camera data based on config file saved to {recording_config_toml}")

        self.controller.process_recordings(recording_path, tracker_enum)

    def active_tracker_config_path(self):
        """
        This will exist of the tracker has been processed. It should have a camera array within it
        that can be displayed to the user and will align
        """
        return Path(self.active_recording_path, self.active_tracker_enum.name, "config.toml")

    def refresh_visualizer(self):
        logger.info("Refreshing vizualizer within post_processing widget")

        if (
            self.archived_config_path.exists()
        ):  # processing has been done and their is a camera array that can be loaded
            stored_config = Configurator(self.archived_config_path.parent)
            presented_camera_array = stored_config.get_camera_array()
        else:
            presented_camera_array = self.controller.camera_array

        self.vis_widget.update_camera_array(presented_camera_array)

        # Check if BGS predictions exist and enable hybrid filtering checkbox if so
        bgs_xyz_path = Path(self.active_recording_path, "FLY", "bgs", "xyz_FLY_bgs_predictions.csv")
        if bgs_xyz_path.exists():
            self.use_hybrid_filtering_checkbox.setEnabled(True)
            logger.info(f"BGS predictions available at {bgs_xyz_path}")
        else:
            self.use_hybrid_filtering_checkbox.setEnabled(False)
            self.use_hybrid_filtering_checkbox.setChecked(False)
            logger.info("No BGS predictions available for this recording")

        self.set_current_xyz()
        self.refresh_environment_structures_panel()
        self.vizualizer_title.setText(self.viz_title_html)
        self.update_enabled_disabled()
        self.update_slider_position()

    def clear_environment_structure_rows(self):
        for row_layout in self.environment_structure_rows:
            while row_layout.count():
                item = row_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.environment_group_layout.removeItem(row_layout)
        self.environment_structure_rows = []
        self.environment_structure_widgets = {}
        self.arena_vertex_widgets = {}

    def refresh_environment_structures_panel(self):
        self.clear_environment_structure_rows()

        structure_settings = load_environment_structure_settings(self.controller.workspace_guide.workspace_dir)
        self.current_arena_vertices = structure_settings.get("arena_vertices", [])
        frame_roi_structures = structure_settings.get("frame_roi_structures", [])

        if len(frame_roi_structures) == 0:
            empty_label = QLabel("No frame ROI labels found")
            empty_label.setStyleSheet("color: #999; font-style: italic;")
            row = QHBoxLayout()
            row.addWidget(empty_label)
            row.addStretch()
            self.environment_group_layout.addLayout(row)
            self.environment_structure_rows.append(row)
            self.push_environment_settings_to_visualizer()
            return

        for structure in frame_roi_structures:
            label_id = int(structure["id"])
            label_name = structure.get("name") or f"Class {label_id}"
            enabled = bool(structure.get("enabled", True))
            geometry = structure.get("geometry", STRUCTURE_GEOMETRY_FLAT)

            row = QHBoxLayout()
            enabled_checkbox = QCheckBox(f"{label_name} (Class {label_id})")
            enabled_checkbox.setChecked(enabled)

            geometry_combo = QComboBox()
            geometry_combo.addItem("Flat Plane", STRUCTURE_GEOMETRY_FLAT)
            geometry_combo.addItem("Semi-sphere", STRUCTURE_GEOMETRY_SEMI_SPHERE)
            geometry_index = geometry_combo.findData(geometry)
            if geometry_index >= 0:
                geometry_combo.setCurrentIndex(geometry_index)

            enabled_checkbox.toggled.connect(
                lambda checked, lid=label_id: self.on_structure_enabled_changed(lid, checked)
            )
            geometry_combo.currentIndexChanged.connect(
                lambda _idx, lid=label_id, combo=geometry_combo: self.on_structure_geometry_changed(
                    lid, combo.currentData()
                )
            )

            row.addWidget(enabled_checkbox)
            row.addWidget(QLabel("Geometry"))
            row.addWidget(geometry_combo)
            row.addStretch()

            self.environment_group_layout.addLayout(row)
            self.environment_structure_rows.append(row)

            self.environment_structure_widgets[label_id] = {
                "label_name": label_name,
                "enabled_checkbox": enabled_checkbox,
                "geometry_combo": geometry_combo,
            }

        arena_vertices = structure_settings.get("arena_vertices", [])
        if len(arena_vertices) > 0:
            header_row = QHBoxLayout()
            header_label = QLabel("Arena Vertices")
            header_label.setStyleSheet("font-weight: bold;")
            header_row.addWidget(header_label)
            header_row.addStretch()
            self.environment_group_layout.addLayout(header_row)
            self.environment_structure_rows.append(header_row)

            for vertex in arena_vertices:
                label_id = int(vertex["id"])
                label_name = vertex.get("name") or f"Class {label_id}"
                enabled = bool(vertex.get("enabled", True))
                add_to_floor = bool(vertex.get("add_to_floor", False))

                row = QHBoxLayout()
                enabled_checkbox = QCheckBox(f"{label_name} (Class {label_id})")
                enabled_checkbox.setChecked(enabled)
                floor_checkbox = QCheckBox("Add to Floor")
                floor_checkbox.setChecked(add_to_floor)

                enabled_checkbox.toggled.connect(
                    lambda checked, lid=label_id: self.on_arena_vertex_enabled_changed(lid, checked)
                )
                floor_checkbox.toggled.connect(
                    lambda checked, lid=label_id: self.on_arena_vertex_floor_changed(lid, checked)
                )

                row.addWidget(enabled_checkbox)
                row.addWidget(floor_checkbox)
                row.addStretch()

                self.environment_group_layout.addLayout(row)
                self.environment_structure_rows.append(row)
                self.arena_vertex_widgets[label_id] = {
                    "label_name": label_name,
                    "enabled_checkbox": enabled_checkbox,
                    "floor_checkbox": floor_checkbox,
                }

        self.push_environment_settings_to_visualizer()

    def push_environment_settings_to_visualizer(self):
        frame_roi_structures = []
        for label_id, widgets in sorted(self.environment_structure_widgets.items()):
            frame_roi_structures.append(
                {
                    "id": label_id,
                    "name": widgets["label_name"],
                    "enabled": widgets["enabled_checkbox"].isChecked(),
                    "geometry": widgets["geometry_combo"].currentData(),
                }
            )

        arena_vertices = []
        for label_id, widgets in sorted(self.arena_vertex_widgets.items()):
            arena_vertices.append(
                {
                    "id": label_id,
                    "name": widgets["label_name"],
                    "enabled": widgets["enabled_checkbox"].isChecked(),
                    "add_to_floor": widgets["floor_checkbox"].isChecked(),
                }
            )

        self.vis_widget.set_environment_structure_settings(
            frame_roi_structures=frame_roi_structures,
            arena_vertices=arena_vertices,
            enabled=self.environment_master_checkbox.isChecked(),
        )

        # Refresh current frame so visibility/geometry changes are visible immediately.
        self.vis_widget.visualizer.display_points(self.vis_widget.slider.value())

    def on_environment_master_toggled(self, _checked: bool):
        self.push_environment_settings_to_visualizer()

    def on_structure_enabled_changed(self, label_id: int, checked: bool):
        self.annotations_config_manager.update_label_structure_metadata(
            label_id=label_id,
            is_ground_truth=True,
            enabled=checked,
        )
        self.push_environment_settings_to_visualizer()

    def on_structure_geometry_changed(self, label_id: int, geometry: str):
        if geometry not in {STRUCTURE_GEOMETRY_FLAT, STRUCTURE_GEOMETRY_SEMI_SPHERE}:
            logger.warning(f"Ignoring unsupported structure geometry '{geometry}' for class {label_id}")
            return

        self.annotations_config_manager.update_label_structure_metadata(
            label_id=label_id,
            is_ground_truth=True,
            geometry=geometry,
        )
        self.push_environment_settings_to_visualizer()

    def on_arena_vertex_enabled_changed(self, label_id: int, checked: bool):
        self.annotations_config_manager.update_label_structure_metadata(
            label_id=label_id,
            is_ground_truth=True,
            enabled=checked,
        )
        self.push_environment_settings_to_visualizer()

    def on_arena_vertex_floor_changed(self, label_id: int, checked: bool):
        self.annotations_config_manager.update_label_structure_metadata(
            label_id=label_id,
            is_ground_truth=True,
            add_to_floor=checked,
        )
        self.push_environment_settings_to_visualizer()

    def disable_all_inputs(self):
        """used to toggle off all inputs will processing is going on"""
        self.recording_folders.setEnabled(False)
        self.tracker_combo.setEnabled(False)
        # self.export_btn.setEnabled(False)
        self.process_current_btn.setEnabled(False)
        self.vis_widget.slider.setEnabled(False)

    def enable_all_inputs(self):
        """
        after processing completes, swithes everything on again,
        but fine tuning of enable/disable will happen with self.update_enabled_disabled
        """
        self.recording_folders.setEnabled(True)
        self.tracker_combo.setEnabled(True)
        self.process_current_btn.setEnabled(True)
        self.vis_widget.slider.setEnabled(True)

    def update_enabled_disabled(self):
        # set availability of metarig generation
        logger.info("Checking if metarig config can be created...")
        tracker = self.tracker_combo.currentData().value()  # error here...
        logger.info(tracker)
        if tracker.metarig_mapped() and self.xyz_processed_path.exists() and not self.metarig_config_path.exists():
            self.generate_metarig_config_btn.setEnabled(True)
            self.generate_metarig_config_btn.setToolTip("Creation of metarig configuration file is now available")
        else:
            self.generate_metarig_config_btn.setEnabled(False)

        if not tracker.metarig_mapped():
            self.generate_metarig_config_btn.setToolTip("Tracker is not set up to scale to a metarig")
        elif self.metarig_config_path.exists():
            self.generate_metarig_config_btn.setToolTip(
                "The Metarig configuration json file has already been created. Check the tracker subfolder in the recording directory."  # noqa E501
            )
        elif not self.xyz_processed_path.exists():
            self.generate_metarig_config_btn.setToolTip(
                "Must process recording to create xyz estimates for metarig configuration"
            )
        else:
            self.generate_metarig_config_btn.setToolTip(
                "Click to create a file in the tracker subfolder that can be used to scale a Blender metarig"
            )
        # set availability of Proecssing and slider
        if self.xyz_processed_path.exists():
            self.process_current_btn.setEnabled(False)
            self.vis_widget.slider.setEnabled(True)
        elif self.xy_base_path.exists() and not self.xyz_processed_path.exists():
            # nothing available to triangulate
            self.process_current_btn.setEnabled(False)
            self.vis_widget.slider.setEnabled(False)

        else:
            self.process_current_btn.setEnabled(True)
            self.vis_widget.slider.setEnabled(False)

    def update_slider_position(self):
        # update slider value to stored value if it exists
        if self.xyz_processed_path in self.sync_index_cursors.keys():
            active_sync_index = self.sync_index_cursors[self.xyz_processed_path]
            self.vis_widget.slider.setValue(active_sync_index)
            self.vis_widget.visualizer.display_points(active_sync_index)
        else:
            pass

    def create_metarig_config(self):
        logger.info(f"Beginning metarig_config creation in {self.processed_subfolder}")
        tracker_enum = self.tracker_combo.currentData()
        xyz_csv_path = Path(self.processed_subfolder, f"xyz_{tracker_enum.name}_labelled.csv")
        generate_metarig_config(tracker_enum, xyz_csv_path)
        self.update_enabled_disabled()
