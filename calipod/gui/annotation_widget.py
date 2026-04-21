"""This widget will supply a basic annotation tool and also point to CVAT for extended functionality."""

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from calipod.annotation_management import LabelEditorWidget
from calipod.annotation_management.annotation_checker import AnnotationChecker
from calipod.annotation_management.annotations_config_manager import AnnotationsConfigManager
from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.styles import create_styled_groupbox

logger = calipod_logger.get(__name__)


class AnnotationWidget(QWidget):
    def __init__(self, controller: Controller):
        super(AnnotationWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.top_vbox = QVBoxLayout()
        self.centre_vbox = QVBoxLayout()
        self.bottom_vbox = QVBoxLayout()

        self.anno_labelling_widget()
        self.annotation_checker_widget()
        self.external_links_widget()

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addLayout(self.centre_vbox, stretch=1)
        self.layout().addLayout(self.bottom_vbox, stretch=1)

    def anno_labelling_widget(self):
        """This widget allows the user to log string labels for annotation class index labels."""
        anno_label_group, anno_label_layout = create_styled_groupbox("Labels")

        # Create label editor widget
        label_editor = LabelEditorWidget(str(self.controller.workspace))
        anno_label_layout.addWidget(label_editor)

        self.top_vbox.addWidget(anno_label_group)

    def annotation_checker_widget(self):
        """This widget checks if annotations are correct by showing bounding boxes overlaid on frames."""
        anno_check_group, anno_check_layout = create_styled_groupbox("Annotation Checker")

        # Initialize annotation checker
        try:
            self.annotation_checker = AnnotationChecker(self.controller)
        except Exception as e:
            logger.error(f"Failed to initialize annotation checker: {e}")
            anno_check_layout.addWidget(QLabel("Error: Could not initialize annotation checker"))
            self.centre_vbox.addWidget(anno_check_group)
            return

        # Create horizontal layout for class selector and examine button
        selector_layout = QHBoxLayout()

        # Class label
        class_label = QLabel("Class:")
        selector_layout.addWidget(class_label)

        # Class dropdown
        self.class_combo = QComboBox()
        self._populate_class_dropdown()
        selector_layout.addWidget(self.class_combo, stretch=1)

        # Focus to bbox checkbox
        self.focus_bbox_checkbox = QCheckBox("Focus to bbox")
        self.focus_bbox_checkbox.setChecked(True)  # Default to checked
        selector_layout.addWidget(self.focus_bbox_checkbox)

        # Examine button
        self.examine_button = QPushButton("Examine")
        self.examine_button.clicked.connect(self._on_examine_clicked)
        selector_layout.addWidget(self.examine_button)

        anno_check_layout.addLayout(selector_layout)

        # Create layout for frames display
        self.frames_layout = QHBoxLayout()
        self.frames_layout.setContentsMargins(0, 0, 0, 0)

        # Create labels for the 3 frames
        self.frame_labels = []
        for i in range(3):
            frame_label = QLabel()
            frame_label.setAlignment(Qt.AlignCenter)
            frame_label.setMinimumSize(200, 200)
            frame_label.setScaledContents(False)
            self.frames_layout.addWidget(frame_label, stretch=1)
            self.frame_labels.append(frame_label)

        anno_check_layout.addLayout(self.frames_layout)
        self.centre_vbox.addWidget(anno_check_group)

    def _populate_class_dropdown(self):
        """Populate the class dropdown with available annotation classes and their names."""
        self.class_combo.clear()

        try:
            classes = self.annotation_checker.get_available_classes()
            if not classes:
                self.class_combo.addItem("No annotations available")
            else:
                # Load label config to get names
                config_manager = AnnotationsConfigManager(self.controller.workspace_guide.workspace_dir)
                gt_labels = config_manager.get_ground_truth_labels() or {}
                pred_labels = config_manager.get_predictions_labels() or {}

                for class_name in classes:
                    # Parse format: "groundtruth/ID" or "predictions/ID"
                    parts = class_name.split("/")
                    if len(parts) == 2:
                        annotation_type, class_id = parts[0], parts[1]
                        try:
                            class_id_int = int(class_id)
                            # Get label name from appropriate config
                            if annotation_type == "groundtruth":
                                label_name = gt_labels.get(class_id_int)
                            else:  # predictions
                                label_name = pred_labels.get(class_id_int)

                            # Build display string with name if available
                            if label_name:
                                display_text = f"{annotation_type}/{class_id}/{label_name}"
                            else:
                                display_text = class_name
                        except ValueError:
                            display_text = class_name
                    else:
                        display_text = class_name

                    self.class_combo.addItem(display_text)
        except Exception as e:
            logger.error(f"Error populating class dropdown: {e}")
            self.class_combo.addItem("Error loading classes")

    def _on_examine_clicked(self):
        """Handle examine button click to visualize annotations."""
        if self.class_combo.count() == 0:
            QMessageBox.warning(self, "No Classes", "No annotation classes available.")
            return

        display_text = self.class_combo.currentText()
        if display_text in ["No annotations available", "Error loading classes"]:
            QMessageBox.warning(self, "Invalid Class", "Please select a valid annotation class.")
            return

        try:
            # Extract base class label from display text
            # Format could be "groundtruth/ID" or "groundtruth/ID/name"
            parts = display_text.split("/")
            if len(parts) < 2:
                QMessageBox.critical(self, "Error", f"Invalid class format: {display_text}")
                return

            # Reconstruct base class label as "groundtruth/ID" or "predictions/ID"
            class_label = f"{parts[0]}/{parts[1]}"

            # Try each port to find annotations
            frames_found = False
            for port in range(10):  # Try ports 0-9
                frames = self.annotation_checker.get_frames_for_class(class_label, port)
                if frames:
                    frames_found = True
                    logger.info(f"Found {len(frames)} frames with class {class_label} on port {port}")

                    # Get frames with annotations drawn
                    # Determine area_ratio based on focus_bbox checkbox state
                    area_ratio = 3.0 if self.focus_bbox_checkbox.isChecked() else None

                    frame_data = self.annotation_checker.get_frames_with_annotations(
                        class_label, port, num_frames=3, area_ratio=area_ratio
                    )

                    if frame_data:
                        # Clear existing frames
                        for label in self.frame_labels:
                            label.setPixmap(QPixmap())

                        # Display frames
                        for i, (frame_bgr, frame_idx) in enumerate(frame_data):
                            if i >= len(self.frame_labels):
                                break

                            # Convert BGR to RGB for QPixmap
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            h, w, ch = frame_rgb.shape
                            bytes_per_line = ch * w
                            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                            # Scale to fit label while maintaining aspect ratio
                            pixmap = QPixmap.fromImage(q_img)
                            scaled_pixmap = pixmap.scaledToWidth(
                                self.frame_labels[i].width() - 10,
                                Qt.TransformationMode.SmoothTransformation
                            )

                            self.frame_labels[i].setPixmap(scaled_pixmap)
                            self.frame_labels[i].setToolTip(f"Frame {frame_idx}")

                        return

            if not frames_found:
                QMessageBox.information(
                    self,
                    "No Frames Found",
                    f"No annotations found for class {class_label} across all ports.",
                )
        except Exception as e:
            logger.error(f"Error examining annotations: {e}")
            QMessageBox.critical(self, "Error", f"Failed to examine annotations: {str(e)}")

    def external_links_widget(self):
        """This widget points to external links for annotations e.g. CVAT."""
        ext_links_group, ext_links_layout = create_styled_groupbox("Links")

        # Create a clickable link to CVAT documentation
        cvat_link = QLabel()
        cvat_link.setText(
            '<a href="https://docs.cvat.ai/docs/administration/community/basics/installation/">'
            'CVAT Installation Guide</a>'
        )
        cvat_link.setOpenExternalLinks(True)
        cvat_link.setAlignment(Qt.AlignCenter)

        ext_links_layout.addWidget(cvat_link)

        self.bottom_vbox.addWidget(ext_links_group)

