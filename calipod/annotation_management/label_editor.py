"""Label editor widget for managing annotation class labels."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from calipod.annotation_management import AnnotationsConfigManager
from calipod.core import logger as calipod_logger
from calipod.gui.utils.styles import create_styled_groupbox

logger = calipod_logger.get(__name__)


class LabelEditorWidget(QWidget):
    """Widget for editing annotation class labels with ground truth and predictions sections."""

    labels_changed = Signal()

    def __init__(self, workspace_dir: str):
        """
        Initialize the label editor widget.

        Args:
            workspace_dir: Path to the workspace directory
        """
        super().__init__()
        self.workspace_dir = workspace_dir
        self.config_manager = AnnotationsConfigManager(workspace_dir)
        self.label_inputs = {}  # Track label input widgets: {(section, label_id): QLineEdit}
        self.category_dropdowns = {}  # Track category dropdowns: {(section, label_id): QComboBox}
        self.CATEGORY_OPTIONS = ["animal", "arena vertex", "frame roi"]
        self.setup_ui()

    def setup_ui(self):
        """Set up the label editor UI with ground truth and predictions sections side-by-side."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area for label sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout(scroll_widget)

        # Add ground truth section on the left
        gt_group, gt_layout = create_styled_groupbox(
            "Ground Truth", title_level="subsection"
        )
        self.populate_label_section(gt_layout, "ground_truth")
        scroll_layout.addWidget(gt_group, stretch=1)

        # Add predictions section on the right
        pred_group, pred_layout = create_styled_groupbox(
            "Predictions", title_level="subsection"
        )
        self.populate_label_section(pred_layout, "predictions")
        scroll_layout.addWidget(pred_group, stretch=1)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

    def populate_label_section(self, section_layout: QVBoxLayout, section_name: str):
        """
        Populate a label section with label ID, name input fields, and category dropdowns.

        Args:
            section_layout: Layout to add labels to
            section_name: Either "ground_truth" or "predictions"
        """
        # Load labels for this section
        if section_name == "ground_truth":
            labels = self.config_manager.get_ground_truth_labels()
        else:
            labels = self.config_manager.get_predictions_labels()

        if not labels:
            not_found_label = QLabel("Not found")
            not_found_label.setStyleSheet("color: #999; font-style: italic;")
            section_layout.addWidget(not_found_label)
            section_layout.addStretch()
            return

        # Create label rows
        for label_id in sorted(labels.keys()):
            label_name = labels[label_id]
            row_layout = QHBoxLayout()

            # Label ID on the left
            id_label = QLabel(f"Class {label_id}:")
            id_label.setMinimumWidth(80)
            row_layout.addWidget(id_label)

            # Text input for name
            name_input = QLineEdit()
            name_input.setText(label_name)
            name_input.setPlaceholderText(f"Name for class {label_id}")

            # Connect text edit finished signal to save
            def make_save_handler(section, lid, input_widget):
                def on_text_edited():
                    new_name = input_widget.text()
                    self.config_manager.update_label_name(
                        lid, new_name, is_ground_truth=(section == "ground_truth")
                    )
                    logger.debug(f"Updated {section} label {lid} to '{new_name}'")
                    self.labels_changed.emit()

                return on_text_edited

            name_input.editingFinished.connect(
                make_save_handler(section_name, label_id, name_input)
            )

            row_layout.addWidget(name_input)

            # Category dropdown on the right
            category_dropdown = QComboBox()
            category_dropdown.addItems([""] + self.CATEGORY_OPTIONS)

            # Load the current category from config
            current_category = self.config_manager.get_label_category(
                label_id, is_ground_truth=(section_name == "ground_truth")
            )
            if current_category:
                index = category_dropdown.findText(current_category)
                if index >= 0:
                    category_dropdown.setCurrentIndex(index)

            # Connect category change signal to save
            def make_category_handler(section, lid, combo_widget):
                def on_category_changed(index):
                    new_category = combo_widget.currentText() or None
                    self.config_manager.update_label_category(
                        lid, new_category, is_ground_truth=(section == "ground_truth")
                    )
                    logger.debug(
                        f"Updated {section} label {lid} category to '{new_category}'"
                    )
                    self.labels_changed.emit()

                return on_category_changed

            category_dropdown.currentIndexChanged.connect(
                make_category_handler(section_name, label_id, category_dropdown)
            )

            row_layout.addWidget(category_dropdown)

            # Store references to input and dropdown
            self.label_inputs[(section_name, label_id)] = name_input
            self.category_dropdowns[(section_name, label_id)] = category_dropdown

            section_layout.addLayout(row_layout)

        # Add stretch to push labels to top
        section_layout.addStretch()

    def refresh_labels(self):
        """Refresh the label editor with current config data."""
        # Clear existing widgets
        layout = self.layout()
        while layout.count() > 0:
            layout.takeAt(0).widget().deleteLater()

        # Clear input tracking
        self.label_inputs.clear()
        self.category_dropdowns.clear()

        # Recreate UI
        self.setup_ui()

    def get_label_name(self, section: str, label_id: int) -> str:
        """
        Get the current name for a label from the input field.

        Args:
            section: Either "ground_truth" or "predictions"
            label_id: The label ID

        Returns:
            The name from the input field, or empty string if not found
        """
        key = (section, label_id)
        if key in self.label_inputs:
            return self.label_inputs[key].text()
        return ""

    def get_label_category(self, section: str, label_id: int) -> str | None:
        """
        Get the current category for a label from the dropdown.

        Args:
            section: Either "ground_truth" or "predictions"
            label_id: The label ID

        Returns:
            The category from the dropdown, or None if not set
        """
        key = (section, label_id)
        if key in self.category_dropdowns:
            category = self.category_dropdowns[key].currentText()
            return category if category else None
        return None
