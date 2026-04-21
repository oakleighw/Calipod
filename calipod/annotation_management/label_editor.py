"""Label editor widget for managing annotation class labels."""

from PySide6.QtWidgets import (
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
        Populate a label section with label ID and name input fields.

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

            # Text input on the right
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

                return on_text_edited

            name_input.editingFinished.connect(
                make_save_handler(section_name, label_id, name_input)
            )

            row_layout.addWidget(name_input)

            # Store reference to input for potential future use
            self.label_inputs[(section_name, label_id)] = name_input

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
