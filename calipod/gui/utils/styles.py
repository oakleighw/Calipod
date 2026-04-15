"""Helper functions for consistent styling across the GUI."""

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

# Helper function to create section titles
def create_section_font(point_size: int = 11) -> QFont:
    """Create a styled font for section headers."""
    font = QFont()
    font.setBold(True)
    font.setPointSize(point_size)
    return font

def create_section_title(text: str) -> QLabel:
    """Create a styled section title label."""
    title = QLabel(text)
    title.setFont(create_section_font(11))
    return title

# Helper function to create subsection titles
def create_subsection_title(text: str, color: str | None = None) -> QLabel:
    """Create a styled subsection title label."""
    title = QLabel(text)
    title.setFont(create_section_font(10))
    if color is not None:
        title.setStyleSheet(f"color: {color};")
    return title

def create_styled_groupbox(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    """Create a QGroupBox with styled section title (returns group and layout)."""
    group = QGroupBox()
    layout = QVBoxLayout()

    # Add styled title as a label, not as QGroupBox title
    title_label = create_section_title(title)
    layout.addWidget(title_label)

    group.setLayout(layout)
    return group, layout