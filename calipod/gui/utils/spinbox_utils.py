from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QAbstractSpinBox


def calculate_spinbox_width(spin_box, min_width=None, padding=None):
    """
    Calculate appropriate width for a spin box based on its content.

    Args:
        spin_box: QSpinBox or QDoubleSpinBox instance
        min_width: Optional minimum width to enforce
        padding: Optional padding to add to calculated width

    Returns:
        Calculated minimum width in pixels
    """
    fm = spin_box.fontMetrics()

    # Get the maximum possible string length
    if isinstance(spin_box, QDoubleSpinBox):
        # For double, consider decimal places
        decimals = spin_box.decimals()
        # Format with maximum digits: -1234.56
        max_str = f"{spin_box.minimum():.{decimals}f}"
        min_str = f"{spin_box.maximum():.{decimals}f}"
    else:
        max_str = str(spin_box.minimum())
        min_str = str(spin_box.maximum())

    # Get width of the longest possible string
    max_width = max(fm.horizontalAdvance(max_str),
                   fm.horizontalAdvance(min_str))

    # Add space for spin arrows and frame
    if padding is None:
        padding = fm.height() + 20  # Default padding based on font height

    width = max_width + padding

    # Enforce minimum width if specified
    if min_width is not None:
        width = max(width, min_width)

    return width

def setup_spinbox_sizing(spin_box, centered=True, min_value=None, max_value=None, min_width=None, padding=None):
    """
    Configure sizing and alignment for a spin box.

    Args:
        spin_box: QSpinBox or QDoubleSpinBox instance
        centered: Whether to center-align the text
        min_width: Optional minimum width to enforce
        padding: Optional padding to add to calculated width
    """
    if centered:
        spin_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

    if min_value is not None:
        spin_box.setMinimum(min_value)
    if max_value is not None:
        spin_box.setMaximum(max_value)

    width = calculate_spinbox_width(spin_box, min_width, padding)
    spin_box.setMinimumWidth(width)

    return width

# Helper function to create a labeled spinbox row
def create_labeled_spinbox_row(parent_layout, label_text, min_value, max_value, decimals=2, max_width=130):
    """
    Create and add a labeled spinbox row to a layout.
    
    Args:
        parent_layout: QLayout to add the row to (typically QVBoxLayout)
        label_text: Text for the label
        min_value: Minimum value for spinbox
        max_value: Maximum value for spinbox
        decimals: Number of decimal places (default 2)
        max_width: Maximum width in pixels (default 130)
    
    Returns:
        The configured QDoubleSpinBox
    """
    h_box = QHBoxLayout()
    label = QLabel(label_text)
    spinbox = QDoubleSpinBox()
    spinbox.setDecimals(decimals)
    spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
    setup_spinbox_sizing(spinbox, min_value=min_value, max_value=max_value)
    spinbox.setMaximumWidth(max_width)
    h_box.addWidget(label, stretch=1)
    h_box.addWidget(spinbox, stretch=1)
    h_box.addStretch()
    parent_layout.addLayout(h_box)
    return spinbox

