from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QWidget,
)

import caliscope.logger
from caliscope.controller import Controller

logger = caliscope.logger.get(__name__)

# BGS Processing Widget - used for supplementing deep learning annotations processing with background subtraction techniques.
class BGSProcessingWidget(QWidget):
    def __init__(self, controller: Controller):
        super(BGSProcessingWidget, self).__init__()
        self.controller = controller
        self.config = self.controller.config

        # Set up the layout
        layout = QVBoxLayout(self)
        
        # Add a placeholder label
        label = QLabel("Background Subtraction Processing")
        layout.addWidget(label)
        
        # Add BGS processing widgets here
        
        self.setLayout(layout)
