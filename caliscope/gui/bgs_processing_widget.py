from pathlib import Path
import re

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
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

        # Create tree widget for recording and video selection
        self.recording_tree = QTreeWidget()
        self.recording_tree.setHeaderLabel("Recordings")
        self.recording_tree.setColumnCount(1)
        self.populate_recording_tree()

        # Set up the layout with tree on left and content on right
        main_layout = QHBoxLayout(self)
        
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(QLabel("Select Recording & Video:"))
        left_vbox.addWidget(self.recording_tree)
        left_vbox.addStretch()
        
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(QLabel("Background Subtraction Processing"))
        right_vbox.addStretch()
        
        main_layout.addLayout(left_vbox, stretch=1)
        main_layout.addLayout(right_vbox, stretch=2)
        
        self.setLayout(main_layout)

    def populate_recording_tree(self):
        """Populate tree with recordings and their associated camera video files"""
        self.recording_tree.clear()
        
        # Get list of recording directories
        recording_dirs = self.controller.workspace_guide.valid_recording_dirs()
        
        for recording_name in recording_dirs:
            recording_path = Path(
                self.controller.workspace_guide.recording_dir,
                recording_name
            )
            
            # Create recording root item
            recording_item = QTreeWidgetItem()
            recording_item.setText(0, recording_name)
            self.recording_tree.addTopLevelItem(recording_item)
            
            # Find all port_*.mp4 files in the recording directory
            if recording_path.exists():
                video_files = sorted(recording_path.glob("port_*.mp4"))
                
                for video_file in video_files:
                    # Extract port number from filename (e.g., "port_0.mp4" -> "0")
                    match = re.search(r"port_(\d+)\.mp4", video_file.name)
                    if match:
                        port_num = match.group(1)
                        video_item = QTreeWidgetItem()
                        video_item.setText(0, f"Camera {port_num}")
                        video_item.setData(0, 32, str(video_file))  # Store full path in role 32 (user role)
                        recording_item.addChild(video_item)
            
            # Expand recording items by default
            self.recording_tree.expandItem(recording_item)
