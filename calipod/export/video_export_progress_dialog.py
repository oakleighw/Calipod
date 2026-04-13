"""Progress dialog for video export operations."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QPushButton, QHBoxLayout


class VideoExportProgressDialog(QDialog):
    """Modal dialog for displaying video export progress with a detailed progress bar."""
    
    def __init__(self, parent=None, title="Video Export Progress", allow_cancel=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(150)
        self.allow_cancel = allow_cancel
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title/status label
        self.display_text = QLabel("Initializing...")
        self.display_text.setWordWrap(True)
        main_layout.addWidget(self.display_text)
        
        # Progress bar with percentage text
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setFormat("%p%")
        main_layout.addWidget(self.progress_bar)
        
        # Frame counter label
        self.frame_counter = QLabel()
        self.frame_counter.setStyleSheet("color: gray; font-size: 11px;")
        main_layout.addWidget(self.frame_counter)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        if allow_cancel:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.setMaximumWidth(100)
            button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        # Store cancel state
        self.cancelled = False
        if allow_cancel:
            self.cancel_button.clicked.connect(self.on_cancel_clicked)
    
    def on_cancel_clicked(self):
        """Handle cancel button click - closes dialog and signals cancellation."""
        self.cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelled")
        self.close()  # Close/destroy the dialog
    
    def set_status(self, status_text: str):
        """Set the status text."""
        self.display_text.setText(status_text)
    
    def set_progress(self, current_frame: int, total_frames: int):
        """Update progress bar and frame counter."""
        if total_frames > 0:
            percentage = int((current_frame / total_frames) * 100)
            self.progress_bar.setValue(percentage)
            self.frame_counter.setText(f"Frame {current_frame} of {total_frames}")
        else:
            self.progress_bar.setValue(0)
            self.frame_counter.setText("")
    
    def set_progress_percentage(self, percentage: int):
        """Set progress bar to a specific percentage (0-100)."""
        self.progress_bar.setValue(max(0, min(100, percentage)))
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        event.accept()
