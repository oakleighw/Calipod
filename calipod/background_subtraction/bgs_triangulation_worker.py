"""
BGS Triangulation Worker - triangulates BGS detections to 3D in background thread
"""

from pathlib import Path
from PySide6.QtCore import QThread, Signal

import calipod.logger
from calipod.background_subtraction.bgs_triangulator import BGSTriangulator

logger = calipod.logger.get(__name__)


class BGSTriangulationWorker(QThread):
    """
    Worker thread for BGS triangulation.
    Triangulates 2D BGS detections to 3D without blocking the UI.
    """
    
    progress_updated = Signal(str)  # Log message
    triangulation_complete = Signal(str)  # Output file path
    triangulation_error = Signal(str)  # Error message
    
    def __init__(self, recording_path: Path, config_path: Path, yolo_xyz_path: Path, 
                 bbox=None, region: str = "full"):
        """
        Initialize the triangulation worker.
        
        Parameters:
            recording_path: Path to recording directory
            config_path: Path to config.toml
            yolo_xyz_path: Path to YOLO predictions CSV
            bbox: Bounding box tuple (x1, y1, x2, y2) or None
            region: Region name ("full", "leaves", "fruit")
        """
        super().__init__()
        self.recording_path = Path(recording_path)
        self.config_path = Path(config_path)
        self.yolo_xyz_path = Path(yolo_xyz_path)
        self.bbox = bbox
        self.region = region
    
    def run(self):
        """Main triangulation process"""
        try:
            self.progress_updated.emit("Initializing BGS triangulation...")
            logger.info("Starting BGS triangulation worker")
            
            # Create triangulator
            triangulator = BGSTriangulator(self.recording_path, self.config_path)
            
            self.progress_updated.emit(f"Processing BGS detections from port-specific labels...")
            logger.info("Triangulating BGS detections from port-specific labels")
            
            # Perform triangulation
            output_path = triangulator.triangulate_bgs_detections_and_save(
                yolo_xyz_path=self.yolo_xyz_path,
                bbox=self.bbox,
                region=self.region
            )
            
            if output_path is None or not output_path.exists():
                self.progress_updated.emit("[ERROR] Triangulation failed - no output generated")
                logger.error("Triangulation returned None or output does not exist")
                self.triangulation_error.emit("Triangulation failed - no output generated")
                return
            
            self.progress_updated.emit(f"[SUCCESS] Triangulation complete!")
            self.progress_updated.emit(f"Output saved to: {output_path}")
            logger.info(f"Triangulation successful: {output_path}")
            
            self.triangulation_complete.emit(str(output_path))
            
        except Exception as e:
            error_msg = f"Error during triangulation: {str(e)}"
            self.progress_updated.emit(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            self.triangulation_error.emit(error_msg)
