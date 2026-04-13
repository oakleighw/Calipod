"""
BGS Triangulation module - triangulates BGS detections to 3D for hybrid filtering.

This module:
1. Scans BGS 2D detections from port-specific label files
2. Triangulates them to 3D using camera calibration
3. Saves 3D points for use in hybrid YOLO+BGS filtering

Hybrid selection (choosing between YOLO and BGS at each Kalman step) happens in post-processing,
not in this module. This module only provides the triangulated BGS points.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from calipod.core import logger as calipod_logger
from calipod.triangulate.triangulation import triangulate_from_files

logger = calipod_logger.get(__name__)


class BGSTriangulator:
    """
    Triangulates BGS 2D detections to 3D points for hybrid YOLO+BGS filtering.
    
    This class handles the triangulation of background subtraction detections
    from multiple camera views. The resulting 3D points are used in post-processing
    for hybrid measurement selection during Kalman filtering.
    """
    
    def __init__(self, recording_path: Path, config_path: Path):
        """
        Initialize the BGS supplementor.
        
        Parameters:
            recording_path: Path to the recording directory
            config_path: Path to config.toml
        """
        self.recording_path = Path(recording_path)
        self.config_path = Path(config_path)
        self.fly_dir = self.recording_path / "FLY"
        self.bgs_dir = self.fly_dir / "bgs"
    
    def create_bgs_xy_input(self, region: str = "full") -> Optional[Path]:
        """
        Convert BGS detections from YOLO labels to xy format for triangulation.
        
        Scans FLY/bgs/port_X/labels/train/ directories for frame_*.txt files and converts them to
        xy format (sync_index, port, point_id, img_loc_x, img_loc_y).
        
        Each port has its own label directory to avoid overwriting conflicts when processing
        multiple camera views with the same frame numbers.
        
        Parameters:
            region: Region used for detections ("full", "leaves", "fruit")
            
        Returns:
            Path to temporary xy CSV file, or None if no detections found
        """
        # Scan all port-specific label directories
        # This matches the structure where each camera (port) has its own labels
        all_label_files = []
        port_nums = []
        
        for port_dir in sorted(self.bgs_dir.glob("port_*")):
            if port_dir.is_dir():
                labels_dir = port_dir / "labels" / "train"
                
                if labels_dir.exists():
                    label_files = sorted(labels_dir.glob("frame_*.txt"))
                    if label_files:
                        # Extract port number (e.g., "port_1" -> 1)
                        port_match = re.search(r"port_(\d+)", port_dir.name)
                        if port_match:
                            port_num = int(port_match.group(1))
                            all_label_files.extend(label_files)
                            port_nums.extend([port_num] * len(label_files))
        
        if not all_label_files:
            logger.warning(f"No label files found in {self.bgs_dir}/port_*/labels/train/")
            return None
        
        xy_data = []
        
        logger.info(f"Found {len(all_label_files)} BGS label files across all ports")
        
        # Process each label file
        # The labels contain normalized coordinates (0-1), so we need to convert back to pixels
        # using the original video dimensions for each port
        for label_file, port_num in zip(all_label_files, port_nums):
            frame_num_str = label_file.stem.replace("frame_", "")
            frame_num = int(frame_num_str)
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    class_id = int(parts[0])
                    norm_x = float(parts[1])
                    norm_y = float(parts[2])
                    # width = float(parts[3])  # Fixed at 0.1 for point detections
                    # height = float(parts[4])
                    
                    # Get the original video file to retrieve frame dimensions for denormalization
                    recording_path = self.recording_path
                    video_file = recording_path / f"port_{port_num}.mp4"
                    
                    if video_file.exists():
                        # Get frame dimensions from video metadata
                        import cv2
                        cap = cv2.VideoCapture(str(video_file))
                        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        # Convert normalized (0-1) coordinates back to pixel coordinates
                        pixel_x = norm_x * frame_w
                        pixel_y = norm_y * frame_h
                        
                        xy_data.append({
                            'sync_index': frame_num,
                            'port': port_num,
                            'point_id': class_id,
                            'img_loc_x': pixel_x,
                            'img_loc_y': pixel_y,
                        })
        
        # Create temporary xy CSV for triangulation input
        xy_df = pd.DataFrame(xy_data)
        
        if xy_df.empty:
            logger.warning("No detections found in BGS labels")
            return None
        
        temp_xy_path = self.bgs_dir / "bgs_detections_xy_temp.csv"
        xy_df.to_csv(temp_xy_path, index=False)
        logger.info(f"Created temporary xy file at {temp_xy_path} with {len(xy_df)} detections")
        
        return temp_xy_path
    
    def triangulate_bgs_detections(self) -> Optional[pd.DataFrame]:
        """
        Triangulate BGS detections from YOLO labels in 3 views.
        
        Returns:
            DataFrame with triangulated 3D points, or None if triangulation failed
        """
        # Create xy input from BGS labels
        xy_path = self.create_bgs_xy_input()
        
        if xy_path is None or not xy_path.exists():
            logger.warning("Could not create xy input for BGS triangulation")
            return None
        
        try:
            # Triangulate using the standard triangulation function
            xyz_bgs = triangulate_from_files(
                config_path=self.config_path,
                xy_path=xy_path,
                output_path=None  # Don't save yet, will save after supplementation
            )
            
            logger.info(f"Triangulated {len(xyz_bgs)} BGS points")
            
            # Clean up temporary file
            xy_path.unlink()
            
            return xyz_bgs
            
        except Exception as e:
            logger.error(f"Error during BGS triangulation: {e}")
            return None
    
    def triangulate_bgs_detections_and_save(self, 
                              yolo_xyz_path: Path, 
                              bbox: Tuple[int, int, int, int],
                              region: str = "full") -> Optional[Path]:
        """
        Triangulate BGS detections to 3D and save for hybrid filtering.
        
        NOTE: This method only triangulates BGS detections. 
        Hybrid selection between YOLO and BGS happens during Kalman filtering in post-processing.
        
        Parameters:
            yolo_xyz_path: Path to xyz_FLY_predictions.csv (unused, kept for API compatibility)
            bbox: Bounding box tuple (unused, kept for API compatibility)
            region: Region name (unused, kept for API compatibility)
            
        Returns:
            Path to BGS xyz file (FLY/bgs/xyz_FLY_bgs_predictions.csv)
        """
        
        # Triangulate BGS detections
        bgs_xyz = self.triangulate_bgs_detections()
        
        if bgs_xyz is None or bgs_xyz.empty:
            logger.warning("No BGS triangulations available")
            return None
        
        logger.info(f"Triangulated {len(bgs_xyz)} BGS 3D points")
        
        # Save BGS predictions without any supplementation/replacement
        output_path = self.bgs_dir / "xyz_FLY_bgs_predictions.csv"
        bgs_xyz.to_csv(output_path, index=False)
        logger.info(f"Saved BGS triangulated predictions to {output_path}")
        
        return output_path



