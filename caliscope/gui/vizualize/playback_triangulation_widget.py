from pathlib import Path
from time import time
import os

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt # Keep Qt
from PySide6.QtWidgets import (
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton
)
from PySide6.QtGui import QImage, QColorConstants # Added QColorConstants for potential debug, though not used in core logic

import caliscope.logger
from caliscope.cameras.camera_array import CameraArray
from caliscope.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from caliscope.motion_trial import MotionTrial

import cv2 # Import OpenCV
import rtoml # <--- ADD THIS IMPORT
import pandas as pd # <--- ADD THIS IMPORT
from typing import Optional # <--- ADD THIS IMPORT


logger = caliscope.logger.get(__name__)

# <--- REMOVE THIS LINE IF IT WAS ADDED BY MISTAKE:
# from caliscope.gui.triangulation_visualizer import TriangulationVisualizer # This import is not needed as it's defined locally


class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.visualizer = TriangulationVisualizer(self.camera_array)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.export_button = QPushButton("Export Video")
        self.export_button.setCheckable(True)

        self.export_video_mode = False
        self.video_framerate = 60 # You can make this configurable in the GUI later

        # NEW: Attributes to store the full session frame range and the path
        self._session_start_frame: Optional[int] = None
        self._session_end_frame: Optional[int] = None
        self.xyz_history_path: Optional[Path] = None # NEW: Store the path for later inference

        self.setMinimumSize(500, 500)

        self.place_widgets()
        self.connect_widgets()
        if xyz_history_path is not None:
            self.update_motion_trial(xyz_history_path)
        else:
            self.motion_trial = None
            # NEW: Set initial slider range if no trial is loaded at startup
            self.slider.setMinimum(0)
            self.slider.setMaximum(0) # Null range if nothing loaded
            self.slider.setValue(0)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.export_button)

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.slider.valueChanged.connect(self.visualizer.update_segment_lines)
        self.export_button.toggled.connect(self.toggle_export_mode)

    def toggle_export_mode(self, checked):
        """Toggles video export mode based on button state."""
        self.export_video_mode = checked
        self.visualizer.set_export_mode(checked) # Tell visualizer to collect frames

        if checked:
            # Clear any previously collected frames in the visualizer
            self.visualizer.clear_collected_frames()
            logger.info("Starting video export process (collecting frames in memory)...")

            # --- MODIFIED: Determine the range of frames to export ---
            export_start_frame = 0
            export_end_frame = 0 # Initialize with safe defaults
            
            # Priority 1: Use the full session range from frame_time_history.csv if available
            if self._session_start_frame is not None and self._session_end_frame is not None:
                export_start_frame = self._session_start_frame
                export_end_frame = self._session_end_frame
                logger.info(f"Exporting video based on full session frame range: {export_start_frame} to {export_end_frame}.")
                
                # Ensure visualizer motion_trial is set up for handling potentially empty frames
                # If no motion_trial was loaded, but we have a session range, create an empty one
                # for display_points to work without error when iterating over potentially non-existent sync_indices.
                if self.visualizer.motion_trial is None: # Check visualizer's motion_trial, not self's
                    self.visualizer.motion_trial = MotionTrial() 
                
            # Priority 2: Fallback to range from loaded MotionTrial (if no session range was found)
            elif self.motion_trial and not self.motion_trial.is_empty:
                export_start_frame = self.motion_trial.start_index
                export_end_frame = self.motion_trial.end_index
                logger.info(f"Exporting video based on loaded motion trial range: {export_start_frame} to {export_end_frame}.")
            
            # Priority 3: Default to a short duration if no motion trial or session range is known
            else:
                export_start_frame = 0
                export_end_frame = self.video_framerate * 3 # Export 3 seconds
                logger.warning(f"No valid motion trial or session range available. Exporting {export_end_frame - export_start_frame + 1} frames of empty scene.")
                # Ensure visualizer motion_trial is set up for empty frames
                if self.visualizer.motion_trial is None: # Check visualizer's motion_trial, not self's
                    self.visualizer.motion_trial = MotionTrial() 

            # The loop now uses the determined export_start_frame and export_end_frame
            for i in range(export_start_frame, export_end_frame + 1):
                self.slider.setValue(i) # This will trigger display_points for each frame
            logger.info("Finished collecting frames.")

            # Define the output path for the exported video
            if self.xyz_history_path: # Use self.xyz_history_path, not self.motion_trial.xyz_csv
                video_name = self.xyz_history_path.stem.replace("xyz_", "exported_") + ".mp4"
                output_video_path = self.xyz_history_path.parent / video_name
            else:
                output_video_path = Path.cwd() / "exported_motion_video.mp4" # Default if no xyz_history_path was loaded
                logger.warning(f"Motion trial path not available. Saving video to current working directory: {output_video_path}")

            
            collected_frames = self.visualizer.get_collected_frames()

            if collected_frames:
                try:
                    # Get dimensions from the first frame
                    first_frame = collected_frames[0]
                    height, width, _ = first_frame.shape
                    size = (width, height)

                    # Define the codec and create VideoWriter object
                    # This codec (MP4V) generally works well for .mp4 containers
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(output_video_path), fourcc, self.video_framerate, size)

                    if not out.isOpened():
                        logger.error(f"Failed to open video writer for {output_video_path}. Check codec availability (e.g., system FFmpeg) and path.")
                    else:
                        logger.info(f"Writing {len(collected_frames)} frames to video: {output_video_path}")
                        for frame in collected_frames:
                            out.write(frame)
                        out.release() # Release the video writer
                        logger.info(f"Video saved to: {output_video_path}")
                except Exception as e:
                    logger.error(f"Failed to create video from collected frames: {e}")
            else:
                logger.warning("No frames were collected to combine into video.")

            # Reset button state
            self.export_button.setChecked(False) # Turn off button after auto-export

        else:
            logger.info("Video export mode disabled.")


    def update_motion_trial(self, xyz_history_path):
        tic = time()
        logger.info(f"Beginning to load in motion trial: {time()}")
        
        # NEW: Store the path to the xyz_history.csv
        self.xyz_history_path = xyz_history_path
        self.motion_trial = MotionTrial(xyz_history_path) # Uses the path to load the trial
        logger.info(f"Motion trial loading complete: {time()} ")
        toc = time()
        logger.info(f"Elapsed time to load: {toc-tic}")

        self.visualizer.update_motion_trial(self.motion_trial)

        # Load framerate from project-specific config.toml
        # MODIFIED: Use self.xyz_history_path for consistency
        if self.xyz_history_path: 
            project_config_path = self.xyz_history_path.parent / "config.toml"
            try:
                if project_config_path.exists():
                    project_config_data = rtoml.load(project_config_path)
                    self.video_framerate = project_config_data.get("fps_sync_stream_processing", 60) # Changed default to 60 to match your code
                    logger.info(f"Video export framerate set from project config ({project_config_path}) to: {self.video_framerate}")
                else:
                    logger.info(f"Project config.toml not found at {project_config_path}. Using default video framerate: {self.video_framerate}")
            except Exception as e:
                logger.error(f"Error reading project config.toml for framerate: {e}. Using default video framerate: {self.video_framerate}")
        else:
            logger.warning("Motion trial path not available, cannot load project-specific config.toml for framerate. Using default.")

        # --- NEW: Load session frame range from frame_time_history.csv ---
        self._session_start_frame = None # Reset previous values
        self._session_end_frame = None   # Reset previous values

        if self.xyz_history_path: # Check if a path was provided to infer from
            frame_time_history_path = self.xyz_history_path.parent / "frame_time_history.csv"
            try:
                if frame_time_history_path.exists():
                    # Read only 'sync_index' column to minimize memory usage
                    frame_time_df = pd.read_csv(frame_time_history_path, usecols=["sync_index"], engine="pyarrow")
                    if not frame_time_df.empty and "sync_index" in frame_time_df.columns:
                        self._session_start_frame = int(frame_time_df["sync_index"].min())
                        self._session_end_frame = int(frame_time_df["sync_index"].max())
                        logger.info(f"Session frame range loaded from {frame_time_history_path}: {self._session_start_frame} to {self._session_end_frame}")
                    else:
                        logger.warning(f"frame_time_history.csv at {frame_time_history_path} is empty or missing 'sync_index' column. Cannot determine full session frame range.")
                else:
                    logger.warning(f"frame_time_history.csv not found at {frame_time_history_path}. Cannot determine full session frame range.")
            except Exception as e:
                logger.error(f"Error reading frame_time_history.csv: {e}. Cannot determine full session frame range.")
        # --- END NEW ---

        # NEW: Set the slider range based on the loaded session frame range, or motion trial range, or a null default
        if self._session_start_frame is not None and self._session_end_frame is not None:
             self.slider.setMinimum(self._session_start_frame)
             self.slider.setMaximum(self._session_end_frame)
             self.slider.setValue(self._session_start_frame) # Move slider to start of session
        elif self.motion_trial.is_empty: # This would happen if xyz_history.csv was empty or invalid
            self.slider.setMinimum(0)
            self.slider.setMaximum(0) # Null range if no session range and empty motion trial
            self.slider.setValue(0)
        else: # Motion trial exists (not empty), but no frame_time_history was found
            self.slider.setMinimum(self.motion_trial.start_index)
            self.slider.setMaximum(self.motion_trial.end_index)
            self.slider.setValue(self.motion_trial.start_index)

    def update_camera_array(self, camera_array: CameraArray):
        self.visualizer.update_camera_array(camera_array)


class TriangulationVisualizer:
    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()
        self.export_video_mode = False
        self.collected_frames = [] # List to store frames in memory
        self.motion_trial: Optional[MotionTrial] = None # NEW: Initialize motion_trial as None

    def build_scene(self):
        if hasattr(self, "scene"):
            logger.info("Clearing scene in capture volume visualizer")
            self.scene.clear()
        else:
            logger.info("Creating initial scene in capture volume visualizer")
            self.scene = gl.GLViewWidget()
            self.scene.setCameraPosition(distance=4)
        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        if self.camera_array.all_extrinsics_calibrated():
            self.meshes = {}
            for port, cam in self.camera_array.cameras.items():
                mesh: CameraMesh = mesh_from_camera(cam)
                self.meshes[port] = mesh
                self.scene.addItem(mesh)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.array([0, 0, 0]),
            color=[1, 1, 1, 1],
            size=0.01,
            pxMode=False,
        )

        self.segments = {}
        self.scene.addItem(self.scatter)
        self.scatter.setData(pos=None)

    def update_camera_array(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()

    def update_motion_trial(self, motion_trial: MotionTrial):
        logger.info("Updating xyz history in playback widget")
        self.motion_trial: MotionTrial = motion_trial

        if hasattr(self.motion_trial.tracker, "wireframe"):
            for segment_line in self.motion_trial.tracker.wireframe.line_plots.values():
                self.scene.addItem(segment_line)

        # self.sync_index = self.motion_trial.start_index # Removed - sync_index is passed directly
        self.display_points(self.motion_trial.start_index) # Call display_points with start_index

    def set_export_mode(self, enabled: bool):
        self.export_video_mode = enabled

    def clear_collected_frames(self):
        self.collected_frames = []
        logger.debug("Cleared collected frames list.")

    def get_collected_frames(self):
        return self.collected_frames

    def qimage_to_cv2(self, qimage: QImage):
        """
        Converts a QImage to a NumPy array suitable for OpenCV (BGR format).
        Handles ARGB32 format, which is common for QImage grabbed from GL.
        """
        if qimage.isNull():
            logger.warning("qimage_to_cv2 received a null QImage.")
            return None

        # Ensure the QImage is in a format that can be easily converted to a byte array
        if qimage.format() != QImage.Format_ARGB32:
            qimage = qimage.convertToFormat(QImage.Format_ARGB32)

        # Get the raw pixel data as a memoryview
        ptr = qimage.constBits()
        
        # Create a NumPy array from the memoryview.
        # Reshape to (height, width, 4) for ARGB (or BGRA)
        # .as_array() ensures a proper NumPy array view from the memoryview
        arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)

        # Convert BGRA (which is how ARGB32 is typically stored on little-endian) to BGR for OpenCV
        bgr_frame = arr[:, :, :3].copy() # Take BGR channels (0, 1, 2). .copy() ensures it's contiguous.
        
        return bgr_frame

    def display_points(self, sync_index: int):
        logger.debug(f"display_points called for sync_index: {sync_index}")

        # Check if motion_trial is even loaded before trying to access its properties
        if self.motion_trial is None or self.motion_trial.is_empty:
            logger.debug(f"Motion trial is not loaded or is empty for sync_index: {sync_index}. Skipping point display.")
            self.scatter.setData(pos=None) # Ensure scatter plot is empty
        else:
            # self.sync_index = sync_index # Not strictly needed as sync_index is passed
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_coords = self.motion_trial.get_xyz(sync_index).point_xyz
            self.scatter.setData(pos=xyz_coords)

            if self.export_video_mode:
                logger.debug(f"Export mode is active for sync_index: {sync_index}. Attempting to grab framebuffer.")
                image = self.scene.grabFramebuffer()

                if image.isNull():
                    logger.warning(f"grabFramebuffer returned a null image for sync_index: {sync_index}!")
                    # Do not append to collected_frames if image is null
                else:
                    cv2_frame = self.qimage_to_cv2(image)
                    if cv2_frame is not None and cv2_frame.size > 0:
                        self.collected_frames.append(cv2_frame)
                        logger.debug(f"Successfully collected frame {sync_index} to memory. Total frames: {len(self.collected_frames)}")
                    else:
                        logger.warning(f"qimage_to_cv2 returned an empty or invalid frame for sync_index: {sync_index}!")
            else:
                logger.debug(f"Export mode is INACTIVE for sync_index: {sync_index}.")

    def update_segment_lines(self, sync_index: int):
        if (self.motion_trial and
            hasattr(self.motion_trial, 'tracker') and
            hasattr(self.motion_trial.tracker, 'wireframe') and
            self.motion_trial.tracker.wireframe is not None):
            self.motion_trial.update_wireframe(sync_index)
        else:
            logger.debug(f"No wireframe to update from PlaybackTriangulationWidget for sync index {sync_index}.")