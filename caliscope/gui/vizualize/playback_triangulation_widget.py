# caliscope/gui/vizualize/playback_triangulation_widget.py

from pathlib import Path
from time import time
import os

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg 
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton
)
from PySide6.QtGui import QImage, QColorConstants, QColor, QVector3D


import caliscope.logger
from caliscope.cameras.camera_array import CameraArray
from caliscope.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from caliscope.motion_trial import MotionTrial

import cv2
import rtoml
import pandas as pd
from typing import Optional


logger = caliscope.logger.get(__name__)


class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.visualizer = TriangulationVisualizer(self.camera_array)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.export_button = QPushButton("Export Video")
        self.export_button.setCheckable(True)

        self.measurement_view_button = QPushButton("Measurement View") 

        self.export_video_mode = False
        self.video_framerate = 60 

        self._session_start_frame: Optional[int] = None
        self._session_end_frame: Optional[int] = None
        self.xyz_history_path: Optional[Path] = None 

        self.setMinimumSize(500, 500)

        self.place_widgets()
        self.connect_widgets()
        if xyz_history_path is not None:
            self.update_motion_trial(xyz_history_path)
        else:
            self.motion_trial = None
            self.slider.setMinimum(0)
            self.slider.setMaximum(0) 
            self.slider.setValue(0)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.export_button)
        self.layout().addWidget(self.measurement_view_button)

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.slider.valueChanged.connect(self.visualizer.update_segment_lines)
        self.export_button.toggled.connect(self.toggle_export_mode)
        self.measurement_view_button.clicked.connect(self.visualizer.toggle_measurement_mode)

    def toggle_export_mode(self, checked):
        """Toggles video export mode based on button state."""
        self.export_video_mode = checked
        self.visualizer.set_export_mode(checked) 

        if checked:
            self.visualizer.clear_collected_frames()
            logger.info("Starting video export process (collecting frames in memory)...")

            export_start_frame = 0
            export_end_frame = 0 
            
            if self._session_start_frame is not None and self._session_end_frame is not None:
                export_start_frame = self._session_start_frame
                export_end_frame = self._session_end_frame
                logger.info(f"Exporting video based on full session frame range: {export_start_frame} to {export_end_frame}.")
                
                if self.visualizer.motion_trial is None: 
                    self.visualizer.motion_trial = MotionTrial() 
                
            elif self.motion_trial and not self.motion_trial.is_empty:
                export_start_frame = self.motion_trial.start_index
                export_end_frame = self.motion_trial.end_index
                logger.info(f"Exporting video based on loaded motion trial range: {export_start_frame} to {export_end_frame}.")
            
            else:
                export_start_frame = 0
                export_end_frame = self.video_framerate * 3 
                logger.warning(f"No valid motion trial or session range available. Exporting {export_end_frame - export_start_frame + 1} frames of empty scene.")
                if self.visualizer.motion_trial is None: 
                    self.visualizer.motion_trial = MotionTrial() 

            for i in range(export_start_frame, export_end_frame + 1):
                self.slider.setValue(i) 
            logger.info("Finished collecting frames.")

            if self.xyz_history_path: 
                video_name = self.xyz_history_path.stem.replace("xyz_", "exported_") + ".mp4"
                output_video_path = self.xyz_history_path.parent / video_name
            else:
                output_video_path = Path.cwd() / "exported_motion_video.mp4" 
                logger.warning(f"Motion trial path not available. Saving video to current working directory: {output_video_path}")

            
            collected_frames = self.visualizer.get_collected_frames()

            if collected_frames:
                try:
                    first_frame = collected_frames[0]
                    height, width, _ = first_frame.shape
                    size = (width, height)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(output_video_path), fourcc, self.video_framerate, size)

                    if not out.isOpened():
                        logger.error(f"Failed to open video writer for {output_video_path}. Check codec availability (e.g., system FFmpeg) and path.")
                    else:
                        logger.info(f"Writing {len(collected_frames)} frames to video: {output_video_path}")
                        for frame in collected_frames:
                            out.write(frame)
                        out.release() 
                        logger.info(f"Video saved to: {output_video_path}")
                except Exception as e:
                    logger.error(f"Failed to create video from collected frames: {e}")
            else:
                logger.warning("No frames were collected to combine into video.")

            self.export_button.setChecked(False) 

        else:
            logger.info("Video export mode disabled.")


    def update_motion_trial(self, xyz_history_path):
        tic = time()
        logger.info(f"Beginning to load in motion trial: {time()}")
        
        self.xyz_history_path = xyz_history_path
        self.motion_trial = MotionTrial(xyz_history_path) 
        logger.info(f"Motion trial loading complete: {time()} ")
        toc = time()
        logger.info(f"Elapsed time to load: {toc-tic}")

        self.visualizer.update_motion_trial(self.motion_trial)

        if self.xyz_history_path: 
            project_config_path = self.xyz_history_path.parent / "config.toml"
            try:
                if project_config_path.exists():
                    project_config_data = rtoml.load(project_config_path)
                    self.video_framerate = project_config_data.get("fps_sync_stream_processing", 60) 
                    logger.info(f"Video export framerate set from project config ({project_config_path}) to: {self.video_framerate}")
                else:
                    logger.info(f"Project config.toml not found at {project_config_path}. Using default video framerate: {self.video_framerate}")
            except Exception as e:
                logger.error(f"Error reading project config.toml for framerate: {e}. Using default video framerate: {self.video_framerate}")
        else:
            logger.warning("Motion trial path not available, cannot load project-specific config.toml for framerate. Using default.")

        self._session_start_frame = None 
        self._session_end_frame = None   

        if self.xyz_history_path: 
            frame_time_history_path = self.xyz_history_path.parent / "frame_time_history.csv"
            try:
                if frame_time_history_path.exists():
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

        if self._session_start_frame is not None and self._session_end_frame is not None:
             self.slider.setMinimum(self._session_start_frame)
             self.slider.setMaximum(self._session_end_frame)
             self.slider.setValue(self._session_start_frame) 
        elif self.motion_trial.is_empty: 
            self.slider.setMinimum(0)
            self.slider.setMaximum(0) 
            self.slider.setValue(0)
        else: 
            self.slider.setMinimum(self.motion_trial.start_index)
            self.slider.setMaximum(self.motion_trial.end_index)
            self.slider.setValue(self.motion_trial.start_index)

    def update_camera_array(self, camera_array: CameraArray):
        self.visualizer.update_camera_array(camera_array)


class TriangulationVisualizer:
    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.default_scatter_color = (1, 1, 1, 1) # White
        self.default_mesh_color = (1, 1, 1, 1)    # White
        self.build_scene()
        self.export_video_mode = False
        self.collected_frames = [] 
        self.motion_trial: Optional[MotionTrial] = None 

        self.grid_labels = []
        self.axis_labels = []
        self.is_measurement_mode_active = False 
        self.xy_grid: Optional[gl.GLGridItem] = None
        self.xz_grid: Optional[gl.GLGridItem] = None

    def build_scene(self):
        if hasattr(self, "scene"):
            logger.info("Clearing scene in capture volume visualizer")
            self.scene.clear()
        else:
            logger.info("Creating initial scene in capture volume visualizer")
            self.scene = gl.GLViewWidget()
            self.scene.setCameraPosition(distance=4)
        
        # Default background color
        self.scene.setBackgroundColor(QColorConstants.Black) 

        axis = gl.GLAxisItem()
        
        self.scene.addItem(axis)

        grid_total_extent_m = 10
        grid_spacing_m = 0.1 

        # Initialize XY and XZ grids with a color visible on both backgrounds
        # Using a medium grey for visibility against both black and white backgrounds
        # This initial color will be set upon creation, but we'll change it in toggle_measurement_mode
        grid_line_initial_color = (0.5, 0.5, 0.5, 0.7) # R, G, B, A (medium grey, semi-transparent)

        self.xy_grid = gl.GLGridItem(size=QVector3D(grid_total_extent_m, grid_total_extent_m, 1), color=grid_line_initial_color)
        self.xy_grid.setSpacing(grid_spacing_m, grid_spacing_m, grid_spacing_m) # Set spacing after creation
        
        self.xz_grid = gl.GLGridItem(size=QVector3D(grid_total_extent_m, grid_total_extent_m, 1), color=grid_line_initial_color)
        self.xz_grid.setSpacing(grid_spacing_m, grid_spacing_m, grid_spacing_m) # Set spacing after creation
        self.xz_grid.rotate(90, 1, 0, 0) 

        self.scene.addItem(self.xy_grid)
        self.scene.addItem(self.xz_grid)
        self.xy_grid.setVisible(False)
        self.xz_grid.setVisible(False)

        if self.camera_array.all_extrinsics_calibrated():
            self.meshes = {}
            for port, cam in self.camera_array.cameras.items():
                mesh: CameraMesh = mesh_from_camera(cam)
                mesh.setColor(self.default_mesh_color) # Set initial mesh color
                self.meshes[port] = mesh
                self.scene.addItem(mesh)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.array([0, 0, 0]),
            color=self.default_scatter_color, # Set initial scatter color
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

        self.display_points(self.motion_trial.start_index) 

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

        if qimage.format() != QImage.Format_ARGB32:
            qimage = qimage.convertToFormat(QImage.Format_ARGB32)

        ptr = qimage.constBits()
        arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)
        bgr_frame = arr[:, :, :3].copy() 
        
        return bgr_frame

    def display_points(self, sync_index: int):
        logger.debug(f"display_points called for sync_index: {sync_index}")

        if self.motion_trial is None or self.motion_trial.is_empty:
            logger.debug(f"Motion trial is not loaded or is empty for sync_index: {sync_index}. Skipping point display.")
            self.scatter.setData(pos=None) 
        else:
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_coords = self.motion_trial.get_xyz(sync_index).point_xyz
            self.scatter.setData(pos=xyz_coords)

            if self.export_video_mode:
                logger.debug(f"Export mode is active for sync_index: {sync_index}. Attempting to grab framebuffer.")
                image = self.scene.grabFramebuffer()

                if image.isNull():
                    logger.warning(f"grabFramebuffer returned a null image for sync_index: {sync_index}!")
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
            self.motion_trial.tracker.update_wireframe_data(sync_index) 
        else:
            logger.debug(f"No wireframe to update from PlaybackTriangulationWidget for sync index {sync_index}.")

    def clear_grid_labels(self): # <--- ADD THIS ENTIRE METHOD
        for label in self.grid_labels:
            self.scene.removeItem(label)
        self.grid_labels = []
        logger.info("Cleared existing grid labels.")

        

    def add_grid_labels(self, grid_total_extent_m=10, grid_spacing_m=0.1, text_color='white'): # <--- ADD THIS ENTIRE METHOD
        self.clear_grid_labels() # Clear existing labels before adding new ones

        label_interval_m = 0.5 # Label every 0.5 meters (50 cm) for less clutter. Adjust as needed.

        # Determine the range for labels based on half the grid extent
        half_extent_m = grid_total_extent_m / 2

        # Iterate for X, Y, and Z axes
        # Use np.arange for float steps, and round to avoid floating point display issues
        for i in np.arange(-half_extent_m, half_extent_m + label_interval_m, label_interval_m):
            i = round(i, 2) 
            if i == 0.0: # Skip label at origin as axis item provides a visual cue
                continue
            
            # X-axis labels (on XY and XZ planes)
            # Position for X-axis labels (offset slightly in Y or Z for clarity)
            text_pos_x = (i, -0.05, 0) # Offset -0.05m in Y
            text_item_x = gl.GLTextItem(pos=text_pos_x, text=f"{i:.1f} m", color=text_color)
            self.scene.addItem(text_item_x)
            self.grid_labels.append(text_item_x)

            # For the XZ grid, we'll place X labels on the XZ plane
            text_pos_x_xz = (i, 0, -0.05) # Offset -0.05m in Z
            text_item_x_xz = gl.GLTextItem(pos=text_pos_x_xz, text=f"{i:.1f} m", color=text_color)
            self.scene.addItem(text_item_x_xz)
            self.grid_labels.append(text_item_x_xz)


        # Y-axis labels (on XY plane)
        for i in np.arange(-half_extent_m, half_extent_m + label_interval_m, label_interval_m):
            i = round(i, 2)
            if i == 0.0:
                continue
            text_pos_y = (-0.05, i, 0) # Offset -0.05m in X
            text_item_y = gl.GLTextItem(pos=text_pos_y, text=f"{i:.1f} m", color=text_color)
            self.scene.addItem(text_item_y)
            self.grid_labels.append(text_item_y)

        # Z-axis labels (for XZ plane, which uses global Z)
        for i in np.arange(-half_extent_m, half_extent_m + label_interval_m, label_interval_m):
            i = round(i, 2)
            if i == 0.0:
                continue
            text_pos_z = (-0.05, 0, i) # Offset -0.05m in X
            text_item_z = gl.GLTextItem(pos=text_pos_z, text=f"{i:.1f} m", color=text_color)
            self.scene.addItem(text_item_z)
            self.grid_labels.append(text_item_z)

        logger.info(f"Added {len(self.grid_labels)} grid labels.")

    
    def toggle_measurement_mode(self):
        self.is_measurement_mode_active = not self.is_measurement_mode_active
        logger.info(f"Toggling measurement mode. New state: {self.is_measurement_mode_active}")

        grid_total_extent_m = 10.0 # 10 meters extent
        grid_spacing_m = 0.1 # 0.1 meters (10 centimeters)

        if self.is_measurement_mode_active:
            self.scene.setBackgroundColor(QColorConstants.Black)
            self.scatter.setData(color=self.default_scatter_color)
            for mesh in self.meshes.values():
                mesh.setColor(self.default_mesh_color)
            
            if self.xy_grid:
                self.scene.removeItem(self.xy_grid)
                # --- CORRECTED GLGridItem INSTANTIATION AND SPACING ---
                self.xy_grid = gl.GLGridItem(size=QVector3D(grid_total_extent_m, grid_total_extent_m, 1), color='white') 
                self.xy_grid.setSpacing(grid_spacing_m, grid_spacing_m, grid_spacing_m) # Set spacing after creation
                self.scene.addItem(self.xy_grid)
                self.xy_grid.setVisible(True)
                logger.info(f"XY grid recreated (white, size={grid_total_extent_m}m, spacing={grid_spacing_m}m). Visible: {self.xy_grid.visible}")
                
            if self.xz_grid:
                self.scene.removeItem(self.xz_grid)
                # --- CORRECTED GLGridItem INSTANTIATION AND SPACING ---
                self.xz_grid = gl.GLGridItem(size=QVector3D(grid_total_extent_m, grid_total_extent_m, 1), color='white') 
                self.xz_grid.setSpacing(grid_spacing_m, grid_spacing_m, grid_spacing_m) # Set spacing after creation
                self.xz_grid.rotate(90, 1, 0, 0)
                self.scene.addItem(self.xz_grid)
                self.xz_grid.setVisible(True)
                logger.info(f"XZ grid recreated (white, size={grid_total_extent_m}m, spacing={grid_spacing_m}m). Visible: {self.xz_grid.visible}")

            self.add_grid_labels(grid_total_extent_m=grid_total_extent_m, text_color='white')
        else:
            self.scene.setBackgroundColor(QColorConstants.Black)
            self.scatter.setData(color=self.default_scatter_color)
            for mesh in self.meshes.values():
                mesh.setColor(self.default_mesh_color)

            if self.xy_grid:
                self.scene.removeItem(self.xy_grid)
                grid_total_extent_m = 10.0 
                grid_spacing_m = 0.1
                grid_line_initial_color = (0.5, 0.5, 0.5, 0.7)
                # --- CORRECTED GLGridItem INSTANTIATION AND SPACING ---
                self.xy_grid = gl.GLGridItem(size=QVector3D(grid_total_extent_m, grid_total_extent_m, 1), color=grid_line_initial_color) 
                self.xy_grid.setSpacing(grid_spacing_m, grid_spacing_m, grid_spacing_m) # Set spacing after creation
                self.scene.addItem(self.xy_grid)
                self.xy_grid.setVisible(False)
                logger.info(f"XY grid recreated (grey, size={grid_total_extent_m}m, spacing={grid_spacing_m}m) and re-hidden.")

            if self.xz_grid:
                self.scene.removeItem(self.xz_grid)
                grid_total_extent_m = 10.0 
                grid_spacing_m = 0.1
                grid_line_initial_color = (0.5, 0.5, 0.5, 0.7)
                # --- CORRECTED GLGridItem INSTANTIATION AND SPACING ---
                self.xz_grid = gl.GLGridItem(size=QVector3D(grid_total_extent_m, grid_total_extent_m, 1), color=grid_line_initial_color)
                self.xz_grid.setSpacing(grid_spacing_m, grid_spacing_m, grid_spacing_m) # Set spacing after creation
                self.xz_grid.rotate(90, 1, 0, 0) 
                self.scene.addItem(self.xz_grid)
                self.xz_grid.setVisible(False)
                logger.info(f"XZ grid recreated (grey, size={grid_total_extent_m}m, spacing={grid_spacing_m}m) and re-hidden.")
            
            self.clear_grid_labels()