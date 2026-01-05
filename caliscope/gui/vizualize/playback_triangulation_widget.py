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
    QPushButton,
    QApplication
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
            self.export_button.setEnabled(False)
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

            # --- MODIFIED EXPORT LOOP START ---
            # Temporarily disconnect display_points and update_segment_lines from the slider's valueChanged signal
            # This prevents multiple updates per frame (one from setValue, one from direct call)
            # and gives more direct control over when rendering happens for export.
            try:
                self.slider.valueChanged.disconnect(self.visualizer.display_points)
                self.slider.valueChanged.disconnect(self.visualizer.update_segment_lines)
            except TypeError: # Disconnect might fail if not connected, ignore.
                pass 

            for i in range(export_start_frame, export_end_frame + 1):
                # Update slider value for visual feedback, but block its signals
                # to prevent re-triggering display_points/update_segment_lines via the signal.
                self.slider.blockSignals(True) 
                self.slider.setValue(i)
                self.slider.blockSignals(False) 

                # Directly call display_points and update_segment_lines to render and collect the frame
                # This ensures the visualizer state is updated for framebuffer grab.
                self.visualizer.display_points(i) 
                self.visualizer.update_segment_lines(i) 

                # Process events to keep the GUI responsive during the potentially long export loop.
                # This allows camera panning or other UI interactions to be handled,
                # which can help prevent the "EError" by not starving the event loop.
                QApplication.processEvents() 

            # Reconnect display_points and update_segment_lines to the slider after export is complete
            self.slider.valueChanged.connect(self.visualizer.display_points)
            self.slider.valueChanged.connect(self.visualizer.update_segment_lines)
            # --- MODIFIED EXPORT LOOP END ---


            # for i in range(export_start_frame, export_end_frame + 1):
            #     self.slider.setValue(i) 
            # logger.info("Finished collecting frames.")

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

            self.export_button.setEnabled(True)
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
        
        # Storage for custom mesh items for special labels
        self.custom_mesh_items = []  # Store references to added mesh items

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
            self.origin_points = {}
            for port, cam in self.camera_array.cameras.items():
                mesh, origin_point = mesh_from_camera(cam)
                mesh: CameraMesh = mesh
                origin_point: CameraMesh = origin_point
                mesh.setColor(self.default_mesh_color) # Set initial mesh color
                origin_point.setColor(cam.color)
                self.meshes[port] = mesh
                self.origin_points[port] = origin_point
                self.scene.addItem(mesh)
                self.scene.addItem(origin_point)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3)),  # Start with empty array instead of None
            color=self.default_scatter_color, # Set initial scatter color
            size=0.01,
            pxMode=False,
        )
        self.scatter.setVisible(False)  # Hide until we have data

        self.segments = {}
        self.scene.addItem(self.scatter)

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

        # Clear previous custom meshes
        for mesh_item in self.custom_mesh_items:
            self.scene.removeItem(mesh_item)
        self.custom_mesh_items = []

        if self.motion_trial is None or self.motion_trial.is_empty:
            logger.debug(f"Motion trial is not loaded or is empty for sync_index: {sync_index}. Skipping point display.")
            self.scatter.setVisible(False)  # Hide scatter when no data
            self.scatter.setData(pos=np.empty((0, 3)))  # Use empty array instead of None
        else:
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_packet = self.motion_trial.get_xyz(sync_index)
            xyz_coords = xyz_packet.point_xyz
            point_ids = xyz_packet.point_ids
            
            # Check if we're using FlyTracker with special labels
            has_fly_tracker = (hasattr(self.motion_trial, 'tracker') and 
                             hasattr(self.motion_trial.tracker, 'name') and
                             self.motion_trial.tracker.name == "FLY")
            
            logger.info(f"has_fly_tracker: {has_fly_tracker}, tracker: {self.motion_trial.tracker if hasattr(self.motion_trial, 'tracker') else 'None'}, point_ids: {point_ids}, num_points: {len(xyz_coords)}")
            
            if has_fly_tracker and len(xyz_coords) > 0:
                # Get average bbox dimensions for this tracker
                try:
                    avg_bbox_by_id = self.motion_trial.tracker.get_average_bbox_by_point_id()
                    logger.info(f"Average bbox data retrieved: {avg_bbox_by_id}")
                except Exception as e:
                    logger.warning(f"Could not get bbox data from tracker: {e}")
                    avg_bbox_by_id = {}
                
                # If bbox_data is empty (tracking done before bbox capture was added),
                # use default sizes for fruit and leaves
                if not avg_bbox_by_id:
                    logger.info("Using default bbox sizes for fruit and leaves (no captured bbox data)")
                    # Default sizes in pixels - adjust these based on your typical object sizes
                    # These are reasonable defaults for fruit/leaves at typical camera distances
                    avg_bbox_by_id = {
                        9: (100.0, 100.0),   # fruit: 100x100 pixels default
                        10: (150.0, 150.0)   # leaves: 150x150 pixels default
                    }
                
                # Separate special labels (9=fruit, 10=leaves) from regular points
                regular_mask = np.ones(len(point_ids), dtype=bool)
                
                for i, (point_id, xyz) in enumerate(zip(point_ids, xyz_coords)):
                    logger.info(f"Processing point {i}: point_id={point_id} (type: {type(point_id)}), xyz={xyz}")
                    # Convert point_id to int for dictionary lookup
                    point_id_int = int(point_id)
                    
                    # Check if this is a center point for fruit or leaves
                    if point_id_int in [9, 10]:
                        regular_mask[i] = False
                        
                        # Find the 4 corner points for this object
                        # Corner IDs are: point_id * 1000 + [0, 1, 2, 3]
                        corner_ids = [point_id_int * 1000 + j for j in range(4)]
                        corner_xyzs = []
                        
                        for corner_id in corner_ids:
                            corner_mask = point_ids == corner_id
                            if np.any(corner_mask):
                                corner_xyz = xyz_coords[corner_mask][0]
                                corner_xyzs.append(corner_xyz)
                                # Mark corners as not regular points
                                corner_idx = np.where(point_ids == corner_id)[0]
                                if len(corner_idx) > 0:
                                    regular_mask[corner_idx[0]] = False
                        
                        # If we found all 4 corners, use them to create the mesh
                        if len(corner_xyzs) == 4:
                            logger.info(f"Found all 4 triangulated corners for point_id={point_id_int}")
                            
                            if point_id_int == 10:  # leaves - flat green square using triangulated corners
                                vertices = np.array(corner_xyzs, dtype=np.float32)
                                
                                # Two triangles to form the square
                                faces = np.array([
                                    [0, 1, 2],  # First triangle: TL, TR, BR
                                    [0, 2, 3],  # Second triangle: TL, BR, BL
                                ], dtype=np.uint32)
                                
                                colors = np.array([(0, 1, 0, 0.6), (0, 1, 0, 0.6)], dtype=np.float32)
                                
                                mesh_item = gl.GLMeshItem(
                                    vertexes=vertices,
                                    faces=faces,
                                    faceColors=colors,
                                    smooth=False,
                                    drawEdges=True,
                                    edgeColor=(0, 0.5, 0, 1)
                                )
                                mesh_item.setGLOptions("translucent")
                                self.scene.addItem(mesh_item)
                                self.custom_mesh_items.append(mesh_item)
                                logger.info(f"Created flat square mesh for leaves from triangulated corners")
                                
                            elif point_id_int == 9:  # fruit - red hemisphere using triangulated corners
                                # Use the 4 corners to determine the actual 3D extent
                                corners_array = np.array(corner_xyzs)
                                
                                # Calculate the actual 3D width and height from corners
                                width_3d = np.linalg.norm(corners_array[1] - corners_array[0])  # TR - TL
                                height_3d = np.linalg.norm(corners_array[3] - corners_array[0])  # BL - TL
                                radius_3d = min(width_3d, height_3d) * 0.5
                                
                                # Use center point as peak of hemisphere
                                vertices, faces, colors = self.create_hemisphere_from_center(
                                    xyz, radius_3d, color=(1, 0, 0, 0.6), segments=16
                                )
                                
                                mesh_item = gl.GLMeshItem(
                                    vertexes=vertices,
                                    faces=faces,
                                    faceColors=colors,
                                    smooth=True,
                                    drawEdges=False
                                )
                                mesh_item.setGLOptions("translucent")
                                self.scene.addItem(mesh_item)
                                self.custom_mesh_items.append(mesh_item)
                                logger.info(f"Created hemisphere mesh for fruit with radius={radius_3d:.4f}")
                        else:
                            logger.warning(f"Could not find all 4 corners for point_id={point_id_int}, found {len(corner_xyzs)} corners")
                            # Fall back to showing just the center point
                    
                    # Skip corner points (IDs >= 1000) - they're already handled above
                    elif point_id_int >= 1000:
                        regular_mask[i] = False
                
                # Display regular points (excluding special labels)
                regular_coords = xyz_coords[regular_mask]
                if len(regular_coords) > 0:
                    self.scatter.setVisible(True)
                    self.scatter.setData(pos=regular_coords)
                else:
                    self.scatter.setVisible(False)
                    self.scatter.setData(pos=np.empty((0, 3)))
            else:
                # Default behavior for non-FlyTracker or when no special handling needed
                self.scatter.setVisible(True)  # Make visible when we have data
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

    def create_hemisphere_from_center(self, center_xyz, radius_3d, color=(1, 0, 0, 0.6), segments=16):
        """Create a hemisphere mesh with a specified 3D radius.
        
        Args:
            center_xyz: (x, y, z) center position (peak of hemisphere)
            radius_3d: radius in 3D world coordinates
            color: RGBA color tuple
            segments: number of segments for hemisphere smoothness
        
        Returns:
            tuple: (vertices, faces, colors) for GLMeshItem
        """
        cx, cy, cz = center_xyz
        
        vertices = []
        faces = []
        
        # Add center peak point of hemisphere
        vertices.append([cx, cy, cz])
        
        # Generate hemisphere vertices
        for i in range(segments // 2 + 1):  # From equator to pole
            lat = i * (np.pi / 2) / (segments // 2)  # 0 to pi/2
            z_offset = -radius_3d * np.cos(lat)  # Negative because hemisphere extends down from peak
            ring_radius = radius_3d * np.sin(lat)
            
            for j in range(segments):
                lon = j * (2 * np.pi) / segments
                x_offset = ring_radius * np.cos(lon)
                y_offset = ring_radius * np.sin(lon)
                
                vertices.append([cx + x_offset, cy + y_offset, cz + z_offset])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Create faces
        # Connect peak to first ring
        for j in range(segments):
            next_j = (j + 1) % segments
            faces.append([0, j + 1, next_j + 1])
        
        # Connect rings
        for i in range(segments // 2):
            for j in range(segments):
                next_j = (j + 1) % segments
                
                current_base = 1 + i * segments
                next_base = 1 + (i + 1) * segments
                
                v1 = current_base + j
                v2 = current_base + next_j
                v3 = next_base + next_j
                v4 = next_base + j
                
                # Two triangles per quad
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        faces = np.array(faces, dtype=np.uint32)
        colors = np.array([color] * len(faces), dtype=np.float32)
        
        return vertices, faces, colors

    def clear_grid_labels(self): # <--- ADD THIS ENTIRE METHOD
        for label in self.grid_labels:
            self.scene.removeItem(label)
        self.grid_labels = []
        logger.info("Cleared existing grid labels.")

        

    def add_grid_labels(self, grid_total_extent_m=10, grid_spacing_m=0.1, text_color='white'):
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