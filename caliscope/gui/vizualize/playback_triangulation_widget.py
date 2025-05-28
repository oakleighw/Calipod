from pathlib import Path
from time import time
import os

import numpy as np
import pyqtgraph.opengl as gl
# from PySide6.QtCore import Qt, QBuffer, QIODevice # QBuffer, QIODevice are not strictly needed here with the updated conversion
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

logger = caliscope.logger.get(__name__)


class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.visualizer = TriangulationVisualizer(self.camera_array)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.export_button = QPushButton("Export Video") # Changed button text
        self.export_button.setCheckable(True)

        self.export_video_mode = False
        self.video_framerate = 60 # You can make this configurable in the GUI later

        self.setMinimumSize(500, 500)

        self.place_widgets()
        self.connect_widgets()
        if xyz_history_path is not None:
            self.update_motion_trial(xyz_history_path)
        else:
            self.motion_trial = None

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

            if self.motion_trial:
                logger.info("Automatically playing through frames for export...")
                for i in range(self.motion_trial.start_index, self.motion_trial.end_index + 1):
                    # Setting slider value triggers display_points, which now collects frames
                    self.slider.setValue(i)
                logger.info("Finished collecting frames.")

                if self.motion_trial and self.motion_trial.xyz_csv:
                    # Save video to the directory of the loaded motion trial
                    video_output_dir = self.motion_trial.xyz_csv.parent
                else:
                    # Fallback to current working directory if motion_trial path is not available
                    logger.warning("Motion trial path not available. Saving video to current working directory.")
                    video_output_dir = Path.cwd()

                output_video_path = video_output_dir / "exported_motion_video.mp4"

            
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
        self.motion_trial = MotionTrial(xyz_history_path)
        logger.info(f"Motion trial loading complete: {time()} ")
        toc = time()
        logger.info(f"Elapsed time to load: {toc-tic}")

        self.visualizer.update_motion_trial(self.motion_trial)

        if self.motion_trial.is_empty:
            self.slider.setMinimum(0)
            self.slider.setMaximum(100)
        else:
            self.slider.setMinimum(self.motion_trial.start_index)
            self.slider.setMaximum(self.motion_trial.end_index)

    def update_camera_array(self, camera_array: CameraArray):
        self.visualizer.update_camera_array(camera_array)


class TriangulationVisualizer:
    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()
        self.export_video_mode = False
        self.collected_frames = [] # List to store frames in memory

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

        self.sync_index = self.motion_trial.start_index
        self.display_points(self.sync_index)

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

        if self.motion_trial.is_empty:
            logger.debug(f"Motion trial is empty for sync_index: {sync_index}. Skipping point display.")
            self.scatter.setData(pos=None)
        else:
            self.sync_index = sync_index
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_coords = self.motion_trial.get_xyz(self.sync_index).point_xyz
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
                        logger.debug(f"Successfully collected frame {self.sync_index} to memory. Total frames: {len(self.collected_frames)}")
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