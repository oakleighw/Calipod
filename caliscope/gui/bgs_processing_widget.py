from pathlib import Path
import re
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QMessageBox,
    QSlider,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

import caliscope.logger
from caliscope.controller import Controller
from caliscope.background_subtraction import BGSProcessor

logger = caliscope.logger.get(__name__)

# Region class mapping for YOLO labels (from fly_tracker)
# Labels 9 (fruit) and 10 (leaves) are used for 3D object visualization
REGION_CLASS_MAP = {
    "leaves": 10,
    "fruit": 9,
}

# BGS Processing Widget - used for supplementing deep learning annotations processing with background subtraction techniques.
class BGSProcessingWidget(QWidget):
    def __init__(self, controller: Controller):
        super(BGSProcessingWidget, self).__init__()
        self.controller = controller
        self.config = self.controller.config
        self.bgs_processor = None

        # Create tree widget for recording and video selection
        self.recording_tree = QTreeWidget()
        self.recording_tree.setHeaderLabel("Recordings")
        self.recording_tree.setColumnCount(1)
        self.populate_recording_tree()

        # Create parameter control widgets
        self.param_layout = self._create_parameter_controls()

        # Create process button
        self.process_btn = QPushButton("&Process")
        self.process_btn.setMaximumHeight(35)
        
        # Create output display title
        self.output_title = QLabel()
        self.update_output_title()
        
        # Video player components
        self.video_frames = []  # Store loaded frames
        self.current_frame_idx = 0
        
        # Create video display label
        self.video_display_label = QLabel()
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setMinimumHeight(400)
        self.video_display_label.setText("Processing output will display here when complete")
        
        # Create scroll area for video display
        self.output_scroll = QScrollArea()
        self.output_scroll.setWidgetResizable(True)
        self.output_scroll.setWidget(self.video_display_label)
        
        # Create frame slider for video navigation
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setVisible(False)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)
        
        # Create frame info label
        self.frame_info_label = QLabel()
        self.frame_info_label.setText("No video loaded")
        self.frame_info_label.setVisible(False)
        
        # Create play/pause button
        self.play_btn = QPushButton("Play")
        self.play_btn.setMaximumWidth(80)
        self.play_btn.setVisible(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.is_playing = False
        
        # Create timer for playback animation
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_frame)
        self.playback_fps = 30  # Playback speed

        # Set up the layout with tree on left and content on right
        main_layout = QHBoxLayout(self)
        
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(QLabel("Select Recording & Video:"))
        left_vbox.addWidget(self.recording_tree)
        left_vbox.addWidget(QLabel("Processing Parameters:"))
        left_vbox.addLayout(self.param_layout)
        left_vbox.addWidget(self.process_btn)
        left_vbox.addStretch()
        
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(self.output_title)
        right_vbox.addWidget(self.output_scroll, stretch=1)
        right_vbox.addWidget(self.frame_info_label)
        
        # Layout for slider and play button
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.play_btn)
        slider_layout.addWidget(self.frame_slider)
        right_vbox.addLayout(slider_layout)
        
        main_layout.addLayout(left_vbox, stretch=1)
        main_layout.addLayout(right_vbox, stretch=2)
        
        self.setLayout(main_layout)
        
        # Connect signals
        self.connect_widgets()

    def _create_parameter_controls(self):
        """Create the parameter input controls"""
        layout = QVBoxLayout()
        
        # Region selection
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Region:"))
        self.region_combo = QComboBox()
        self.region_combo.addItem("Full", "full")
        self.region_combo.addItem("Leaves", "leaves")
        self.region_combo.addItem("Fruit", "fruit")
        region_layout.addWidget(self.region_combo)
        layout.addLayout(region_layout)
        
        # Alpha (learning rate)
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha:"))
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setMinimum(0.001)
        self.alpha_spinbox.setMaximum(1.0)
        self.alpha_spinbox.setSingleStep(0.001)
        self.alpha_spinbox.setValue(0.002)
        self.alpha_spinbox.setDecimals(4)
        alpha_layout.addWidget(self.alpha_spinbox)
        layout.addLayout(alpha_layout)
        
        # N_Sigma (sensitivity)
        nsigma_layout = QHBoxLayout()
        nsigma_layout.addWidget(QLabel("N-Sigma:"))
        self.nsigma_spinbox = QDoubleSpinBox()
        self.nsigma_spinbox.setMinimum(1.0)
        self.nsigma_spinbox.setMaximum(10.0)
        self.nsigma_spinbox.setSingleStep(0.5)
        self.nsigma_spinbox.setValue(3.0)
        nsigma_layout.addWidget(self.nsigma_spinbox)
        layout.addLayout(nsigma_layout)
        
        # Bright Cutoff
        bright_layout = QHBoxLayout()
        bright_layout.addWidget(QLabel("Bright Cutoff:"))
        self.bright_spinbox = QSpinBox()
        self.bright_spinbox.setMinimum(0)
        self.bright_spinbox.setMaximum(255)
        self.bright_spinbox.setValue(200)
        bright_layout.addWidget(self.bright_spinbox)
        layout.addLayout(bright_layout)
        
        # Replacement value
        repl_layout = QHBoxLayout()
        repl_layout.addWidget(QLabel("Replacement:"))
        self.replacement_spinbox = QSpinBox()
        self.replacement_spinbox.setMinimum(0)
        self.replacement_spinbox.setMaximum(255)
        self.replacement_spinbox.setValue(0)
        repl_layout.addWidget(self.replacement_spinbox)
        layout.addLayout(repl_layout)
        
        # Warmup frames
        warmup_layout = QHBoxLayout()
        warmup_layout.addWidget(QLabel("Warmup (s):"))
        self.warmup_spinbox = QDoubleSpinBox()
        self.warmup_spinbox.setMinimum(0.0)
        self.warmup_spinbox.setMaximum(60.0)
        self.warmup_spinbox.setSingleStep(0.1)
        self.warmup_spinbox.setValue(5.0)
        warmup_layout.addWidget(self.warmup_spinbox)
        layout.addLayout(warmup_layout)
        
        # Opening size (morphological kernel)
        opening_layout = QHBoxLayout()
        opening_layout.addWidget(QLabel("Opening Size:"))
        self.opening_spinbox = QSpinBox()
        self.opening_spinbox.setMinimum(0)
        self.opening_spinbox.setMaximum(20)
        self.opening_spinbox.setValue(2)
        opening_layout.addWidget(self.opening_spinbox)
        layout.addLayout(opening_layout)
        
        # Start time with unit selector
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start:"))
        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setMinimum(0.0)
        self.start_spinbox.setMaximum(3600.0)
        self.start_spinbox.setSingleStep(0.1)
        self.start_spinbox.setValue(0.0)
        start_layout.addWidget(self.start_spinbox)
        self.start_unit_combo = QComboBox()
        self.start_unit_combo.addItem("Seconds", "seconds")
        self.start_unit_combo.addItem("Frames", "frames")
        start_layout.addWidget(self.start_unit_combo)
        layout.addLayout(start_layout)
        
        # End time with unit selector
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End:"))
        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setMinimum(0.0)
        self.end_spinbox.setMaximum(3600.0)
        self.end_spinbox.setSingleStep(0.1)
        self.end_spinbox.setValue(30.0)
        end_layout.addWidget(self.end_spinbox)
        self.end_unit_combo = QComboBox()
        self.end_unit_combo.addItem("Seconds", "seconds")
        self.end_unit_combo.addItem("Frames", "frames")
        end_layout.addWidget(self.end_unit_combo)
        layout.addLayout(end_layout)
        
        return layout

    def connect_widgets(self):
        """Connect widget signals to slots"""
        self.recording_tree.itemSelectionChanged.connect(self.on_video_selected)
        self.process_btn.clicked.connect(self.process_selected_video)

    def on_video_selected(self):
        """Handle video selection - load existing processed video if available"""
        self.update_output_title()
        self.load_existing_processed_video()

    def get_selected_video(self):
        """Get the currently selected video file path"""
        selected_items = self.recording_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            video_path = item.data(0, 32)  # Get stored path
            if video_path:
                return video_path
        return None

    def load_existing_processed_video(self):
        """Check if a processed video exists for the selected video and load it"""
        video_path = self.get_selected_video()
        if not video_path:
            # Clear display if no video selected
            self.video_frames = []
            self.frame_slider.setVisible(False)
            self.play_btn.setVisible(False)
            self.frame_info_label.setVisible(False)
            self.video_display_label.setText("Select a video to process")
            return
        
        # Check for processed video in FLY directory
        recording_path = Path(video_path).parent
        video_stem = Path(video_path).stem
        processed_video_path = recording_path / "FLY" / f"{video_stem}_bgs.mp4"
        
        logger.info(f"Checking for processed video: {processed_video_path}")
        
        if processed_video_path.exists():
            logger.info(f"Found existing processed video, loading: {processed_video_path}")
            self.load_video_for_display(str(processed_video_path))
        else:
            logger.info(f"No processed video found. Please process this video first.")
            self.video_frames = []
            self.frame_slider.setVisible(False)
            self.play_btn.setVisible(False)
            self.frame_info_label.setVisible(False)
            self.video_display_label.setText("No processed video found.\nSelect 'Process' to generate BGS output.")

    def find_bounding_box_from_annotations(self, video_path: str, region: str):
        """
        Find bounding box for a region (leaves/fruit) from YOLO annotations.
        Annotations are stored in: {calibration_project}/annotations_dir/port_{port}/labels/train/
        Returns (x1, y1, x2, y2) in pixel coordinates, or None if not found.
        """
        if region == "full":
            return None
        
        region_class = REGION_CLASS_MAP.get(region)
        if region_class is None:
            return None
        
        # Extract port number from video filename (e.g., "port_0.mp4" -> "0")
        video_name = Path(video_path).stem
        match = re.search(r"port_(\d+)", video_name)
        if not match:
            logger.warning(f"Could not extract port number from video name: {video_name}")
            return None
        
        port = match.group(1)
        logger.info(f"Extracted port {port} from video filename")
        
        # Get annotations directory from workspace guide (calibration project)
        try:
            annotations_root = self.controller.workspace_guide.annotations_dir
            logger.info(f"Using annotations directory from controller: {annotations_root}")
        except Exception as e:
            logger.error(f"Could not get annotations directory from controller: {str(e)}")
            return None
        
        if not annotations_root or not annotations_root.exists():
            logger.warning(f"Annotations directory does not exist: {annotations_root}")
            return None
        
        # Build path: {annotations_dir}/port_{port}/labels/train/
        label_path = annotations_root / f"port_{port}" / "labels" / "train"
        logger.info(f"Looking for labels in: {label_path}")
        
        if not label_path.exists():
            logger.warning(f"Label path does not exist: {label_path}")
            return None
        
        # Get any label file from the directory (they should all have the same object)
        label_files = list(label_path.glob("frame_*.txt"))
        if not label_files:
            logger.warning(f"No label files found in {label_path}")
            return None
        
        # Parse the first label file to get bounding box
        try:
            logger.info(f"Found {len(label_files)} label files, checking first one: {label_files[0]}")
            with open(label_files[0], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    class_id = int(parts[0])
                    logger.info(f"Found label class {class_id}, looking for {region_class}")
                    
                    if class_id == region_class:
                        # YOLO format: class_id center_x center_y width height (normalized 0.0-1.0)
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to pixel coordinates
                        cap = cv2.VideoCapture(str(video_path))
                        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        # Convert normalized to pixel coordinates
                        x1 = max(0, int((center_x - width/2) * frame_w))
                        y1 = max(0, int((center_y - height/2) * frame_h))
                        x2 = min(frame_w, int((center_x + width/2) * frame_w))
                        y2 = min(frame_h, int((center_y + height/2) * frame_h))
                        
                        logger.info(f"Found {region.upper()} bounding box: ({x1}, {y1}, {x2}, {y2})")
                        return (x1, y1, x2, y2)
        except Exception as e:
            logger.error(f"Error parsing label file: {str(e)}")
            return None
        
        logger.warning(f"No {region.upper()} annotations (class {region_class}) found for port {port}")
        return None

    def update_output_title(self):
        """Update the title based on selected video"""
        video_path = self.get_selected_video()
        if video_path:
            title = f"<div align='center'><b>Processing: {Path(video_path).name}</b></div>"
        else:
            title = "<div align='center'><b>Select a video to process</b></div>"
        self.output_title.setText(title)

    def process_selected_video(self):
        """Process the currently selected video"""
        video_path = self.get_selected_video()
        if not video_path:
            QMessageBox.warning(self, "Warning", "Please select a video to process")
            return
        
        # Get selected region
        region = self.region_combo.currentData()
        
        # Find bounding box if region is not "full"
        bbox = None
        if region != "full":
            bbox = self.find_bounding_box_from_annotations(video_path, region)
            if bbox is None:
                QMessageBox.warning(
                    self,
                    "Bounding Box Not Found",
                    f"No {region.upper()} bounding box annotations found for this video.\n\n"
                    "Please ensure ground truth labels are available in the recording directory."
                )
                return
        
        # Disable button during processing
        self.process_btn.setEnabled(False)
        
        try:
            # Determine output directory (FLY folder in recording)
            recording_path = Path(video_path).parent
            fly_dir = recording_path / "FLY"
            fly_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Beginning BGS processing for {video_path}")
            logger.info(f"Region: {region}, Bounding box: {bbox}")
            logger.info(f"Output directory: {fly_dir}")
            
            # Get parameters from GUI
            alpha = self.alpha_spinbox.value()
            n_sigma = self.nsigma_spinbox.value()
            bright_cutoff = self.bright_spinbox.value()
            replacement = self.replacement_spinbox.value()
            warmup_secs = self.warmup_spinbox.value()
            opening_size = self.opening_spinbox.value()
            
            # Get start and end times, converting from frames if needed
            start_value = self.start_spinbox.value()
            end_value = self.end_spinbox.value()
            start_unit = self.start_unit_combo.currentData()
            end_unit = self.end_unit_combo.currentData()
            
            # Get video FPS from the actual video file
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Ensure fps is valid
            if fps <= 0:
                logger.warning(f"Could not read FPS from video {video_path}, using config value")
                try:
                    fps = self.controller.config.get_fps_sync_stream_processing()
                except:
                    fps = 100  # Final fallback
            else:
                # Save the detected FPS to config for future reference
                try:
                    self.controller.config.dict['fps_recording'] = int(fps)
                    self.controller.config.update_config_toml()
                    logger.info(f"Saved detected video FPS ({fps}) to config")
                except Exception as e:
                    logger.warning(f"Could not save FPS to config: {e}")
            
            # Convert to seconds if unit is frames
            if start_unit == "frames":
                start_sec = start_value / fps
            else:
                start_sec = start_value
            
            if end_unit == "frames":
                end_sec = end_value / fps
            else:
                end_sec = end_value
            
            logger.info(f"BGS Parameters: alpha={alpha}, n_sigma={n_sigma}, bright_cutoff={bright_cutoff}, "
                       f"replacement={replacement}, warmup={warmup_secs}s, opening_size={opening_size}, "
                       f"start={start_sec}s (from {start_value} {start_unit}), end={end_sec}s (from {end_value} {end_unit})")
            
            # Create processor
            self.bgs_processor = BGSProcessor(
                video_path=video_path,
                output_dir=str(fly_dir),
                alpha=alpha,
                n_sigma=n_sigma,
                bright_cutoff=bright_cutoff,
                replacement=replacement,
                start_sec=start_sec,
                end_sec=end_sec,
                opening_size=opening_size,
                warmup_secs=warmup_secs,
                bounding_box=bbox
            )
            
            # Connect signals
            self.bgs_processor.progress_updated.connect(self.on_progress_update)
            self.bgs_processor.processing_complete.connect(self.on_processing_complete)
            self.bgs_processor.processing_error.connect(self.on_processing_error)
            
            # Start processing
            self.bgs_processor.start()
            
        except Exception as e:
            logger.error(f"Error starting BGS processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error starting processing: {str(e)}")
            self.process_btn.setEnabled(True)

    def on_progress_update(self, frame_num: int, timestamp: float):
        """Handle progress updates"""
        pass  # No logging to avoid overhead

    def load_video_for_display(self, video_path: str):
        """Load video frames for display"""
        try:
            logger.info(f"Loading video for display: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {total_frames} frames, {fps} fps")
            
            # Load all frames (or sample for very long videos)
            self.video_frames = []
            frame_count = 0
            sample_rate = max(1, total_frames // 100)  # Load at most 100 frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    self.video_frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Loaded {len(self.video_frames)} frames for display (sampled from {total_frames})")
            
            if self.video_frames:
                # Set up slider
                self.frame_slider.setMaximum(len(self.video_frames) - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setVisible(True)
                self.frame_info_label.setVisible(True)
                self.play_btn.setVisible(True)
                self.play_btn.setText("Play")
                self.is_playing = False
                self.playback_timer.stop()
                
                # Display first frame
                self.current_frame_idx = 0
                self.display_frame(0)
            
        except Exception as e:
            logger.error(f"Error loading video for display: {str(e)}")

    def display_frame(self, frame_idx: int):
        """Display a specific frame from the loaded video"""
        if not self.video_frames or frame_idx >= len(self.video_frames):
            return
        
        frame = self.video_frames[frame_idx]
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Scale to fit display (max 800 width)
        max_width = 800
        if w > max_width:
            scale = max_width / w
            rgb_frame = cv2.resize(rgb_frame, (int(w * scale), int(h * scale)))
        
        # Convert to QPixmap
        bytes_per_line = rgb_frame.shape[1] * 3
        qt_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], 
                         bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Display
        self.video_display_label.setPixmap(pixmap)
        
        # Update slider and info
        self.current_frame_idx = frame_idx
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        self.frame_info_label.setText(f"Frame {frame_idx + 1} / {len(self.video_frames)}")

    def on_frame_slider_moved(self, value: int):
        """Handle frame slider movement"""
        self.display_frame(value)

    def toggle_playback(self):
        """Toggle play/pause"""
        if not self.video_frames:
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("Pause")
            self.playback_timer.start(1000 // self.playback_fps)
        else:
            self.play_btn.setText("Play")
            self.playback_timer.stop()

    def advance_frame(self):
        """Advance to next frame during playback"""
        if not self.video_frames:
            return
        
        next_idx = self.current_frame_idx + 1
        if next_idx >= len(self.video_frames):
            # Loop back to start
            next_idx = 0
            self.is_playing = False
            self.play_btn.setText("Play")
            self.playback_timer.stop()
        
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_idx)
        self.frame_slider.blockSignals(False)
        self.display_frame(next_idx)

    def on_processing_complete(self, output_path: str):
        """Handle processing completion"""
        logger.info(f"BGS processing complete! Output saved to: {output_path}")
        QMessageBox.information(self, "Success", f"Processing complete!\n\nOutput saved to:\n{output_path}")
        self.process_btn.setEnabled(True)
        
        # Load and display the output video
        self.load_video_for_display(output_path)

    def on_processing_error(self, error_msg: str):
        """Handle processing errors"""
        logger.error(f"BGS processing error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{error_msg}")
        self.process_btn.setEnabled(True)

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
