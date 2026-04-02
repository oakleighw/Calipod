"""Video export orchestration for 3D visualization rendering to MP4."""

from pathlib import Path
from typing import Optional
import subprocess

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, QObject

import calipod.logger

logger = calipod.logger.get(__name__)


class VideoExportWorker(QObject):
    """Worker to run video export in a background thread."""
    finished = Signal()
    error = Signal(str)
    result = Signal(Path)
    progress = Signal(str)
    
    def __init__(self, exporter, visualizer, slider, motion_trial, 
                 export_start_frame, export_end_frame, output_path):
        super().__init__()
        self.exporter = exporter
        self.visualizer = visualizer
        self.slider = slider
        self.motion_trial = motion_trial
        self.export_start_frame = export_start_frame
        self.export_end_frame = export_end_frame
        self.output_path = output_path
    
    def run(self):
        """Execute the export in this worker thread."""
        try:
            self.progress.emit("Starting video export...")
            exported_path = self.exporter.export_motion_video(
                self.visualizer,
                self.slider,
                self.motion_trial,
                self.export_start_frame,
                self.export_end_frame,
                self.output_path,
            )
            if exported_path:
                self.progress.emit(f"Export complete: {exported_path}")
                self.result.emit(exported_path)
            else:
                self.error.emit("Export failed - check logs for details")
        except Exception as e:
            logger.error(f"Export worker error: {e}", exc_info=True)
            self.error.emit(f"Export failed: {e}")
        finally:
            self.finished.emit()


class CompareVideoExportWorker(QObject):
    """Worker to run quad-split compare video export in a background thread."""
    finished = Signal()
    error = Signal(str)
    result = Signal(str)  # Returns message, not path
    progress = Signal(str)
    
    def __init__(self, widget_method, port_video_paths, start_frame, end_frame, output_path):
        """
        Args:
            widget_method: The widget's method to call (create_quad_split_video_streaming)
            port_video_paths: List of port video paths
            start_frame: First frame to export
            end_frame: Last frame to export
            output_path: Output video path
        """
        super().__init__()
        self.widget_method = widget_method
        self.port_video_paths = port_video_paths
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output_path = output_path
    
    def run(self):
        """Execute the compare video export in this worker thread."""
        try:
            self.progress.emit("Starting compare video export (this may take a minute)...")
            self.widget_method(self.port_video_paths, self.start_frame, self.end_frame, self.output_path)
            self.progress.emit("Compare video export complete!")
            self.result.emit("Compare video export completed successfully")
        except Exception as e:
            logger.error(f"Compare export worker error: {e}", exc_info=True)
            self.error.emit(f"Compare export failed: {e}")
        finally:
            self.finished.emit()


class GenericWorker(QObject):
    """Generic worker for running any callable in a background thread."""
    finished = Signal()
    error = Signal(str)
    result = Signal(object)  # Can emit any object
    progress = Signal(str)
    
    def __init__(self, callable_func, *args, **kwargs):
        super().__init__()
        self.callable_func = callable_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """Execute the callable in this worker thread."""
        try:
            result = self.callable_func(*self.args, **self.kwargs)
            self.result.emit(result)
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class VideoExporter:
    """Orchestrates rendering visualization frames to video file via FFmpeg."""
    
    def __init__(self, video_framerate: int = 100):
        """
        Initialize the video exporter.
        
        Args:
            video_framerate: Frame rate for output video (fps)
        """
        self.video_framerate = video_framerate
        self.last_exported_path: Optional[Path] = None
    
    def derive_export_paths(self, xyz_history_path: Optional[Path] = None):
        """
        Derive export paths based on motion trial path.
        
        Args:
            xyz_history_path: Path to xyz_*.csv file
            
        Returns:
            Tuple of (export_video_path, compare_video_path, recording_dir)
        """
        if xyz_history_path:
            tracker_suffix = xyz_history_path.stem.replace("xyz_", "")
            export_video_path = xyz_history_path.parent / f"exported_{tracker_suffix}.mp4"
            compare_video_path = xyz_history_path.parent / f"exported_compare_{tracker_suffix}.mp4"
            recording_dir = xyz_history_path.parent.parent
        else:
            export_video_path = Path.cwd() / "exported_motion_video.mp4"
            compare_video_path = Path.cwd() / "exported_real_compare.mp4"
            recording_dir = None

        return export_video_path, compare_video_path, recording_dir
    
    def export_motion_video(
        self, 
        visualizer,
        slider,
        motion_trial,
        export_start_frame: int,
        export_end_frame: int,
        output_path: Path,
    ) -> Optional[Path]:
        """
        Render visualization frames to MP4 video using FFmpeg.
        
        Connects to Qt visualizer to collect frames at each frame index,
        then pipes them to FFmpeg for compression.
        
        Args:
            visualizer: TriangulationVisualizer with display_points() and update_segment_lines()
            slider: QSlider for frame control
            motion_trial: MotionTrial data object
            export_start_frame: First frame index to export
            export_end_frame: Last frame index to export
            output_path: Path to save MP4 output
            
        Returns:
            Path to exported video if successful, None otherwise
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean up old/corrupt video file if it exists (especially 0-byte files from failed exports)
        if output_path.exists():
            try:
                old_size = output_path.stat().st_size
                if old_size == 0:
                    logger.warning(f"Removing corrupt 0-byte file: {output_path}")
                    output_path.unlink()
                else:
                    logger.info(f"Old video file exists ({old_size} bytes); FFmpeg will overwrite with -y flag")
            except Exception as e:
                logger.warning(f"Could not remove old video file: {e}")
        
        # Set up visualizer for export
        visualizer.set_export_mode(True)
        visualizer.clear_collected_frames()
        
        logger.info("Starting video export process (collecting frames in memory)...")
        
        # Temporarily disconnect slider signals to avoid UI updates
        reconnect_needed = False
        try:
            slider.valueChanged.disconnect(visualizer.display_points)
            slider.valueChanged.disconnect(visualizer.update_segment_lines)
            reconnect_needed = True
        except TypeError:
            pass
        
        exported_path: Optional[Path] = None
        
        try:
            # Collect frames by stepping through slider
            frame_count = 0
            for i in range(export_start_frame, export_end_frame + 1):
                slider.blockSignals(True)
                slider.setValue(i)
                slider.blockSignals(False)
                
                visualizer.display_points(i)
                visualizer.update_segment_lines(i)
                
                QApplication.processEvents()
                frame_count += 1
            
            logger.info(f"Stepped through {frame_count} frame indices")
            
            collected_frames = visualizer.get_collected_frames()
            logger.info(f"Collected {len(collected_frames)} frames from visualizer")
            
            if not collected_frames:
                logger.error(f"No frames were collected to combine into video. Export mode: {visualizer.export_video_mode}, motion_trial empty: {motion_trial.is_empty if motion_trial else 'N/A'}")
                return None
            
            # Validate frame consistency before writing to FFmpeg
            logger.info("Validating collected frames...")
            first_valid_frame = None
            valid_frame_count = 0
            invalid_frame_indices = []
            
            for idx, frame in enumerate(collected_frames):
                if frame is None or frame.size == 0:
                    invalid_frame_indices.append(idx)
                elif frame.dtype != np.uint8:
                    logger.warning(f"Frame {idx} has unexpected dtype {frame.dtype}, converting to uint8")
                    collected_frames[idx] = frame.astype(np.uint8)
                    valid_frame_count += 1
                    if first_valid_frame is None:
                        first_valid_frame = collected_frames[idx]
                else:
                    valid_frame_count += 1
                    if first_valid_frame is None:
                        first_valid_frame = frame
            
            if invalid_frame_indices:
                logger.warning(f"Found {len(invalid_frame_indices)} invalid/empty frames at indices: {invalid_frame_indices[:20]}{'...' if len(invalid_frame_indices) > 20 else ''}")
                # Filter out invalid frames
                collected_frames = [f for f in collected_frames if f is not None and f.size > 0]
                logger.info(f"Filtered to {len(collected_frames)} valid frames")
            
            if not collected_frames or first_valid_frame is None:
                logger.error("No valid frames collected - cannot encode video")
                return None
            
            logger.info(f"Frame validation complete: {len(collected_frames)} valid frames, first frame shape: {first_valid_frame.shape}")
            
            # Render collected frames to video via FFmpeg
            exported_path = self._render_frames_to_video(
                collected_frames, 
                output_path,
                self.video_framerate
            )
            self.last_exported_path = exported_path
            
            return exported_path
            
        finally:
            # Restore slider connections
            if reconnect_needed:
                slider.valueChanged.connect(visualizer.display_points)
                slider.valueChanged.connect(visualizer.update_segment_lines)
            
            visualizer.set_export_mode(False)
    
    def _render_frames_to_video(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        framerate: int,
    ) -> Optional[Path]:
        """
        Pipe frames to FFmpeg for encoding to H.264 MP4.
        
        Args:
            frames: List of BGR numpy arrays
            output_path: Path to save output MP4
            framerate: Video framerate in fps
            
        Returns:
            output_path if successful, None otherwise
        """
        if not frames or len(frames) == 0:
            logger.warning("No frames to render")
            return None
        
        # Validate first frame exists and has valid shape
        if frames[0].size == 0 or len(frames[0].shape) != 3:
            logger.error(f"Invalid first frame: size={frames[0].size}, shape={frames[0].shape if hasattr(frames[0], 'shape') else 'N/A'}")
            return None
        
        try:
            first_frame = frames[0]
            
            # Check for 3-channel BGR data
            if len(first_frame.shape) != 3 or first_frame.shape[2] != 3:
                logger.error(f"Frame has wrong shape: {first_frame.shape}. Expected (height, width, 3) for BGR24 data")
                return None
            
            height, width, channels = first_frame.shape
            
            # Validate ALL frames have consistent dimensions - this is critical!
            inconsistent_indices = []
            for idx, frame in enumerate(frames):
                if frame.shape != (height, width, channels):
                    inconsistent_indices.append((idx, frame.shape))
            
            if inconsistent_indices:
                logger.error(f"Found {len(inconsistent_indices)}/{len(frames)} frames with inconsistent dimensions:")
                for idx, shape in inconsistent_indices[:5]:
                    logger.error(f"  Frame {idx}: {shape} (expected ({height}, {width}, {channels}))")
                
                # Try to recover by resizing all inconsistent frames to match first frame
                logger.info(f"Attempting recovery: resizing {len(inconsistent_indices)} frames to match first frame dimensions ({height}x{width})")
                import cv2
                frames_fixed = []
                for idx, frame in enumerate(frames):
                    if frame.shape != (height, width, channels):
                        try:
                            resized = cv2.resize(frame, (width, height))
                            frames_fixed.append(resized)
                            logger.debug(f"  Resized frame {idx} from {frame.shape} to ({height}, {width}, {channels})")
                        except Exception as e:
                            logger.warning(f"  Failed to resize frame {idx}: {e}, skipping")
                    else:
                        frames_fixed.append(frame)
                
                frames = frames_fixed
                logger.info(f"After recovery: {len(frames)} frames")
                
                if len(frames) == 0:
                    logger.error("No valid frames after attempting to fix mismatches")
                    return None
            
            # H.264 requires even dimensions; pad if necessary
            if width % 2 != 0 or height % 2 != 0:
                logger.warning(f"Frame dimensions {width}x{height} are odd; h264 encoding requires even dimensions. Padding to even size.")
                # Round each dimension up to nearest even number
                padded_width = width if width % 2 == 0 else width + 1
                padded_height = height if height % 2 == 0 else height + 1
                
                # Pad all frames
                padded_frames = []
                for frame in frames:
                    padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
                    padded[:height, :width, :] = frame
                    padded_frames.append(padded)
                frames = padded_frames
                width = padded_width
                height = padded_height
                logger.info(f"Padded frames to {width}x{height}")
            
            logger.info(f"First frame shape: {frames[0].shape} after validation (height={height}, width={width})")
            
            # FFmpeg command for H.264 compression via pipe
            # Using "faster" preset for stability, "superfast" if even that fails
            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(framerate),
                "-i", "-",
                "-c:v", "libx264",
                "-b:v", "5000k",  # Reduced from 8000k for stability
                "-preset", "faster",  # Changed from medium for more stable real-time encoding
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-threads", "4",  # Limit threads to reduce memory usage
                "-y",
                str(output_path)
            ]
            
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            try:
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except FileNotFoundError:
                logger.error("FFmpeg not found. Install FFmpeg or add it to PATH.")
                raise IOError("FFmpeg is required for video encoding but was not found.")
            
            logger.info(f"Writing {len(frames)} frames to video: {output_path}")
            
            # Write frames with error handling for pipe issues
            frames_written = 0
            try:
                for frame_idx, frame in enumerate(frames):
                    try:
                        frame_bytes = frame.tobytes()
                        proc.stdin.write(frame_bytes)
                        frames_written += 1
                        if frame_idx % 300 == 0 and frame_idx > 0:
                            logger.debug(f"  Progress: {frame_idx}/{len(frames)} frames written ({100*frame_idx/len(frames):.1f}%)")
                    except BrokenPipeError as e:
                        logger.error(f"FFmpeg pipe closed after writing {frame_idx}/{len(frames)} frames: {e}")
                        # Try to get error info from FFmpeg
                        try:
                            _, stderr = proc.communicate(timeout=5)
                            if stderr:
                                logger.error(f"FFmpeg stderr at failure: {stderr.decode()[:500]}")
                        except:
                            pass
                        break
            except Exception as e:
                logger.error(f"Error while writing frames: {e}")
            
            if frames_written < len(frames):
                logger.warning(f"Only wrote {frames_written}/{len(frames)} frames before pipe closure")
            
            logger.info(f"Closing FFmpeg stdin pipe (wrote {frames_written} frames)...")
            try:
                proc.stdin.close()
            except Exception as e:
                logger.warning(f"Error closing stdin: {e}")
            
            QApplication.processEvents()
            
            logger.info("Waiting for FFmpeg to finish encoding...")
            try:
                stdout, stderr = proc.communicate(timeout=600)  # Increased timeout for large videos
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg encoding timed out after 600 seconds")
                proc.kill()
                stdout, stderr = proc.communicate()
            
            logger.info(f"FFmpeg process completed with return code: {proc.returncode}")
            
            if stderr:
                stderr_text = stderr.decode() if isinstance(stderr, bytes) else stderr
                if proc.returncode == 0:
                    logger.debug(f"FFmpeg stderr (informational): {stderr_text[:300]}")
                else:
                    logger.error(f"FFmpeg stderr (first 1000 chars): {stderr_text[:1000]}")
            
            if proc.returncode == 0:
                # Verify file was actually created and has content
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    logger.info(f"Video saved successfully to: {output_path} ({file_size} bytes)")
                    if file_size == 0:
                        logger.error("Video file is 0 bytes - FFmpeg may have failed silently")
                        return None
                    return output_path
                else:
                    logger.error(f"FFmpeg reported success but file not found: {output_path}")
                    return None
            else:
                logger.error(f"FFmpeg encoding failed with return code {proc.returncode}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create video from collected frames: {e}")
            try:
                proc.stdin.close()
                proc.terminate()
            except:
                pass
            return None
