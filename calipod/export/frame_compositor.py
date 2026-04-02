"""Frame composition and resizing utilities for multi-video editing."""

from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication

import calipod.logger

logger = calipod.logger.get(__name__)


class FrameCompositor:
    """Composes and resizes video frames for split-screen and comparison videos."""
    
    @staticmethod
    def resize_and_center_crop(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Resize with aspect ratio preservation and center-crop/pad to target size.
        
        Args:
            frame: Input BGR image
            target_w: Target width in pixels
            target_h: Target height in pixels
            
        Returns:
            Resized and cropped frame
        """
        h, w = frame.shape[:2]
        if w == 0 or h == 0 or target_w == 0 or target_h == 0:
            return frame

        target_aspect = target_w / target_h
        src_aspect = w / h

        if src_aspect > target_aspect:
            # Crop width
            new_w = int(h * target_aspect)
            x0 = max((w - new_w) // 2, 0)
            frame = frame[:, x0:x0 + new_w]
        elif src_aspect < target_aspect:
            # Crop height
            new_h = int(w / target_aspect)
            y0 = max((h - new_h) // 2, 0)
            frame = frame[y0:y0 + new_h, :]

        return cv2.resize(frame, (target_w, target_h))
    
    @staticmethod
    def collect_port_videos(recording_dir: Path, expected_count: int = 3) -> list[Path]:
        """
        Discover port_*.mp4 video files in directory.
        
        Args:
            recording_dir: Path to directory containing port_N.mp4 files
            expected_count: Expected number of video files
            
        Returns:
            Sorted list of video paths
        """
        port_videos = []

        for video_file in recording_dir.glob("port_*.mp4"):
            try:
                port_number = int(video_file.stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning(f"Skipping unexpected video file name: {video_file}")
                continue

            port_videos.append((port_number, video_file))

        port_videos = [path for _, path in sorted(port_videos, key=lambda x: x[0])]

        if len(port_videos) < expected_count:
            logger.error(
                f"Found {len(port_videos)} camera videos at {recording_dir}, but {expected_count} are required."
            )
            return []

        return port_videos[:expected_count]
    
    @staticmethod
    def create_quad_split_video(video_paths: list[Path], output_path: Path, framerate: Optional[int] = None):
        """
        Compose 4 videos into 2x2 grid split-screen video.
        
        Args:
            video_paths: List of exactly 4 video paths (real 3, sim 1)
            output_path: Path to save combined video
            framerate: Output framerate (auto-detected if None)
            
        Raises:
            ValueError: If not exactly 4 video paths provided
            IOError: If video opening/writing fails
        """
        if len(video_paths) != 4:
            raise ValueError("Please provide exactly 4 video paths.")

        caps = []
        opened_meta = []
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                for c in caps:
                    c.release()
                raise IOError(f"Failed to open video: {video_path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            opened_meta.append((video_path, width, height, fps))
            caps.append(cap)

        # Choose base size from videos (smallest to avoid upscaling)
        valid_sizes = [(w, h) for (_, w, h, _) in opened_meta if w > 0 and h > 0]
        if valid_sizes:
            quad_width, quad_height = min(valid_sizes, key=lambda wh: wh[0] * wh[1])
        else:
            quad_width, quad_height = 640, 480

        output_resolution = (quad_width * 2, quad_height * 2)
        if output_resolution[0] <= 0 or output_resolution[1] <= 0:
            for cap in caps:
                cap.release()
            raise IOError(f"Invalid output resolution: {output_resolution}")

        target_fps = int(opened_meta[0][3]) if opened_meta and opened_meta[0][3] > 0 else (framerate or 30)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, output_resolution)
        if not out.isOpened():
            for cap in caps:
                cap.release()
            raise IOError(f"Failed to open video writer for {output_path}")

        frame_count = 0
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    frames = []
                    break
                frames.append(frame)

            if len(frames) < 4:
                break

            # Resize first 3 (real) with simple resize, crop only the 4th (sim)
            resized_frames = [
                cv2.resize(frames[i], (quad_width, quad_height)) if i < 3 
                else FrameCompositor.resize_and_center_crop(frames[i], quad_width, quad_height) 
                for i in range(4)
            ]
            top_row = np.hstack((resized_frames[0], resized_frames[1]))
            bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
            combined_frame = np.vstack((top_row, bottom_row))

            out.write(combined_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                QApplication.processEvents()

        logger.info(f"Finished processing {frame_count} frames, finalizing video file...")
        QApplication.processEvents()
        
        for cap in caps:
            cap.release()
        out.release()
        
        QApplication.processEvents()

        if frame_count == 0:
            logger.error(f"No frames written to combined video: {output_path}")
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            return

        logger.info(
            f"Combined video saved to: {output_path} ({frame_count} frames at {target_fps} fps)"
        )
    
    @staticmethod
    def create_quad_split_video_streaming(
        port_video_paths: list[Path], 
        render_callback, 
        start_frame: int, 
        end_frame: int, 
        output_path: Path,
        framerate: int = 100,
    ):
        """
        Compose port videos with on-demand sim frame rendering (streaming/low-memory).
        
        Args:
            port_video_paths: List of exactly 3 port video paths
            render_callback: Callable(frame_index) -> np.ndarray for sim frame rendering
            start_frame: Start frame index
            end_frame: End frame index
            output_path: Path to save combined video
            framerate: Output framerate
            
        Raises:
            ValueError: If not exactly 3 video paths provided
            IOError: If video opening fails
        """
        if len(port_video_paths) != 3:
            raise ValueError("Please provide exactly 3 port video paths.")

        caps = []
        opened_meta = []
        for video_path in port_video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                for c in caps:
                    c.release()
                raise IOError(f"Failed to open video: {video_path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            opened_meta.append((video_path, width, height, fps))
            caps.append(cap)

        valid_sizes = [(w, h) for (_, w, h, _) in opened_meta if w > 0 and h > 0]
        if valid_sizes:
            quad_width, quad_height = min(valid_sizes, key=lambda wh: wh[0] * wh[1])
        else:
            quad_width, quad_height = 640, 480

        output_resolution = (quad_width * 2, quad_height * 2)
        if output_resolution[0] <= 0 or output_resolution[1] <= 0:
            for cap in caps:
                cap.release()
            raise IOError(f"Invalid output resolution: {output_resolution}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use FFmpeg for compressed output via pipe
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{output_resolution[0]}x{output_resolution[1]}",
            "-r", str(framerate),
            "-i", "-",
            "-c:v", "libx264",
            "-b:v", "8000k",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y",
            str(output_path)
        ]

        try:
            proc = FrameCompositor._start_ffmpeg_encoder(ffmpeg_cmd)
        except Exception as e:
            for cap in caps:
                cap.release()
            raise

        try:
            frame_count = 0
            for frame_idx in range(start_frame, end_frame + 1):
                # Get port video frames
                port_frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        port_frames = []
                        break
                    port_frames.append(frame)
                
                if len(port_frames) < 3:
                    break
                
                # Get sim frame via callback
                try:
                    sim_frame = render_callback(frame_idx)
                except Exception as e:
                    logger.error(f"Failed to render frame {frame_idx}: {e}")
                    break
                
                # Compose 2x2 grid
                resized_port = [
                    cv2.resize(port_frames[i], (quad_width, quad_height)) 
                    for i in range(3)
                ]
                resized_sim = FrameCompositor.resize_and_center_crop(sim_frame, quad_width, quad_height)
                
                # Create 2x2 grid (3 real + 1 sim, sim in bottom-right)
                top_row = np.hstack((resized_port[0], resized_port[1]))
                bottom_row = np.hstack((resized_port[2], resized_sim))
                combined = np.vstack((top_row, bottom_row))
                
                proc.stdin.write(combined.tobytes())
                frame_count += 1
                
                if frame_count % 30 == 0:
                    QApplication.processEvents()
            
            proc.stdin.close()
            proc.wait(timeout=300)
            logger.info(f"Streaming video composition complete: {frame_count} frames written")
            
        finally:
            for cap in caps:
                cap.release()
            try:
                proc.terminate()
            except:
                pass
    
    @staticmethod
    def _start_ffmpeg_encoder(ffmpeg_cmd: list[str]):
        """Start FFmpeg encoding subprocess."""
        import subprocess
        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return proc
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install FFmpeg or add it to PATH.")
            raise IOError("FFmpeg is required for video encoding but was not found.")
