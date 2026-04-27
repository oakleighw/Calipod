import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import QObject, Signal


class MetadataWorker(QObject):
    finished = Signal(dict)
    def __init__(self, path, count_keyframes=False):
        super().__init__()
        self.path = path
        self.count_keyframes = count_keyframes
        self._proc = None
        self._cancelled = False

    def run(self):
        try:
            meta = get_video_metadata_ffprobe(self.path, self.count_keyframes, self)
        except Exception as e:
            meta = {"error": str(e)}
        self.finished.emit(meta)

    def cancel(self):
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


def get_video_metadata_ffprobe(video_path: str, count_keyframes: bool = False,
                                worker: MetadataWorker = None) -> Dict[str, Any]:
    """
    Extracts video metadata using ffprobe (from ffmpeg).
    Returns a dictionary with codec, fps, keyframe count (optional), image format, etc.
    If worker is provided, checks for cancellation.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Always get basic info
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,avg_frame_rate,pix_fmt",
        "-of", "json",
        video_path
    ]
    try:
        # Use Popen for cancellation
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if worker is not None:
            worker._proc = proc
        out, err = proc.communicate()
        info = json.loads(out)
        stream = info.get("streams", [{}])[0]
        codec = stream.get("codec_name", "?")
        pix_fmt = stream.get("pix_fmt", "?")
        avg_frame_rate = stream.get("avg_frame_rate", "0/0")
        try:
            num, denom = map(int, avg_frame_rate.split("/"))
            fps = num / denom if denom else 0
        except Exception:
            fps = 0
        meta = {
            "codec": codec,
            "fps": fps,
            "pix_fmt": pix_fmt,
        }
        if count_keyframes:
            # Keyframe count: separate call for speed
            keyframe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-skip_frame", "nokey",
                "-show_frames",
                "-of", "json",
                video_path
            ]
            proc2 = subprocess.Popen(keyframe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if worker is not None:
                worker._proc = proc2
            out2, err2 = proc2.communicate()
            if worker is not None and worker._cancelled:
                return {"error": "Cancelled"}
            frames_info = json.loads(out2)
            keyframes = frames_info.get("frames", [])
            meta["keyframe_count"] = len(keyframes)
        return meta
    except Exception as e:
        return {"error": str(e)}
