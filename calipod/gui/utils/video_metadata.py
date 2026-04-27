import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import QObject, Signal


class MetadataWorker(QObject):
    finished = Signal(dict)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        meta = get_video_metadata_ffprobe(self.path)
        self.finished.emit(meta)

def get_video_metadata_ffprobe(video_path: str) -> Dict[str, Any]:
    """
    Extracts video metadata using ffprobe (from ffmpeg).
    Returns a dictionary with codec, fps, keyframe count, image format, etc.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,pix_fmt",
        "-show_entries",
        "frame=pict_type",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        codec = stream.get("codec_name", "?")
        pix_fmt = stream.get("pix_fmt", "?")
        # FPS calculation
        avg_frame_rate = stream.get("avg_frame_rate", "0/0")
        try:
            num, denom = map(int, avg_frame_rate.split("/"))
            fps = num / denom if denom else 0
        except Exception:
            fps = 0
        # Keyframe count
        keyframes = [f for f in info.get("frames", []) if f.get("pict_type") == "I"]
        keyframe_count = len(keyframes)
        return {
            "codec": codec,
            "fps": fps,
            "keyframe_count": keyframe_count,
            "pix_fmt": pix_fmt,
        }
    except Exception as e:
        return {"error": str(e)}
