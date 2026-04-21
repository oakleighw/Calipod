"""File utility functions for annotation management.

Provides shared functions for extracting metadata from filenames used in annotation workflows.
"""

import re
from pathlib import Path
from typing import Optional


def extract_port_from_filename(filename: str) -> Optional[int]:
    """
    Extract port number from a video filename like 'port_0.mp4'.

    Args:
        filename: Filename or path string

    Returns:
        Port number (int), or None if not found
    """
    match = re.search(r"port_(\d+)", filename)
    return int(match.group(1)) if match else None


def extract_frame_index_from_filename(filename: str) -> Optional[int]:
    """
    Extract frame index from various naming conventions.

    Supports:
    - frame_000123.txt (convention 1)
    - something_123.txt (last underscore before extension)
    - 123.txt (simple numeric)

    Args:
        filename: Filename without path (e.g., 'frame_000123.txt')

    Returns:
        Frame index (int), or None if not parsed
    """
    stem = Path(filename).stem

    # Convention 1: frame_000123
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_")[1])
        except (ValueError, IndexError):
            pass

    # Convention 2: ..._{number} (last underscore before extension)
    if "_" in stem:
        try:
            last_part = stem.split("_")[-1]
            return int(last_part)
        except ValueError:
            pass

    # Convention 3: simple numeric like 123
    if stem.isdigit():
        return int(stem)

    return None
