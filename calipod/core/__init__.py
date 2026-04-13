"""
Core modules for Calipod.

Contains utility modules, configuration, logging, and data structures
used across the application.

Note: Heavy modules like Configurator and Controller are NOT eagerly imported
to avoid circular imports. Import them explicitly when needed:
    from calipod.core.configurator import Configurator
    from calipod.core.controller import Controller
"""

from calipod.core.logger import get as get_logger
from calipod.core.packets import (
    FramePacket,
    PointPacket,
    SyncPacket,
    XYZPacket,
)

__all__ = [
    "get_logger",
    "PointPacket",
    "FramePacket",
    "SyncPacket",
    "XYZPacket",
]
