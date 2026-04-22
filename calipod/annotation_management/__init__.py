"""Annotation management utilities for handling annotation configurations and label management."""

from calipod.annotation_management.annotation_checker import AnnotationChecker
from calipod.annotation_management.annotation_index import AnnotationIndex
from calipod.annotation_management.annotations_config_manager import (
    ANNOTATIONS_CONFIG_FILENAME,
    AnnotationsConfigManager,
)
from calipod.annotation_management.file_utils import (
    extract_frame_index_from_filename,
    extract_port_from_filename,
)
from calipod.annotation_management.label_editor import LabelEditorWidget
from calipod.annotation_management.yolo_utils import (
    denormalize_bbox,
    load_yolo_file,
    parse_yolo_line,
)

__all__ = [
    "AnnotationChecker",
    "AnnotationIndex",
    "AnnotationsConfigManager",
    "ANNOTATIONS_CONFIG_FILENAME",
    "LabelEditorWidget",
    "parse_yolo_line",
    "denormalize_bbox",
    "load_yolo_file",
    "extract_port_from_filename",
    "extract_frame_index_from_filename",
]
