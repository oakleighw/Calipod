"""Annotation management utilities for handling annotation configurations and label management."""

from calipod.annotation_management.annotations_config_manager import (
    ANNOTATIONS_CONFIG_FILENAME,
    AnnotationsConfigManager,
)
from calipod.annotation_management.label_editor import LabelEditorWidget

__all__ = [
    "AnnotationsConfigManager",
    "ANNOTATIONS_CONFIG_FILENAME",
    "LabelEditorWidget",
]
