from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from calipod.annotation_management import AnnotationsConfigManager
from calipod.annotation_management.annotations_config_manager import (
    STRUCTURE_GEOMETRY_FLAT,
)


def load_environment_structure_settings(workspace_dir: Path | str) -> Dict:
    """Load frame-ROI and arena-vertex structure settings from annotations config."""
    manager = AnnotationsConfigManager(workspace_dir)

    frame_roi = manager.get_frame_roi_structure_labels(is_ground_truth=True)
    arena_vertices = manager.get_arena_vertex_structure_labels(is_ground_truth=True)

    structures: List[Dict] = []
    for label_id, label_data in sorted(frame_roi.items()):
        structure_data = label_data.get("structure", {})
        geometry = structure_data.get("geometry", STRUCTURE_GEOMETRY_FLAT)
        structures.append(
            {
                "id": int(label_id),
                "name": label_data.get("name", "") or f"Class {label_id}",
                "enabled": bool(structure_data.get("enabled", True)),
                "geometry": geometry,
                "role": structure_data.get("role"),
            }
        )

    arena_vertex_structures: List[Dict] = []
    for label_id, label_data in sorted(arena_vertices.items()):
        structure_data = label_data.get("structure", {})
        arena_vertex_structures.append(
            {
                "id": int(label_id),
                "name": label_data.get("name", "") or f"Class {label_id}",
                "enabled": bool(structure_data.get("enabled", True)),
                "add_to_floor": bool(structure_data.get("add_to_floor", False)),
            }
        )

    return {
        "frame_roi_structures": structures,
        "arena_vertices": arena_vertex_structures,
    }
