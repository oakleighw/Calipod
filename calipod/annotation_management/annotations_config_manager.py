"""
Manager for handling annotation configuration and label management.

Saves ground truth and prediction labels to a config JSON file in the project's
annotations folder. Labels are indexed by ID with space for string names.
"""

import json
from pathlib import Path
from typing import Dict, Set

from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)

# Default config filename
ANNOTATIONS_CONFIG_FILENAME = "annotations_config.json"


class AnnotationsConfigManager:
    """Manages saving and loading annotation configuration with label metadata."""

    def __init__(self, workspace_dir: Path | str):
        """
        Initialize the config manager for a workspace.

        Args:
            workspace_dir: Path to the workspace root directory
        """
        self.workspace_dir = Path(workspace_dir)
        self.annotations_dir = self.workspace_dir / "annotations"
        self.config_path = self.annotations_dir / ANNOTATIONS_CONFIG_FILENAME

    def save_annotations_config(
        self,
        ground_truth_labels: Dict[int, str] | Set[int] = None,
        ground_truth_format: str = None,
        predictions_labels: Dict[int, str] | Set[int] = None,
        predictions_format: str = None,
    ) -> Path:
        """
        Save annotation configuration with ground truth and prediction labels.

        Converts label sets or dicts to a structured config with ID->name mappings.
        String names are initialized as empty strings for later population.

        Args:
            ground_truth_labels: Dict of {id: name} or Set of label IDs for ground truth
            ground_truth_format: Annotation format for ground truth (e.g., "YOLO", "COCO")
            predictions_labels: Dict of {id: name} or Set of label IDs for predictions
            predictions_format: Annotation format for predictions

        Returns:
            Path to the saved config file
        """
        # Ensure annotations directory exists
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        # Build config structure
        config = {
            "ground_truth": self._build_label_section(
                ground_truth_labels, ground_truth_format
            ),
            "predictions": self._build_label_section(predictions_labels, predictions_format),
        }

        # Save to JSON
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved annotations config to {self.config_path}")
        return self.config_path

    def _build_label_section(
        self, labels: Dict[int, str] | Set[int] | None, format_type: str | None
    ) -> Dict:
        """
        Build a label section for the config.

        Args:
            labels: Dict of {id: name} or Set of label IDs, or None
            format_type: Annotation format name

        Returns:
            Dict with format and labels structure
        """
        section = {"format": format_type, "labels": {}}

        if labels is None or len(labels) == 0:
            return section

        # Convert set to dict with empty names
        if isinstance(labels, set):
            labels = {label_id: "" for label_id in sorted(labels)}
        elif isinstance(labels, dict):
            # Ensure all values are strings and empty if not already set
            labels = {
                label_id: (name if isinstance(name, str) else "")
                for label_id, name in labels.items()
            }

        # Build labels dictionary with ID and name
        for label_id, name in sorted(labels.items()):
            section["labels"][str(label_id)] = {"id": label_id, "name": name}

        return section

    def load_annotations_config(self) -> Dict | None:
        """
        Load the annotations config from JSON file.

        Returns:
            Dict with config structure, or None if file doesn't exist
        """
        if not self.config_path.exists():
            logger.debug(f"Annotations config not found at {self.config_path}")
            return None

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded annotations config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading annotations config: {e}")
            return None

    def get_ground_truth_labels(self) -> Dict[int, str] | None:
        """
        Get ground truth labels from the config.

        Returns:
            Dict mapping label ID to name, or None if not available
        """
        config = self.load_annotations_config()
        if config is None or "ground_truth" not in config:
            return None

        gt_labels = config["ground_truth"].get("labels", {})
        return {int(label_id): data["name"] for label_id, data in gt_labels.items()}

    def get_predictions_labels(self) -> Dict[int, str] | None:
        """
        Get prediction labels from the config.

        Returns:
            Dict mapping label ID to name, or None if not available
        """
        config = self.load_annotations_config()
        if config is None or "predictions" not in config:
            return None

        pred_labels = config["predictions"].get("labels", {})
        return {int(label_id): data["name"] for label_id, data in pred_labels.items()}

    def update_label_name(self, label_id: int, name: str, is_ground_truth: bool = True):
        """
        Update the name for a specific label ID.

        Args:
            label_id: The label ID to update
            name: The new name for the label
            is_ground_truth: If True, update ground truth labels; else prediction labels
        """
        config = self.load_annotations_config()
        if config is None:
            logger.warning("No annotations config found to update")
            return

        section_key = "ground_truth" if is_ground_truth else "predictions"
        if section_key not in config:
            logger.warning(f"Section '{section_key}' not found in config")
            return

        label_key = str(label_id)
        if label_key not in config[section_key]["labels"]:
            logger.warning(f"Label ID {label_id} not found in {section_key}")
            return

        config[section_key]["labels"][label_key]["name"] = name

        # Save updated config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Updated label {label_id} in {section_key} to '{name}'")
