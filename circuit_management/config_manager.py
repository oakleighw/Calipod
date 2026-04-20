"""Manager for circuit management configuration persistence."""

import json
from pathlib import Path
from typing import Any


class CircuitManagementConfigManager:
    """Handles loading and saving circuit management configuration."""

    def __init__(self, workspace_dir: Path):
        self.config_dir = Path(workspace_dir) / "circuit_management"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.config_file = self.config_dir / "circuit_management_config.json"
        self._config: dict[str, Any] = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file or return empty dict if file doesn't exist."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading circuit management config: {e}")
                return {}
        return {}

    def _save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error saving circuit management config: {e}")

    def get_camera_config(self, camera_port: int) -> dict[str, Any]:
        """Get configuration for a specific camera."""
        camera_key = str(camera_port)
        if camera_key not in self._config:
            self._config[camera_key] = {
                "connector_type": None,
                "wire_rows": {},
                "delay_ms": None,
            }
        elif "delay_ms" not in self._config[camera_key]:
            self._config[camera_key]["delay_ms"] = None
        return self._config[camera_key]

    def set_connector_type(self, camera_port: int, connector_type: str) -> None:
        """Set the connector type for a camera."""
        camera_config = self.get_camera_config(camera_port)
        camera_config["connector_type"] = connector_type
        self._save_config()

    def get_connector_type(self, camera_port: int) -> str | None:
        """Get the connector type for a camera. Returns None if not yet set."""
        camera_config = self.get_camera_config(camera_port)
        return camera_config.get("connector_type")

    def get_connector_configuration(self, camera_port: int) -> tuple[str | None, dict[str, dict[str, str]]]:
        """Get connector type and all wire rows for a camera."""
        camera_config = self.get_camera_config(camera_port)
        connector_type = camera_config.get("connector_type")
        wire_rows = camera_config.get("wire_rows", {})
        return connector_type, wire_rows

    def save_connector_configuration(
        self,
        camera_port: int,
        connector_type: str,
        wire_rows: dict[str, dict[str, str]],
    ) -> None:
        """Save connector type and all applicable wire rows in one write."""
        camera_config = self.get_camera_config(camera_port)
        camera_config["connector_type"] = connector_type
        camera_config["wire_rows"] = wire_rows
        self._save_config()

    def set_wire_row(self, camera_port: int, row_index: int, wire_type: str, colour: str) -> None:
        """Set wire configuration for a specific row of a camera."""
        camera_config = self.get_camera_config(camera_port)
        if "wire_rows" not in camera_config:
            camera_config["wire_rows"] = {}

        row_key = str(row_index)
        camera_config["wire_rows"][row_key] = {
            "wire_type": wire_type,
            "colour": colour,
        }
        self._save_config()

    def get_wire_row(self, camera_port: int, row_index: int) -> dict[str, str]:
        """Get wire configuration for a specific row of a camera."""
        camera_config = self.get_camera_config(camera_port)
        row_key = str(row_index)
        return camera_config.get("wire_rows", {}).get(row_key, {"wire_type": "", "colour": ""})

    def get_all_wire_rows(self, camera_port: int) -> dict[str, dict[str, str]]:
        """Get all wire configurations for a camera."""
        camera_config = self.get_camera_config(camera_port)
        return camera_config.get("wire_rows", {})

    def clear_all_wire_rows(self, camera_port: int) -> None:
        """Clear all wire row configurations for a camera."""
        camera_config = self.get_camera_config(camera_port)
        camera_config["wire_rows"] = {}
        self._save_config()

    def set_delay_ms(self, camera_port: int, delay_ms: int | None) -> None:
        """Set optional delay (in milliseconds) for a camera."""
        camera_config = self.get_camera_config(camera_port)
        camera_config["delay_ms"] = delay_ms
        self._save_config()

    def get_delay_ms(self, camera_port: int) -> int | None:
        """Get optional delay (in milliseconds) for a camera."""
        camera_config = self.get_camera_config(camera_port)
        delay_ms = camera_config.get("delay_ms")
        if isinstance(delay_ms, int):
            return delay_ms
        return None

    def clear_camera_config(self, camera_port: int) -> None:
        """Clear all configuration for a camera."""
        camera_key = str(camera_port)
        if camera_key in self._config:
            del self._config[camera_key]
            self._save_config()
