"""Persistence helper for arena simulation widget configuration."""

import json
from pathlib import Path

import calipod.logger

logger = calipod.logger.get(__name__)


class ArenaConfigManager:
    """Load and save arena simulation config JSON in the workspace arena_sim directory."""

    FILE_NAME = "arena_config.json"

    def __init__(self, arena_sim_dir: Path):
        self.arena_sim_dir = Path(arena_sim_dir)

    @property
    def config_path(self) -> Path:
        return self.arena_sim_dir / self.FILE_NAME

    def load(self) -> dict:
        """Load arena config JSON, returning an empty dict when unavailable."""
        if not self.config_path.exists():
            logger.debug(f"No arena config found at {self.config_path}; using defaults")
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded arena config from {self.config_path}")
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to load arena config from {self.config_path}: {e}")
            return {}

    def save(self, config: dict):
        """Save arena config JSON to disk."""
        self.arena_sim_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved arena config to {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to save arena config to {self.config_path}: {e}")
