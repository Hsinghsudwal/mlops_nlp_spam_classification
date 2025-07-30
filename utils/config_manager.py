import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from utils.logger import logger


class ConfigManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info("ConfigManager initialized with configuration.")

    @classmethod
    def load_file(cls, path: str) -> 'ConfigManager':
        """Load configuration from a YAML or JSON file."""
        if not os.path.isfile(path):
            logger.error(f"Config file not found: {path}")
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, "r") as file:
                if path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                    logger.info(f"Loaded YAML config from {path}")
                elif path.endswith(".json"):
                    config_data = json.load(file)
                    logger.info(f"Loaded JSON config from {path}")
                else:
                    raise ValueError("Unsupported config file format. Use .json, .yml or .yaml")

            return cls(config_data)

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing config file: {e}")
            raise ValueError(f"Error parsing config file: {e}")
                   
    def get(self, key_path: str, default=None):
        """Support nested keys like 'base.separator' or 'artifact_path.train'"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logger.warning(f"Config key path '{key_path}' is invalid at '{key}'")
                return default

        logger.debug(f"Retrieved config key '{key}' with value: {value}")
        return value


