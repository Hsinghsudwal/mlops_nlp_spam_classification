import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from utils.logger import logger


class ConfigManager:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config_dict = config_dict or {}
        logger.info("ConfigManager initialized with configuration.")

    @classmethod
    def load_file(cls, config_file: str) -> 'ConfigManager':
        """Load configuration from a YAML or JSON file."""
        if not os.path.isfile(config_file):
            logger.error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            with open(config_file, "r") as file:
                if config_file.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                    logger.info(f"Loaded YAML config from {config_file}")
                elif config_file.endswith(".json"):
                    config_data = json.load(file)
                    logger.info(f"Loaded JSON config from {config_file}")
                else:
                    raise ValueError("Unsupported config file format. Use .json, .yml or .yaml")

            return cls(config_data)

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing config file: {e}")
            raise ValueError(f"Error parsing config file: {e}")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Access nested config values using dot notation."""
        keys = key.split('.')
        value = self.config_dict

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                logger.warning(f"Config key path '{key}' is invalid at '{k}'")
                return default

        logger.debug(f"Retrieved config key '{key}' with value: {value}")
        return value
