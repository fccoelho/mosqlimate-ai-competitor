"""Configuration management for Mosqlimate AI Competitor.

Provides utilities for loading and merging configuration from YAML files
with CLI arguments.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path("mosqlimate.yaml"),
    Path(".mosqlimate.yaml"),
    Path("config.yaml"),
    Path(".config/mosqlimate/config.yaml"),
]


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    If no path is provided, searches for config files in default locations.

    Args:
        config_path: Path to config file, or None to search defaults

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config()
        >>> print(config["models"]["xgboost"]["n_estimators"])
        500
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        return _load_yaml(path)

    # Search default locations
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            logger.debug(f"Loading config from: {path}")
            return _load_yaml(path)

    logger.debug("No config file found in default locations")
    return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file safely.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with config values
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                return {}
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {path}: {e}")
        return {}


def get_cli_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Get CLI-specific configuration.

    Extracts CLI-relevant settings from the full config.

    Args:
        config_path: Path to config file, or None for defaults

    Returns:
        Dictionary with CLI configuration
    """
    full_config = load_config(config_path)

    # Extract CLI-relevant sections
    cli_config = {
        "data": full_config.get("data", {}),
        "models": full_config.get("models", {}),
        "ensemble": full_config.get("ensemble", {}),
        "evaluation": full_config.get("evaluation", {}),
        "paths": full_config.get("paths", {}),
        "states": full_config.get("states", []),
    }

    return cli_config


def merge_with_defaults(
    user_value: Any,
    config_value: Any,
    default_value: Any,
) -> Any:
    """Merge user CLI value with config value and default.

    Priority: user_value > config_value > default_value

    Args:
        user_value: Value from CLI argument
        config_value: Value from config file
        default_value: Default value

    Returns:
        Merged value with proper priority

    Example:
        >>> merge_with_defaults(None, 100, 50)  # From config
        100
        >>> merge_with_defaults(200, 100, 50)   # User override
        200
        >>> merge_with_defaults(None, None, 50)  # Use default
        50
    """
    if user_value is not None:
        return user_value
    if config_value is not None:
        return config_value
    return default_value


class ConfigManager:
    """Manages configuration loading and merging.

    Provides a convenient interface for accessing configuration
    with proper precedence (CLI args > config file > defaults).

    Example:
        >>> config = ConfigManager("config.yaml")
        >>> n_estimators = config.get("models.xgboost.n_estimators", default=500)
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager.

        Args:
            config_path: Path to config file, or None to search defaults
        """
        self._config = load_config(config_path)
        self._config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "models.xgboost.n_estimators")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("models.xgboost.max_depth", default=6)
            6
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value if value is not None else default

    def get_path(self, key: str, default: Optional[Path] = None) -> Optional[Path]:
        """Get path configuration value.

        Args:
            key: Dot-separated key
            default: Default path

        Returns:
            Path object or None
        """
        value = self.get(key, default)
        if value is None:
            return None
        return Path(value)

    def get_models_dir(self) -> Path:
        """Get models directory path."""
        return self.get_path("paths.models_dir", Path("models"))

    def get_forecasts_dir(self) -> Path:
        """Get forecasts directory path."""
        return self.get_path("paths.forecasts_dir", Path("forecasts"))

    def get_cache_dir(self) -> Optional[Path]:
        """Get data cache directory path."""
        return self.get_path("data.cache_dir")

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get configuration for specific model.

        Args:
            model_name: Model name (e.g., "xgboost", "lstm")

        Returns:
            Model configuration dictionary
        """
        return self.get(f"models.{model_name}", {})

    def get_states(self) -> list[str]:
        """Get list of states from config."""
        return self.get("states", [])

    @property
    def config(self) -> dict[str, Any]:
        """Access full configuration dictionary."""
        return self._config


def save_example_config(output_path: Path = Path("mosqlimate.yaml")) -> None:
    """Create an example configuration file.

    Args:
        output_path: Where to save the example config

    Example:
        >>> save_example_config()
        # Creates mosqlimate.yaml with default settings
    """
    example_config = """# Mosqlimate AI Competitor Configuration
# =====================================

# Data settings
data:
  cache_dir: "./data"

# Model hyperparameters
models:
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    epochs: 100

# Ensemble settings
ensemble:
  method: "weighted_average"
  weight_metric: "crps"

# Evaluation settings
evaluation:
  confidence_levels: [0.50, 0.80, 0.95]

# Path settings
paths:
  models_dir: "./models"
  forecasts_dir: "./forecasts"
  submissions_dir: "./submissions"

# States to process (empty = all states)
states: []

# API settings
api:
  base_url: "https://api.mosqlimate.org/api"
  timeout: 30
"""

    with open(output_path, "w") as f:
        f.write(example_config)

    logger.info(f"Example config saved to: {output_path}")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get or create global config manager.

    Args:
        config_path: Path to config file

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None or (
        config_path is not None and _config_manager._config_path != config_path
    ):
        _config_manager = ConfigManager(config_path)
    return _config_manager
