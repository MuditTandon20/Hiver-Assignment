"""Configuration and logging utilities."""
import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent

def resolve_path(relative_or_absolute: str) -> Path:
    """Resolve a path relative to project root if it is not already absolute."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return get_project_root() / p

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file with environment variable overrides."""
    full_path = resolve_path(config_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
    
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Environment variable overrides
    if os.getenv("SELECTED_BRAND"):
        config.setdefault("project", {})["selected_brand"] = os.getenv("SELECTED_BRAND")
    if os.getenv("INTENT_CONFIDENCE_THRESHOLD"):
        config.setdefault("intents", {})["confidence_threshold"] = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD"))
    if os.getenv("GENERATION_PROVIDER"):
        config.setdefault("generation", {})["provider"] = os.getenv("GENERATION_PROVIDER")
        
    return config

def setup_logger(name: str = "support_agent", level: int = logging.INFO) -> logging.Logger:
    """Set up structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
