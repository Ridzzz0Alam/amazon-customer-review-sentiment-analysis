"""Configuration for training (params.yaml) and serving (environment variables)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path = PROJECT_ROOT / "params.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Settings(BaseSettings):
    """Serving settings. Override any field with an env var prefixed SENTIMENT_."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env", extra="ignore")

    model_path: Path = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
    model_meta_path: Path = PROJECT_ROOT / "models" / "model_meta.json"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]
    max_batch_size: int = 1000
    max_text_chars: int = 5000
    log_predictions: bool = True
    prediction_log_path: Path = PROJECT_ROOT / "logs" / "predictions.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
