"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    """Small dependency-free settings object for the take-home service."""

    def __init__(self) -> None:
        configured_path = os.getenv("DATABASE_PATH", "data/warehouse.sqlite3")
        path = Path(configured_path)
        self.database_path = path if path.is_absolute() else PROJECT_ROOT / path
        self.api_version = os.getenv("API_VERSION", "0.1.0")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        configured_artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))
        self.model_artifact_dir = (
            configured_artifact_dir if configured_artifact_dir.is_absolute()
            else PROJECT_ROOT / configured_artifact_dir
        )
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.assistant_api_base_url = os.getenv("ASSISTANT_API_BASE_URL", "http://127.0.0.1:8000")
        configured_log_path = Path(os.getenv("GENAI_AUDIT_LOG", "logs/genai_assistant.jsonl"))
        self.genai_audit_log = configured_log_path if configured_log_path.is_absolute() else PROJECT_ROOT / configured_log_path


settings = Settings()
