"""Application configuration loaded from environment variables."""
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    NEON_DATABASE_URL: str
    UPLOAD_DIR: str = "./uploads"
    TOP_K_CHUNKS: int = 5

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="allow",
    )


settings = Settings()
