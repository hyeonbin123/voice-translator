import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment variables override defaults; no .env file is loaded implicitly.
    database_url: str = "postgresql+asyncpg://vtuser:vtpass@localhost:55442/voicetranslator"
    jwt_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(48)))
    access_token_expire_minutes: int = Field(default=30, gt=0)
    refresh_token_expire_days: int = Field(default=7, gt=0)
    audio_dir: Path = Path(__file__).resolve().parents[2] / "work" / "audio"

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 UTF-8 bytes")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
