import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment variables override defaults; no .env file is loaded implicitly.
    database_url: str = "postgresql+asyncpg://vtuser:vtpass@localhost:55442/voicetranslator"
    jwt_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(48)))
    access_token_expire_minutes: int = Field(default=30, gt=0)
    refresh_token_expire_days: int = Field(default=7, gt=0)
    audio_dir: Path = Path(__file__).resolve().parents[2] / "work" / "audio"

    # Models (T11, chosen in docs/experiments.md). Off by default so tests and CI need no GPU or model
    # files; the server sets LOAD_MODELS=true. Without them every translation answers 503.
    load_models: bool = False
    model_device: Literal["cuda", "cpu"] = "cuda"
    stt_model: str = "large-v3-turbo"
    ct2_dir: Path = Path(__file__).resolve().parents[2] / "data" / "models" / "ct2"
    tts_enabled: bool = True
    # Freeze the loaded models out of the garbage collector's reach (T23). Off only to measure without it.
    gc_freeze: bool = True
    # Pass only the speech parts to Whisper (T14, docs/experiments.md 1-1): no empty results on speech, and
    # silence comes back as "no speech" instead of made-up text.
    stt_vad_filter: bool = True
    # Decode uploads in app/services/stt.py instead of inside faster-whisper (T23 candidate C).
    stt_own_decode: bool = False
    # Run every model once after loading, so the first request does not pay for lazy loading (T25).
    warm_up: bool = True
    # Fix typos in typed English before translating it, with a small LLM served by Ollama (T32,
    # docs/experiments.md 6). When Ollama cannot be reached, text is translated as typed.
    typo_correction: bool = True
    ollama_url: str = "http://localhost:11434"
    correction_model: str = "qwen2.5:1.5b-instruct"
    correction_timeout_s: float = Field(default=10, gt=0)
    # Limit for each slow preparation call (download, load, first correction); a failed try repeats (T37).
    correction_prepare_timeout_s: float = Field(default=600, gt=0)

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 UTF-8 bytes")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
