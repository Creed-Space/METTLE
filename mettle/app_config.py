"""Pydantic Settings for METTLE standalone."""

from pydantic_settings import BaseSettings


class MettleSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    api_keys: str = ""
    dev_mode: bool = False
    cors_origins: str = "*"
    vcp_signing_key: str = ""
    model_config = {"env_prefix": "METTLE_", "env_file": ".env", "env_file_encoding": "utf-8"}


settings = MettleSettings()
