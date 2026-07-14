"""Pydantic Settings for METTLE standalone."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MettleSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    redis_namespace: str = Field(
        default="mettle",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9:_-]{0,63}$",
    )
    api_keys: str = ""
    dev_mode: bool = False
    cors_origins: str = "*"
    vcp_signing_key: str = ""
    vcp_signing_key_id: str = Field(
        default="mettle-vcp-v1",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    model_config = SettingsConfigDict(
        env_prefix="METTLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        # The project environment can contain many non-METTLE keys.
        # Ignore unknown keys so settings import remains stable.
        extra="ignore",
    )


settings = MettleSettings()
