"""METTLE configuration management."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Enable debug mode")

    # API
    api_title: str = Field(default="METTLE", description="API title")
    api_version: str = Field(default="2.0.0", description="API version")

    # CORS
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed origins, or * for all",
    )

    # Rate Limiting
    rate_limit_sessions: str = Field(
        default="10/minute",
        description="Rate limit for session creation",
    )
    rate_limit_answers: str = Field(
        default="60/minute",
        description="Rate limit for answer submission",
    )

    # Security
    secret_key: str = Field(
        default="",
        description="Secret key for badge signing. Required in production.",
    )

    # Badge settings
    badge_expiry_seconds: int = Field(
        default=86400,
        description="Badge expiry time in seconds (default: 24 hours)",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {
        "env_prefix": "METTLE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed origins into a list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
