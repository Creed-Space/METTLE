"""METTLE configuration management."""

import enum
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class RuntimeEnvironment(str, enum.Enum):
    """Recognised deployment environments.

    A closed set prevents a typo such as ``prodution`` from silently acquiring
    development behaviour.
    """

    LOCAL = "local"
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


def normalize_runtime_environment(value: object) -> RuntimeEnvironment:
    """Normalise a runtime environment or fail closed on unknown values."""

    if isinstance(value, RuntimeEnvironment):
        return value
    if not isinstance(value, str):
        raise ValueError("METTLE_ENVIRONMENT must be a recognised string")
    normalized = value.strip().lower()
    try:
        return RuntimeEnvironment(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RuntimeEnvironment)
        raise ValueError(f"METTLE_ENVIRONMENT must be one of: {allowed}") from exc


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: RuntimeEnvironment = Field(
        default=RuntimeEnvironment.DEVELOPMENT,
        description="Runtime environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # API
    api_title: str = Field(default="METTLE", description="API title")
    api_version: str = Field(default="0.2.0", description="API version")

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

    # API Key for admin operations
    admin_api_key: str = Field(
        default="",
        description="Admin API key for tier management and admin operations",
    )

    # Ed25519 signing key for VCP attestations
    vcp_signing_key: str = Field(
        default="",
        repr=False,
        description="PEM-encoded Ed25519 private key. Required in production.",
    )
    vcp_signing_key_id: str = Field(
        default="mettle-vcp-v1",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
        description="Stable identifier for the active Ed25519 issuer key.",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///mettle.db",
        description="Database URL (sqlite:// or postgresql://)",
    )
    use_database: bool = Field(
        default=False,
        description="Enable database persistence (default: in-memory)",
    )

    # Redis persistence for v2 session state
    redis_url: str = Field(
        default="",
        repr=False,
        description="Redis URL. Required in production for v2 session persistence.",
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
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment is RuntimeEnvironment.PRODUCTION

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(_cls, value: object) -> RuntimeEnvironment:
        return normalize_runtime_environment(value)

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        """SECURITY: Validate security-critical settings in production."""
        if self.is_production:
            if self.allowed_origins == "*" or not self.allowed_origins_list:
                raise ValueError(
                    "METTLE_ALLOWED_ORIGINS must list trusted origins in production"
                )
            if any(
                not origin.startswith("https://")
                for origin in self.allowed_origins_list
            ):
                raise ValueError("METTLE_ALLOWED_ORIGINS must use HTTPS in production")
            if len(self.secret_key) < 32:
                raise ValueError(
                    "METTLE_SECRET_KEY must be at least 32 characters in production"
                )
            if len(self.admin_api_key) < 32:
                raise ValueError(
                    "METTLE_ADMIN_API_KEY must be at least 32 characters in production"
                )
            if not self.vcp_signing_key:
                raise ValueError("METTLE_VCP_SIGNING_KEY is required in production")
            if not self.use_database:
                raise ValueError("METTLE_USE_DATABASE must be enabled in production")
            if not self.redis_url.strip():
                raise ValueError("METTLE_REDIS_URL is required in production")
            database_scheme = urlparse(self.database_url).scheme
            if database_scheme not in {"postgres", "postgresql"}:
                raise ValueError(
                    "METTLE_DATABASE_URL must use PostgreSQL in production"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
