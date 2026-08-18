"""METTLE configuration management."""

from functools import lru_cache
from urllib.parse import parse_qs, urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


def _validate_redis_tls_query(redis_url: str) -> None:
    """Reject URL query values that can override certificate verification."""
    query = parse_qs(urlparse(redis_url).query, keep_blank_values=True)
    certificate_requirements = query.get("ssl_cert_reqs", [])
    if certificate_requirements and (
        len(certificate_requirements) != 1
        or certificate_requirements[0].strip().lower()
        not in {"required", "cert_required"}
    ):
        raise ValueError("METTLE_REDIS_URL must require TLS certificate verification")
    hostname_checks = query.get("ssl_check_hostname", [])
    if hostname_checks and (
        len(hostname_checks) != 1 or hostname_checks[0].strip().lower() != "true"
    ):
        raise ValueError("METTLE_REDIS_URL must enable TLS hostname verification")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Enable debug mode")

    # API
    api_title: str = Field(default="METTLE", description="API title")
    api_version: str = Field(default="0.4.7", description="API version")

    # CORS
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed origins, or * for all",
    )
    trusted_hosts: str = Field(
        default="*",
        description="Comma-separated HTTP Host values accepted by the service",
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
    credential_issuance_enabled: bool = Field(
        default=True,
        description="Emergency switch for all new credential issuance",
    )

    # Badge settings
    badge_expiry_seconds: int = Field(
        default=86400,
        description="Badge expiry time in seconds (default: 24 hours)",
    )
    private_data_retention_seconds: int = Field(
        default=86400,
        ge=1800,
        le=2592000,
        description="Maximum retention for persisted sessions and challenge data",
    )
    verification_record_retention_seconds: int = Field(
        default=86400,
        ge=3600,
        le=2592000,
        description="Maximum retention for collusion-detection events",
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
    vcp_verifying_keys: str = Field(
        default="",
        description="JSON object of retired key IDs to Ed25519 public PEM values",
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
    redis_url: str = Field(
        default="",
        repr=False,
        description="Redis URL for durable v2 session and rate-limit authority",
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
    def trusted_hosts_list(self) -> list[str]:
        """Parse accepted HTTP hostnames for TrustedHostMiddleware."""
        if self.trusted_hosts == "*":
            return ["*"]
        return [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

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
            if self.trusted_hosts == "*" or not self.trusted_hosts_list:
                raise ValueError(
                    "METTLE_TRUSTED_HOSTS must list accepted hosts in production"
                )
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
            database_scheme = urlparse(self.database_url).scheme
            if database_scheme not in {"postgres", "postgresql"}:
                raise ValueError(
                    "METTLE_DATABASE_URL must use PostgreSQL in production"
                )
            database_query = parse_qs(urlparse(self.database_url).query)
            if database_query.get("sslmode", [""])[-1] != "verify-full":
                raise ValueError(
                    "METTLE_DATABASE_URL must set sslmode=verify-full in production"
                )
            if urlparse(self.redis_url).scheme != "rediss":
                raise ValueError("METTLE_REDIS_URL must use rediss TLS in production")
            _validate_redis_tls_query(self.redis_url)
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
