"""Version and time semantics shared by METTLE credential formats."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

CREDENTIAL_SCHEMA_VERSION = "1.1"
SUITE_POLICY_VERSION = "2026-08-14"
SUPPORTED_CREDENTIAL_SCHEMA_VERSIONS = frozenset({CREDENTIAL_SCHEMA_VERSION})
SUPPORTED_SUITE_POLICY_VERSIONS = frozenset({SUITE_POLICY_VERSION})

# Verifiers may tolerate a small amount of ordinary clock drift. The value is
# deliberately bounded and published in the credential transparency contract.
CREDENTIAL_CLOCK_SKEW_SECONDS = 30


def utc_now() -> datetime:
    """Return an aware UTC timestamp through one patchable protocol clock."""
    return datetime.now(timezone.utc)


def credential_versions_supported(metadata: dict[str, object]) -> bool:
    """Accept only credentials using the verifier's exact current semantics."""
    schema_version = metadata.get("credential_schema_version")
    policy_version = metadata.get("suite_policy_version")
    return (
        isinstance(schema_version, str)
        and schema_version in SUPPORTED_CREDENTIAL_SCHEMA_VERSIONS
        and isinstance(policy_version, str)
        and policy_version in SUPPORTED_SUITE_POLICY_VERSIONS
    )


def credential_time_window_valid(
    *,
    reviewed_at: datetime,
    expires_at: datetime,
    now: datetime | None = None,
    clock_skew_seconds: int = CREDENTIAL_CLOCK_SKEW_SECONDS,
) -> bool:
    """Validate issuance and expiry with an explicit bounded skew allowance."""
    if not 0 <= clock_skew_seconds <= CREDENTIAL_CLOCK_SKEW_SECONDS:
        raise ValueError(
            f"clock_skew_seconds must be between 0 and {CREDENTIAL_CLOCK_SKEW_SECONDS}"
        )
    current = now or utc_now()
    if (
        current.tzinfo is None
        or reviewed_at.tzinfo is None
        or expires_at.tzinfo is None
    ):
        return False
    skew = timedelta(seconds=clock_skew_seconds)
    return (
        expires_at > reviewed_at
        and reviewed_at <= current + skew
        and expires_at + skew > current
    )
