"""Ed25519 signing for VCP attestations.

Loads a signing key from METTLE_VCP_SIGNING_KEY env var (PEM format)
or generates an ephemeral key pair in dev mode.

Requires: pip install cryptography
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Module-level state
_private_key: Any = None
_public_key: Any = None
_key_id: str = "mettle-vcp-v1"
_initialized: bool = False


def init_signing() -> bool:
    """Initialize the Ed25519 signing key.

    Loads from METTLE_VCP_SIGNING_KEY env var (PEM) or generates ephemeral key.

    Returns:
        True if signing is available, False otherwise.
    """
    global _private_key, _public_key, _initialized

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
        )
    except ImportError:
        logger.warning(
            "cryptography package not installed. VCP attestation signing disabled. "
            "Install with: pip install cryptography"
        )
        _initialized = True
        return False

    # Try settings first (.env support), fall back to raw env var
    pem_key = None
    try:
        from mettle.app_config import settings
        pem_key = settings.vcp_signing_key or None
    except Exception:
        pass
    if not pem_key:
        pem_key = os.environ.get("METTLE_VCP_SIGNING_KEY")

    if pem_key:
        try:
            _private_key = load_pem_private_key(pem_key.encode(), password=None)
            _public_key = _private_key.public_key()
            logger.info("VCP signing key loaded from METTLE_VCP_SIGNING_KEY")
        except Exception:
            logger.error("Failed to load METTLE_VCP_SIGNING_KEY", exc_info=True)
            _initialized = True
            return False
    else:
        # Generate ephemeral key for dev
        _private_key = Ed25519PrivateKey.generate()
        _public_key = _private_key.public_key()
        logger.info("Generated ephemeral Ed25519 key for VCP attestation signing (dev mode)")

    _initialized = True
    return True


def sign_attestation(data: bytes) -> str:
    """Sign data with the Ed25519 private key.

    Args:
        data: Bytes to sign.

    Returns:
        Base64-encoded signature string.

    Raises:
        RuntimeError: If signing is not initialized or unavailable.
    """
    if not _initialized:
        init_signing()

    if _private_key is None:
        raise RuntimeError("VCP attestation signing not available")

    signature = _private_key.sign(data)
    return base64.b64encode(signature).decode("ascii")


def get_public_key_pem() -> str | None:
    """Get the public key in PEM format for trust config discovery.

    Returns:
        PEM-encoded public key string, or None if signing unavailable.
    """
    if not _initialized:
        init_signing()

    if _public_key is None:
        return None

    try:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        return _public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("ascii")
    except Exception:
        return None


def get_public_key_info() -> dict[str, Any]:
    """Get public key info for the .well-known endpoint.

    Returns:
        Dict with key_id, algorithm, and public key (PEM).
    """
    pem = get_public_key_pem()
    return {
        "key_id": _key_id,
        "algorithm": "Ed25519",
        "public_key_pem": pem,
        "available": pem is not None,
    }


def is_available() -> bool:
    """Check if signing is initialized and available."""
    if not _initialized:
        init_signing()
    return _private_key is not None
