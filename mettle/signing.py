"""Ed25519 signing for VCP attestations.

Loads a signing key from METTLE_VCP_SIGNING_KEY env var (PEM format)
or generates an ephemeral key pair in dev mode.

Requires: pip install cryptography
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Module-level state
_private_key: Any = None
_public_key: Any = None
_key_id: str = "mettle-vcp-v1"
_initialized: bool = False
KEY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def init_signing() -> bool:
    """Initialize the Ed25519 signing key.

    Loads from METTLE_VCP_SIGNING_KEY env var (PEM) or generates ephemeral key.

    Returns:
        True if signing is available, False otherwise.
    """
    global _private_key, _public_key, _key_id, _initialized

    _private_key = None
    _public_key = None
    _key_id = "mettle-vcp-v1"

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
    configured_key_id = os.environ.get("METTLE_VCP_SIGNING_KEY_ID")
    dev_mode = (os.environ.get("METTLE_DEV_MODE") or "false").lower() == "true"
    try:
        from mettle.app_config import settings

        pem_key = settings.vcp_signing_key or None
        configured_key_id = configured_key_id or getattr(
            settings, "vcp_signing_key_id", None
        )
        dev_mode = getattr(settings, "dev_mode", False) or dev_mode
    except Exception as settings_error:
        logger.debug(
            "Mettle settings unavailable for VCP signing key lookup: %s", settings_error
        )
    if not pem_key:
        pem_key = os.environ.get("METTLE_VCP_SIGNING_KEY")
    configured_key_id = configured_key_id or "mettle-vcp-v1"
    if KEY_ID_PATTERN.fullmatch(configured_key_id) is None:
        logger.error("METTLE_VCP_SIGNING_KEY_ID is invalid")
        _initialized = True
        return False
    _key_id = configured_key_id

    if pem_key:
        try:
            loaded_key = load_pem_private_key(pem_key.encode(), password=None)
            if not isinstance(loaded_key, Ed25519PrivateKey):
                raise ValueError("VCP signing key must use Ed25519")
            _private_key = loaded_key
            _public_key = _private_key.public_key()
            logger.info("VCP signing key loaded from METTLE_VCP_SIGNING_KEY")
        except Exception:
            logger.error("Failed to load METTLE_VCP_SIGNING_KEY", exc_info=True)
            _initialized = True
            return False
    else:
        if not dev_mode:
            logger.error(
                "METTLE_VCP_SIGNING_KEY is required outside explicit development mode"
            )
            _initialized = True
            return False
        # Ephemeral keys are suitable only for explicit development sessions.
        _private_key = Ed25519PrivateKey.generate()
        _public_key = _private_key.public_key()
        logger.info(
            "Generated ephemeral Ed25519 key for VCP attestation signing (dev mode)"
        )

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

    pem: str | None = None
    try:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        pem = _public_key.public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        ).decode("ascii")
    except Exception as exc:
        logger.debug("Failed to serialize VCP public key: %s", exc)
    return pem


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


def verify_signature(public_key_pem: str, data: bytes, signature_b64: str) -> bool:
    """Verify a base64 Ed25519 signature against data using a PEM public key.

    Returns True if the signature is valid, False otherwise.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError:
        return False

    if (
        not isinstance(public_key_pem, str)
        or not isinstance(data, bytes)
        or not isinstance(signature_b64, str)
    ):
        return False

    try:
        public_key = load_pem_public_key(public_key_pem.encode("ascii"))
        if not isinstance(public_key, Ed25519PublicKey):
            logger.debug("Public key is not Ed25519: %s", type(public_key).__name__)
            return False
        signature = base64.b64decode(signature_b64, validate=True)
        if len(signature) != 64:
            return False
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False
    except (AttributeError, binascii.Error, UnicodeError, ValueError, TypeError) as e:
        logger.debug("Signature verification error: %s", e)
        return False
