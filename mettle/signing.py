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
    dev_mode = (os.environ.get("METTLE_DEV_MODE") or "false").lower() == "true"
    try:
        from mettle.app_config import settings
        pem_key = settings.vcp_signing_key or None
        dev_mode = settings.dev_mode or dev_mode
    except Exception as settings_error:
        logger.debug("Mettle settings unavailable for VCP signing key lookup: %s", settings_error)
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
        if not dev_mode:
            logger.error(
                "METTLE_VCP_SIGNING_KEY is required outside explicit development mode"
            )
            _initialized = True
            return False
        # Ephemeral keys are suitable only for explicit development sessions.
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

    pem: str | None = None
    try:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        pem = _public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("ascii")
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


# === CLI self-signed credential key management ===

CLI_KEY_ID = "mettle-cli-ed25519-v1"


def cli_key_dir() -> "os.PathLike[str] | str":
    """Return the directory holding the CLI's persistent signing key.

    Defaults to ``~/.mettle`` but can be overridden with ``METTLE_HOME``
    (useful for tests and sandboxed runs).
    """
    from pathlib import Path

    override = os.environ.get("METTLE_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".mettle"
    return base


def load_or_create_cli_keypair() -> tuple[Any, str]:
    """Load (or create on first run) the persistent Ed25519 CLI signing key.

    The private key is stored at ``<key_dir>/ed25519_private.pem`` with 0600
    permissions; the public key at ``<key_dir>/ed25519_public.pem``.

    Returns:
        Tuple of (Ed25519PrivateKey, public_key_pem_str).

    Raises:
        RuntimeError: If the cryptography package is unavailable.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
            load_pem_private_key,
        )
    except ImportError as e:
        raise RuntimeError("The 'cryptography' package is required for credential signing") from e

    from pathlib import Path

    key_dir = Path(cli_key_dir())
    key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkdir mode is ignored for pre-existing dirs; enforce it. Fatal on failure —
    # a traversable key dir undermines the 0600 key file below.
    os.chmod(key_dir, 0o700)

    priv_path = key_dir / "ed25519_private.pem"
    pub_path = key_dir / "ed25519_public.pem"

    if priv_path.exists():
        private_key = load_pem_private_key(priv_path.read_bytes(), password=None)
    else:
        private_key = Ed25519PrivateKey.generate()
        priv_bytes = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        # Create atomically with 0600 — never a window where the key is
        # world-readable (no write-then-chmod TOCTOU). O_EXCL: refuse to
        # follow/overwrite anything racing us at this path.
        fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(priv_bytes)
        except BaseException:
            priv_path.unlink(missing_ok=True)
            raise

    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")
    pub_path.write_text(public_pem)

    return private_key, public_pem


def sign_bytes(private_key: Any, data: bytes) -> str:
    """Sign bytes with an Ed25519 private key, returning a base64 signature."""
    return base64.b64encode(private_key.sign(data)).decode("ascii")


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

    try:
        public_key = load_pem_public_key(public_key_pem.encode("ascii"))
        if not isinstance(public_key, Ed25519PublicKey):
            logger.debug("Public key is not Ed25519: %s", type(public_key).__name__)
            return False
        public_key.verify(base64.b64decode(signature_b64), data)
        return True
    except InvalidSignature:
        return False
    except (ValueError, TypeError) as e:
        logger.debug("Signature verification error: %s", e)
        return False
