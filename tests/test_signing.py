"""Tests for mettle/signing.py — Ed25519 signing for VCP attestations."""

import base64
import builtins
import json
from unittest.mock import patch

import pytest

import mettle.signing as signing


@pytest.fixture(autouse=True)
def reset_signing_state(monkeypatch):
    """Reset module-level state between tests."""
    monkeypatch.setenv("METTLE_DEV_MODE", "true")
    signing._private_key = None
    signing._public_key = None
    signing._key_id = "mettle-vcp-v1"
    signing._initialized = False
    yield
    signing._private_key = None
    signing._public_key = None
    signing._key_id = "mettle-vcp-v1"
    signing._initialized = False


# --- init_signing ---


class TestInitSigning:
    def test_ephemeral_key_generation(self):
        """init_signing generates ephemeral key when no env var set."""
        result = signing.init_signing()
        assert result is True
        assert signing._private_key is not None
        assert signing._public_key is not None
        assert signing._initialized is True

    def test_missing_key_fails_closed_outside_dev_mode(self, monkeypatch):
        monkeypatch.setenv("METTLE_DEV_MODE", "false")
        mock_settings = type(
            "MockSettings", (), {"vcp_signing_key": "", "dev_mode": False}
        )()

        with patch("mettle.app_config.settings", mock_settings):
            result = signing.init_signing()

        assert result is False
        assert signing._private_key is None

    def test_with_valid_pem_env_var(self, monkeypatch):
        """init_signing loads key from METTLE_VCP_SIGNING_KEY env var."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        ).decode()
        monkeypatch.setenv("METTLE_VCP_SIGNING_KEY", pem)

        # Patch settings to return empty so env var path is used
        mock_settings = type("MockSettings", (), {"vcp_signing_key": ""})()
        with patch("mettle.app_config.settings", mock_settings):
            result = signing.init_signing()

        assert result is True
        assert signing._private_key is not None
        assert signing._public_key is not None

    def test_with_invalid_pem_returns_false(self, monkeypatch):
        """init_signing returns False for invalid PEM key."""
        monkeypatch.setenv("METTLE_VCP_SIGNING_KEY", "not-a-valid-pem-key")

        # Patch settings to return the invalid key (settings is imported inside init_signing)
        mock_settings = type(
            "MockSettings", (), {"vcp_signing_key": "not-a-valid-pem-key"}
        )()
        with patch("mettle.app_config.settings", mock_settings):
            result = signing.init_signing()

        assert result is False
        assert signing._initialized is True
        assert signing._private_key is None

    def test_without_cryptography_returns_false(self):
        """init_signing returns False when cryptography is not installed."""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "cryptography" in name:
                raise ImportError("No module named 'cryptography'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = signing.init_signing()

        assert result is False
        assert signing._initialized is True
        assert signing._private_key is None

    def test_with_valid_pem_via_settings(self, monkeypatch):
        """init_signing loads key from settings.vcp_signing_key."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        ).decode()

        mock_settings = type("MockSettings", (), {"vcp_signing_key": pem})()
        with patch("mettle.app_config.settings", mock_settings):
            result = signing.init_signing()

        assert result is True
        assert signing._private_key is not None

    def test_settings_import_error_falls_back_to_env(self, monkeypatch):
        """When settings import raises, falls back to METTLE_VCP_SIGNING_KEY env var."""
        monkeypatch.delenv("METTLE_VCP_SIGNING_KEY", raising=False)

        # Make settings raise an exception when accessed, triggering except path
        # The code does: from mettle.app_config import settings; settings.vcp_signing_key
        # If settings.vcp_signing_key raises, it falls through to os.environ.get
        class MockSettings:
            @property
            def vcp_signing_key(self) -> str:
                raise Exception("settings broken")

        mock_settings = MockSettings()

        with patch("mettle.app_config.settings", mock_settings):
            result = signing.init_signing()

        # No env var set either, so ephemeral key generated
        assert result is True
        assert signing._private_key is not None


# --- sign_attestation ---


class TestSignAttestation:
    def test_sign_after_init(self):
        """sign_attestation returns base64 string after init."""
        signing.init_signing()
        result = signing.sign_attestation(b"test data")
        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) == 64  # Ed25519 signatures are 64 bytes

    def test_sign_auto_initializes(self):
        """sign_attestation auto-initializes if not yet initialized."""
        assert signing._initialized is False
        result = signing.sign_attestation(b"test data")
        assert isinstance(result, str)
        assert signing._initialized is True

    def test_sign_when_key_is_none_raises(self):
        """sign_attestation raises RuntimeError when key is None."""
        # Mark as initialized but with no key (simulates failed init)
        signing._initialized = True
        signing._private_key = None

        with pytest.raises(RuntimeError, match="VCP attestation signing not available"):
            signing.sign_attestation(b"test data")


# --- get_public_key_pem ---


class TestGetPublicKeyPem:
    def test_returns_pem_after_init(self):
        """get_public_key_pem returns PEM string after successful init."""
        signing.init_signing()
        pem = signing.get_public_key_pem()
        assert pem is not None
        assert "BEGIN PUBLIC KEY" in pem

    def test_returns_none_when_unavailable(self):
        """get_public_key_pem returns None when signing unavailable."""
        signing._initialized = True
        signing._public_key = None
        result = signing.get_public_key_pem()
        assert result is None

    def test_auto_initializes(self):
        """get_public_key_pem auto-initializes if not yet initialized."""
        assert signing._initialized is False
        pem = signing.get_public_key_pem()
        assert pem is not None
        assert signing._initialized is True

    def test_returns_none_on_serialization_error(self):
        """get_public_key_pem returns None if public_bytes raises."""
        signing._initialized = True
        # Set a mock public key that will fail on public_bytes
        signing._public_key = "not-a-real-key"
        result = signing.get_public_key_pem()
        assert result is None


# --- get_public_key_info ---


class TestGetPublicKeyInfo:
    def test_returns_dict_after_init(self):
        """get_public_key_info returns correct dict structure."""
        signing.init_signing()
        info = signing.get_public_key_info()
        assert info["key_id"] == "mettle-vcp-v1"
        assert info["algorithm"] == "Ed25519"
        assert info["public_key_pem"] is not None
        assert info["available"] is True
        assert info["credential_schema_versions"] == ["1.1"]
        assert info["suite_policy_versions"] == ["2026-08-14"]

    def test_returns_unavailable_when_no_key(self):
        """get_public_key_info shows unavailable when no key."""
        signing._initialized = True
        signing._public_key = None
        info = signing.get_public_key_info()
        assert info["public_key_pem"] is None
        assert info["available"] is False

    def test_configured_key_id_is_published(self, monkeypatch):
        monkeypatch.setenv("METTLE_VCP_SIGNING_KEY_ID", "mettle-vcp-2026-02")
        signing.init_signing()
        info = signing.get_public_key_info()
        assert info["key_id"] == "mettle-vcp-2026-02"

    def test_rotation_keyring_publishes_active_and_verify_only_keys(self, monkeypatch):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        old_key = Ed25519PrivateKey.generate()
        old_pem = (
            old_key.public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            .decode("ascii")
        )
        monkeypatch.setenv(
            "METTLE_VCP_VERIFYING_KEYS", json.dumps({"mettle-vcp-old": old_pem})
        )
        signing.init_signing()

        info = signing.get_public_key_info()

        assert {item["key_id"] for item in info["keys"]} == {
            "mettle-vcp-v1",
            "mettle-vcp-old",
        }
        assert {item["status"] for item in info["keys"]} == {
            "active",
            "verify-only",
        }

    def test_malformed_rotation_keyring_fails_closed(self, monkeypatch):
        monkeypatch.setenv("METTLE_VCP_VERIFYING_KEYS", "not-json")
        signing.init_signing()
        with pytest.raises(RuntimeError, match="valid JSON"):
            signing.get_public_keyring()

    def test_invalid_configured_key_id_fails_closed(self, monkeypatch):
        signing._private_key = object()
        signing._public_key = object()
        signing._key_id = "stale-key"
        monkeypatch.setenv("METTLE_VCP_SIGNING_KEY_ID", "bad key id")
        assert signing.init_signing() is False
        assert signing.is_available() is False
        assert signing._private_key is None
        assert signing._public_key is None
        assert signing._key_id == "mettle-vcp-v1"


# --- is_available ---


class TestIsAvailable:
    def test_true_after_successful_init(self):
        """is_available returns True after successful init."""
        signing.init_signing()
        assert signing.is_available() is True

    def test_false_after_failed_init(self):
        """is_available returns False when private key is None."""
        signing._initialized = True
        signing._private_key = None
        assert signing.is_available() is False

    def test_auto_initializes(self):
        """is_available auto-initializes if not yet initialized."""
        assert signing._initialized is False
        result = signing.is_available()
        assert result is True
        assert signing._initialized is True


def test_verify_signature_rejects_tampered_data_and_invalid_base64() -> None:
    """The standalone verifier must fail closed for altered or malformed input."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    private_key = Ed25519PrivateKey.generate()
    public_key_pem = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    signature = base64.b64encode(private_key.sign(b"original")).decode("ascii")

    assert signing.verify_signature(public_key_pem, b"original", signature)
    assert not signing.verify_signature(public_key_pem, b"tampered", signature)
    assert not signing.verify_signature(public_key_pem, b"original", "not-base64!")
