"""Tests for mettle.vcp module - VCP integration, CSM-1 parsing, attestation, tier computation."""

import json
from datetime import datetime
from typing import Any, cast

import pytest
from mettle.vcp import (
    SUITE_ORDER,
    TIER_RANGES,
    VCPTokenClaim,
    _canonical_bytes,
    build_mettle_attestation,
    compute_tier,
    format_csm1_line,
    parse_csm1_token,
    verify_mettle_attestation,
)


class TestVCPTokenClaim:
    """Test VCPTokenClaim dataclass and its properties."""

    def test_constitution_ref_with_version(self):
        claim = VCPTokenClaim(
            version="3.1",
            profile_id="agent-1",
            constitution_id="professional.safe",
            constitution_version="2.0.0",
        )
        assert claim.constitution_ref == "professional.safe@2.0.0"

    def test_constitution_ref_without_version(self):
        claim = VCPTokenClaim(
            version="3.1",
            profile_id="agent-1",
            constitution_id="professional.safe",
        )
        assert claim.constitution_ref == "professional.safe"

    def test_constitution_ref_no_id(self):
        claim = VCPTokenClaim(version="3.1", profile_id="agent-1")
        assert claim.constitution_ref is None

    def test_defaults(self):
        claim = VCPTokenClaim(version="3.1", profile_id="agent-1")
        assert claim.constitution_id is None
        assert claim.constitution_version is None
        assert claim.persona is None
        assert claim.adherence is None
        assert claim.goal is None
        assert claim.extra_lines == {}
        assert claim.raw == ""


class TestParseCSM1Token:
    """Test parse_csm1_token with valid tokens, edge cases, and error handling."""

    def test_full_token(self):
        token = (
            "VCP:3.1:agent-42\n"
            "C:professional.safe.balanced@2.0.0\n"
            "P:advisor:4\n"
            "G:assist:expert:analytical\n"
            "X:none\n"
            "F:none\n"
            "S:none\n"
            "R:none\n"
        )
        claim = parse_csm1_token(token)
        assert claim.version == "3.1"
        assert claim.profile_id == "agent-42"
        assert claim.constitution_id == "professional.safe.balanced"
        assert claim.constitution_version == "2.0.0"
        assert claim.persona == "advisor"
        assert claim.adherence == 4
        assert claim.goal == "assist:expert:analytical"
        assert claim.extra_lines["X"] == "none"
        assert claim.extra_lines["F"] == "none"
        assert claim.extra_lines["S"] == "none"
        assert claim.extra_lines["R"] == "none"

    def test_header_only(self):
        claim = parse_csm1_token("VCP:3.1:agent-1")
        assert claim.version == "3.1"
        assert claim.profile_id == "agent-1"
        assert claim.constitution_id is None
        assert claim.persona is None

    def test_constitution_without_version(self):
        token = "VCP:3.1:agent-1\nC:simple-constitution"
        claim = parse_csm1_token(token)
        assert claim.constitution_id == "simple-constitution"
        assert claim.constitution_version is None

    def test_persona_without_adherence(self):
        token = "VCP:3.1:agent-1\nP:advisor"
        claim = parse_csm1_token(token)
        assert claim.persona == "advisor"
        assert claim.adherence is None

    def test_persona_with_non_numeric_adherence(self):
        token = "VCP:3.1:agent-1\nP:advisor:high"
        claim = parse_csm1_token(token)
        assert claim.persona == "advisor"
        assert claim.adherence is None

    def test_mt_line_stored_in_extra(self):
        token = "VCP:3.1:agent-1\nMT:gold:abc123def456:2025-01-01T00:00:00Z"
        claim = parse_csm1_token(token)
        assert claim.extra_lines["MT"] == "gold:abc123def456:2025-01-01T00:00:00Z"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_csm1_token("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_csm1_token(cast(Any, None))

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_csm1_token(cast(Any, 123))

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_csm1_token("   \n  \n  ")

    def test_invalid_header_raises(self):
        with pytest.raises(ValueError, match="Invalid VCP header"):
            parse_csm1_token("NOT_VCP_TOKEN")

    def test_missing_profile_id_raises(self):
        with pytest.raises(ValueError, match="Invalid VCP header"):
            parse_csm1_token("VCP:3.1")

    def test_lines_without_colon_skipped(self):
        token = "VCP:3.1:agent-1\nno-colon-here\nC:my-const"
        claim = parse_csm1_token(token)
        assert claim.constitution_id == "my-const"

    def test_whitespace_stripped(self):
        token = "  VCP:3.1:agent-1  \n  C:my-const@1.0  \n"
        claim = parse_csm1_token(token)
        assert claim.version == "3.1"
        assert claim.constitution_id == "my-const"
        assert claim.constitution_version == "1.0"

    def test_raw_preserved(self):
        token = "VCP:3.1:agent-1\nC:test@1.0"
        claim = parse_csm1_token(token)
        assert claim.raw == token.strip()

    def test_case_insensitive_prefix(self):
        token = "VCP:3.1:agent-1\nc:my-const@1.0\np:role:5\ng:help"
        claim = parse_csm1_token(token)
        assert claim.constitution_id == "my-const"
        assert claim.persona == "role"
        assert claim.adherence == 5
        assert claim.goal == "help"

    def test_constitution_with_multiple_at_signs(self):
        token = "VCP:3.1:agent-1\nC:ns@scope@2.0"
        claim = parse_csm1_token(token)
        assert claim.constitution_id == "ns@scope"
        assert claim.constitution_version == "2.0"


class TestComputeTier:
    """Tiers require every suite in their contiguous challenge range."""

    @pytest.mark.parametrize(
        ("count", "tier"),
        [
            (0, "none"),
            (4, "none"),
            (5, "bronze"),
            (7, "silver"),
            (9, "gold"),
            (11, "platinum"),
        ],
    )
    def test_contiguous_tier_ranges(self, count, tier):
        suites = list(SUITE_ORDER)[:count]
        assert compute_tier(suites) == tier

    def test_gap_drops_to_previous_complete_tier(self):
        suites = list(SUITE_ORDER)[:9]
        suites.remove("anti-thrall")
        assert compute_tier(suites) == "bronze"

    def test_llm_dynamic_alone_never_earns_a_tier(self):
        assert compute_tier(["llm-dynamic"]) == "none"


class TestBuildMettleAttestation:
    """Only tier-qualifying server results become signed credentials."""

    def test_qualifying_result_is_signed_and_verifiable(self, monkeypatch):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from mettle import signing

        private_key = Ed25519PrivateKey.generate()
        monkeypatch.setattr(signing, "_private_key", private_key)
        monkeypatch.setattr(signing, "_public_key", private_key.public_key())
        monkeypatch.setattr(signing, "_initialized", True)
        att = build_mettle_attestation(
            session_id="ses-123",
            subject_id="test-user",
            entity_id="agent-1",
            difficulty="basic",
            suites_passed=list(SUITE_ORDER),
            suites_failed=[],
            pass_rate=1.0,
        )
        assert att["attestation_type"] == "mettle-verification-credential"
        assert att["signature"].startswith("ed25519:")
        assert att["credential_issued"] is True
        assert att["metadata"]["tier"] == "platinum"
        assert att["metadata"]["assurance"] == "mettle_behavioral_verification"
        assert att["metadata"]["credential_eligible"] is True
        assert att["content_hash"].startswith("sha256:")
        assert datetime.fromisoformat(att["expires_at"]) > datetime.fromisoformat(
            att["reviewed_at"]
        )
        public_key = signing.get_public_key_pem()
        assert public_key is not None
        assert verify_mettle_attestation(att, public_key)

        att["metadata"]["pass_rate"] = 0.1
        assert not verify_mettle_attestation(att, public_key)

    def test_nonqualifying_result_is_unsigned_evidence(self):
        att = build_mettle_attestation(
            session_id="ses-123",
            subject_id="test-user",
            difficulty="basic",
            suites_passed=["adversarial"],
            suites_failed=["native"],
            pass_rate=0.5,
        )
        assert att["attestation_type"] == "mettle-evidence-receipt"
        assert att["signature"] is None
        assert att["credential_issued"] is False
        assert att["metadata"]["credential_eligible"] is False

    @pytest.mark.parametrize(
        "changes",
        [
            {"session_id": ""},
            {"subject_id": ""},
            {"pass_rate": -0.1},
            {"pass_rate": 1.1},
            {"suites_failed": ["adversarial"]},
        ],
    )
    def test_malformed_issuance_inputs_are_rejected(self, changes):
        values: dict[str, Any] = {
            "session_id": "ses-123",
            "subject_id": "test-user",
            "difficulty": "basic",
            "suites_passed": ["adversarial"],
            "suites_failed": [],
            "pass_rate": 1.0,
        }
        values.update(changes)
        with pytest.raises(ValueError):
            build_mettle_attestation(**values)

    def test_content_hash_changes_with_subject(self):
        common: dict[str, Any] = dict(
            session_id="s",
            difficulty="basic",
            suites_passed=[],
            suites_failed=[],
            pass_rate=0.0,
        )
        one = build_mettle_attestation(
            session_id=cast(str, common["session_id"]),
            subject_id="one",
            difficulty=cast(str, common["difficulty"]),
            suites_passed=cast(list[str], common["suites_passed"]),
            suites_failed=cast(list[str], common["suites_failed"]),
            pass_rate=cast(float, common["pass_rate"]),
        )
        two = build_mettle_attestation(
            session_id=cast(str, common["session_id"]),
            subject_id="two",
            difficulty=cast(str, common["difficulty"]),
            suites_passed=cast(list[str], common["suites_passed"]),
            suites_failed=cast(list[str], common["suites_failed"]),
            pass_rate=cast(float, common["pass_rate"]),
        )
        assert one["content_hash"] != two["content_hash"]


class TestFormatCSM1Line:
    def test_none_tier_format(self):
        line = format_csm1_line("none", "session-abc-12345", "2025-01-01T00:00:00Z")
        assert line == "MT:none:session-abc-:2025-01-01T00:00:00Z"

    def test_default_timestamp(self):
        line = format_csm1_line("none", "short")
        assert line.startswith("MT:none:short:")

    @pytest.mark.parametrize("tier", list(TIER_RANGES))
    def test_tiers_are_formatted(self, tier):
        assert format_csm1_line(tier, "session").startswith(f"MT:{tier}:session:")

    def test_unknown_tier_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown METTLE tier"):
            format_csm1_line("diamond", "session")


class TestCanonicalBytes:
    """Test _canonical_bytes deterministic serialization."""

    def test_sorted_keys(self):
        result = _canonical_bytes({"z": 1, "a": 2})
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "z"]

    def test_compact_separators(self):
        result = _canonical_bytes({"key": "val"})
        assert result == b'{"key":"val"}'

    def test_deterministic(self):
        data = {"b": 2, "a": 1, "c": [3, 2, 1]}
        assert _canonical_bytes(data) == _canonical_bytes(data)

    def test_empty_dict(self):
        assert _canonical_bytes({}) == b"{}"
