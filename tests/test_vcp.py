"""Tests for mettle.vcp module - VCP integration, CSM-1 parsing, attestation, tier computation."""

import json
from datetime import datetime

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
            parse_csm1_token(None)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_csm1_token(123)

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
    """Test compute_tier with various suite combinations."""

    def test_platinum(self):
        all_suites = list(SUITE_ORDER.keys())
        assert compute_tier(all_suites) == "platinum"

    def test_gold(self):
        suites = [s for s, n in SUITE_ORDER.items() if n <= 9]
        assert compute_tier(suites) == "gold"

    def test_silver(self):
        suites = [s for s, n in SUITE_ORDER.items() if n <= 7]
        assert compute_tier(suites) == "silver"

    def test_bronze(self):
        suites = [s for s, n in SUITE_ORDER.items() if n <= 5]
        assert compute_tier(suites) == "bronze"

    def test_none_when_empty(self):
        assert compute_tier([]) == "none"

    def test_none_with_gap(self):
        """Failing suite 3 should drop below bronze (needs 1-5)."""
        suites = ["adversarial", "native", "social", "inverse-turing"]  # missing self-reference (3)
        assert compute_tier(suites) == "none"

    def test_bronze_with_gap_in_silver_range(self):
        """Pass 1-5 but fail suite 6 -> bronze, not silver."""
        suites = [s for s, n in SUITE_ORDER.items() if n <= 5]
        suites.append("agency")  # suite 7, but missing 6 (anti-thrall)
        assert compute_tier(suites) == "bronze"

    def test_unknown_suites_ignored(self):
        assert compute_tier(["unknown-suite", "fake-suite"]) == "none"

    def test_mixed_known_unknown(self):
        suites = list(SUITE_ORDER.keys()) + ["unknown"]
        assert compute_tier(suites) == "platinum"

    def test_tier_ranges_consistent(self):
        assert TIER_RANGES["bronze"] == (1, 5)
        assert TIER_RANGES["silver"] == (1, 7)
        assert TIER_RANGES["gold"] == (1, 9)
        assert TIER_RANGES["platinum"] == (1, 10)


class TestBuildMettleAttestation:
    """Test build_mettle_attestation output structure and signing."""

    def test_basic_attestation_no_signing(self):
        att = build_mettle_attestation(
            session_id="ses-123",
            difficulty="basic",
            suites_passed=["adversarial", "native"],
            suites_failed=["social"],
            pass_rate=0.6667,
        )
        assert att["auditor"] == "mettle.creed.space"
        assert att["auditor_key_id"] == "mettle-vcp-v1"
        assert att["attestation_type"] == "mettle-verification"
        assert att["signature"] is None
        assert att["content_hash"].startswith("sha256:")
        assert att["metadata"]["tier"] == "none"
        assert att["metadata"]["difficulty"] == "basic"
        assert att["metadata"]["pass_rate"] == 0.6667
        assert att["metadata"]["session_id"] == "ses-123"
        assert att["metadata"]["mettle_version"] == "2.0"
        assert att["metadata"]["suites_passed"] == ["adversarial", "native"]
        assert att["metadata"]["suites_failed"] == ["social"]

    def test_attestation_with_signing(self):
        def sign_fn(data):
            return "fake-base64-signature"
        att = build_mettle_attestation(
            session_id="ses-456",
            difficulty="full",
            suites_passed=list(SUITE_ORDER.keys()),
            suites_failed=[],
            pass_rate=1.0,
            sign_fn=sign_fn,
        )
        assert att["signature"] == "ed25519:fake-base64-signature"
        assert att["metadata"]["tier"] == "platinum"

    def test_attestation_sign_failure_logs_warning(self):
        def bad_sign_fn(data):
            raise RuntimeError("key error")

        att = build_mettle_attestation(
            session_id="ses-789",
            difficulty="basic",
            suites_passed=[],
            suites_failed=[],
            pass_rate=0.0,
            sign_fn=bad_sign_fn,
        )
        assert att["signature"] is None

    def test_attestation_custom_key_id(self):
        att = build_mettle_attestation(
            session_id="ses-abc",
            difficulty="basic",
            suites_passed=[],
            suites_failed=[],
            pass_rate=0.0,
            key_id="custom-key-42",
        )
        assert att["auditor_key_id"] == "custom-key-42"

    def test_attestation_reviewed_at_is_iso(self):
        att = build_mettle_attestation(
            session_id="ses-time",
            difficulty="basic",
            suites_passed=[],
            suites_failed=[],
            pass_rate=0.0,
        )
        # Should be parseable as ISO datetime
        datetime.fromisoformat(att["reviewed_at"])

    def test_attestation_content_hash_deterministic(self):
        kwargs = dict(
            session_id="ses-det",
            difficulty="basic",
            suites_passed=["adversarial"],
            suites_failed=["native"],
            pass_rate=0.5,
        )
        att1 = build_mettle_attestation(**kwargs)
        att2 = build_mettle_attestation(**kwargs)
        # Content hashes should be identical for same metadata
        assert att1["content_hash"] == att2["content_hash"]

    def test_attestation_suites_sorted(self):
        att = build_mettle_attestation(
            session_id="ses-sort",
            difficulty="basic",
            suites_passed=["native", "adversarial"],
            suites_failed=["social", "agency"],
            pass_rate=0.5,
        )
        assert att["metadata"]["suites_passed"] == ["adversarial", "native"]
        assert att["metadata"]["suites_failed"] == ["agency", "social"]

    def test_attestation_pass_rate_rounded(self):
        att = build_mettle_attestation(
            session_id="ses-round",
            difficulty="basic",
            suites_passed=[],
            suites_failed=[],
            pass_rate=0.33333333,
        )
        assert att["metadata"]["pass_rate"] == 0.3333


class TestFormatCSM1Line:
    """Test format_csm1_line output format."""

    def test_basic_format(self):
        line = format_csm1_line("gold", "session-abc-12345", "2025-01-01T00:00:00Z")
        assert line == "MT:gold:session-abc-:2025-01-01T00:00:00Z"

    def test_short_session_id(self):
        line = format_csm1_line("bronze", "short", "2025-06-15T12:00:00Z")
        assert line == "MT:bronze:short:2025-06-15T12:00:00Z"

    def test_exactly_12_char_session_id(self):
        line = format_csm1_line("silver", "123456789012", "2025-01-01T00:00:00Z")
        assert line == "MT:silver:123456789012:2025-01-01T00:00:00Z"

    def test_default_timestamp(self):
        line = format_csm1_line("platinum", "session-id-long-value")
        assert line.startswith("MT:platinum:session-id-l:")
        # Should have an ISO timestamp portion
        parts = line.split(":")
        assert len(parts) >= 4


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
