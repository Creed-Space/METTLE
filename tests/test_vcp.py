"""Tests for mettle.vcp module - VCP integration, CSM-1 parsing, attestation, tier computation."""

import base64
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from hypothesis import given, settings, strategies as st
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


def _signed_credential(
    monkeypatch: pytest.MonkeyPatch,
    *,
    presence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Ed25519PrivateKey, str]:
    from mettle import signing

    private_key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(signing, "_private_key", private_key)
    monkeypatch.setattr(signing, "_public_key", private_key.public_key())
    monkeypatch.setattr(signing, "_initialized", True)
    public_key = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    attestation = build_mettle_attestation(
        session_id="semantic-verifier-session",
        subject_id="semantic-subject",
        entity_id="semantic-entity",
        difficulty="standard",
        suites_passed=list(SUITE_ORDER),
        suites_failed=[],
        pass_rate=1.0,
        presence=presence,
    )
    assert attestation["credential_issued"] is True
    return attestation, private_key, public_key


def _resign(attestation: dict[str, Any], private_key: Ed25519PrivateKey) -> None:
    metadata = attestation["metadata"]
    attestation["content_hash"] = (
        "sha256:" + hashlib.sha256(_canonical_bytes(metadata)).hexdigest()
    )
    unsigned = dict(attestation)
    unsigned.pop("signature", None)
    attestation["signature"] = "ed25519:" + base64.b64encode(
        private_key.sign(_canonical_bytes(unsigned))
    ).decode("ascii")


def _freeze_vcp_clock(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Any:
            return now if tz is None else now.astimezone(tz)

    monkeypatch.setattr("mettle.vcp.datetime", FrozenDateTime)


JSON_VALUES = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=64),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(max_size=32), children, max_size=6),
    ),
    max_leaves=24,
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
        assert "no-colon-here" not in claim.extra_lines

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
            {"pass_rate": True},
            {"pass_rate": float("nan")},
            {"difficulty": "unsupported"},
            {"suites_passed": ["adversarial", {}]},
            {"suites_passed": ["adversarial", "adversarial"]},
            {"suites_failed": "native"},
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

    def test_nonfinite_numbers_and_non_objects_are_rejected(self):
        with pytest.raises(ValueError):
            _canonical_bytes({"score": float("nan")})
        with pytest.raises(ValueError, match="object"):
            _canonical_bytes([])  # type: ignore[arg-type]


class TestVerifyMettleAttestationAdversarial:
    @given(JSON_VALUES)
    @settings(max_examples=100, deadline=None)
    def test_verifier_never_raises_for_json_values(self, candidate: Any) -> None:
        assert verify_mettle_attestation(candidate, "not-a-public-key") is False

    def test_verifier_is_total_over_malformed_json_shapes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attestation, _private_key, public_key = _signed_credential(monkeypatch)
        assert verify_mettle_attestation(attestation, public_key) is True
        assert verify_mettle_attestation(None, public_key) is False
        assert verify_mettle_attestation([], public_key) is False
        assert verify_mettle_attestation(attestation, None) is False

        malformed: list[dict[str, Any]] = []
        for field, value in (
            ("suites_passed", [{}]),
            ("suites_failed", "none"),
            ("tier", []),
            ("difficulty", {}),
            ("pass_rate", float("nan")),
            ("entity_id", "e" * 257),
        ):
            candidate = copy.deepcopy(attestation)
            candidate["metadata"][field] = value
            malformed.append(candidate)
        malformed.append({"metadata": {"suites_passed": []}})
        recursive: dict[str, Any] = copy.deepcopy(attestation)
        recursive["metadata"]["recursive"] = recursive
        malformed.append(recursive)

        for candidate in malformed:
            assert verify_mettle_attestation(candidate, public_key) is False

    def test_profile_expectations_are_signed_and_enforced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attestation, private_key, public_key = _signed_credential(monkeypatch)
        assert verify_mettle_attestation(
            attestation,
            public_key,
            expected_subject_id="semantic-subject",
            expected_entity_id="semantic-entity",
            expected_key_id="mettle-vcp-v1",
            expected_difficulty="standard",
        )
        assert not verify_mettle_attestation(
            attestation, public_key, expected_subject_id="another-subject"
        )
        assert not verify_mettle_attestation(
            attestation, public_key, expected_entity_id="another-entity"
        )
        assert not verify_mettle_attestation(
            attestation, public_key, expected_key_id="another-key"
        )
        assert not verify_mettle_attestation(
            attestation, public_key, expected_difficulty="hard"
        )

        for path, value in (
            (("auditor",), "another-auditor"),
            (("auditor_key_id",), "bad key id"),
            (("metadata", "assurance"), "another-profile"),
            (("metadata", "difficulty"), "unsupported"),
        ):
            candidate = copy.deepcopy(attestation)
            target: dict[str, Any] = candidate
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            _resign(candidate, private_key)
            assert verify_mettle_attestation(candidate, public_key) is False

    def test_presence_profile_rejects_boolean_and_incoherent_signed_timing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        holder_key = Ed25519PrivateKey.generate()
        holder_public_key = (
            holder_key.public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            .decode("ascii")
        )
        from mettle.presence import key_fingerprint

        transcript_hash = "sha256:" + "a" * 64
        presence = {
            "protocol": "mettle-presence-v1",
            "public_key_pem": holder_public_key,
            "key_fingerprint": key_fingerprint(holder_public_key),
            "audience": "service.example",
            "credential_jti": "c" * 32,
            "transcript_hash": transcript_hash,
            "sequence": 1,
            "started_at_unix_ms": 1_000,
            "submissions": [
                {
                    "sequence": 1,
                    "action": "suite:adversarial",
                    "response_time_ms": 250,
                    "accepted_at_unix_ms": 1_250,
                    "transcript_hash": transcript_hash,
                }
            ],
        }
        attestation, private_key, public_key = _signed_credential(
            monkeypatch, presence=presence
        )
        assert verify_mettle_attestation(attestation, public_key) is True

        mutations: list[tuple[tuple[Any, ...], Any]] = [
            (("metadata", "proof_of_possession", "sequence"), True),
            (
                (
                    "metadata",
                    "proof_of_possession",
                    "server_timing",
                    "total_elapsed_ms",
                ),
                True,
            ),
            (
                (
                    "metadata",
                    "proof_of_possession",
                    "server_timing",
                    "total_elapsed_ms",
                ),
                249,
            ),
            (
                (
                    "metadata",
                    "proof_of_possession",
                    "server_timing",
                    "submissions",
                    0,
                    "sequence",
                ),
                True,
            ),
            (
                (
                    "metadata",
                    "proof_of_possession",
                    "server_timing",
                    "submissions",
                    0,
                    "response_time_ms",
                ),
                True,
            ),
            (
                (
                    "metadata",
                    "proof_of_possession",
                    "server_timing",
                    "submissions",
                    0,
                    "action",
                ),
                "suite:",
            ),
            (("metadata", "audience"), "a" * 257),
            (
                ("metadata", "proof_of_possession", "public_key_pem"),
                "p" * 4097,
            ),
        ]
        for path, value in mutations:
            candidate = copy.deepcopy(attestation)
            target: Any = candidate
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            _resign(candidate, private_key)
            assert verify_mettle_attestation(candidate, public_key) is False

    def test_time_boundaries_are_explicit_and_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
        _freeze_vcp_clock(monkeypatch, now)
        attestation, private_key, public_key = _signed_credential(monkeypatch)

        exact = copy.deepcopy(attestation)
        exact["reviewed_at"] = (now + timedelta(minutes=5)).isoformat()
        exact["expires_at"] = (now + timedelta(minutes=65)).isoformat()
        _resign(exact, private_key)
        assert verify_mettle_attestation(exact, public_key) is True

        outside_skew = copy.deepcopy(exact)
        outside_skew["reviewed_at"] = (
            now + timedelta(minutes=5, microseconds=1)
        ).isoformat()
        outside_skew["expires_at"] = (
            now + timedelta(minutes=65, microseconds=1)
        ).isoformat()
        _resign(outside_skew, private_key)
        assert verify_mettle_attestation(outside_skew, public_key) is False

        overlong = copy.deepcopy(attestation)
        overlong["reviewed_at"] = now.isoformat()
        overlong["expires_at"] = (now + timedelta(hours=1, microseconds=1)).isoformat()
        _resign(overlong, private_key)
        assert verify_mettle_attestation(overlong, public_key) is False

        expired = copy.deepcopy(attestation)
        expired["reviewed_at"] = (now - timedelta(hours=1)).isoformat()
        expired["expires_at"] = now.isoformat()
        _resign(expired, private_key)
        assert verify_mettle_attestation(expired, public_key) is False

        non_utc = copy.deepcopy(attestation)
        non_utc["reviewed_at"] = "2026-08-20T13:00:00+01:00"
        non_utc["expires_at"] = "2026-08-20T14:00:00+01:00"
        _resign(non_utc, private_key)
        assert verify_mettle_attestation(non_utc, public_key) is False
