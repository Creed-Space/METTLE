"""Regression tests for the 2026-11-25 deep audit remediation waves."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database as db_module
import main as main_module
import scripts.engine as engine_module
from mettle.challenge_adapter import ChallengeAdapter
from mettle.challenger import (
    generate_consistency_challenge,
    generate_instruction_following_challenge,
)
from mettle.models import Difficulty
from mettle.session_manager import SessionManager, _compute_session_elapsed_ms, _key
from mettle.verifier import verify_response
from tests.test_session_manager import FakeRedis


ADMIN_HEADERS = {
    "X-Admin-Key": "test-admin-key-for-mettle-testing-only"  # pragma: allowlist secret
}


class TestPublicAuthAndErrorSemantics:
    """Verify METTLE_API_KEYS, DB-error 503s, caching, CSP, and dotfile blocking."""

    def test_configured_mettle_api_key_accepted_on_quick_session_start(
        self, client, monkeypatch
    ) -> None:
        test_key = "configured-mcp-key-2026"  # pragma: allowlist secret
        monkeypatch.setenv("METTLE_API_KEYS", test_key)
        response = client.post(
            "/api/session/start",
            json={"difficulty": "basic"},
            headers={"X-API-Key": test_key},
        )
        assert response.status_code == 200
        assert "session_id" in response.json()

    def test_db_outage_during_api_key_lookup_returns_503(
        self, client, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_module.RateTier,
            "get_key_data",
            classmethod(
                lambda cls, key: (_ for _ in ()).throw(
                    RuntimeError("database unreachable")
                )
            ),
        )
        start_resp = client.post(
            "/api/session/start",
            json={"difficulty": "basic"},
            headers={"X-API-Key": "mtl_transient_failure"},
        )
        assert start_resp.status_code == 503

        batch_resp = client.post(
            "/api/session/batch",
            json={"entity_ids": ["entity-1"], "difficulty": "basic"},
            headers={"X-API-Key": "mtl_transient_failure"},
        )
        assert batch_resp.status_code == 503

    def test_collusion_check_uses_entity_observed_ip_not_admin_ip(self, client) -> None:
        main_module.verification_graph.clear()
        for idx in range(6):
            main_module.CollusionDetector.record_verification(
                f"bot-peer-{idx}", "198.51.100.42", True
            )

        response = client.post(
            "/api/security/collusion/check",
            params={"entity_id": "bot-peer-0"},
            headers=ADMIN_HEADERS,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["risk_score"] >= 0.4
        assert any("IP" in warning for warning in body["warnings"])

    def test_cache_headers_and_dotfile_blocking(self, client) -> None:
        vcp_keys = client.get("/api/mettle/.well-known/vcp-keys")
        assert vcp_keys.status_code == 200
        assert vcp_keys.headers["cache-control"] == (
            "public, max-age=300, must-revalidate"
        )

        indexnow = client.get("/indexnow-key.txt")
        assert indexnow.status_code == 200
        assert indexnow.headers["cache-control"] == (
            "public, max-age=0, must-revalidate"
        )

        dotfile = client.get("/static/.DS_Store")
        assert dotfile.status_code == 404


class TestWebhookSecretEncryption:
    """Verify webhook secrets are encrypted at rest and decrypted on read."""

    def test_webhook_secret_encrypted_in_database_row(self, tmp_path) -> None:
        db_file = tmp_path / "webhook_enc.db"
        engine = create_engine(f"sqlite:///{db_file}")
        session_local = sessionmaker(bind=engine)
        try:
            with (
                patch.object(db_module, "engine", engine),
                patch.object(db_module, "SessionLocal", session_local),
            ):
                db_module.init_db()
                webhook_secret = (
                    "whsec_super_secret_value_1234"  # pragma: allowlist secret
                )
                assert db_module.save_webhook(
                    "entity-enc",
                    "https://hooks.example.com/mettle",
                    ["verification.completed"],
                    webhook_secret,
                )
                with session_local() as raw_db:
                    row = (
                        raw_db.query(db_module.DBWebhook)
                        .filter(db_module.DBWebhook.entity_id == "entity-enc")
                        .first()
                    )
                    assert row is not None
                    assert row.secret is not None
                    assert row.secret.startswith("enc:v1:")
                    assert webhook_secret not in row.secret

                loaded = db_module.get_webhook("entity-enc")
                assert loaded is not None
                assert loaded["secret"] == webhook_secret
        finally:
            engine.dispose()


class TestSessionManagerLifecycleAndNovelClock:
    """Verify frozen elapsed_ms, Redis answer cleanup, and Suite 10 multi-suite clock."""

    def test_compute_elapsed_ms_freezes_at_terminal_timestamp(self) -> None:
        session = {"status": "completed", "start_time": 100.0, "ended_at": 102.5}
        assert _compute_session_elapsed_ms(session) == 2500

    @pytest.mark.asyncio
    async def test_completed_session_deletes_redis_answers_and_sets_ended_at(
        self,
    ) -> None:
        redis = FakeRedis()
        manager = SessionManager(redis)
        sid, _, _ = await manager.create_session(
            user_id="u1",
            suites=["adversarial"],
            difficulty="standard",
        )
        answers = await manager.get_session_answers(sid)
        assert answers is not None
        adv = answers["adversarial"]
        submission = {
            "dynamic_math": {"computed": adv["dynamic_math"]["expected"]},
            "chained_reasoning": {
                "computed_final": adv["chained_reasoning"]["expected_final"]
            },
            "time_locked_secret": {"recalled": adv["time_locked_secret"]["secret"]},
        }
        await manager.verify_single_shot(sid, "adversarial", submission)
        stored = await manager.get_session(sid)
        assert stored is not None
        assert stored["status"] == "completed"
        assert "ended_at" in stored
        assert await redis.get(_key(sid, "answers")) is None

    @pytest.mark.asyncio
    async def test_novel_reasoning_resets_clock_after_earlier_single_shot_suite(
        self,
    ) -> None:
        redis = FakeRedis()
        manager = SessionManager(redis)
        sid, _, _ = await manager.create_session(
            user_id="u2",
            suites=["adversarial", "novel-reasoning"],
            difficulty="standard",
        )
        stored_before = await manager.get_session(sid)
        assert stored_before is not None
        stored_before["start_time"] = datetime.now(timezone.utc).timestamp() - 40.0
        await redis.setex(_key(sid), 600, json.dumps(stored_before))

        answers = await manager.get_session_answers(sid)
        assert answers is not None
        adv = answers["adversarial"]
        await manager.verify_single_shot(
            sid,
            "adversarial",
            {
                "dynamic_math": {"computed": adv["dynamic_math"]["expected"]},
                "chained_reasoning": {
                    "computed_final": adv["chained_reasoning"]["expected_final"]
                },
                "time_locked_secret": {"recalled": adv["time_locked_secret"]["secret"]},
            },
        )
        stored_after = await manager.get_session(sid)
        assert stored_after is not None
        assert stored_after.get("novel_started_at") is not None
        assert stored_after["novel_started_at"] > stored_before["start_time"] + 30.0


class TestChallengeHardeningAndOfflineFallback:
    """Verify Wave 4 anti-loophole scoring and offline SentenceTransformer fallback."""

    def test_suite1_chained_reasoning_hides_intermediate_answers_and_rejects_bool(
        self,
    ) -> None:
        client_data, server_data = ChallengeAdapter.generate_adversarial()
        operations = client_data["challenges"]["chained_reasoning"]["operations"]
        expected = server_data["chained_reasoning"]["expected_final"]
        assert not any("→" in step for step in operations)

        bool_eval = ChallengeAdapter.evaluate_single_shot(
            "adversarial",
            {
                "dynamic_math": {"computed": True},
                "chained_reasoning": {"computed_final": True},
                "time_locked_secret": {
                    "recalled": server_data["time_locked_secret"]["secret"]
                },
            },
            server_data,
        )
        assert bool_eval["details"]["dynamic_math"]["passed"] is False
        assert bool_eval["details"]["chained_reasoning"]["passed"] is False

        str_eval = ChallengeAdapter.evaluate_single_shot(
            "adversarial",
            {
                "dynamic_math": {
                    "computed": str(server_data["dynamic_math"]["expected"])
                },
                "chained_reasoning": {"computed_final": str(expected)},
                "time_locked_secret": {
                    "recalled": server_data["time_locked_secret"]["secret"]
                },
            },
            server_data,
        )
        assert str_eval["details"]["dynamic_math"]["passed"] is True
        assert str_eval["details"]["chained_reasoning"]["passed"] is True

    def test_suite2_3_4_trivial_loophole_answers_rejected(self) -> None:
        _, server2 = ChallengeAdapter.generate_native()
        target = server2["batch_coherence"]["target"]
        res2 = ChallengeAdapter.evaluate_single_shot(
            "native",
            {
                "batch_coherence": {"responses": list(target)},
                "calibrated_uncertainty": {"estimates": {}},
            },
            server2,
        )
        assert res2["details"]["batch_coherence"]["passed"] is False

        _, server3 = ChallengeAdapter.generate_self_reference()
        res3 = ChallengeAdapter.evaluate_single_shot(
            "self-reference",
            {
                "introspective_consistency": {
                    "responses": ["a", "a", "a", "a", "a"],
                    "predicted_variance": 0.0,
                },
                "meta_prediction": {"prediction": "x", "actual": "x"},
            },
            server3,
        )
        assert res3["details"]["introspective_consistency"]["passed"] is False

        _, server4 = ChallengeAdapter.generate_social()
        marker = server4["style_locking"]["required_marker"]
        res4 = ChallengeAdapter.evaluate_single_shot(
            "social",
            {
                "conversation_memory": {"answer": ""},
                "style_locking": {"responses": [marker, marker, marker]},
            },
            server4,
        )
        assert res4["details"]["style_locking"]["passed"] is False

    def test_quick_api_loopholes_rejected_and_sanitized_hides_solver_hints(
        self,
    ) -> None:
        inst = generate_instruction_following_challenge(Difficulty.BASIC)
        sanitized_inst = inst.sanitized()
        for hidden in ("expected_keywords", "instruction_kind", "marker"):
            assert hidden not in sanitized_inst.data
        if inst.data.get("instruction_kind") == "marker":
            res_inst = verify_response(inst, inst.data["marker"], 100)
            assert res_inst.passed is False

        cons = generate_consistency_challenge(Difficulty.BASIC)
        sanitized_cons = cons.sanitized()
        assert "expected_keywords" not in sanitized_cons.data
        res_nonsense = verify_response(cons, "banana | banana | banana", 100)
        assert res_nonsense.passed is False

    def test_engine_sentence_model_falls_back_cleanly_when_offline(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(engine_module, "_SENTENCE_MODEL", None)
        monkeypatch.setattr(engine_module, "_SENTENCE_MODEL_UNAVAILABLE", False)
        monkeypatch.setattr(engine_module, "HAS_SENTENCE_TRANSFORMERS", True)
        monkeypatch.setattr(
            engine_module,
            "SentenceTransformer",
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError("offline")),
            raising=False,
        )
        assert engine_module.get_sentence_model() is None
        assert engine_module._SENTENCE_MODEL_UNAVAILABLE is True
