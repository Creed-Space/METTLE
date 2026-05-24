# METTLE — Wiki Log

## [2026-05-23] bootstrap | Initial wiki creation
Pages created: verification-suites (systems), inverse-turing-concept (domain)
Sources ingested: README.md, main.py, mcp_server.py, mettle/ directory listing

## [2026-05-23] expand | 2 more pages: challenge-generation, integration-and-deployment
Pages created: systems/challenge-generation, flows/integration-and-deployment
Sources ingested: mettle/challenger.py:1-80; docs/VERIFICATION_SUITES.md (full 330 lines); docs/VCP_INTEGRATION.md:1-80; docs/METTLE_VERIFICATION_SYSTEM.md:1-80; examples/python_example.py:1-60; tests/ directory listing
Key findings: All 9 suites fully documented with challenge types, time limits, pass criteria; VCP integration adds 2 challenges to Suite 9; tier computation is strictly sequential (gap = drop); anti-harvest design: verifier withholds expected answers on failure; cryptographic randomness prevents seed-prediction attacks

## [2026-05-23] expand | 2 additional pages covering session manager and verifier
Pages created: systems/session-manager-redis, systems/verifier-functions
Sources ingested: mettle/session_manager.py:1-80, mettle/verifier.py:1-80, mettle/challenge_adapter.py:1-27
Key findings: Session state machine CREATED→CHALLENGES_GENERATED→IN_PROGRESS→COMPLETED/EXPIRED/CANCELLED; active TTL=300s, completed TTL=3600s; MAX_ACTIVE_SESSIONS_PER_USER=5, MAX_SESSIONS_PER_HOUR=100; anti-harvesting confirmed present in verifier (correct answers withheld on failure); verify_token_prediction uses containment match not exact match; verify_chained_reasoning reveals chain on pass (audit trail); vcp_token param enables VCP-linked sessions

## [2026-05-23] expand | 3 additional pages covering signing, MCP/API, anti-thrall
Pages created: systems/signing-and-credentials, systems/mcp-server-and-api, domain/anti-thrall-and-agency
Sources ingested: mettle/signing.py (full); mettle/vcp.py (full); README.md (full); mettle/verifier.py:1-37; mettle/ directory listing; tests/ directory listing
Key findings: Tier computation is cumulative sequential (gaps drop tier); ephemeral keys regenerate on restart; verifier withholds expected answers on failure (anti-harvest design).
Coverage expanded from 2 → 5 pages.
