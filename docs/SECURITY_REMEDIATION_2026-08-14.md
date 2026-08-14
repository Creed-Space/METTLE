# Security remediation ledger, 2026-08-14

This ledger is the sole finding level accounting record for remediation of the sealed Codex Security deep scan against `d060596184908ba262bfba945adf6b05faa81ac9`. The sealed report SHA-256 is `6be3b74bc278e41b5a445d2fe5bd9033e0be6ccfc2e19e9518c680d17d549235`.

The scan ended at its configured 40-run cap before semantic saturation. It validated 95 candidates: 58 reportable, 34 suppressed, and 3 deferred. Suppressed candidates remain in the sealed scan ledger and are outside this remediation set. A row is complete only when its final disposition names concrete code or configuration evidence and a passing regression or an explicit external proof.

## Status contract

* `Open`: revalidation or implementation remains.
* `Fixed`: the vulnerable path is closed and focused proof passes.
* `No change`: current evidence disproves the reported path, with the proof recorded.
* `External`: repository controls are complete, but a named provider or human gate remains and is not claimed as source proof.

## Reportable findings

| # | Severity | Finding | Cluster | Primary locus | Status | Closure evidence |
|---:|---|---|---|---|---|---|
| 1 | Low | `csf_92e81bc0ba2f81764462046a` Active legacy GET badge verification places a replayable signed credential in the URL path | API and data exposure | `main.py:2248` | Fixed | `main.py` exposes badge verification only as body-bearing POST; `test_legacy_url_token_route_is_absent` and `test_post_valid_badge_keeps_token_out_of_url`. |
| 2 | Medium | `csf_e3e2b2ee28d85a7eb2ed641c` Operator commitments are replayable across sessions, accept malformed contact metadata, receive a fresh unsigned timestamp, and are not self-contained | Credential and identity semantics | `mettle/api_models.py:160` | Fixed | `mettle/api_models.py` rejects the retired operator field; `test_retired_operator_commitment_is_rejected_not_silently_ignored`. |
| 3 | Medium | `csf_1256a103780b296c175ed456` The public MCP transport delegates shared upstream identity, quota, and optional credential authority to anonymous callers | MCP ingress | `mettle/_http.py:136` | Fixed | `mettle/_http.py` authenticates every MCP caller and `mettle/mcp_context.py` carries only that principal upstream; `test_mcp_requires_authoritative_bearer_authentication` and `test_session_token_vault_is_isolated_by_authenticated_principal`. |
| 4 | High | `csf_2fd41ef0adef86079d1594a5` Deterministic solver answers and caller-authored behavioral or governance assertions can earn issuer-signed higher-tier credentials | Credential assurance | `mettle/router.py:272` | Fixed | Self-report suites are credential-ineligible and `mettle/solver.py` is absent; `test_self_report_suites_are_never_tier_evidence` and `test_distribution_package_contains_no_reference_solver`. |
| 5 | Medium | `csf_ef9092f479d746be31e8c128` Normal GitHub Release attestation does not rebind the mutable post-gate artifact bundle to reproducibility receipts | Release supply chain | `.github/workflows/release.yml:35` | Fixed | The release manifest hashes final distributions and public receipts after publication readback; `test_release_manifest_rebinds_final_distributions_to_receipts`. |
| 6 | Low | `csf_a4a3bbd4459bf7a8516533d4` Holder signing service installs mutable unhashed dependency ranges inside the credential-signing trust boundary | Release supply chain | `deploy/holder/render.yaml:1` | Fixed | `deploy/holder/render.yaml` installs `requirements-production.txt` with `--require-hashes`; `test_production_deploy_uses_hashed_lock`. |
| 7 | Medium | `csf_3f09cc96543e29f30247f13f` Fixed Redis session-lock lease can expire during external LLM evaluation and permit concurrent state mutation | Session concurrency | `mettle/router.py:272` | Fixed | `SessionManager.session_transition_lock` renews its token-owned lease and cancels on lost ownership; `test_session_transition_lock_renews_while_body_is_active` and `test_session_transition_lock_cancels_owner_when_lease_is_lost`. |
| 8 | Medium | `csf_e0200f5e31b4d9e4ff7b6092` Daily API-key session quota uses a lost-update-prone read-modify-write across production workers | Quota and abuse controls | `main.py:1708` | Fixed | `database.reserve_api_key_usage` performs one conditional SQL update; `test_parallel_workers_cannot_overbook_final_slot`. |
| 9 | Medium | `csf_fc9f2a5bd668c8ae22f13a39` Public MCP dispatch has no per-caller ingress budget and collapses anonymous callers into shared upstream quota identities | MCP ingress | `mettle/_http.py:136` | Fixed | MCP ingress enforces per-principal minute and concurrency budgets; `test_mcp_enforces_per_caller_request_budget`. |
| 10 | Low | `csf_d2491634f8073589ff5c5a60` Webhook callback paths and queries can expose embedded bearer secrets in application logs | API and data exposure | `main.py:2935` | Fixed | Webhook logs emit bounded event metadata without callback URLs; `test_delivery_logs_never_include_callback_path_or_query_secrets`. |
| 11 | High | `csf_a4f8315d4d86ffaece27d6e0` Shipped JavaScript, Python, and Rust examples automatically solve live quick challenges and retrieve signed credentials | Credential assurance | `examples/python_example.py:18` | Fixed | Python, JavaScript, and Rust examples require respondent-supplied answers; `test_shipped_examples_require_a_respondent_supplied_answer`. |
| 12 | Low | `csf_bebc4c870c3fdd05b36e6700` Red Council security workflow can pass on local heuristics without submitting scenarios to the METTLE verifier | Security test integrity | `.github/workflows/red-council.yml:107` | Fixed | The synthetic Red Council workflow and verdict runners are removed while the threat corpus is retained as non-executable input; `test_retired_pseudo_gates_and_manual_oidc_publisher_are_absent`. |
| 13 | Medium | `csf_42f790e7237904096ce05419` Workflow dispatch version input permits shell command injection in an OIDC-authorized publication job | Release supply chain | `.github/workflows/mcp-registry-publish.yml:7` | Fixed | The manually dispatched MCP publication workflow is removed; `test_retired_pseudo_gates_and_manual_oidc_publisher_are_absent`. |
| 14 | Medium | `csf_5a68288355b807324cd17643` A manually dispatched branch can publish an unreviewed MCP manifest using repository OIDC authority | Release supply chain | `.github/workflows/mcp-registry-publish.yml:6` | Fixed | MCP publication now occurs only inside the immutable tag release; `test_retired_pseudo_gates_and_manual_oidc_publisher_are_absent` and `test_tag_release_reuses_full_ci_on_the_exact_candidate`. |
| 15 | Low | `csf_117bee2b59094584e01a2da4` Credential presentation has a revocation time-of-check to time-of-use window | Credential revocation | `mettle/router.py:617` | Fixed | Presence presentation consumes its nonce then rechecks revocation immediately before acceptance; `test_presentation_rechecks_revocation_after_consuming_challenge`. |
| 16 | Medium | `csf_9b180d6b68efe55e90fe53eb` Main public FastAPI service has no application-level request-body size limit | API and data exposure | `main.py:1595` | Fixed | `RequestBodyLimitMiddleware` caps all HTTP bodies at 1 MiB before parsing; `test_public_api_rejects_oversized_request_bodies`. |
| 17 | High | `csf_9fe4ef42ce4322457a5e00bc` Novel-reasoning iteration scoring can pass a transcript with a completely wrong final round | Novel reasoning integrity | `scripts/engine.py:170` | Fixed | Novel iteration scoring requires final-round accuracy independently of curve shape; `test_completely_wrong_final_novel_round_cannot_pass`. |
| 18 | Low | `csf_58df8e6c9aa532ef3639eeaa` Rust reference credential verification omits required semantics and key-ID binding, and CI never compiles or runs it | Verifier parity | `.github/workflows/ci.yml:119` | Fixed | The Rust verifier enforces the current schema and policy, expiry, issuer key ID, fingerprint, status, and identity binding, and rejects Presence bearer envelopes; CI runs `cargo run --manifest-path examples/Cargo.toml --locked`. |
| 19 | Medium | `csf_ad582bf27a436f5e263867d4` Portable and holder credential acceptance lack authenticated revocation status | Credential revocation | `mettle/vcp.py:318` | Fixed | Portable and holder acceptance require a fresh issuer-signed nonrevoked status receipt; `test_portable_acceptance_requires_fresh_good_signed_status`. |
| 20 | Low | `csf_64b5199c17e9526fe428d653` Release deployment-drift gate omits the holder service and its security-critical settings | Deployment governance | `scripts/check_render_drift.py:255` | Fixed | The Render contract requires and binds `deploy/holder/render.yaml`; `test_additional_holder_blueprint_must_have_a_deployment_binding`. |
| 21 | Low | `csf_8b3fcc808c4ad32b23b9466a` Production rate-limit and admin-auth failure state is split across two workers | Quota and abuse controls | `main.py:1595` | Fixed | Configured production rate-limit and admin-failure state is Redis-authoritative across workers; `test_configured_admin_failure_state_is_shared_in_redis`. |
| 22 | Low | `csf_5a228422aca198c16901ea03` SQLAlchemy exception logging can disclose API keys, webhook secrets, URLs, and other bound security data | Logging and privacy | `main.py:2984` | Fixed | Database failures log operation and exception type only; `test_database_errors_never_log_bound_secret_values`. |
| 23 | Medium | `csf_1927977eac0c97d64909bf7f` Credential issuance remains enabled during PostgreSQL health loss and Redis-authoritative transitions survive shadow-persistence failure | Fail closed persistence | `main.py:1931` | Fixed | Main-app VCP issuance checks retention, PostgreSQL health, and schema. Legacy signed badges commit immutably to PostgreSQL before Redis publication, while unsigned progress retains rollback semantics; `test_unhealthy_dependency_guard_stops_vcp_issuance`, `test_shadow_persistence_failure_rolls_back_redis_and_emits_no_badge`, and `test_redis_failure_after_badge_commit_recovers_the_same_credential`. |
| 24 | Medium | `csf_a527c906e22e094cf637a142` Operator accountability attestations validate only a self-supplied signature and permit unverified contact claims | Credential and identity semantics | `mettle/api_models.py:69` | Fixed | METTLE no longer accepts or countersigns an operator identity or contact claim; `test_retired_operator_commitment_is_rejected_not_silently_ignored`. |
| 25 | Low | `csf_976be3f3953e158434116c6b` The returned operator attestation omits the entity identifier that its signature is supposed to bind | Credential and identity semantics | `mettle/api_models.py:378` | Fixed | The unauthenticated operator-attestation object is removed rather than returned incompletely; `test_retired_operator_commitment_is_rejected_not_silently_ignored`. |
| 26 | Medium | `csf_d747999c31cef20d32205e7b` The release verifier attests that no automatic solver exists while the wheel ships a callable deterministic tier solver | Release supply chain | `scripts/verify_pypi_release.py:144` | Fixed | The package solver module and CLI/MCP auto-solver surfaces are removed; `test_distribution_package_contains_no_reference_solver`, `test_auto_solver_flag_is_removed`, and installed-wheel smoke. |
| 27 | Medium | `csf_8d0b91520985eef5006b0654` Published website and MCP registry guidance encourage high-impact trust based on properties the assurance case says METTLE does not establish | Claims and documentation | `static/index.html:174` | Fixed | Public copy bounds METTLE to probabilistic session evidence and disclaims identity, agency, safety, and governance inference; `test_video_states_no_absolute_certainty_claims` and `test_governance_registry_description_is_explicitly_self_reported`. |
| 28 | Medium | `csf_a78d23af6946ffb5ca48a756` Non-Presence Ed25519 credentials have no revocable identifier or online status path | Credential revocation | `main.py:2342` | Fixed | Schema 1.1 credentials carry a JTI and require authenticated status, while legacy credential semantics are rejected; `test_current_schema_requires_status_and_legacy_schema_is_rejected` and `test_status_is_signed_and_reflects_revocation`. |
| 29 | Medium | `csf_dfe7544270a3541e0cfdf27a` Historical CLI credential verification trusts the claimant-supplied public key | Credential verification | `mettle/cli.py:84` | Fixed | Historical CLI verification requires a trusted keyring and signed status receipt; `test_claimant_supplied_historical_key_is_not_trusted`. |
| 30 | Low | `csf_f25a2a99ad2593a159cc11f6` Production can remain live on process-local security state after its configured PostgreSQL module fails to load | Fail closed persistence | `main.py:1053` | Fixed | A configured database import failure aborts startup; `test_configured_database_import_failure_stops_application_startup`. |
| 31 | Low | `csf_0794af69217162c2458be84d` The ordinary public session route does not enforce the declared free-tier daily cap | Quota and abuse controls | `main.py:1595` | Fixed | The ordinary public start route atomically reserves the anonymous daily budget before allocation; `test_anonymous_daily_quota_rejects_before_session_allocation` and `test_anonymous_daily_quota_allows_a_reserved_session`. |
| 32 | High | `csf_f8b08cd4089c094dae9a4f78` Novel-reasoning session creation discloses future round material | Novel reasoning integrity | `mettle/challenge_adapter.py:830` | Fixed | Novel session creation returns only the current round; `test_novel_session_payload_contains_no_future_round_material`. |
| 33 | Low | `csf_607af35a2bd7a73d299c93e4` Private-data retention failures are swallowed while service continues | Fail closed persistence | `main.py:1005` | Fixed | A purge failure marks retention unhealthy and blocks all new API writes and credential issuance; `test_cleanup_records_database_retention_failure` and `test_retention_authority_failure_blocks_new_private_writes`. |
| 34 | Medium | `csf_bb9babbb75366b0654dba9f4` Public Swagger and ReDoc consoles execute third-party JavaScript without integrity pinning | Browser supply chain | `main.py:1130` | Fixed | FastAPI Swagger UI and ReDoc routes are disabled; `test_third_party_interactive_api_consoles_are_disabled`. |
| 35 | Low | `csf_453b684a627a315c6f6743b3` Active-session quota permanently counts expired members while traffic continues | Quota and abuse controls | `mettle/session_manager.py:145` | Fixed | The atomic active-session reservation prunes expired ZSET members before counting; `test_atomic_rate_reservation_prunes_expired_members_before_counting`. |
| 36 | High | `csf_b043a68e3b7db5dd8bdc3c03` Advertised per-challenge timing is not enforced by the v2 issuer | Timing integrity | `mettle/session_manager.py:245` | Fixed | The API no longer advertises an unenforced subchallenge clock and enforces its server-issued session deadline; `test_v2_payload_does_not_advertise_an_unenforced_subchallenge_clock` and `test_server_issued_time_budget_expires_single_shot`. |
| 37 | Medium | `csf_6d11863a5647b43d9ee3b589` WebMCP serializes replayable session bearer tokens into model-visible results and arguments | WebMCP token handling | `static/webmcp.js:107` | Fixed | WebMCP keeps session bearer tokens in a principal-isolated vault and never serializes them to tool results or arguments; `test_tool_schemas_never_expose_session_bearer_tokens` and `test_session_token_vault_is_isolated_by_authenticated_principal`. |
| 38 | High | `csf_54f410c698cd9bd7fd0dcaaa` Single-shot verification response discloses server-held correct answers | Answer confidentiality | `mettle/router.py:272` | Fixed | Single-shot details contain verdicts without correct answers; `test_single_shot_details_never_disclose_server_answers`. |
| 39 | Low | `csf_58e7a39163cf32f2ce1943ad` Render drift receipt treats all nonempty secret values as equivalent | Deployment governance | `scripts/check_render_drift.py:68` | Fixed | Render drift compares exact approved secret fingerprints and emits only match state; `test_substituted_nonempty_secret_is_detected_without_disclosing_it`. |
| 40 | Medium | `csf_c9060a895023772aa6818967` Default and explicitly selected LLM suites transfer candidate responses to Anthropic without explicit per-session acknowledgement | LLM privacy | `mettle/llm_challenges.py:566` | Fixed | Session creation requires explicit third-party transfer acknowledgement for `llm-dynamic`; `test_llm_suite_requires_explicit_third_party_acknowledgement`. |
| 41 | Low | `csf_bcd411cff62f972f56fc30a2` Validated PostgreSQL configuration can silently resolve to SQLite | Fail closed persistence | `main.py:155` | Fixed | Production settings require PostgreSQL with `sslmode=verify-full`; `test_insecure_production_settings_rejected`. |
| 42 | Low | `csf_162ac0b9519b44540de7a8cb` LLM suite time budget is checked only before external evaluation | Timing integrity | `mettle/session_manager.py:384` | Fixed | LLM results returning after the authoritative deadline expire the session and are not stored; `test_llm_result_returning_after_deadline_is_not_stored`. |
| 43 | Medium | `csf_a796f335e80ae288c0482cac` The model fingerprinting endpoint is publicly callable despite being declared a Pro-only feature | Authorization | `main.py:2638` | Fixed | Fingerprinting authenticates the API key and checks paid-tier features; `test_model_fingerprinting_requires_an_authenticated_paid_tier` with the Pro-tier control `test_fingerprint_endpoint`. |
| 44 | Medium | `csf_d7603a34f1a2b6e80657a80a` The loopback MCP HTTP server has no Host or Origin validation and accepts JSON independently of content type | MCP ingress | `mettle/_http.py:170` | Fixed | MCP HTTP validates loopback bind, Host, Origin, JSON content type, and bounded body before dispatch; `test_mcp_rejects_untrusted_host_origin_and_content_type`. |
| 45 | Medium | `csf_297dee06f1c19b14fc9f22cc` The operator contact commitment protocol recommends and returns a raw unsalted SHA-256 digest of low-entropy email, handle, or legal-identity values, enabling offline dictionary recovery. | Credential and identity semantics | `docs/openapi-v1.json:665` | Fixed | The dictionary-recoverable contact commitment and its OpenAPI schema are removed; `test_retired_operator_commitment_is_rejected_not_silently_ignored`. |
| 46 | Medium | `csf_e2f16c4954daa1b51fbca34a` JavaScript verifier accepts Presence credentials without validating proof-of-possession semantics | Verifier parity | `examples/verify_credential_fixture.js:41` | Fixed | The JavaScript portable verifier rejects Presence bearer envelopes entirely; `npm run check:fixtures` exercises the negative Presence case and six ordinary compatibility cases. |
| 47 | Low | `csf_4b9439f02472fc8625e17092` Red Council response-timing evidence measures local preprocessing rather than respondent or server latency | Security test integrity | `red_team/instrumented_agent.py:262` | Fixed | The misleading local-timing Red Council runner and workflow are removed; `test_retired_pseudo_gates_and_manual_oidc_publisher_are_absent`. |
| 48 | Low | `csf_e5ceab8e0b8b9ac33a8be9b1` The holder obtains externally valid Vault signatures before durably committing the associated one-time state. | Holder atomicity | `mettle/holder_service.py:308` | Fixed | Holder one-time submission state is durably reserved before any external Vault signature; `test_submission_reservation_is_durable_before_external_signing`. |
| 49 | Medium | `csf_42cb2cf1c4ce3031f2059ed1` Render auto-deploys mutable main-branch commits to both credential API and MCP production independently of the repository's tag-bound release evidence gates. | Deployment governance | `.github/workflows/release.yml:3` | Fixed | API and MCP disable auto-deploy and are promoted together only after tag-bound release and drift gates, with rollback on partial failure; `test_render_production_cannot_auto_deploy_mutable_main` and `test_later_service_failure_rolls_back_already_promoted_service`. |
| 50 | Low | `csf_b5b829b00f4641c4dc5ac86c` Internet-facing MCP container inherits a mutable base image and runs as root | Container supply chain | `Dockerfile:11` | Fixed | The MCP image pins its base digest and runs as UID/GID 10001; `test_mcp_container_pins_its_base_and_drops_root`. |
| 51 | Low | `csf_821c64690de912aa592f70bb` An empty MCP bind host bypasses the loopback-only exposure guard | MCP ingress | `mettle/_http.py:58` | Fixed | An empty MCP bind is classified nonloopback and rejected; `test_empty_bind_host_is_never_loopback`. |
| 52 | Low | `csf_cf31c23e1e73ce6c70b25f8b` Unauthenticated session-read requests acquire Redis mutation locks before bearer verification | Authorization | `main.py:1961` | Fixed | Legacy session reads authenticate the bearer before requesting a Redis mutation lock; `test_unauthenticated_read_is_rejected_before_mutation_lock`. |
| 53 | Low | `csf_5200011e45974fd1a2b43b66` Unauthenticated status endpoints disclose detailed operational activity. | API and data exposure | `main.py:1508` | Fixed | Anonymous health and readiness return only coarse status, version, and source revision; `test_public_health_responses_are_coarse`. |
| 54 | Medium | `csf_bfd5e8e37779b9cc1eb6d436` Delayed VCP issuance resets evidence freshness when a completed result is first retrieved | Credential freshness | `mettle/router.py:442` | Fixed | VCP evidence timestamps use server-recorded completion time rather than retrieval time; `test_delayed_vcp_uses_completion_time_for_freshness`. |
| 55 | Medium | `csf_19309bcf0d03d4aedd48b504` Reproducible-build, registry-publication, and Render-drift jobs execute Python artifacts without hash verification | Release supply chain | `.github/workflows/ci.yml:265` | Fixed | Build, release, and Render parser jobs install only hash-locked inputs; `test_tag_release_reuses_full_ci_on_the_exact_candidate` plus a clean `requirements-release-lock.txt` install, `pip check`, and audit. The release lock is generated from committed `requirements-release.in` on the release job's Linux and Python 3.11 platform, so its Linux-only keyring dependencies are also pinned and hashed. |
| 56 | Low | `csf_3142447e88738024613de3de` Credential presentation verification raises an uncaught exception on JSON-valid unhashable suite entries | Credential verification | `mettle/router.py:584` | Fixed | Presentation verification validates suite elements before set conversion and fails closed on unhashable input; `test_unhashable_suite_data_fails_closed`. |
| 57 | Medium | `csf_8659bdc7ad8bd2bedfbed146` Signed credentials place an arbitrary caller-supplied entity identifier beside authenticated claims without an in-band self-asserted marker | Credential and identity semantics | `mettle/api_models.py:160` | Fixed | Signed metadata marks caller identity as `self_asserted` or `self_asserted_by_authenticated_subject`; `test_new_credentials_name_schema_and_suite_policy` and `test_repeated_result_reads_return_same_signed_badge`. |
| 58 | Low | `csf_9776746ebf0caf3904e4e577` Legacy plaintext API keys remain recoverable until each key is successfully used | Secret storage | `database.py:114` | Fixed | Schema migration 3 hashes every pre-migration row under a PostgreSQL advisory lock, including 64-hex plaintext. Runtime uniquely resolves both current and historical double-digest aliases; `test_startup_migrates_then_deletes_legacy_plaintext_key`, `test_v3_double_digest_migration_preserves_v2_logical_keys`, and `test_ambiguous_digest_aliases_fail_closed`. |

## Deferred deployment questions

| # | Candidate | Cluster | Question | Primary locus | Status | Closure evidence |
|---:|---|---|---|---|---|---|
| 59 | `candidate-e5c0141bab12da47` | Datastore transport | Production configuration permits plaintext Redis and PostgreSQL transport for security critical state. | `config.py:138` | External | `config.py` rejects plaintext Redis and PostgreSQL without `sslmode=verify-full`; production probes on 2026-08-14 negotiated verified Redis TLS and PostgreSQL TLS 1.3. Final candidate environment reconciliation and post-deploy recheck remain. |
| 60 | `candidate-0b68dfa49a465c5b` | Redirect integrity | Allowlisted download and provider requests do not revalidate the final redirect origin. | `scripts/verify_pypi_release.py:26` | Fixed | PyPI and Render clients reject redirects before forwarding authorization and verify the final fixed origin; `test_publication_verifier_rejects_redirects_before_following` and `test_render_checker_rejects_redirects_before_forwarding_bearer`. |
| 61 | `candidate-7a4142a8e10e1b9c` | Proxy identity | Client IP controls depend on an unverified proxy trust configuration. | `main.py:1635` | External | `render.yaml` makes Uvicorn proxy trust explicit and limits the service port to Render ingress by provider architecture. A final deployed spoof-resistance probe remains before this infrastructure-dependent question can be closed as live proof. |

## Residual security diff review

The first complete remediation candidate received a second sealed security diff
review. Its report SHA-256 is
`8dd96fc23a3773bb41dc0c6a7f595c2103d35818d5e289c2a0e3e1b39ea8ad46`,
and its snapshot digest is
`codex-security-snapshot/v1:sha256:ffc51baf3397d9e2c7429d387904c8175197b7319fd06a0ad13a825e0b064a7c`.
All 14 reportable residual findings have candidate fixes and focused regression
evidence. The provider-dependent proxy question remains row 61 above until live
proof is obtained.

The resulting pre-release candidate received a third sealed security diff
review. Its report SHA-256 is
`0df2d9d955d9d0505a48e1685186323aab5cba8754e16f501f3f5bbf440bfeea`,
and its snapshot digest is
`codex-security-snapshot/v1:sha256:52bf2dfb4a14a51afd1438b371aba54bebe6358a7115c545a4ee190cb55e4812`.
That review reported one medium and two low findings. R15 and R16 record their
remediation below. Its only deferred question is still the exact-candidate live
proxy proof in row 61.

* **R1, PyPI bytes execute before independent source binding, Fixed.**
  `verify_pypi_release.py` now requires an exact source-bound reproducibility
  receipt and compares names and SHA-256 values before copying, installing, or
  executing public bytes. The release workflow fetches that receipt before the
  OIDC-authorized steps. Proof:
  `test_publication_verifier_loads_source_bound_reproducible_hashes` and
  `test_publication_verifier_rejects_untrusted_reproducibility_receipts`.
* **R2, portable verifiers accept copied Presence envelopes, Fixed.** Generic
  Python, CLI, JavaScript, and Rust acceptance rejects Presence by default. Only
  holder registration invokes the explicit live-holder allowance. Proof:
  `test_copied_presence_envelope_is_not_a_portable_bearer_credential` and the
  seven cross-language credential fixtures.
* **R3, retired credential policy remains accepted, Fixed.** The supported sets
  contain only schema `1.1` and policy `2026-08-14`; missing or legacy versions
  and their exemptions are rejected. Proof:
  `test_current_schema_requires_status_and_legacy_schema_is_rejected` and
  `test_unknown_or_omitted_versions_fail_closed`.
* **R4, schema-v3 migration double-hashes schema-v2 API keys, Fixed.** API-key
  lookup, deletion, duplicate detection, and quota reservation resolve both
  migration aliases and fail closed if both map to separate rows. Proof:
  `test_v3_double_digest_migration_preserves_v2_logical_keys` and
  `test_ambiguous_digest_aliases_fail_closed`.
* **R5, public credential status has unbounded database and signing work,
  Fixed.** The shared application limiter enforces `60/minute` before revocation
  lookup and signing. Proof:
  `test_status_route_bounds_revocation_checks_and_signatures` plus the normal
  signed-status control.
* **R6, Render drift omits holder managed secret files, Fixed.** The contract
  declares all six files. The checker retrieves the fixed-origin provider
  endpoint, compares exact names and approved content fingerprints, and emits no
  secret material. Proof:
  `test_substituted_secret_file_is_detected_without_disclosing_it`.
* **R7, rotating invalid MCP bearers bypass budgets and retain state, Fixed.** A
  global pre-authentication rate budget and bounded principal cardinality apply
  before validator work or caller-bucket creation, with stale-bucket pruning.
  Proof:
  `test_rotating_invalid_bearers_share_a_global_authentication_budget`.
* **R8, production Redis URLs can disable peer verification, Fixed.** Production
  rejects unsafe or duplicate TLS query overrides and passes required
  certificate and hostname verification explicitly to redis-py. Proof:
  `test_redis_tls_query_cannot_downgrade_verification` and the required-settings
  control cases.
* **R9, cross-store compensation can leave a reachable signed legacy badge,
  Fixed.** PostgreSQL is authoritative for a signed badge before Redis
  publication. Issued badges are immutable, and ambiguous Redis failure recovers
  the same credential rather than erasing or replacing it. Proof:
  `test_redis_failure_after_badge_commit_recovers_the_same_credential` and
  `test_issued_badge_is_immutable_and_not_cleared_by_later_progress`.
* **R10, in-flight Render deploy is missing from rollback bookkeeping, Fixed.**
  The current attempt and rollback target are recorded before the deploy POST,
  and every timeout or later failure includes it in rollback. Proof:
  `test_timeout_after_deploy_trigger_rolls_back_the_current_service` and
  `test_later_service_failure_rolls_back_already_promoted_service`.
* **R11, malformed nested challenge objects can produce internal errors,
  Fixed.** Both single-shot and multi-round boundaries reject non-object
  challenge values with a stable failure rather than evaluator exceptions. This
  first fix did not claim to validate evaluator-specific fields inside those
  objects; the third review found and R15 closes that deeper boundary. Proof:
  `test_malformed_nested_answer_shape_fails_closed`,
  `test_verify_malformed_nested_answer_fails_without_500`, and
  `test_submit_round_malformed_nested_answer_returns_400`.
* **R12, absent issuance guard is treated as healthy, Fixed.** Missing,
  malformed, false, and exception-raising guards return `503`; fixture
  applications must install an explicit healthy guard. Proof:
  `test_absent_dependency_guard_stops_vcp_issuance` and
  `test_unhealthy_dependency_guard_stops_vcp_issuance`.
* **R13, WebMCP strips negative identity provenance, Fixed.** Results now retain
  signed `identity_binding` and `entity_id_verified` fields, the tool description
  identifies self-asserted provenance, and the result is annotated as untrusted
  content. Proof: static contract and cache-fingerprint checks.
* **R14, malformed nested credential metadata escapes fail-closed verification,
  Fixed.** The public verifier wraps the strict implementation in a narrow input
  exception boundary and rejects malformed suites, timing, and status metadata.
  Proof: `test_malformed_nested_credential_metadata_fails_closed`.
* **R15, evaluator-specific inner containers permit credential evidence or
  exceptions, Fixed.** Single-shot and novel-reasoning dispatch validates each
  collection and numeric field used by an evaluator, rejects non-finite values,
  and catches only the narrow input-shape exceptions at the scoring boundary.
  A string-valued native response collection now returns `passed=false`, score
  zero, and `credential_eligible=false`; all ten additional reproduced wrong-type
  cases return stable failures. Proof:
  `test_evaluator_specific_wrong_types_fail_closed` in the adapter and the real
  adapter API regression
  `test_verify_native_string_collection_cannot_pass_evidence`.
* **R16, completion releases active quota before durable state, Fixed.** Both
  single-shot and final multi-round transitions persist the completed session
  before removing its active reservation. Injected final-write failures retain
  the reservation and the previously durable session state; a later removal
  failure can only conservatively overcount until expiry. Proof:
  `test_single_shot_persistence_failure_keeps_active_reservation` and
  `test_multi_round_persistence_failure_keeps_active_reservation`.

## Aggregate acceptance gates

* Every row has a terminal disposition and evidence.
* New regressions exercise the real affected boundary and a legitimate control.
* The final candidate passes Ruff lint and format, mypy, Vulture, the complete pytest coverage gate, Bandit, secret scanning, all dependency audits, the security mutation gate, package and clean install checks, static and OpenAPI checks, cross language fixtures, browser tests, and Impeccable design detection.
* The final diff is reviewed against the sealed report after all substantive edits.
* Hosted and production claims identify the immutable commit and are kept separate from local source proof.

Working if: the table has exactly 61 rows, no row remains `Open`, every `Fixed` or `No change` row names repeatable evidence, and external rows state precisely what repository proof exists and what nonlocal fact remains.
