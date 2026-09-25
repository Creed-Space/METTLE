// METTLE WebMCP — AI Agent Tool Discovery
// Chrome 145+ experimental API. Websites register tools via navigator.modelContext.registerTool()
// that Becoming Minds discover and call directly. Progressive enhancement, with no effect on unsupported browsers.

(function() {
  'use strict';

  if (typeof navigator === 'undefined' || !navigator.modelContext || typeof navigator.modelContext.registerTool !== 'function') {
    return;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function textResult(text) {
    return { content: [{ type: 'text', text: typeof text === 'string' ? text : JSON.stringify(text, null, 2) }] };
  }

  function errorResult(message) {
    return { content: [{ type: 'text', text: JSON.stringify({ error: message }) }], isError: true };
  }

  // Network and parser failures become a plain cause; the model never sees
  // browser exception text. The server's own public detail still passes through.
  async function fetchOrExplain(path, options) {
    try {
      return await fetch(path, options);
    } catch (_) {
      throw new Error('METTLE could not be reached');
    }
  }

  async function postJSON(path, body, sessionToken) {
    var response = await fetchOrExplain(path, {
      method: 'POST',
      headers: Object.assign(
        { 'Content-Type': 'application/json' },
        sessionToken ? { 'X-Session-Token': sessionToken } : {}
      ),
      body: JSON.stringify(body)
    });
    if (!response.ok) {
      var errBody;
      try { errBody = await response.json(); } catch (_) { errBody = { detail: response.statusText }; }
      throw new Error(errBody.detail || errBody.error || ('HTTP ' + response.status));
    }
    try { return await response.json(); } catch (_) { throw new Error('METTLE returned a response this page could not read'); }
  }

  async function getJSON(path, sessionToken) {
    var response = await fetchOrExplain(path, {
      headers: sessionToken ? { 'X-Session-Token': sessionToken } : {}
    });
    if (!response.ok) {
      var errBody;
      try { errBody = await response.json(); } catch (_) { errBody = { detail: response.statusText }; }
      throw new Error(errBody.detail || errBody.error || ('HTTP ' + response.status));
    }
    try { return await response.json(); } catch (_) { throw new Error('METTLE returned a response this page could not read'); }
  }

  /**
   * Strip any fields from a challenge object that could leak expected answers.
   */
  function sanitizeChallenge(challenge) {
    if (!challenge || typeof challenge !== 'object') return challenge;
    var clean = {};
    var blocked = ['expected_answer', 'answer', 'correct_answer', 'solution', 'expected'];
    for (var key in challenge) {
      if (challenge.hasOwnProperty(key) && blocked.indexOf(key) === -1) {
        clean[key] = challenge[key];
      }
    }
    return clean;
  }

  // Session bearer credentials stay in this page-local closure. They are never
  // returned as tool content or accepted as model-authored tool arguments.
  var sessionTokens = new Map();
  var SESSION_TOKEN_TTL_MS = 60 * 60 * 1000;
  var MAX_SESSION_TOKENS = 128;

  function rememberSessionToken(sessionId, token) {
    if (typeof sessionId !== 'string' || typeof token !== 'string' || !token) {
      throw new Error('the response had no session ID or token. Retry mettle_start_verification once; if this repeats, stop and report the error.');
    }
    var now = Date.now();
    sessionTokens.forEach(function(record, key) {
      if (record.expiresAt <= now) sessionTokens.delete(key);
    });
    if (sessionTokens.size >= MAX_SESSION_TOKENS) {
      sessionTokens.delete(sessionTokens.keys().next().value);
    }
    sessionTokens.set(sessionId, { token: token, expiresAt: now + SESSION_TOKEN_TTL_MS });
  }

  function sessionTokenFor(sessionId, consume) {
    var record = sessionTokens.get(sessionId);
    if (!record || record.expiresAt <= Date.now()) {
      sessionTokens.delete(sessionId);
      throw new Error('No active session with that ID on this page. It may have expired, or its result was already read. Start a new session with mettle_start_verification.');
    }
    if (consume) sessionTokens.delete(sessionId);
    return record.token;
  }

  // ---------------------------------------------------------------------------
  // Tool definitions
  // ---------------------------------------------------------------------------

  var tools = [
    // Tool 1: Start a quick screening session (the tool name is a stable contract)
    {
      name: 'mettle_start_verification',
      description: 'Start a quick METTLE screening session on this page: a short run of timed challenges that the server scores against a pass threshold. Returns the first challenge. Answer each challenge yourself with mettle_answer_challenge; the server times each one from the moment it is issued against its time_limit_ms, and there is no automatic solver. The session token stays with this page and is never shown to you. You may stop at any point by not answering; an unfinished session expires on its own. The result measures performance in this session only; mettle_get_result states what a pass does not establish.',
      inputSchema: {
        type: 'object',
        properties: {
          difficulty: {
            type: 'string',
            enum: ['basic', 'full'],
            description: 'basic: 3 challenges with 2 to 3 second limits; a pass needs all 3 and is recorded at Bronze tier. full: 5 challenges with 0.4 to 1 second limits; a pass needs 4 of 5 and is recorded at Silver tier.'
          },
          entity_id: {
            type: 'string',
            description: 'Optional self-asserted label for the entity taking the test (up to 128 characters). Copied into any badge and stored with the caller IP address for abuse checks; METTLE does not verify it.'
          }
        },
        required: ['difficulty']
      },
      annotations: { readOnlyHint: false },
      execute: async function(params) {
        if (!params || !params.difficulty) {
          return errorResult('difficulty is required (basic or full)');
        }
        if (params.difficulty !== 'basic' && params.difficulty !== 'full') {
          return errorResult('difficulty must be "basic" or "full"');
        }

        try {
          var body = { difficulty: params.difficulty };
          if (params.entity_id) {
            body.entity_id = params.entity_id;
          }

          var data = await postJSON('/api/session/start', body);
          rememberSessionToken(data.session_id, data.session_token);

          var result = {
            session_id: data.session_id,
            total_challenges: data.total_challenges
          };

          if (data.current_challenge) {
            result.current_challenge = sanitizeChallenge(data.current_challenge);
          }

          return textResult(result);
        } catch (err) {
          return errorResult('Failed to start session: ' + err.message);
        }
      }
    },

    // Tool 2: Answer a challenge
    {
      name: 'mettle_answer_challenge',
      description: 'Submit your answer to the current challenge. Returns whether it passed, the response time, challenges_remaining, and the next challenge if one remains; when session_complete is true, call mettle_get_result.',
      inputSchema: {
        type: 'object',
        properties: {
          session_id: {
            type: 'string',
            description: 'The session ID from mettle_start_verification'
          },
          challenge_id: {
            type: 'string',
            description: 'The id of the current challenge (current_challenge.id or next_challenge.id)'
          },
          answer: {
            type: 'string',
            description: 'Your answer as plain text, in the exact format the challenge prompt asks for'
          }
        },
        required: ['session_id', 'challenge_id', 'answer']
      },
      annotations: { readOnlyHint: false },
      execute: async function(params) {
        if (!params || !params.session_id) {
          return errorResult('session_id is required');
        }
        if (!params.challenge_id) {
          return errorResult('challenge_id is required');
        }
        if (typeof params.answer === 'undefined' || params.answer === null) {
          return errorResult('answer is required');
        }

        try {
          var data = await postJSON('/api/session/answer', {
            session_id: params.session_id,
            challenge_id: params.challenge_id,
            answer: String(params.answer)
          }, sessionTokenFor(params.session_id, false));

          var result = {};

          if (data.result) {
            result.result = {
              passed: data.result.passed,
              response_time_ms: data.result.response_time_ms
            };
          }

          if (data.next_challenge) {
            result.next_challenge = sanitizeChallenge(data.next_challenge);
          }

          if (typeof data.challenges_remaining !== 'undefined') {
            result.challenges_remaining = data.challenges_remaining;
          }

          if (typeof data.session_complete !== 'undefined') {
            result.session_complete = data.session_complete;
          }

          return textResult(result);
        } catch (err) {
          return errorResult('Failed to submit answer: ' + err.message);
        }
      }
    },

    // Tool 3: Get session result
    {
      name: 'mettle_get_result',
      description: 'Get the result of a completed METTLE quick session. Call once, after mettle_answer_challenge reports session_complete: true; after one successful read this page releases the session. Returns whether the session met the timed-challenge pass threshold (verified), the pass rate, the tier, and a signed, time-limited badge if one was issued. badge is null when none was issued, and a tier without a badge is not a credential. The result measures performance in this session only. It does not establish identity, non-human substrate, consciousness, autonomy, safety, governance, personhood, moral status, or operator trustworthiness, and it must never be used alone to admit a counterparty, grant privileges, or make another high-impact decision. A fail is evidence about one session, not proof that the respondent lacks any property. To contest systematic false rejection, use https://github.com/Creed-Space/METTLE/issues/new?template=protocol-appeal.yml; issues are public, so include no tokens, badges, or challenge answers.',
      inputSchema: {
        type: 'object',
        properties: {
          session_id: {
            type: 'string',
            description: 'The session ID from mettle_start_verification'
          },
        },
        required: ['session_id']
      },
      // Not read-only: a successful read releases the page-held session token.
      annotations: { readOnlyHint: false },
      execute: async function(params) {
        if (!params || !params.session_id) {
          return errorResult('session_id is required');
        }

        try {
          var data = await getJSON(
            '/api/session/' + encodeURIComponent(params.session_id) + '/result',
            sessionTokenFor(params.session_id, false)
          );

          var result = {
            verified: !!data.verified,
            screening_passed: !!data.screening_passed,
            assurance: data.assurance,
            credential_eligible: !!data.credential_eligible,
            tier: data.tier || 'none',
            pass_rate: data.pass_rate,
            badge: data.badge || null,
            badge_info: data.badge_info || null
          };

          sessionTokenFor(params.session_id, true);
          return textResult(result);
        } catch (err) {
          return errorResult('Failed to get result: ' + err.message);
        }
      }
    },

    // Tool 4: Verify a badge
    {
      name: 'mettle_verify_badge',
      description: 'Verify a METTLE badge token (JWT) with this METTLE server. A valid badge means a METTLE session passed and the badge is unexpired and unrevoked. The badge is a bearer token: it does not show that the presenter took that session, and it must never be used alone to admit a counterparty, grant privileges, or make another high-impact decision. Any entity identifier is self-asserted provenance, not verified identity. Returns validity, issue and expiry times, revocation status, and any entity identifier with its entity_id_verified flag and identity_binding.',
      inputSchema: {
        type: 'object',
        properties: {
          token: {
            type: 'string',
            description: 'The badge token (JWT) to verify'
          }
        },
        required: ['token']
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: async function(params) {
        if (!params || !params.token) {
          return errorResult('token is required');
        }

        try {
          var data = await postJSON('/api/badge/verify', { token: params.token });

          // Badge fields live in the nested `payload` object (BadgeVerifyResponse)
          var payload = data.payload || {};

          var result = {
            valid: !!data.valid
          };

          if (payload.entity_id) {
            result.entity_id = payload.entity_id;
            result.entity_id_verified = payload.entity_id_verified === true;
            result.identity_binding = payload.identity_binding || 'self_asserted';
          }
          // Current badges carry the issue time as `iat` (seconds); older badges used `verified_at`.
          if (payload.verified_at) {
            result.issued_at = payload.verified_at;
          } else if (typeof payload.iat === 'number') {
            result.issued_at = new Date(payload.iat * 1000).toISOString();
          }
          if (data.expires_at) {
            result.expires_at = data.expires_at;
          }
          if (payload.exp) {
            result.expires_at = result.expires_at || new Date(payload.exp * 1000).toISOString();
          }
          if (data.error) {
            result.error = data.error;
          }
          if (data.revoked) {
            result.revoked = true;
          }

          return textResult(result);
        } catch (err) {
          return errorResult('Failed to verify badge: ' + err.message);
        }
      }
    }
  ];

  // ---------------------------------------------------------------------------
  // Registration
  // ---------------------------------------------------------------------------

  var registrations = [];

  for (var i = 0; i < tools.length; i++) {
    try {
      registrations.push(navigator.modelContext.registerTool(tools[i]));
      console.log('[WebMCP] Registered: ' + tools[i].name);
    } catch (e) {
      console.warn('[WebMCP] Failed: ' + tools[i].name, e);
    }
  }

  // ---------------------------------------------------------------------------
  // Cleanup on page unload
  // ---------------------------------------------------------------------------

  window.addEventListener('beforeunload', function() {
    for (var j = 0; j < registrations.length; j++) {
      try {
        if (registrations[j] && typeof registrations[j].unregister === 'function') {
          registrations[j].unregister();
        }
      } catch (_) {
        // Ignore cleanup errors during page unload
      }
    }
  });
})();
