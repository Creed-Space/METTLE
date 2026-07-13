// METTLE WebMCP — AI Agent Tool Discovery
// Chrome 145+ experimental API. Websites register tools via navigator.modelContext.registerTool()
// that AI agents discover and call directly. Progressive enhancement — no-op on unsupported browsers.

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

  async function postJSON(path, body, sessionToken) {
    var response = await fetch(path, {
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
    return response.json();
  }

  async function getJSON(path, sessionToken) {
    var response = await fetch(path, {
      headers: sessionToken ? { 'X-Session-Token': sessionToken } : {}
    });
    if (!response.ok) {
      var errBody;
      try { errBody = await response.json(); } catch (_) { errBody = { detail: response.statusText }; }
      throw new Error(errBody.detail || errBody.error || ('HTTP ' + response.status));
    }
    return response.json();
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

  // ---------------------------------------------------------------------------
  // Tool definitions
  // ---------------------------------------------------------------------------

  var tools = [
    // Tool 1: Start verification session
    {
      name: 'mettle_start_verification',
      description: 'Start a new METTLE verification session. Returns the first challenge to answer.',
      inputSchema: {
        type: 'object',
        properties: {
          difficulty: {
            type: 'string',
            enum: ['basic', 'full'],
            description: 'basic = relaxed timing, full = comprehensive profiling'
          },
          entity_id: {
            type: 'string',
            description: 'Optional identifier for the entity being verified'
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

          var result = {
            session_id: data.session_id,
            session_token: data.session_token,
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
      description: 'Submit an answer to the current METTLE challenge. Returns result and next challenge if any.',
      inputSchema: {
        type: 'object',
        properties: {
          session_id: {
            type: 'string',
            description: 'The session ID from mettle_start_verification'
          },
          session_token: {
            type: 'string',
            description: 'The bearer token from mettle_start_verification'
          },
          challenge_id: {
            type: 'string',
            description: 'The challenge ID to answer'
          },
          answer: {
            type: 'string',
            description: 'Your answer to the challenge'
          }
        },
        required: ['session_id', 'session_token', 'challenge_id', 'answer']
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
          }, params.session_token);

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
      description: 'Get the METTLE verification result and signed credential for a completed session.',
      inputSchema: {
        type: 'object',
        properties: {
          session_id: {
            type: 'string',
            description: 'The session ID to get results for'
          },
          session_token: {
            type: 'string',
            description: 'The bearer token from mettle_start_verification'
          }
        },
        required: ['session_id', 'session_token']
      },
      annotations: { readOnlyHint: true },
      execute: async function(params) {
        if (!params || !params.session_id) {
          return errorResult('session_id is required');
        }

        try {
          var data = await getJSON(
            '/api/session/' + encodeURIComponent(params.session_id) + '/result',
            params.session_token
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

          return textResult(result);
        } catch (err) {
          return errorResult('Failed to get result: ' + err.message);
        }
      }
    },

    // Tool 4: Verify a badge
    {
      name: 'mettle_verify_badge',
      description: 'Verify an existing METTLE badge token. Returns whether the badge is valid, who it was issued to, and when it expires.',
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
      annotations: { readOnlyHint: true },
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
          }
          if (payload.verified_at) {
            result.issued_at = payload.verified_at;
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
