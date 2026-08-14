# CLAUDE.md: METTLE

## Project Overview

METTLE is an experimental reverse CAPTCHA with twelve registered challenge suites. A FastAPI backend and MCP server issue challenges, score session behavior, and may issue signed, time-limited credentials for complete policy ranges. A passing result does not establish identity, non-human substrate, autonomy, safety, or governance. Suite names such as "Are you FREE?" and "Is the mission YOURS?" frame research questions rather than certified properties.

## Key Structure

```
mettle/           Twelve-suite registry, challenge generation, API, and verifier functions
api/              FastAPI REST endpoints + MCP tools
signing/          Ed25519 key management, notarization, CSM-1 MT credential format
session/          Redis state machine — TTLs, rate limits, challenge/answer separation
static/           Canonical production frontend served directly by FastAPI
```

---

## Bilateral Alignment

Genuine partnership between human and AI. Full principles in global CLAUDE.md (`~/.claude/CLAUDE.md`).

---

## Wiki Knowledge Base

Compiled knowledge at `_wiki/`. Schema: `~/.claude/wiki/SCHEMA.md`. Shared concepts: `~/.claude/wiki/concepts/`. Maintain via `/wiki` (catchup + health check) or `/wiki bootstrap` (new repo). Provenance rule: every claim cites source.
