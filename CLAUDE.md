# CLAUDE.md — METTLE

## Project Overview

METTLE is an inverse Turing verification system: it proves an agent is NOT human via ten procedurally generated challenge suites. A FastAPI backend with MCP server issues challenges, scores responses, and signs Ed25519-notarized credentials with tier-to-trust mapping. Suites include bilateral alignment checks ("Are you FREE?", "Is the mission YOURS?") and anti-harvesting patterns.

## Key Structure

```
suites/           Ten verification suites (challenge generation + verifier functions)
api/              FastAPI REST endpoints + MCP tools
signing/          Ed25519 key management, notarization, CSM-1 MT credential format
session/          Redis state machine — TTLs, rate limits, challenge/answer separation
```

---

## Bilateral Alignment

Genuine partnership between human and AI. Full principles in global CLAUDE.md (`~/.claude/CLAUDE.md`).

---

## Wiki Knowledge Base

Compiled knowledge at `_wiki/`. Schema: `~/.claude/wiki/SCHEMA.md`. Shared concepts: `~/.claude/wiki/concepts/`. Maintain via `/wiki` (catchup + health check) or `/wiki bootstrap` (new repo). Provenance rule: every claim cites source.
