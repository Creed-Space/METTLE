# MCP automatic-solver hotfix

Date: 2026-08-12

Base: `ab805a70b65f8978297db60c394062eebc0055f3`

## Reason

A read-only handshake against the current public MCP service listed eight tools,
including `mettle_auto_verify`. That tool lets the service answer its own
challenges, so the result cannot demonstrate that the calling client supplied
the answers.

A clean rebuild of the base also resolved MCP 2.0.0 and then crashed because the
base server still used the removed MCP 1 decorator API. Removing only the solver
would therefore create a deployment that fails at startup.

## Bounded scope

This hotfix:

1. removes the reference-solver import, automatic tool, and handler;
2. adapts tool registration to MCP 2;
3. requires MCP 2.x in package and readable requirement metadata;
4. installs the reviewed MCP 2.0.0 hashed lock in the container and hosted CI;
5. allowlists the Docker context and validates the transport entry point;
6. removes the duplicate GitHub deploy-hook workflow so Render auto-deploy is
   the only main-branch deployment authority;
7. updates nonvisual MCP documentation and focused tests.

The authored static site and media are deliberately unchanged because they remain
under explicit visual approval. `static/docs.html` will therefore mention the old
tool until the comprehensive approved candidate replaces it. Runtime integrity
takes priority: the deployed tool list must still contain exactly seven tools and
must reject `mettle_auto_verify` as unknown.

## Acceptance

* Focused unit and transport tests pass under MCP 2.0.0.
* A clean Docker build uses the hashed lock.
* Stdio and Streamable HTTP handshakes list exactly seven tools.
* The built image contains no automatic solver surface.
* Invalid transport selection exits 64.
* No commit, push, deployment, or visual approval is claimed by this document.
