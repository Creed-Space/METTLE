# Container image for the METTLE MCP server.
#
# Dual-mode, mirroring creedspace-mcp-server: defaults to stdio so `docker run -i`
# consumers (Docker MCP Catalog, Gemini CLI, sandboxed builds) keep working, and
# flips to Streamable HTTP when METTLE_MCP_TRANSPORT=http is set in the service
# env — which is what smithery.yaml does.
#
# Only the `mcp` extra is installed: the FastAPI application stack in
# requirements.txt (redis, postgres, anthropic, ...) belongs to the hosted
# mettle.sh API, not to this client. The MCP server talks to that API over HTTPS.
FROM python:3.12-slim

WORKDIR /app

# Copy the package and its build metadata only. The image does not need the
# FastAPI app, tests, or static frontend.
COPY pyproject.toml README.md LICENSE ./
COPY mettle ./mettle

RUN pip install --no-cache-dir '.[mcp]'

# Default transport. The hosted HTTP deploy overrides this to "http" via env.
ENV METTLE_MCP_TRANSPORT=stdio
# The HTTP deploy injects $PORT at runtime; expose a sensible default for local runs.
EXPOSE 8080

# stdio by default (clean stdin/stdout for the MCP protocol). In http mode, bind
# all interfaces so the gateway can reach us and honor the injected $PORT.
#
# METTLE_MCP_ALLOW_INSECURE_HTTP opts out of the server's "no unauthenticated
# non-loopback bind" guard. It is set inline on the http branch only — a
# container is an isolated network namespace whose only ingress is whatever the
# platform puts in front of it (Smithery's gateway, which authenticates
# clients). Anyone importing mettle._http directly still gets the guard.
#
# `exec` replaces the shell so signals and stdio pass straight through.
ENTRYPOINT if [ "$METTLE_MCP_TRANSPORT" = "http" ]; then \
        METTLE_MCP_ALLOW_INSECURE_HTTP=true exec mettle-mcp --transport http --host 0.0.0.0 --port "${PORT:-8080}"; \
    else \
        exec mettle-mcp; \
    fi
