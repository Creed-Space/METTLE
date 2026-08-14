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
FROM python:3.12.14-slim-trixie@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

WORKDIR /app

# Copy the package and its build metadata only. The image does not need the
# FastAPI app, tests, or static frontend.
COPY --chown=10001:10001 pyproject.toml README.md LICENSE requirements-mcp-lock.txt ./
COPY --chown=10001:10001 mettle ./mettle
COPY --chmod=755 deploy/mcp/entrypoint.sh /usr/local/bin/mettle-mcp-entrypoint

RUN pip install --no-cache-dir --require-hashes -r requirements-mcp-lock.txt && \
    pip install --no-cache-dir --no-deps .

# The server has no reason to retain root authority at runtime. Use a fixed
# numeric identity so the invariant is independent of mutable base-image users.
RUN groupadd --gid 10001 mettle && \
    useradd --uid 10001 --gid 10001 --home-dir /home/mettle \
      --create-home --shell /usr/sbin/nologin mettle && \
    chown -R 10001:10001 /app
USER 10001:10001
ENV HOME=/home/mettle

# Default transport. The hosted HTTP deploy overrides this to "http" via env.
ENV METTLE_MCP_TRANSPORT=stdio
# The HTTP deploy injects $PORT at runtime; expose a sensible default for local runs.
EXPOSE 8080

# The wrapper validates the selected transport and uses exec so signals and
# stdio reach the MCP process directly.
ENTRYPOINT ["/usr/local/bin/mettle-mcp-entrypoint"]
