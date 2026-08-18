#!/bin/sh
set -eu

case "${METTLE_MCP_TRANSPORT:-stdio}" in
    stdio)
        exec mettle-mcp
        ;;
    http)
        # A container is an isolated network namespace. The hosted gateway is
        # responsible for authenticating ingress before traffic reaches this
        # otherwise unauthenticated MCP transport.
        export METTLE_MCP_ALLOW_INSECURE_HTTP=true
        exec mettle-mcp --transport http --host 0.0.0.0 --port "${PORT:-8080}"
        ;;
    *)
        printf 'Unsupported METTLE_MCP_TRANSPORT: %s\n' "$METTLE_MCP_TRANSPORT" >&2
        exit 64
        ;;
esac
