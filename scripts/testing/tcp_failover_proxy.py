#!/usr/bin/env python3
"""Tiny TCP forwarder used to model a managed Redis stable endpoint in tests."""

from __future__ import annotations

import argparse
import selectors
import socket
import socketserver
import sys
from typing import cast


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[socketserver.BaseRequestHandler],
        backend_address: tuple[str, int],
    ) -> None:
        self.backend_address = backend_address
        super().__init__(server_address, handler)


class ProxyHandler(socketserver.BaseRequestHandler):
    """Forward bytes bidirectionally until either side closes."""

    server: ProxyServer

    def handle(self) -> None:
        try:
            backend = socket.create_connection(self.server.backend_address, timeout=1.0)
        except OSError:
            return
        with backend:
            self.request.setblocking(False)
            backend.setblocking(False)
            selector = selectors.DefaultSelector()
            selector.register(self.request, selectors.EVENT_READ, backend)
            selector.register(backend, selectors.EVENT_READ, self.request)
            try:
                while True:
                    events = selector.select(timeout=5.0)
                    if not events:
                        continue
                    for key, _mask in events:
                        source = cast(socket.socket, key.fileobj)
                        target: socket.socket = key.data
                        try:
                            data = source.recv(65536)
                            if not data:
                                return
                            target.sendall(data)
                        except OSError:
                            return
            finally:
                selector.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend-port", type=int, required=True)
    args = parser.parse_args()
    with ProxyServer(
        ("127.0.0.1", args.listen_port),
        ProxyHandler,
        ("127.0.0.1", args.backend_port),
    ) as server:
        server.serve_forever(poll_interval=0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
