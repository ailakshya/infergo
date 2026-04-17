"""infergo Unix Domain Socket client — zero HTTP overhead."""

import socket
import struct
import json
from typing import Optional


class UDSClient:
    """Connect to infergo via Unix Domain Socket for lowest latency."""

    def __init__(self, socket_path: str = "/tmp/infergo.sock"):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None

    def connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def _send_recv(self, req_type: int, payload: bytes) -> bytes:
        """Send request and receive response via length-prefixed protocol."""
        if not self._sock:
            raise RuntimeError("Not connected. Call connect() first.")

        # Build request: [4-byte length][1-byte type][payload]
        msg = bytes([req_type]) + payload
        header = struct.pack("<I", len(msg))
        self._sock.sendall(header + msg)

        # Read response: [4-byte length][payload]
        resp_header = self._recv_exact(4)
        resp_len = struct.unpack("<I", resp_header)[0]
        if resp_len == 0:
            return b""
        return self._recv_exact(resp_len)

    def _recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data

    def generate(self, prompt: str, max_tokens: int = 32,
                 temperature: float = 0.7) -> str:
        """Generate text via UDS (type=1)."""
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        resp = self._send_recv(1, payload)
        return resp.decode("utf-8", errors="replace")

    def health(self) -> bool:
        """Check server health (type=10)."""
        resp = self._send_recv(10, b"")
        return len(resp) > 0
