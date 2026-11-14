"""Chrome DevTools Protocol over a tiny stdlib-only WebSocket client.

No third-party packages: RFC 6455 framing (masked client frames, fragmentation,
ping/pong, close) on a plain socket, plus a CDP session that multiplexes
commands and events over one connection in "flattened" target mode.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import socket
import struct
import threading
import time
import urllib.parse
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class CDPError(RuntimeError):
    pass


class WebSocket:
    """Minimal blocking WebSocket client (text/binary/ping/pong/close)."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("ws", "wss"):
            raise ValueError(f"not a websocket url: {url}")
        host, port = u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "wss" else 80)
        sock = socket.create_connection((host, port), timeout=timeout)
        if u.scheme == "wss":
            import ssl

            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        req = f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(4096)
            if not chunk:
                raise CDPError("websocket handshake: connection closed")
            resp += chunk
        head, self._buf = resp.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise CDPError(f"websocket handshake failed: {head[:200]!r}")
        sock.settimeout(None)
        self.sock = sock
        self._send_lock = threading.Lock()
        self.closed = False

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self._buf)))
            if not chunk:
                raise ConnectionError("websocket closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload)) if n < 4096 else _xor_mask(payload, mask)
        with self._send_lock:
            self.sock.sendall(bytes(header) + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode())

    def recv(self) -> str | bytes:
        """Return the next complete text/binary message (handles fragmentation and control frames)."""
        parts: list[bytes] = []
        msg_opcode = None
        while True:
            b1, b2 = self._recv_exact(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if b2 & 0x80 else None
            data = self._recv_exact(n)
            if mask:
                data = _xor_mask(data, mask)
            if opcode == 0x9:  # ping
                self._send_frame(0xA, data)
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode == 0x8:  # close
                self.closed = True
                raise ConnectionError("websocket closed by peer")
            if opcode in (0x1, 0x2):
                msg_opcode = opcode
            parts.append(data)
            if fin:
                payload = b"".join(parts)
                return payload.decode("utf-8", "replace") if msg_opcode == 0x1 else payload

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._send_frame(0x8, struct.pack(">H", 1000))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def _xor_mask(data: bytes, mask: bytes) -> bytes:
    if not data:
        return data
    n = len(data)
    m = int.from_bytes((mask * (n // 4 + 1))[:n], "big")
    return (int.from_bytes(data, "big") ^ m).to_bytes(n, "big")


class CDPConnection:
    """One browser-level CDP connection; commands may target a sessionId (flatten mode)."""

    def __init__(self, ws_url: str) -> None:
        self.ws = WebSocket(ws_url)
        self._id = 0
        self._id_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._listeners: dict[str, list[Callable[[dict, str | None], None]]] = defaultdict(list)
        self._events: queue.Queue = queue.Queue()
        self.alive = True
        threading.Thread(target=self._reader, name="cdp-reader", daemon=True).start()
        threading.Thread(target=self._dispatcher, name="cdp-dispatch", daemon=True).start()

    # -- plumbing -------------------------------------------------------------
    def _reader(self) -> None:
        while self.alive:
            try:
                raw = self.ws.recv()
            except (ConnectionError, OSError):
                self.alive = False
                break
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if "id" in msg and msg["id"] in self._pending:
                self._pending.pop(msg["id"]).put(msg)
            elif "method" in msg:
                self._events.put(msg)
        for q in list(self._pending.values()):
            q.put({"error": {"message": "CDP connection closed"}})
        self._events.put(None)

    def _dispatcher(self) -> None:
        # Events are handled off the reader thread so handlers may issue commands.
        while True:
            msg = self._events.get()
            if msg is None:
                break
            for fn in list(self._listeners.get(msg["method"], ())) + list(self._listeners.get("*", ())):
                try:
                    fn(msg.get("params", {}), msg.get("sessionId"))
                except Exception:  # noqa: BLE001 - a bad handler must not kill the loop
                    pass

    def on(self, method: str, fn: Callable[[dict, str | None], None]) -> None:
        self._listeners[method].append(fn)

    def send(self, method: str, params: dict | None = None, session_id: str | None = None, timeout: float = 30.0) -> dict:
        if not self.alive:
            raise CDPError("CDP connection is closed")
        with self._id_lock:
            self._id += 1
            mid = self._id
        q: queue.Queue = queue.Queue(maxsize=1)
        self._pending[mid] = q
        msg: dict[str, Any] = {"id": mid, "method": method, "params": params or {}}
        if session_id:
            msg["sessionId"] = session_id
        self.ws.send_text(json.dumps(msg))
        try:
            resp = q.get(timeout=timeout)
        except queue.Empty as exc:
            self._pending.pop(mid, None)
            raise CDPError(f"{method}: timed out after {timeout}s") from exc
        if "error" in resp:
            raise CDPError(f"{method}: {resp['error'].get('message')} {resp['error'].get('data', '')}".strip())
        return resp.get("result", {})

    def expect(self, method: str, predicate: Callable[[dict, str | None], bool] | None = None) -> _Waiter:
        """Register interest in an event *before* triggering it; then call .wait()."""
        return _Waiter(self, method, predicate)

    def wait_for(self, method: str, predicate: Callable[[dict, str | None], bool] | None = None, timeout: float = 30.0) -> dict | None:
        w = self.expect(method, predicate)
        try:
            return w.wait(timeout)
        finally:
            w.cancel()

    def close(self) -> None:
        self.alive = False
        self.ws.close()


class _Waiter:
    def __init__(self, conn: CDPConnection, method: str, predicate: Callable[[dict, str | None], bool] | None) -> None:
        self.conn, self.method, self.predicate = conn, method, predicate
        self.q: queue.Queue = queue.Queue(maxsize=1)
        conn.on(method, self._handler)

    def _handler(self, params: dict, sid: str | None) -> None:
        if (self.predicate is None or self.predicate(params, sid)) and self.q.empty():
            self.q.put(params)

    def wait(self, timeout: float = 30.0) -> dict | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def cancel(self) -> None:
        try:
            self.conn._listeners[self.method].remove(self._handler)
        except ValueError:
            pass


def wait_port_file(path: str, timeout: float = 30.0) -> tuple[int, str]:
    """Read Chromium's DevToolsActivePort file -> (port, browser ws path)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path) as f:
                lines = f.read().split("\n")
            if len(lines) >= 2 and lines[0].strip().isdigit() and lines[1].strip():
                return int(lines[0]), lines[1].strip()
        except OSError:
            pass
        time.sleep(0.1)
    raise CDPError("Chromium did not start (no DevToolsActivePort). Run `revapi doctor`.")
