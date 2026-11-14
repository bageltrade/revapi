"""Terminal output (stderr) and NDJSON event streaming (stdout)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, TextIO

_TTY = sys.stderr.isatty() and not os.environ.get("NO_COLOR")


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s: str) -> str:
    return c("2", s)


def bold(s: str) -> str:
    return c("1", s)


def green(s: str) -> str:
    return c("32", s)


def red(s: str) -> str:
    return c("31", s)


def yellow(s: str) -> str:
    return c("33", s)


def cyan(s: str) -> str:
    return c("36", s)


def _short(v: Any, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


class Reporter:
    """Receives agent events; prints human output to stderr and/or NDJSON to stdout."""

    def __init__(self, quiet: bool = False, json_stream: TextIO | None = None, verbose: bool = False) -> None:
        self.quiet = quiet
        self.stream = json_stream
        self.verbose = verbose or bool(os.environ.get("REVAPI_VERBOSE"))
        self._lock = threading.Lock()
        self.usage: dict[str, Any] = {}
        self.t0 = time.time()

    def emit(self, event: str, **fields: Any) -> None:
        if self.stream:
            with self._lock:
                self.stream.write(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str) + "\n")
                self.stream.flush()

    def log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "usage":
            self.usage = data
            return
        if kind == "thinking_start":
            return
        if kind == "thinking":
            self.emit("thinking", text=data["text"][:500])
            if self.verbose:
                self.log(dim(f"  ∴ {_short(data['text'], 200)}"))
        elif kind == "text":
            self.emit("text", text=data["text"])
            self.log(f"{cyan('●')} {data['text'].strip()}")
        elif kind == "tool_start":
            self.emit("tool_start", name=data["name"], input=data["input"])
            args = data["input"]
            brief = next((args[k] for k in ("url", "ref", "command", "path", "query", "expression", "key", "text", "index") if k in args), "")
            self.log(f"  {yellow('→')} {data['name']} {dim(_short(brief))}")
        elif kind == "tool_end":
            self.emit("tool_end", name=data["name"], is_error=data["is_error"], duration_ms=data["duration_ms"], output=data["output"][:400])
            if data["is_error"]:
                self.log(f"    {red('✗')} {_short(data['output'], 160)}")
            elif self.verbose:
                self.log(dim(f"    {_short(data['output'], 160)}"))

    def footer(self) -> None:
        u = self.usage
        if u and not self.quiet:
            cost = u.get("estimated_cost_usd", 0)
            self.log(dim(f"  tokens in {u.get('input_tokens', 0):,} · out {u.get('output_tokens', 0):,} · requests {u.get('requests', 0)} · ~${cost:.4f} · {time.time() - self.t0:.0f}s"))
