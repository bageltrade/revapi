"""The session view: a numbered ledger of steps, streamed Markdown replies, thinking in the margin, a live
status line, and a client card at the end — modelled on reverse-api-engineer's interactive session,
drawn with the standard library only. NDJSON events go to stdout for --json-stream."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from . import term as T
from .term import AMBER, DIM, HAIRLINE, INK, INK_SOFT, KIND_TINT, MINT, PASTEL_OTHER, PINK, SKY, sty

_TTY = bool(T.COLOR)

VERB_COL = 9
INDENT = 6 + VERB_COL
NOTE_LINES = 3
OUTPUT_PREVIEW = 3
CODE_PREVIEW = 8
KIND_VERB = {"read": "reading the site", "write": "writing the client", "run": "testing the client", "other": "thinking"}
LANG_BY_EXT = {
    ".py": "python",
    ".js": "js",
    ".mjs": "js",
    ".ts": "js",
    ".go": "go",
    ".java": "java",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".psm1": "powershell",
    ".ps1": "powershell",
    ".json": "json",
    ".sh": "bash",
    ".md": "markdown",
}
LANG_NAME = {"python": "python", "js": "javascript", "csharp": "c#", "powershell": "powershell"}

LAST_THINKING: list[str] = [""]  # the latest full reasoning, for /thinking
_ACTIVE: list[Reporter] = []


# -- compatibility helpers used across the package --------------------------------------------------


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s: str) -> str:
    return sty(s, DIM)


def bold(s: str) -> str:
    return sty(s, INK, bold=True) if _TTY else s


def green(s: str) -> str:
    return sty(s, MINT)


def red(s: str) -> str:
    return sty(s, PINK)


def yellow(s: str) -> str:
    return sty(s, AMBER)


def cyan(s: str) -> str:
    return sty(s, SKY)


def accent(mode: str) -> str:
    return T.MODE_COLORS.get(mode, PINK)


def _short(v: Any, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def first_line(v: Any) -> str:
    s = str(v if v is not None else "")
    return next((ln.strip() for ln in s.splitlines() if ln.strip()), "")


@contextlib.contextmanager
def suspended() -> Iterator[None]:
    """Pause the live status line while something else owns the terminal (questions, prompts)."""
    rep = _ACTIVE[-1] if _ACTIVE else None
    if rep and rep.live:
        rep.live.pause()
    try:
        yield
    finally:
        if rep and rep.live:
            rep.live.resume()


# -- how each tool shows as a step ------------------------------------------------------------------


def describe(name: str, a: dict[str, Any]) -> tuple[str, str, str, str | None] | None:
    """(verb, target, kind, url) for a tool call; None hides it."""
    url = a.get("url") if isinstance(a.get("url"), str) else None
    path = a.get("path") if isinstance(a.get("path"), str) else None
    table: dict[str, tuple[str, str, str]] = {
        "browser_navigate": ("navigate", first_line(url or "page"), "read"),
        "browser_snapshot": ("look", "page structure", "read"),
        "browser_click": ("click", first_line(a.get("element") or a.get("ref") or "element"), "read"),
        "browser_type": ("type", first_line(a.get("text") or "text") + (f"  → [{a.get('ref')}]" if a.get("ref") else ""), "read"),
        "browser_press_key": ("key", first_line(a.get("key") or ""), "read"),
        "browser_select": ("select", first_line(a.get("value") or ""), "read"),
        "browser_scroll": ("scroll", str(a.get("direction") or "down"), "read"),
        "browser_back": ("navigate", "back", "read"),
        "browser_evaluate": ("script", first_line(a.get("expression") or "page script"), "read"),
        "browser_wait_for": ("wait", first_line(a.get("text") or (f"{a.get('seconds')}s" if a.get("seconds") else "page")), "read"),
        "browser_find_text": ("find", first_line(a.get("text") or ""), "read"),
        "browser_take_screenshot": ("look", "screenshot", "read"),
        "browser_tabs": ("tabs", "list", "read"),
        "browser_switch_tab": ("tabs", f"switch to {a.get('tab')}", "read"),
        "browser_cookies": ("cookies", "all, incl. HttpOnly", "read"),
        "browser_storage": ("storage", "local + session", "read"),
        "browser_fetch": ("request", f"{(a.get('method') or 'GET').upper()} {url or ''}", "read"),
        "browser_network_requests": ("network", a.get("filter") or ("all requests" if a.get("all") else "api requests"), "read"),
        "browser_network_request": ("request", f"#{a.get('index', '?')}", "read"),
        "browser_close": ("close", "browser · HAR saved", "read"),
        "har_overview": ("traffic", "overview", "read"),
        "har_search": ("search", first_line(a.get("query") or ""), "read"),
        "har_entry": ("request", f"#{a.get('index', '?')}" + (f" · {a.get('part')}" if a.get("part") not in (None, "all") else ""), "read"),
        "read_file": ("read", Path(path).name if path else "file", "read"),
        "write_file": ("write", Path(path).name if path else "file", "write"),
        "append_file": ("append", Path(path).name if path else "file", "write"),
        "edit_file": ("edit", Path(path).name if path else "file", "write"),
        "list_dir": ("list", path or ".", "read"),
        "shell": ("run", first_line(a.get("command") or "command"), "run"),
        "web_search": ("search", first_line(a.get("query") or ""), "read"),
        "web_fetch": ("fetch", first_line(url or ""), "read"),
        "save_items": ("save", f"{len(a.get('items') or [])} items", "write"),
        "recall": ("recall", f"#{a['id']}" if a.get("id") is not None else first_line(a.get("query") or "index"), "other"),
    }
    if name in ("finish", "ask_user"):
        return None
    if name in table:
        verb, target, kind = table[name]
        return verb, target, kind, url
    words = name.removeprefix("browser_").replace("_", " ").split() or ["tool"]
    vals = [v for v in a.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
    return " ".join(words)[:VERB_COL], first_line(vals[0]) if vals else "", "other", url


def first_definition(source: str) -> int:
    pat = r"\s*(async\s+def|def|class|function|func|export\s+(default\s+)?(async\s+)?function|export\s+class|public\s+(static\s+)?(class|async|function))\b"
    for n, line in enumerate(source.splitlines(), 1):
        if re.match(pat, line):
            return n
    return 1


class Reporter:
    """Receives agent events; draws the session on stderr and/or writes NDJSON events to stdout."""

    def __init__(self, quiet: bool = False, json_stream: TextIO | None = None, verbose: bool = False, mode: str = "agent", show_thinking: bool | None = None) -> None:
        self.quiet = quiet
        self.stream = json_stream
        self.verbose = verbose or bool(os.environ.get("REVAPI_VERBOSE"))
        self.show_thinking = (os.environ.get("REVAPI_HIDE_THINKING") is None) if show_thinking is None else show_thinking
        self.mode = mode
        self._lock = threading.RLock()
        self.usage: dict[str, Any] = {}
        self.t0 = time.time()
        self.live = T.Live() if not quiet else None
        self.tty = bool(self.live and self.live.enabled)
        # session state
        self.step = 0
        self.verb = "starting"
        self.kind = "other"
        self._verb_shown = ""
        self._verb_at = 0.0
        self.host = ""
        self.client = ""
        self.model = ""
        self.ctx: tuple[int, int] = (0, 0)
        self.retry: dict[str, Any] | None = None
        self.tool_stream: dict[str, Any] | None = None
        self.turn_t0 = time.time()
        self.last_byte = time.time()
        self.awaiting_model = False
        self.think = ""
        self.think_t0 = 0.0
        self.think_done = True
        self.text_partial = ""
        self.md: T.Markdown | None = None
        self.streamed = False
        self.carded = False
        self.compactions = 0
        self.think_settled = False
        self.summary_shown = False

    # -- output primitives --------------------------------------------------------------------------
    def emit(self, event: str, **fields: Any) -> None:
        if self.stream:
            with self._lock:
                self.stream.write(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str) + "\n")
                self.stream.flush()

    def out(self, *lines: str, full: list[str] | None = None) -> None:
        """Print lines. ``full`` is the unfolded version (all of the reasoning, the whole output), which the
        full-screen session keeps for ctrl+o."""
        if self.quiet:
            return
        if self.live is not None and hasattr(self.live, "add"):
            self.live.add(list(lines), full)
        elif self.live:
            self.live.print(*lines)
        else:
            print("\n".join(lines), file=sys.stderr, flush=True)

    def log(self, msg: str) -> None:
        self.out(msg)

    def _start_live(self) -> None:
        if self.live and self.tty and not self.live.render:
            if self not in _ACTIVE:
                _ACTIVE.append(self)
            self.live.start(self._live_lines)

    def stop(self) -> None:
        if self.live:
            self.live.stop()
        if self in _ACTIVE:
            _ACTIVE.remove(self)

    # -- header / run line ----------------------------------------------------------------------------
    def run_header(self, mode: str, run_id: str, model: str, detail: str = "") -> None:
        self.mode, self.model = mode, model
        a = accent(mode)
        bits = " · ".join(x for x in (model, detail) if x)
        self.out("", f" {sty('*', a, bold=True)} {sty(mode, a, italic=True)}  {sty('run ' + run_id, DIM)}  {sty(bits, INK_SOFT)}", "")

    # -- the live status line ---------------------------------------------------------------------------
    def _flap(self, verb: str) -> str:
        if verb != self._verb_shown:
            self._verb_shown, self._verb_at = verb, time.time()
        settled = int((time.time() - self._verb_at) / 0.035)
        tick = int(time.time() * 30)
        return "".join(ch if i < settled or ch == " " else T.FLAP[(tick + i * 7) % len(T.FLAP)] for i, ch in enumerate(verb))

    def _live_lines(self) -> list[str]:
        w = T.cols() - 1
        lines: list[str] = []
        if self.think and not self.think_done and self.show_thinking:
            wrapped = T.wrap(" ".join(self.think.split()), w - 8)
            for ln in wrapped[-NOTE_LINES:]:
                lines.append(f"      {sty('┊', HAIRLINE)} {sty(ln, DIM, italic=True)}")
        if self.text_partial:
            margin = f" {sty('¶', accent(self.mode))} " if self.md is None or self.md.emitted == 0 else "   "
            lines.append(margin + sty(T.tail_fit(self.text_partial, w - 5), INK) + sty("▍", accent(self.mode)))
        lines.append(self._status_line(w))
        return lines

    def _status_line(self, w: int) -> str:
        tint = KIND_TINT.get(self.kind, PASTEL_OTHER)
        now = time.time()
        if self.retry and now < self.retry["until"]:
            left = self.retry["until"] - now
            verb = f"reconnecting {self.retry['attempt']}/{self.retry['max']} · {left:.0f}s"
            head = f" {sty(T.spinner_frame(), AMBER)} {sty(verb, AMBER, italic=True)}"
        else:
            verb = self.verb
            tail = ""
            silent = now - self.last_byte if self.awaiting_model and not self.retry else 0
            if silent >= 20:
                tint = AMBER
                verb = "provider silent"  # the counter stays outside the split-flap, which restarts on every change
                tail = f" {silent:.0f}s" + (" · composing a tool call" if (self.text_partial or self.streamed or self.think_done) else "")
            elif self.tool_stream:
                ts = self.tool_stream
                verb = f"{ts['verb']} · {T.k(ts['chars'])} chars"
            elif not self.think_done and self.think:
                verb = "thinking"
            head = f" {sty(T.spinner_frame(), tint)} {sty(self._flap(verb), INK, italic=True)}" + (sty(tail, AMBER, italic=True) if tail else "")
        right_bits = [T.elapsed(now - self.t0)]
        tok = (self.usage.get("input_tokens", 0) or 0) + (self.usage.get("output_tokens", 0) or 0)
        if tok:
            right_bits.append(f"{T.k(tok)} tok")
        if self.ctx[1]:
            right_bits.append(f"ctx {min(999, round(100 * self.ctx[0] / self.ctx[1]))}%")
        cost = self.usage.get("estimated_cost_usd") or 0
        if cost:
            right_bits.append(T.money(cost))
        right = sty("  " + " · ".join(right_bits), DIM)
        used = T.vis_len(head) + T.vis_len(right)
        source, sink = self.host or "the site", self.client or "your API"
        track = w - used - len(source) - len(sink) - 6
        if track >= 10:
            offset = int(now * 9)
            tape = "".join(sty("▪", tint, bold=True) if (i - offset) % 7 == 0 else sty("┄", HAIRLINE) for i in range(track))
            mid = f"   {sty(source, INK_SOFT)} {tape} {sty(sink, INK_SOFT)}"
        else:
            mid = " " * max(1, w - used)
        return head + mid + right

    # -- thinking ---------------------------------------------------------------------------------------
    def _settle_thinking(self) -> None:
        """The reasoning is complete: print it in the margin, folded to its first lines."""
        if self.think_done:
            return
        self.think_done = True
        self.think_settled = True
        text = self.think.strip()
        if not text:
            return
        LAST_THINKING[0] = text
        self.emit("thinking", text=text[:4000])
        if not self.show_thinking and not self.verbose:
            return
        secs = max(1, int(time.time() - (self.think_t0 or time.time())))
        w = T.cols() - 1
        paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-*•]|\d+\.)", text) if p.strip()]
        wrapped: list[str] = []
        for p in paras:
            p = re.sub(r"\*\*(.+?)\*\*", r"\1", " ".join(p.split())).replace("`", "")
            wrapped += T.wrap(p, w - 8)
        shown = wrapped if self.verbose else wrapped[:NOTE_LINES]
        bar = sty("┊", HAIRLINE)
        head = [f"      {bar} {sty(f'thought for {secs}s', DIM)}"]
        lines = head + [f"      {bar} {sty(ln, DIM, italic=True)}" for ln in shown]
        full = head + [f"      {bar} {sty(ln, DIM, italic=True)}" for ln in wrapped] + [""]
        if len(shown) < len(wrapped):
            how = "ctrl+o full output" if hasattr(self.live, "add") else "/thinking shows all"
            lines.append(f"      {bar} {sty(f'+{len(wrapped) - len(shown)} lines · {how}', DIM)}")
        self.out(*lines, "", full=full if len(shown) < len(wrapped) else None)

    # -- streamed text -----------------------------------------------------------------------------------
    def _md(self) -> T.Markdown:
        if self.md is None:
            self.md = T.Markdown(lambda: T.cols() - 1, margin="   ", first_margin=f" {sty('¶', accent(self.mode))} ")
        return self.md

    def _text_delta(self, delta: str) -> None:
        self._settle_thinking()
        if not self.tty or self.quiet:
            return  # without a live terminal the finished reply is rendered once, at the end
        self.streamed = True
        self.text_partial += delta
        while "\n" in self.text_partial:
            line, self.text_partial = self.text_partial.split("\n", 1)
            rendered = self._md().feed(line)
            if rendered:
                self.out(*rendered)

    def _text_done(self, text: str) -> None:
        self._settle_thinking()
        self.emit("text", text=text)
        if self.quiet:
            self._reset_text()
            return
        if self.streamed and self.tty:
            self._close_stream()
            self.out("")
        else:
            lines = T.render_markdown(text.strip(), T.cols() - 1, "   ", f" {sty('¶', accent(self.mode))} ")
            if lines:
                self.out("", *lines, "")
        self._reset_text()

    def _close_stream(self) -> None:
        """Render the rest of a streamed reply (the unfinished last line, an open table or code block)."""
        if self.md is None and not self.text_partial:
            return
        rest = (self._md().feed(self.text_partial) if self.text_partial else []) + self._md().finish()
        self.text_partial = ""
        if rest:
            self.out(*rest)

    def _reset_text(self) -> None:
        self.text_partial = ""
        self.md = None
        self.streamed = False

    # -- steps and receipts ---------------------------------------------------------------------------
    def _step(self, name: str, args: dict[str, Any]) -> None:
        found = describe(name, args)
        if found is None:
            return
        verb, target, kind, url = found
        self.kind = kind
        self.verb = "reading traffic" if verb in ("network", "request", "traffic") or name.startswith("har_") else KIND_VERB.get(kind, "working")
        if url:
            self.host = urlparse(url).netloc or self.host
        self.step += 1
        tint = KIND_TINT.get(kind, PASTEL_OTHER)
        room = max(10, T.cols() - INDENT - 3)
        target = " ".join(str(target).split())
        target = target if len(target) <= room else target[: room - 1] + "…"
        shown = T.link(target, url) if url else target
        self.out(f" {sty('*', tint)} {sty(f'{self.step:02d}', DIM)}  {sty(f'{verb:<{VERB_COL}}', tint, bold=True)} {sty(shown, INK_SOFT)}")
        path = args.get("path") if isinstance(args.get("path"), str) else None
        if name in ("write_file", "append_file") and path and isinstance(args.get("content"), str):
            self.client = Path(path).name
            self._code(args["content"], LANG_BY_EXT.get(Path(path).suffix.lower(), ""))
        elif name == "edit_file" and isinstance(args.get("old_text"), str) and isinstance(args.get("new_text"), str):
            self._diff(args["old_text"], args["new_text"], LANG_BY_EXT.get(Path(path or "").suffix.lower(), ""))

    def _code(self, source: str, lang: str) -> None:
        lines = source.splitlines() or [""]
        start = first_definition(source) if len(lines) > CODE_PREVIEW else 1
        shown = lines[start - 1 : start - 1 + (len(lines) if self.verbose else CODE_PREVIEW)]
        w = T.cols() - INDENT - 6
        nw = len(str(start + len(shown)))
        out = []
        for i, ln in enumerate(shown, start):
            out.append(" " * INDENT + sty(f"{i:>{nw}} ", HAIRLINE) + T.truncate(T.highlight(ln.replace("\t", "    "), lang), w))
        full = None
        if len(lines) > len(shown):
            out.append(" " * (INDENT + 2) + sty(f"+{len(lines) - len(shown)} lines · {len(lines)} total", DIM))
            fw = len(str(len(lines)))
            full = [" " * INDENT + sty(f"{i:>{fw}} ", HAIRLINE) + T.highlight(ln.replace("\t", "    "), lang) for i, ln in enumerate(lines, 1)]
        self.out(*out, full=full)

    def _diff(self, old: str, new: str, lang: str) -> None:
        w = T.cols() - INDENT - 4
        rows = [" " * INDENT + sty("- ", PINK, bold=True) + T.truncate(T.highlight(x, lang), w) for x in old.splitlines()]
        rows += [" " * INDENT + sty("+ ", MINT, bold=True) + T.truncate(T.highlight(x, lang), w) for x in new.splitlines()]
        cap = len(rows) if self.verbose else OUTPUT_PREVIEW * 2
        more = len(rows) - cap
        self.out(*(rows[:cap] + ([" " * (INDENT + 2) + sty(f"+{more} lines", DIM)] if more > 0 else [])), full=rows if more > 0 else None)

    def _receipt(self, name: str, output: str, is_error: bool) -> None:
        text = (output or "").strip()
        lines = [re.sub(r"^#+\s+", "", ln.rstrip()) for ln in text.splitlines() if ln.strip()]
        if not is_error:
            if name in ("write_file", "append_file", "edit_file", "finish", "ask_user"):
                return
            if name in ("browser_snapshot", "browser_take_screenshot"):
                lines = [f"{len(lines)} lines captured"] if name == "browser_snapshot" else lines[:1]
            elif name == "read_file":
                lines = [f"{len(lines)} lines"]
            elif name == "recall":
                lines = [f"{len(text):,} chars recalled"] + lines[:1]
        lines = lines or ["done"]
        cap = len(lines) if (self.verbose or is_error and len(lines) <= 8) else OUTPUT_PREVIEW
        room = max(20, T.cols() - INDENT - 4)
        out = []
        for i, ln in enumerate(lines[:cap]):
            ln = T.strip_ansi(ln)
            ln = ln if len(ln) <= room else ln[: room - 1] + "…"
            out.append(" " * INDENT + sty("└ " if i == 0 else "  ", HAIRLINE) + T.receipt_line(ln, PINK if is_error else DIM))
        full = None
        if len(lines) > cap:
            out.append(" " * (INDENT + 2) + sty(f"+{len(lines) - cap} lines", DIM))
            full = [" " * INDENT + sty("└ " if i == 0 else "  ", HAIRLINE) + T.receipt_line(T.strip_ansi(ln), PINK if is_error else DIM) for i, ln in enumerate(lines[:2000])]
        self.out(*out, full=full)

    # -- events from the agent ------------------------------------------------------------------------
    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._handle(kind, data)

    def _handle(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "usage":
            self.usage = data
            return
        if kind == "context":
            self.ctx = (data.get("tokens", 0), data.get("window", 0))
            self.emit("context", tokens=self.ctx[0], window=self.ctx[1])
            return
        if kind in ("thinking_start", "thinking_delta", "text_delta", "tool_delta", "stream_reset"):
            self.last_byte = time.time()
            self.awaiting_model = True
        if kind == "thinking_start":
            self.turn_t0 = time.time()
            self.think, self.think_done, self.think_t0 = "", True, 0.0
            self.think_settled = False
            self._reset_text()
            self.tool_stream = None
            self.retry = None
            self.verb, self.kind = "thinking", "other"
            self._start_live()
            return
        if kind == "thinking_delta":
            if self.think_done:
                self.think, self.think_done, self.think_t0 = "", False, time.time()
            self.think += data["text"]
            self.retry = None
            return
        if kind == "thinking":
            if self.think_done and data.get("text") and not self.think_settled:  # not streamed: show the whole reasoning once
                self.think, self.think_done, self.think_t0 = data["text"], False, self.turn_t0
            self._settle_thinking()
            return
        if kind == "text_delta":
            self.retry = None
            self._text_delta(data["text"])
            return
        if kind == "text":
            self._text_done(data["text"])
            return
        if kind == "tool_delta":
            self._settle_thinking()
            self._close_stream()  # the reply's text is complete once a tool call starts streaming
            name = data.get("name") or ""
            m = re.search(r'"path"\s*:\s*"([^"]+)"', data.get("args") or "")
            target = Path(m.group(1)).name if m else ""
            verb = {"write_file": "writing", "append_file": "appending to", "edit_file": "editing", "shell": "preparing a command", "save_items": "collecting"}.get(
                name, f"calling {name}" if name else "calling a tool"
            )
            self.tool_stream = {"verb": f"{verb} {target}".strip(), "chars": data.get("chars", 0)}
            return
        if kind == "stream_reset":
            if self.text_partial or self.streamed or (self.think and not self.think_done):
                self.out(f"      {sty('↻', AMBER)} {sty('connection dropped mid-reply — asking again', DIM)}")
            self.think, self.think_done = "", True
            self._reset_text()
            self.tool_stream = None
            return
        if kind == "retry":
            self.retry = {**data, "until": time.time() + float(data.get("wait", 0))}
            self.emit("retry", attempt=data["attempt"], max=data["max"], wait=round(float(data["wait"]), 1), reason=data["reason"])
            wait = float(data.get("wait", 0))
            note = f"retry {data['attempt']}/{data['max']} in {wait:.0f}s" if data.get("attempt", 0) > 0 else f"waiting {wait:.0f}s"
            self.out(f"   {sty('◇', AMBER)} {sty(data['reason'], INK_SOFT)}  {sty(note, DIM)}")
            return
        if kind == "compaction_start":
            self._settle_thinking()
            self.verb, self.kind = "compacting context", "other"
            self._start_live()
            return
        if kind == "compaction":
            self.compactions = data.get("count", self.compactions + 1)
            self.ctx = (data["after"], data["window"])
            how = {"summary": "summarized", "trimmed": "old outputs shortened", "dropped": "oldest turns archived"}.get(data.get("method"), "compacted")
            self.emit("compaction", before=data["before"], after=data["after"], window=data["window"], method=data.get("method"))
            line = f"   {sty('◇', AMBER)} {sty('context compacted', INK_SOFT)}  {sty(T.k(data['before']) + ' →', DIM)} {sty(T.k(data['after']), INK, bold=True)}  {sty(f'· {how} · nothing lost', DIM)}"
            self.out(T.truncate(line, T.cols() - 1))
            return
        if kind == "progress":
            self.emit("progress", message=data.get("message", ""))
            self.out(T.truncate(f"   {sty(data.get('message', ''), DIM)}", T.cols() - 1))
            return
        if kind == "interrupted":
            self._settle_thinking()
            self._close_stream()
            self._reset_text()
            self.out(f"      {sty('┊', HAIRLINE)} {sty('interrupted', PINK)} {sty('· everything so far is saved; /resume continues', DIM)}", "")
            self.stop()
            return
        if kind == "tool_start":
            self.awaiting_model = False
            self._settle_thinking()
            self._close_stream()
            self._reset_text()
            self.tool_stream = None
            self.emit("tool_start", name=data["name"], input=data["input"])
            if data["name"] == "finish" and isinstance(data["input"], dict) and data["input"].get("summary"):
                self.summary_shown = True
                self.out(*T.render_markdown(str(data["input"]["summary"]).strip(), T.cols() - 1, "   ", f" {sty('¶', accent(self.mode))} "), "")
            self._step(data["name"], data["input"] if isinstance(data["input"], dict) else {})
            self._start_live()
            return
        if kind == "tool_end":
            self.emit("tool_end", name=data["name"], is_error=data["is_error"], duration_ms=data["duration_ms"], output=data["output"][:400])
            self._receipt(data["name"], data["output"], data["is_error"])
            self.verb, self.kind = "thinking", "other"
            return

    # -- the end of a run -------------------------------------------------------------------------------
    def card(self, path: str | Path | None, model: str = "") -> None:
        """The run's receipt as a boarding pass: the client on the left, a perforated stub with the model."""
        self.stop()
        if self.quiet or not path:
            return
        p = Path(path)
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            return
        self.carded = True
        a = accent(self.mode)
        lang = LANG_NAME.get(LANG_BY_EXT.get(p.suffix.lower(), ""), LANG_BY_EXT.get(p.suffix.lower(), p.suffix.lstrip(".")))
        u = self.usage
        tok = (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)
        spend = f"{T.k(tok)} tokens · {T.money(u.get('estimated_cost_usd') or 0)}"
        size = f"{lang} · {src.count(chr(10)) + 1} lines"
        main = [
            (f"{sty('*', a)} {sty(T.link(p.name, p.resolve().as_uri()), INK, bold=True)} {sty('ready', PINK, italic=True)}", f"* {p.name} ready"),
            (f"  {sty(size, INK_SOFT)}", f"  {size}"),
            (f"  {sty(spend, DIM)}", f"  {spend}"),
        ]
        mdl = (model or self.model or "").split("/")[-1]
        stub = [
            (sty(mdl, INK_SOFT), mdl),
            (sty(f"{u.get('requests', 0)} requests", DIM), f"{u.get('requests', 0)} requests"),
            (sty(T.elapsed(time.time() - self.t0), DIM), T.elapsed(time.time() - self.t0)),
        ]
        main_w = max(len(x[1]) for x in main) + 4
        stub_w = max(len(x[1]) for x in stub) + 4
        if main_w + stub_w + 4 > T.cols():  # narrow phone screen: no box
            self.out("", *(m for m, _ in main), f"  {sty(' · '.join(x[1] for x in stub), DIM)}", f"  {sty(str(p), DIM)}", "")
            return
        legend = " client "
        out = ["", " " + sty("╭─", HAIRLINE) + sty(legend, DIM) + sty("─" * (main_w - len(legend) - 1) + "┬" + "─" * stub_w + "╮", HAIRLINE)]
        for (m, mp), (s, sp) in zip(main, stub, strict=True):
            out.append(" " + sty("│", HAIRLINE) + "  " + m + " " * (main_w - 2 - len(mp)) + sty("┊", HAIRLINE) + "  " + s + " " * (stub_w - 2 - len(sp)) + sty("│", HAIRLINE))
        out.append(" " + sty("╰" + "─" * main_w + "┴" + "─" * stub_w + "╯", HAIRLINE))
        out.append(f"   {sty('└', HAIRLINE)} {sty('in', DIM)} {sty(T.link(str(p.parent), p.parent.resolve().as_uri()), INK_SOFT)}")
        self.out(*out, "")

    def footer(self) -> None:
        self.stop()
        u = self.usage
        if u and not self.quiet and not self.carded:
            cost = u.get("estimated_cost_usd", 0) or 0
            bits = [f"tokens in {u.get('input_tokens', 0):,}", f"out {u.get('output_tokens', 0):,}", f"requests {u.get('requests', 0)}"]
            if cost:
                bits.append(f"~${cost:.4f}")
            if self.compactions:
                bits.append(f"compacted ×{self.compactions}")
            bits.append(T.elapsed(time.time() - self.t0))
            self.out(sty("   " + " · ".join(bits), DIM))
