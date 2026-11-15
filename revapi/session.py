"""The full-screen session: the conversation above an input bar fixed to the bottom.

RAE's interactive session, ported to the Python standard library so it runs anywhere revapi does, Termux
included: no prompt_toolkit, no rich. Every run streams into the conversation through the same Reporter the
plain CLI uses; a run works in a worker thread while the main thread reads keys and draws.

What it has, as RAE does:
- the turning asterisk header, the ledger, the split-flap status line above the bar
- a / command menu that narrows as you type; tab completes, ↑↓ move, enter runs
- pickers for /model (searchable, loaded from the provider), /effort, /mode, /language, /settings,
  /history (details, then resume / re-engineer / run), /resume (replays the conversation, then continues it)
- shift+tab next mode, ctrl+e next effort, ctrl+o everything unfolded in a pager, esc stops the run or closes a
  list, pageup/pagedown or the mouse wheel scroll, end jumps back to the latest
- type while a run works: enter queues the prompt; a finished run waits for a follow-up until /new
- the agent's questions (and any prompt a run asks) appear in the bar, options as a list
- click a link or a path to open it; drag over the conversation to select text, copied when you let go
- tap an item in a list to pick it (made for touch screens)
- on exit the session is printed into the terminal's history, like a normal command's output
"""

from __future__ import annotations

import base64
import builtins
import codecs
import contextlib
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__, modes, store
from . import term as T
from .llm import PROVIDERS, LLMError
from .term import AMBER, CHIP_INK, CHIP_LAVENDER, CHIP_YELLOW, DIM, HAIRLINE, INK, INK_SOFT, KIND_TINT, MINT, MODE_COLORS, PASTEL_OTHER, PASTEL_WRITE, PINK, SKY, sty
from .ui import LAST_THINKING, Reporter

MODES = {
    "agent": "the agent browses the site and writes the client",
    "manual": "you browse, revapi records the traffic",
    "engineer": "re-engineer a past run",
    "collector": "collect data into json and csv",
}
EFFORTS = ("auto", *store.EFFORTS)
EFFORT_NOTES = {
    "auto": "the model's own default",
    "low": "fastest, fewer tool calls",
    "medium": "lighter reasoning",
    "high": "deep reasoning on hard captures",
    "xhigh": "deeper still (Anthropic budget 32k; high elsewhere)",
    "max": "the most reasoning the model allows",
}
COMMANDS = {
    "/new": "start a new capture",
    "/model": "choose the model",
    "/effort": "how hard the agent thinks",
    "/mode": "choose the mode",
    "/language": "the client's language",
    "/docs": "write OpenAPI docs instead of a client (toggle)",
    "/mcp": "write an MCP server for the site instead of a client (toggle)",
    "/settings": "all settings",
    "/history": "past runs: their details, resume or re-engineer",
    "/resume": "a past run's conversation, to continue it",
    "/engineer": "re-engineer a past run (/engineer <run_id> [instructions])",
    "/run": "run the latest client (/run [run_id] [args])",
    "/thinking": "the model's full reasoning from the last reply",
    "/context": "how full the context window is",
    "/compact": "summarize the conversation now (nothing is lost)",
    "/models": "list the provider's models (/models <filter>)",
    "/output": "everything from this session, unfolded",
    "/open": "open the latest client",
    "/clear": "clear the conversation and start over",
    "/help": "what revapi does, its modes, commands and keys",
    "/exit": "quit revapi",
}
KEYS = (
    ("enter", "send · answers the run that waits for a follow-up"),
    ("esc", "stop the run · close a list"),
    ("↑ ↓", "earlier prompts · move in a list, around from either end"),
    ("tab", "complete a command"),
    ("shift+tab", "next mode"),
    ("ctrl+e", "next effort"),
    ("ctrl+o", "everything from this session, unfolded"),
    ("pageup pagedown", "scroll, as does the mouse wheel · end jumps to the latest"),
    ("click", "open a file or a URL · pick an item in a list"),
    ("drag", "select text, copied when you let go"),
    ("ctrl+c", "stop the run · clear the bar · quit"),
    ("ctrl+d", "quit"),
)
ALIASES = {"/quit": "/exit", "/q": "/exit", "/commands": "/help"}
PICKER_COMMANDS = {"/model", "/effort", "/mode", "/history", "/resume", "/settings", "/language"}
HISTORY_RUNS = 20
MENU_ROWS = 10
TOP_PADDING = 1
LOGO_COLS, LOGO_ROWS = 24, 11
LOGO_SHADES = ["#ff5c9c", "#ff3d8b", "#e8357c", "#c92f6c", "#a8295c", "#86234d"]
GAUGE_EMPTY = "#a89fba"
URL_IN_PROMPT = re.compile(r"https?://([^/\s]+)|\b((?:[\w-]+\.)+(?:com|io|co|ch|de|fr|uk|org|net|dev|ai|app|so|sh|eu|jobs|ng|me|tv|gg|xyz))\b")
URL_RE = re.compile(r"https?://[^\s'\"<>]+|\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|app|dev|co|xyz|me|tv|gg|so|sh|ng|uk|de|fr)(?:/[^\s]*)?\b", re.I)
SETTINGS = (
    ("provider", "provider", "choice"),
    ("model", "model", "model"),
    ("effort", "effort", "choice"),
    ("engine", "engine", "choice"),
    ("output_language", "output language", "choice"),
    ("output_mode", "output", "choice"),
    ("output_dir", "output directory", "text"),
    ("real_time_sync", "real-time sync", "toggle"),
    ("headless", "headless browser", "toggle"),
    ("stream", "streaming", "toggle"),
    ("show_thinking", "show thinking", "toggle"),
    ("retries", "retries per request", "text"),
    ("context_window", "context window", "text"),
    ("compact_at", "compact at", "text"),
    ("max_turns", "max turns", "text"),
    ("shell_timeout", "shell timeout (s)", "text"),
    ("max_tokens", "max reply tokens", "text"),
    ("collector_model", "collector model", "text"),
    ("api_key", "api key", "secret"),
    ("base_url", "base url", "text"),
    ("proxy", "proxy", "text"),
    ("keep_session", "keep browser session", "toggle"),
    ("handoff", "human checks", "choice"),
    ("handoff_timeout", "live view wait (s)", "text"),
    ("user_data_dir", "browser profile", "text"),
    ("hermes_model", "hermes model", "text"),
    ("hermes_provider", "hermes provider", "text"),
    ("hermes_max_turns", "hermes max turns", "text"),
)
LABELS = {k: label for k, label, _ in SETTINGS}
KINDS = {k: kind for k, _, kind in SETTINGS}
CHOICES = {
    "provider": [("auto", "auto", "detect from your API keys")] + [(p, p, "") for p in PROVIDERS],
    "effort": [("", "auto", "the model's default")] + [(e, e, EFFORT_NOTES[e]) for e in store.EFFORTS],
    "engine": [("revapi", "revapi", "the built-in agent, any provider"), ("hermes", "hermes", "the local Hermes CLI engineers saved captures")],
    "output_language": [(lang, lang, "") for lang in store.LANGUAGES],
    "output_mode": [("client", "client", "a working API client"), ("docs", "docs", "an OpenAPI 3.0 spec"), ("mcp", "mcp", "an MCP server with the tools you ask for")],
    "handoff": [
        ("auto", "auto", "live view when a terminal is attached"),
        ("live", "live", "always open the live view so you can pass the check"),
        ("off", "off", "just report the check and stop"),
    ],
}
SUGGESTIONS = {
    "output_dir": [("", "default", "~/.revapi/runs")],
    "hermes_model": [("", "the Hermes config's own model", "")],
    "hermes_provider": [("", "the Hermes config's own provider", "")],
    "context_window": [("", "auto", "known size for the model")],
}


# ----------------------------------------------------------------------------- helpers


def available() -> bool:
    """The full-screen session needs a real terminal (and termios); otherwise the plain prompt is used."""
    if os.environ.get("REVAPI_CLASSIC") or os.environ.get("TERM") == "dumb":
        return False
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def around(index: int, count: int) -> int:
    return index % count if count else 0


def host_in(prompt: str) -> str:
    m = URL_IN_PROMPT.search(prompt or "")
    return (m.group(1) or m.group(2) or "") if m else ""


def user_words(prompt: Any) -> str:
    """What the user asked: the mission inside an agent's brief, the instructions inside an engineer's, else the message."""
    text = str(prompt or "")
    found = re.search(r"<mission>(.*?)</mission>", text, re.S) or re.search(r"\nAdditional instructions:\n(.*?)(?:\n## |\Z)", text, re.S) or re.search(r"<user_prompt>(.*?)</user_prompt>", text, re.S)
    words = (found.group(1) if found else text).strip()
    return words if len(words) <= 500 else words[:200] + "…"


def run_summary(run: dict[str, Any]) -> str:
    """One line for a past run in a list: when, what was asked, what it cost."""
    when = str(run.get("created") or run.get("timestamp") or "")[5:16].replace("T", " ")
    prompt = " ".join(str(run.get("prompt") or "").split())
    prompt = prompt if len(prompt) <= 44 else prompt[:43] + "…"
    cost = ((run.get("usage") or {}).get("estimated_cost_usd")) or 0
    return f"{when}  {run.get('mode', ''):<9} {prompt:<44}  " + (T.money(cost) if cost else "")


def open_target(target: str) -> None:
    """Open a clicked link: files with the system's default app, URLs in the browser (termux-open on Android)."""
    if os.environ.get("REVAPI_NO_OPEN"):
        return
    path = target
    if target.startswith("file://"):
        from urllib.parse import unquote, urlparse

        path = unquote(urlparse(target).path)
    is_url = re.match(r"https?://", path) is not None
    for cmd in (["termux-open-url"] if is_url else []) + ["termux-open", "xdg-open", "open"]:
        exe = shutil.which(cmd)
        if exe:
            with contextlib.suppress(OSError):
                subprocess.Popen([exe, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
            return


def copy_to_clipboard(text: str, write_raw: Callable[[str], None]) -> None:
    """Termux:API, pbcopy, wl-copy, xclip or xsel; over SSH (or with none of them) the terminal does it with OSC 52."""
    remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
    candidates: list[list[str]] = []
    if not remote:
        candidates = [["termux-clipboard-set"], ["pbcopy"]] + ([["wl-copy"]] if os.environ.get("WAYLAND_DISPLAY") else []) + [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            try:
                env = {**os.environ, "LC_CTYPE": "UTF-8"} if cmd[0] == "pbcopy" else None
                subprocess.run(cmd, input=text.encode(), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                return
            except (OSError, subprocess.SubprocessError):
                continue
    write_raw(f"\x1b]52;c;{base64.b64encode(text.encode()).decode()}\x07")


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def logo_rows(theta: float) -> list[str]:
    """revapi's italic six-arm asterisk with teardrop arms at angle ``theta``, in half blocks (RAE's logo)."""
    w, h = LOGO_COLS, LOGO_ROWS * 2
    cx, cy = (w - 1) / 2, (h - 1) / 2
    radius = min(cx, cy) * 0.98
    tip_c, tip_r = 0.70 * radius, 0.24 * radius
    arms = [(math.cos(theta + k * math.pi / 3 + math.pi / 2), math.sin(theta + k * math.pi / 3 + math.pi / 2)) for k in range(6)]

    def pixel(px: int, py: int) -> str | None:
        x, y = px - cx, cy - py
        x -= y * 0.22  # the italic slant
        r = math.hypot(x, y)
        if r <= radius * 0.13:
            return LOGO_SHADES[0]
        for ux, uy in arms:
            t = x * ux + y * uy
            d = abs(y * ux - x * uy)
            if t <= 0 or t > tip_c + tip_r:
                continue
            half = tip_r * (t / tip_c) ** 0.85 if t <= tip_c else math.sqrt(max(0.0, tip_r**2 - (t - tip_c) ** 2))
            if d <= half:
                shine = 0.5 + 0.5 * math.cos(math.atan2(y, x) + theta * 3)
                return LOGO_SHADES[min(len(LOGO_SHADES) - 1, int((r / radius) * 4 + (1 - shine) * 2))]
        return None

    rows = []
    for j in range(LOGO_ROWS):
        row = []
        for i in range(LOGO_COLS):
            top, bottom = pixel(i, 2 * j), pixel(i, 2 * j + 1)
            if top and bottom:
                row.append(T.seq(top, bottom) + "▀" + T.RESET)
            elif top:
                row.append(T.seq(top) + "▀" + T.RESET)
            elif bottom:
                row.append(T.seq(bottom) + "▄" + T.RESET)
            else:
                row.append(" ")
        rows.append("".join(row))
    return rows


_TOK = re.compile(r"(\x1b\[[0-9;?]*[ -/]*[@-~])|(\x1b\]8;[^\x1b\x07]*(?:\x1b\\|\x07))|(\x1b\][^\x07\x1b]*(?:\x07|\x1b\\))|(.)", re.S)


def hard_wrap(line: str, width: int) -> list[str]:
    """Cut an ANSI line into rows of at most ``width`` columns, carrying colors and links onto each row."""
    if T.vis_len(line) <= width:
        return [line]
    rows: list[str] = []
    cur: list[str] = []
    w = 0
    sgr: list[str] = []
    link = ""
    for m in _TOK.finditer(line):
        csi, osc8, osc, ch = m.groups()
        if csi:
            cur.append(csi)
            if csi.endswith("m"):
                sgr = [] if csi in ("\x1b[0m", "\x1b[m") else sgr + [csi]
            continue
        if osc8:
            cur.append(osc8)
            link = "" if re.match(r"\x1b\]8;[^;]*;(\x1b\\|\x07)", osc8) else osc8
            continue
        if osc:
            continue
        cw = T.char_width(ch)
        if w + cw > width and w > 0:
            rows.append("".join(cur) + (T.RESET if sgr else "") + ("\x1b]8;;\x1b\\" if link else ""))
            cur = list(sgr) + ([link] if link else [])
            w = 0
        cur.append(ch)
        w += cw
    rows.append("".join(cur))
    return rows


def plain_cols(line: str) -> str:
    """The visible text of an ANSI line, one string index per column (wide characters padded)."""
    out = []
    for ch in T.strip_ansi(line):
        cw = T.char_width(ch)
        if cw == 0:
            continue
        out.append(ch + ("\x00" if cw == 2 else ""))
    return "".join(out)


def links_in(line: str) -> list[tuple[int, int, str]]:
    """Clickable spans of a line, by column: OSC 8 links, then bare URLs and absolute paths in the text."""
    spans: list[tuple[int, int, str]] = []
    col, start, url = 0, 0, ""
    for m in _TOK.finditer(line):
        csi, osc8, osc, ch = m.groups()
        if osc8:
            target = re.match(r"\x1b\]8;[^;]*;([^\x1b\x07]*)", osc8)
            target_url = target.group(1) if target else ""
            if url and not target_url:
                spans.append((start, col, url))
            url, start = target_url, col
        elif ch:
            col += T.char_width(ch)
    text = plain_cols(line)
    for m in re.finditer(r"https?://[^\s\x00]+|(?:~|/)[\w.@%+-]*/[\w./@%+-]+", text):
        if not any(a <= m.start() < b for a, b, _ in spans):
            spans.append((m.start(), m.end(), m.group(0).rstrip(".,;:)")))
    return spans


# ----------------------------------------------------------------------------- keys


class Keys:
    """Turns terminal input bytes into key events: ("key", name), ("text", s), ("paste", s), ("mouse", btn, x, y, up)."""

    CSI = re.compile(r"\x1b\[([0-9;]*)([~A-Za-z])")
    MOUSE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
    TILDE = {"1": "home", "7": "home", "4": "end", "8": "end", "3": "delete", "5": "pageup", "6": "pagedown", "2": "insert"}
    FINAL = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end", "Z": "s-tab"}
    CTRL = {"\r": "enter", "\n": "enter", "\t": "tab", "\x7f": "backspace", "\x08": "backspace", "\x00": "c-space"}

    def __init__(self) -> None:
        self.buf = ""
        self.dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.paste: str | None = None

    def feed(self, data: bytes, final: bool = False) -> list[tuple]:
        self.buf += self.dec.decode(data)
        return self.parse(final)

    def parse(self, final: bool = False) -> list[tuple]:  # noqa: C901  (one branch per kind of sequence)
        out: list[tuple] = []
        b = self.buf
        i = 0
        text = []

        def flush_text() -> None:
            if text:
                out.append(("text", "".join(text)))
                text.clear()

        while i < len(b):
            if self.paste is not None:
                j = b.find("\x1b[201~", i)
                if j < 0:
                    self.paste += b[i:]
                    i = len(b)
                    break
                self.paste += b[i:j]
                out.append(("paste", self.paste))
                self.paste = None
                i = j + 6
                continue
            ch = b[i]
            if ch == "\x1b":
                rest = b[i:]
                if len(rest) == 1:
                    if final:
                        flush_text()
                        out.append(("key", "esc"))
                        i += 1
                        continue
                    break
                if rest.startswith("\x1b[200~"):
                    flush_text()
                    self.paste = ""
                    i += 6
                    continue
                m = self.MOUSE.match(rest)
                if m:
                    flush_text()
                    out.append(("mouse", int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) == "m"))
                    i += m.end()
                    continue
                if rest.startswith("\x1b[<") and not final:
                    break  # a mouse report still arriving
                m = self.CSI.match(rest)
                if m:
                    flush_text()
                    params, fin = m.group(1), m.group(2)
                    mod = params.split(";")[1] if ";" in params else ""
                    if fin == "~":
                        name = self.TILDE.get(params.split(";")[0], "")
                    else:
                        name = self.FINAL.get(fin, "")
                    if name in ("left", "right") and mod in ("3", "5", "9"):
                        name = "c-" + name
                    if name:
                        out.append(("key", name))
                    i += m.end()
                    continue
                if rest.startswith("\x1b[") and not final and re.fullmatch(r"\x1b\[[0-9;<]*", rest):
                    break  # incomplete
                if rest.startswith("\x1bO") and len(rest) >= 3:
                    flush_text()
                    name = self.FINAL.get(rest[2], "")
                    if name:
                        out.append(("key", name))
                    i += 3
                    continue
                if rest.startswith("\x1bO") and not final:
                    break
                flush_text()
                nxt = rest[1]
                out.append(("key", "alt-backspace" if nxt in "\x7f\x08" else f"alt-{nxt}"))
                i += 2
                continue
            if ch in self.CTRL:
                flush_text()
                out.append(("key", self.CTRL[ch]))
                i += 1
                continue
            if ord(ch) < 32:
                flush_text()
                out.append(("key", "c-" + chr(ord(ch) + 96)))
                i += 1
                continue
            text.append(ch)
            i += 1
        flush_text()
        self.buf = b[i:]
        return out


# ----------------------------------------------------------------------------- transcript


@dataclass
class Item:
    lines: list[str]
    full: list[str] | None = None
    cache: tuple[int, list[str]] | None = None


class Transcript:
    def __init__(self) -> None:
        self.items: list[Item] = []
        self.lock = threading.RLock()
        self.version = 0
        self._flat: tuple[int, int, list[str]] | None = None

    def add(self, lines: list[str], full: list[str] | None = None) -> None:
        with self.lock:
            clean = [part.replace("\r", "").replace("\t", "    ") for ln in lines for part in str(ln).split("\n")]
            self.items.append(Item(clean, full))
            self.version += 1

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.version += 1

    def rows(self, width: int) -> list[str]:
        with self.lock:
            if self._flat and self._flat[0] == width and self._flat[1] == self.version:
                return self._flat[2]
            out: list[str] = []
            for it in self.items:
                if not it.cache or it.cache[0] != width:
                    wrapped: list[str] = []
                    for ln in it.lines:
                        wrapped += hard_wrap(ln, width) if ln else [""]
                    it.cache = (width, wrapped)
                out += it.cache[1]
            self._flat = (width, self.version, out)
            return out

    def full_lines(self) -> list[str]:
        with self.lock:
            out: list[str] = []
            for it in self.items:
                out += [part for ln in it.full for part in str(ln).split("\n")] if it.full is not None else it.lines
            return out


class SessionLive:
    """What the Reporter draws through inside the session: finished lines go to the conversation, the live lines
    (reasoning as it streams, the reply being typed, the status line) are drawn above the input bar."""

    enabled = True

    def __init__(self, session: Session) -> None:
        self.session = session
        self.render: Callable[[], list[str]] | None = None
        self.lock = threading.RLock()

    def start(self, render: Callable[[], list[str]]) -> None:
        self.render = render
        self.session.dirty = True

    def stop(self) -> None:
        self.render = None
        self.session.dirty = True

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def print(self, *lines: str) -> None:
        self.add(list(lines))

    def add(self, lines: list[str], full: list[str] | None = None) -> None:
        self.session.add(lines, full)


class Capture:
    """sys.stdout / sys.stderr while the session owns the screen: stray prints become dim notices."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.buf = ""
        self.encoding = "utf-8"

    def write(self, s: str) -> int:
        self.buf += str(s)
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.session.add_stray(line)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


@dataclass
class Picker:
    title: str
    items: list[tuple[str, str, str]]  # value, label, description
    current: str | None
    on_pick: Callable[[str | None], None]
    index: int = 0
    question: bool = False  # an agent question, which ends with its run
    typed: bool = False  # enter sends what was typed in the bar, when anything was
    search: bool = False  # what is typed in the bar narrows the items
    hint: str = ""  # shown in the empty bar: what typing does here


@dataclass
class Selection:
    anchor: tuple[int, int]
    head: tuple[int, int]
    dragging: bool = True

    def span(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.anchor, self.head) if self.anchor <= self.head else (self.head, self.anchor)

    def cols(self, row: int, width: int) -> tuple[int, int] | None:
        (r0, c0), (r1, c1) = self.span()
        if not r0 <= row <= r1:
            return None
        return (c0 if row == r0 else 0, c1 + 1 if row == r1 else width)


@dataclass
class State:
    mode: str = "agent"
    running: bool = False
    waiting: bool = False  # a run is open and waits for a follow-up
    stopping: bool = False
    started: float = 0.0
    verb: str = "starting"
    kind: str = "other"
    host: str = ""
    client: str = ""
    queue: list[str] = field(default_factory=list)
    scroll: int = 0
    flash: tuple[str, float] | None = None
    palette_index: int = 0
    picker: Picker | None = None
    question: Future | None = None
    run_id: str | None = None
    cost: float = 0.0  # this session's spend before the open run
    last_client: str | None = None


# ----------------------------------------------------------------------------- the session


class Session:
    def __init__(self, cfg: store.Config, mode: str | None = None) -> None:
        self.cfg = cfg
        m = mode or cfg.get("repl_mode", "agent")
        self.st = State(mode=m if m in MODES else "agent")
        self.transcript = Transcript()
        self.live = SessionLive(self)
        self.keys = Keys()
        self.buf = ""
        self.cur = 0
        self.history = self._load_history()
        self.hist_i: int | None = None
        self.hist_draft = ""
        self.lock = threading.RLock()
        self.calls: deque[Callable[[], None]] = deque()
        self.dirty = True
        self.worker: threading.Thread | None = None
        self.rep: Reporter | None = None
        self.keep: dict[str, Any] = {}
        self.results: list[dict] = []
        self.runners: dict[str, Callable[[str, Reporter, dict], dict]] = {}
        self.first: tuple[str, str | None] | None = None
        self.after_run: Callable[[], None] | None = None
        self.selection: Selection | None = None
        self.view_rows: list[str] = []
        self.view_start = 0
        self.view_end = 0
        self.transcript_h = 0
        self.menu_rows: dict[int, int] = {}
        self.frame: list[str] = []
        self.size = (0, 0)
        self.quitting = False
        self.copies = 0
        self.model_desc = ""
        self.model_short = ""
        self.real_out = sys.stdout
        self.real_err = sys.stderr
        try:
            self.out_fd = sys.__stdout__.fileno() if sys.__stdout__ else 1
        except (OSError, ValueError):
            self.out_fd = 1
        try:
            self.in_fd = sys.stdin.fileno()
        except (OSError, ValueError, AttributeError):
            self.in_fd = 0
        self.saved_tty: Any = None
        self.mouse = not os.environ.get("REVAPI_NO_MOUSE")
        self._interrupted = False
        self._refresh_model()

    # -- plumbing ------------------------------------------------------------------------------------
    def call_soon(self, fn: Callable[[], None]) -> None:
        self.calls.append(fn)

    def add(self, lines: list[str], full: list[str] | None = None) -> None:
        self.transcript.add(lines, full)
        self.dirty = True

    def add_stray(self, line: str) -> None:
        line = line.replace("\r", "")
        if "\x1b[2J" in line:
            return
        if not line.strip():
            self.add([""])
        elif "\x1b[" in line:
            self.add([line])
        else:
            self.add([f"   {sty(line, DIM)}"])

    def _refresh_model(self) -> None:
        try:
            if modes.hermes_engine(self.cfg) and self.st.mode in ("engineer", "manual"):
                self.model_desc = f"hermes · {self.cfg.get('hermes_model') or 'default'}"
                self.model_short = self.cfg.get("hermes_model") or "hermes"
            else:
                llm = modes.make_llm(self.cfg, collector=self.st.mode == "collector")
                self.model_desc = llm.describe()
                self.model_short = llm.model.split("/")[-1] if llm.model else llm.provider
        except LLMError as e:
            self.model_desc = f"no model ({e})"
            self.model_short = "no model"
        self.dirty = True

    def _load_history(self) -> list[str]:
        try:
            lines = (store.app_dir() / "repl_history").read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [ln for ln in lines if ln.strip() and not ln.startswith("_HiStOrY_V2_")][-500:]

    def _save_history(self) -> None:
        with contextlib.suppress(OSError):
            (store.app_dir() / "repl_history").write_text("\n".join(self.history[-500:]) + "\n", encoding="utf-8")

    @property
    def accent(self) -> str:
        return MODE_COLORS.get(self.st.mode, PINK)

    @property
    def effort(self) -> str:
        return self.cfg.get("effort") or "auto"

    # -- the ledger's own lines ---------------------------------------------------------------------------
    def gap(self) -> None:
        rows = self.transcript.items
        if rows and rows[-1].lines and rows[-1].lines[-1].strip():
            self.add([""])

    def user(self, text: str) -> None:
        self.gap()
        body = T.wrap(" ".join(str(text).split()), T.cols() - 5)
        self.add([f" {sty('›', self.accent, bold=True)} {sty(body[0] if body else '', INK)}"] + [f"   {sty(ln, INK)}" for ln in body[1:]] + [""])

    def command(self, cmd: str, arg: str = "") -> None:
        self.gap()
        self.add([f" {sty('›', DIM)} {sty(cmd, self.accent)}" + (f" {sty(arg, INK_SOFT)}" if arg else "")])

    def notice(self, text: str) -> None:
        self.add([f"   {sty(ln, DIM)}" for ln in T.wrap(text, T.cols() - 4)])

    def error(self, text: str) -> None:
        self.gap()
        body = T.wrap(str(text), T.cols() - 6) or [""]
        self.add([f" {sty('!', PINK, bold=True)}  {sty(body[0], PINK)}"] + [f"    {sty(ln, PINK)}" for ln in body[1:]] + [""])

    def event(self, key: str, old: str, new: str, note: str = "") -> None:
        tail = f"  {sty('· ' + note, DIM)}" if note else ""
        self.add([f"   {sty('◇', AMBER)} {sty(key, INK_SOFT)}  {sty(str(old) + ' →', DIM)} {sty(str(new), INK, bold=True)}{tail}"])

    def flash(self, message: str) -> None:
        self.st.flash = (message, time.time() + 2)
        self.dirty = True

    def help(self) -> None:
        a = self.accent
        about = (
            "revapi turns a website into an API client. Describe the API you want: the agent opens the site in headless "
            "Chromium, reads the requests it makes, writes a client and runs it against the live site before it is done."
        )
        out = [f"   {sty(ln, INK_SOFT)}" for ln in T.wrap(about, T.cols() - 5)]
        sections = [
            ("modes", "shift+tab switches", [(m, d, a if m == self.st.mode else INK) for m, d in MODES.items()]),
            ("commands", "type / for the list", [(c, d, a) for c, d in COMMANDS.items()]),
            ("keys", "", [(k, d, INK) for k, d in KEYS]),
        ]
        name_w = max(len(n) for _, _, rows in sections for n, _, _ in rows)
        for title, note, rows in sections:
            out += ["", f"   {sty(title, INK, bold=True)}" + (f"  {sty('· ' + note, DIM)}" if note else "")]
            for name, desc, color in rows:
                cur = f"  {sty('●', a)}" if title == "modes" and name == self.st.mode else ""
                room = max(10, T.cols() - name_w - 9)
                d = desc if len(desc) <= room else desc[: room - 1] + "…"
                out.append(f"     {sty(f'{name:<{name_w}}', color, bold=color == a and title == 'modes')}  {sty(d, DIM)}{cur}")
        out += ["", f"   {sty('docs', DIM)}  {sty(T.link('github.com/bageltrade/revapi', 'https://github.com/bageltrade/revapi#readme'), INK_SOFT)}", ""]
        self.add(out)

    def run_details(self, run: dict[str, Any]) -> None:
        usage = run.get("usage") or {}
        cost = usage.get("estimated_cost_usd") or 0
        tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        when = str(run.get("created") or "")[:16].replace("T", " ")
        facts = " · ".join(x for x in (when, run.get("model") or "", run.get("mode") or "", run.get("status") or "") if x)
        spend = " · ".join(x for x in (T.money(cost) if cost else "", f"{T.k(tokens)} tokens" if tokens else "") if x)
        script = (run.get("paths") or {}).get("script_path")
        lines = [f" {sty('*', self.accent)} {sty(str(run.get('run_id', '')), INK, bold=True)}  {sty(facts, DIM)}"]
        prompt = " ".join(str(run.get("prompt") or "").split())
        lines.append(f"   {sty('└', HAIRLINE)} {sty(prompt[:200], INK_SOFT)}")
        if spend or script:
            client = T.link(Path(script).name, Path(script).resolve().as_uri()) if script and Path(script).exists() else "no client"
            lines.append(f"     {sty(spend + ' · ' if spend else '', DIM)}{sty(client, INK_SOFT)}")
        self.add(lines)

    # -- header ----------------------------------------------------------------------------------------
    def header_rows(self, theta: float, width: int, height: int) -> list[str]:
        effort = f" · effort {self.effort}" if self.cfg.get("effort") else ""
        text = [
            "",
            "",
            f" {sty('*', PINK, bold=True)} {sty('revapi', INK, bold=True)}{sty(f'  v{__version__}', DIM)}{sty('   turn websites into ', INK_SOFT)}{sty('APIs', PINK, italic=True)}",
            f"   {sty(T.truncate(self.model_desc + effort, max(10, width - 4)), DIM)}",
            "",
            f"   {sty('describe the API you want · / for commands', DIM)}",
        ]
        widest = max(T.vis_len(r) for r in text)
        logo = logo_rows(theta) if width >= widest + LOGO_COLS + 10 and height >= 24 and T.COLOR else []
        rows = [""] * TOP_PADDING
        for r in range(max(len(text), len(logo))):
            row = text[r] if r < len(text) else ""
            if r < len(logo):
                row += " " * max(1, width - LOGO_COLS - 5 - T.vis_len(row)) + logo[r]
            rows.append(row)
        rows.append("")
        return rows

    # -- the bar ---------------------------------------------------------------------------------------
    def _flap(self, verb: str) -> str:
        if verb != getattr(self, "_verb", None):
            self._verb, self._verb_at = verb, time.time()
        settled = int((time.time() - self._verb_at) / 0.035)
        tick = int(time.time() * 30)
        return "".join(ch if i < settled or ch == " " else T.FLAP[(tick + i * 7) % len(T.FLAP)] for i, ch in enumerate(verb))

    def spinner(self, width: int) -> list[str]:
        st = self.st
        rows: list[str] = [""]
        rep = self.rep
        live = None
        if rep is not None and self.live.render is not None:
            try:
                live = self.live.render()
            except Exception:  # noqa: BLE001  (a render bug never takes the screen down)
                live = None
        if live:
            if st.stopping:
                live = live[:-1] + [f" {sty(T.spinner_frame(), PINK)} {sty('stopping…', PINK, italic=True)}"]
            rows += live
        else:
            tint = KIND_TINT.get(st.kind, PASTEL_OTHER)
            verb = "stopping…" if st.stopping else st.verb
            source, sink, clock = st.host or "the site", st.client or "your API", T.elapsed(time.time() - st.started)
            head = f" {sty(self._flap(verb), INK, italic=True)}" + " " * max(2, 22 - len(verb))
            track = max(8, width - T.vis_len(head) - len(source) - len(sink) - len(clock) - 6)
            offset = int(time.time() * 9)
            tape = "".join(sty("▪", tint, bold=True) if (i - offset) % 7 == 0 else sty("┄", HAIRLINE) for i in range(track))
            rows.append(head + sty(source, INK_SOFT) + " " + tape + " " + sty(sink, INK_SOFT) + sty(f"   {clock}", DIM))
        for q in st.queue:
            rows.append(f"   {sty('queued ›', DIM)} {sty(q, INK_SOFT)}")
        rows.append("")
        return rows

    def seam(self, width: int) -> str:
        left = sty(" ┄┄ ", HAIRLINE) + sty(self.st.mode, self.accent, italic=True) + " "
        model = self.model_short
        if self.keep.get("agent") is not None:
            model = self.keep["agent"].llm.model.split("/")[-1]
        elif self.keep.get("hermes") is not None:
            model = self.keep["hermes"].label
        right = " " + sty(model, INK_SOFT) + sty(" ┄┄", HAIRLINE)
        used = T.vis_len(left) + T.vis_len(right)
        if used + 2 > width:
            right = sty(" ┄┄", HAIRLINE)
            used = T.vis_len(left) + 3
        return left + sty("┄" * max(1, width - used), HAIRLINE) + right

    def seam_under(self, width: int) -> str:
        return sty(" " + "┄" * max(1, width - 1), HAIRLINE)

    @staticmethod
    def pill(color: str, body: str) -> str:
        """A pastel chip with rounded half-block ends, like a small post-it."""
        return sty("▐", color) + T.paint(body, T.seq(None, color)) + sty("▌", color)

    def status(self, width: int) -> str:
        st = self.st
        level = EFFORTS.index(self.effort) if self.effort in EFFORTS else 0
        gauge = sty(" ", CHIP_INK) + sty("▰" * level, PINK, bold=True) + sty("▱" * (len(EFFORTS) - 1 - level), GAUGE_EMPTY) + sty(f" {self.effort} ", CHIP_INK)
        left = "   " + self.pill(CHIP_LAVENDER, gauge) + sty(" ⌃e  ", DIM)
        cost = st.cost + (((self.rep.usage or {}).get("estimated_cost_usd") or 0) if self.rep is not None and st.running else 0)
        if cost:
            left += self.pill(CHIP_YELLOW, sty(f" {T.money(cost)} ", CHIP_INK, bold=True)) + "  "
        if st.flash and time.time() < st.flash[1]:
            hint = sty(f"{st.flash[0]} ", PASTEL_WRITE)
        elif st.scroll:
            hint = sty("end ↓ back to the latest ", AMBER)
        elif st.question is not None:
            hint = sty("type your answer · esc skips ", AMBER)
        elif st.running:
            hint = sty("esc stops · enter queues ", DIM)
        elif st.waiting:
            hint = sty("type a follow-up · /new starts over · ⌃o expand ", DIM)
        else:
            hint = sty("/ commands · ⌃o expand ", DIM)
        used = T.vis_len(left) + T.vis_len(hint)
        run = f"· run {st.run_id} " if st.run_id else ""
        if run and used + 1 + len(run) <= width:
            hint, used = hint + sty(run, DIM), used + len(run)
        if used > width:
            left = "   "
            used = 3 + T.vis_len(hint)
        return left + " " * max(1, width - used) + hint

    def menu(self, rows: list[tuple[str, str, bool, bool]], title: str | None, limit: int, empty: str, first_row: int) -> list[str]:
        out: list[str] = []
        self.menu_rows = {}
        if title:
            out.append(sty(f"   {T.truncate(title, T.cols() - 5)}", DIM, italic=True))
        label_w = min(max((T.vis_len(r[0]) for r in rows), default=0) + 3, 40)
        sel = next((i for i, r in enumerate(rows) if r[2]), 0)
        start = max(0, min(sel - limit // 2, len(rows) - limit))
        hidden = max(0, len(rows) - limit)
        for i, (label, desc, selected, current) in enumerate(rows[start : start + limit], start):
            label = T.truncate(label, label_w - 1)
            pad = " " * max(1, label_w - T.vis_len(label))
            if selected:
                line = sty(" › ", self.accent, bold=True) + sty(label, INK, bold=True) + pad + sty(desc, INK_SOFT)
            else:
                line = "   " + sty(label, INK_SOFT) + pad + sty(desc, DIM)
            if current:
                line += sty("  ●", self.accent)
            self.menu_rows[first_row + len(out)] = i
            out.append(line)
        if not rows:
            out.append(sty(f"   {empty}", DIM))
        if hidden:
            out.append(sty(f"   {hidden} more · ↑↓", DIM))
        return out

    def palette_active(self) -> bool:
        return self.st.picker is None and self.st.question is None and self.buf.startswith("/") and " " not in self.buf

    def palette_items(self) -> list[tuple[str, str]]:
        typed = self.buf.strip().lower()
        aliased = {t for a, t in ALIASES.items() if a.startswith(typed)}
        items = [(c, d) for c, d in COMMANDS.items() if c.startswith(typed) or c in aliased]
        return sorted(items, key=lambda it: it[0] != ALIASES.get(typed, typed))

    def picker_items(self, p: Picker) -> list[tuple[str, str, str]]:
        typed = self.buf.strip().lower()
        if not (p.search and typed):
            return p.items
        return [it for it in p.items if typed in f"{it[0]} {it[1]} {it[2]}".lower()]

    def input_rows(self, width: int) -> tuple[list[str], int, int]:
        """The bar: the marker and what is typed (highlighted), wrapped; returns rows and the cursor's row and column."""
        st = self.st
        marker = sty(" › ", AMBER if st.question is not None else self.accent, bold=True)
        room = max(8, width - 4)
        if not self.buf:
            if st.picker is not None and st.picker.hint:
                ph = st.picker.hint
            elif st.question is not None:
                ph = "type your answer"
            elif st.waiting:
                ph = "a follow-up for this run, or /new"
            else:
                ph = "describe the API you want, e.g. all open jobs on jobs.ashbyhq.com/openai"
            return [marker + sty(T.truncate(ph, room), DIM, italic=True)], 0, 3
        chunks = [self.buf[i : i + room] for i in range(0, len(self.buf), room)] or [""]
        rows = []
        for n, chunk in enumerate(chunks):
            rows.append((marker if n == 0 else "   ") + self._lex(chunk, n == 0))
        crow, ccol = divmod(self.cur, room)
        if crow >= len(rows):
            rows.append("   ")
        return rows, crow, 3 + T.vis_len(self.buf[crow * room : self.cur]) if crow < len(chunks) else 3

    def _lex(self, text: str, first: bool) -> str:
        out, pos = [], 0
        for m in re.finditer(r"(^/\w*)|(https?://\S+)|(\b[\w.-]+\.(?:com|ch|io|dev|org|net|ai|co|ng|app)\S*)", text):
            out.append(sty(text[pos : m.start()], INK))
            if m.group(1) and first:
                out.append(sty(m.group(0), self.accent, bold=True))
            else:
                out.append(sty(m.group(0), SKY, underline=True))
            pos = m.end()
        out.append(sty(text[pos:], INK))
        return "".join(out)

    # -- composing a frame -----------------------------------------------------------------------------
    def compose(self, width: int, height: int) -> tuple[list[str], int, int]:
        st = self.st
        w = width - 1  # never write the last column: some terminals wrap there
        inp, crow, ccol = self.input_rows(w)
        tail: list[str] = []
        avail_menu = max(3, min(MENU_ROWS, height - 10))
        bottom_head = (self.spinner(w) if st.running else []) + [self.seam(w)]
        menu_first = len(bottom_head) + len(inp) + 1
        if self.palette_active():
            items = self.palette_items()
            st.palette_index = min(st.palette_index, max(0, len(items) - 1))
            tail = self.menu([(c, d, i == st.palette_index, False) for i, (c, d) in enumerate(items)], None, min(len(COMMANDS), avail_menu), "no matching command", 0)
        elif st.picker is not None:
            p = st.picker
            items = self.picker_items(p)
            p.index = min(p.index, max(0, len(items) - 1))
            rows = [(label, desc, i == p.index, value == p.current and value is not None) for i, (value, label, desc) in enumerate(items)]
            tail = self.menu(rows, p.title, avail_menu, "nothing matches what you typed", 0)
        else:
            self.menu_rows = {}
            tail = [self.status(w)]
        bottom = bottom_head + inp + [self.seam_under(w)] + tail
        if len(bottom) > height - 1:  # a tiny window: drop the live lines first
            bottom = bottom[len(bottom) - (height - 1) :]
        top_h = height - len(bottom)
        self.transcript_h = top_h
        lines = self.header_rows(time.time() * 0.35, w, height) + self.transcript.rows(w)
        if lines and lines[-1].strip():
            lines = lines + [""]
        st.scroll = max(0, min(st.scroll, len(lines) - top_h))
        end = len(lines) - st.scroll
        start = max(0, end - top_h)
        self.view_rows, self.view_start, self.view_end = lines, start, end
        shown = lines[start:end]
        if self.selection:
            shown = [self._highlight(start + i, ln, w) for i, ln in enumerate(shown)]
        frame = shown + [""] * (top_h - len(shown)) + bottom
        offset = height - len(bottom)
        self.menu_rows = {offset + menu_first - (len(bottom_head) + len(inp) + 1) + len(bottom_head) + len(inp) + 1 + k: v for k, v in self.menu_rows.items()}
        cursor_row = offset + len(bottom_head) + crow
        return [T.truncate(r, w) for r in frame], cursor_row, ccol

    def _highlight(self, row: int, line: str, width: int) -> str:
        cols = self.selection.cols(row, width) if self.selection else None
        if not cols:
            return line
        text = plain_cols(line)
        a, b = cols
        a, b = max(0, min(a, len(text))), max(0, min(b, len(text)))
        clean = lambda s: s.replace("\x00", "")  # noqa: E731
        return sty(clean(text[:a]), INK_SOFT) + "\x1b[7m" + clean(text[a:b]) + T.RESET + sty(clean(text[b:]), INK_SOFT)

    def draw(self, force: bool = False) -> None:
        try:
            size = os.get_terminal_size(self.out_fd)
            width, height = size.columns, size.lines
        except OSError:
            width, height = 80, 24
        if (width, height) != self.size:
            self.size = (width, height)
            self.frame = []
            self.selection = None
            force = True
        frame, crow, ccol = self.compose(width, height)
        parts = ["\x1b[?2026h\x1b[?25l"]
        for r, row in enumerate(frame):
            if force or r >= len(self.frame) or self.frame[r] != row:
                parts.append(f"\x1b[{r + 1};1H{T.RESET}{row}{T.RESET}\x1b]8;;\x1b\\\x1b[K")
        parts.append(f"\x1b[{crow + 1};{ccol + 1}H\x1b[?25h\x1b[?2026l")
        self.frame = frame
        self._write("".join(parts))

    def _write(self, s: str) -> None:
        data = s.encode("utf-8", errors="replace")
        while data:
            try:
                n = os.write(self.out_fd, data)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except OSError:
                return
            data = data[n:]

    def write_raw(self, s: str) -> None:
        self.call_soon(lambda: self._write(s))

    # -- terminal --------------------------------------------------------------------------------------
    def enter_screen(self) -> None:
        import termios

        self.saved_tty = termios.tcgetattr(self.in_fd)
        mode = termios.tcgetattr(self.in_fd)
        mode[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
        mode[2] |= termios.CS8
        mode[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
        mode[6][termios.VMIN] = 1
        mode[6][termios.VTIME] = 0
        termios.tcsetattr(self.in_fd, termios.TCSAFLUSH, mode)
        self._write("\x1b[?1049h\x1b[?2004h\x1b[2J" + ("\x1b[?1000h\x1b[?1002h\x1b[?1006h" if self.mouse else ""))
        self.frame = []

    def leave_screen(self) -> None:
        import termios

        self._write("\x1b[?1000l\x1b[?1002l\x1b[?1006l\x1b[?2004l\x1b[0m\x1b[?25h\x1b[?1049l")
        if self.saved_tty is not None:
            with contextlib.suppress(termios.error):
                termios.tcsetattr(self.in_fd, termios.TCSADRAIN, self.saved_tty)

    @contextlib.contextmanager
    def suspended(self) -> Any:
        """Give the terminal back for a moment (the pager), then take the screen again."""
        self.leave_screen()
        sys.stdout, sys.stderr = self.real_out, self.real_err
        try:
            yield
        finally:
            sys.stdout = sys.stderr = self.capture  # type: ignore[assignment]
            self.enter_screen()
            self.dirty = True

    # -- input -----------------------------------------------------------------------------------------
    def set_buf(self, text: str, cur: int | None = None) -> None:
        if text != self.buf:
            self.st.palette_index = 0
            if self.st.picker is not None and self.st.picker.search:
                self.st.picker.index = 0
            self.selection = None
        self.buf = text
        self.cur = len(text) if cur is None else max(0, min(cur, len(text)))
        self.dirty = True

    def on_event(self, ev: tuple) -> None:  # noqa: C901  (one branch per key)
        kind = ev[0]
        st = self.st
        self.dirty = True
        if kind == "text":
            text = ev[1]
            if st.picker is not None and not (st.picker.search or st.picker.typed or st.picker.question) and text.isdigit():
                n = int(text) - 1  # a number picks that item in a short list
                items = self.picker_items(st.picker)
                if 0 <= n < len(items):
                    st.picker.index = n
                    self.choose()
                return
            self.set_buf(self.buf[: self.cur] + text + self.buf[self.cur :], self.cur + len(text))
            return
        if kind == "paste":
            text = " ".join(ev[1].replace("\r\n", "\n").split("\n")).replace("\t", " ")
            self.set_buf(self.buf[: self.cur] + text + self.buf[self.cur :], self.cur + len(text))
            return
        if kind == "mouse":
            self.on_mouse(*ev[1:])
            return
        key = ev[1]
        palette, picker = self.palette_active(), st.picker is not None
        if key == "enter":
            if palette:
                items = self.palette_items()
                cmd = items[st.palette_index][0] if items else self.buf.strip()
                self.set_buf("")
                self.run_command(cmd)
            elif picker:
                self.choose()
            else:
                text = self.buf.strip()
                if text and (not self.history or self.history[-1] != text) and st.question is None:
                    self.history.append(text)
                    self._save_history()
                self.hist_i = None
                self.set_buf("")
                self.submit(text)
        elif key == "tab":
            if palette:
                items = self.palette_items()
                if items:
                    self.set_buf(items[st.palette_index][0] + " ")
        elif key in ("up", "c-p"):
            if palette:
                st.palette_index = around(st.palette_index - 1, len(self.palette_items()))
            elif picker:
                self.move_in_picker(-1)
            else:
                self.history_move(-1)
        elif key in ("down", "c-n"):
            if palette:
                st.palette_index = around(st.palette_index + 1, len(self.palette_items()))
            elif picker:
                self.move_in_picker(1)
            else:
                self.history_move(1)
        elif key == "left":
            self.cur = max(0, self.cur - 1)
        elif key == "right":
            self.cur = min(len(self.buf), self.cur + 1)
        elif key in ("c-left", "alt-b"):
            m = list(re.finditer(r"\S+", self.buf[: self.cur]))
            self.cur = m[-1].start() if m else 0
        elif key in ("c-right", "alt-f"):
            m = re.search(r"\S+", self.buf[self.cur :])
            self.cur = self.cur + m.end() if m else len(self.buf)
        elif key in ("home", "c-a"):
            if not self.buf:
                st.scroll = 10**9  # the top of the conversation
            self.cur = 0
        elif key == "end":
            if self.buf:
                self.cur = len(self.buf)
            else:
                st.scroll = 0
        elif key == "backspace":
            if self.cur:
                self.set_buf(self.buf[: self.cur - 1] + self.buf[self.cur :], self.cur - 1)
        elif key == "delete":
            self.set_buf(self.buf[: self.cur] + self.buf[self.cur + 1 :], self.cur)
        elif key in ("c-w", "alt-backspace"):
            left = re.sub(r"\S+\s*$", "", self.buf[: self.cur].rstrip()) if self.buf[: self.cur].strip() else ""
            self.set_buf(left + self.buf[self.cur :], len(left))
        elif key == "c-u":
            self.set_buf(self.buf[self.cur :], 0)
        elif key == "c-k":
            self.set_buf(self.buf[: self.cur], self.cur)
        elif key == "c-l":
            self.frame = []
        elif key == "pageup":
            st.scroll += max(1, self.transcript_h // 2)
        elif key == "pagedown":
            st.scroll = max(0, st.scroll - max(1, self.transcript_h // 2))
        elif key == "s-tab":
            names = list(MODES)
            self.set_mode(names[(names.index(st.mode) + 1) % len(names)], quiet=True)
        elif key == "c-e":
            self.cycle_effort()
        elif key == "c-o":
            self.show_output()
        elif key == "esc":
            if st.picker is not None:
                p, st.picker = st.picker, None
                self.set_buf("")
                if p.question and isinstance(st.question, Future):
                    pass
                p.on_pick(None)
            elif isinstance(st.question, Future):
                fut, st.question = st.question, None
                if not fut.done():
                    fut.set_result("")
                self.add([f"      {sty('┊', HAIRLINE)} {sty('skipped', DIM)}", ""])
            elif self.buf.startswith("/"):
                self.set_buf("")
            elif self.selection is not None:
                self.selection = None
            else:
                self.stop()
        elif key == "c-c":
            if st.running or st.waiting:
                self.stop()
            elif self.buf:
                self.set_buf("")
            else:
                self.quit()
        elif key == "c-d":
            if self.buf:
                self.set_buf(self.buf[: self.cur] + self.buf[self.cur + 1 :], self.cur)
            else:
                self.quit()

    def history_move(self, step: int) -> None:
        if not self.history:
            return
        if self.hist_i is None:
            if step > 0:
                return
            self.hist_draft = self.buf
            self.hist_i = len(self.history)
        self.hist_i = max(0, min(len(self.history), self.hist_i + step))
        self.set_buf(self.hist_draft if self.hist_i == len(self.history) else self.history[self.hist_i])
        if self.hist_i == len(self.history):
            self.hist_i = None

    def on_mouse(self, btn: int, x: int, y: int, released: bool) -> None:
        st = self.st
        row, col = y - 1, x - 1
        if btn in (64, 65):  # the wheel
            if row < self.transcript_h:
                st.scroll = max(0, st.scroll + (3 if btn == 64 else -3))
            elif st.picker is not None:
                self.move_in_picker(-1 if btn == 64 else 1)
            return
        left = btn & 3 == 0
        motion = bool(btn & 32)
        if row >= self.transcript_h:  # under the conversation: a list item, or the end of a drag
            if self.selection is not None and self.selection.dragging:
                self.selection.head = (self.view_end - 1, 10**6)
                if released:
                    self.end_drag()
                return
            if released and left and row in self.menu_rows:
                idx = self.menu_rows[row]
                if self.palette_active():
                    items = self.palette_items()
                    if idx < len(items):
                        self.set_buf("")
                        self.run_command(items[idx][0])
                elif st.picker is not None:
                    st.picker.index = idx
                    self.choose()
            return
        at = (self.view_start + row, col)
        if not released and not motion and left:
            self.selection = Selection(at, at)
        elif motion and left and self.selection is not None and self.selection.dragging:
            self.selection.head = at
        elif released and self.selection is not None:
            self.selection.head = at
            self.end_drag()

    def end_drag(self) -> None:
        sel = self.selection
        if sel is None:
            return
        sel.dragging = False
        if sel.anchor == sel.head:
            self.selection = None
            r, c = sel.anchor
            if 0 <= r < len(self.view_rows):
                for a, b, url in links_in(self.view_rows[r]):
                    if a <= c < b:
                        open_target(url)
                        self.flash(f"opening {T.truncate(url, 50)}")
                        return
            return
        (r0, c0), (r1, c1) = sel.span()
        out = []
        for r in range(r0, min(r1, len(self.view_rows) - 1) + 1):
            text = plain_cols(self.view_rows[r])
            a = c0 if r == r0 else 0
            b = c1 + 1 if r == r1 else len(text)
            out.append(text[a:b].replace("\x00", "").rstrip())
        text = "\n".join(out).strip("\n")
        if not text.strip():
            return
        rows = text.count("\n") + 1
        done = f"copied {rows} lines" if rows > 1 else f"copied {len(text)} characters"
        self.copies += 1
        ticket = self.copies

        def copy() -> None:
            if ticket != self.copies:
                return  # a newer selection came while this one waited
            copy_to_clipboard(text, self.write_raw)
            self.call_soon(lambda: self.flash(done))

        threading.Thread(target=copy, name="revapi-copy", daemon=True).start()

    # -- pickers ---------------------------------------------------------------------------------------
    def open_picker(self, title: str, items: list[tuple[str, str, str]], current: str | None, on_pick: Callable[[str | None], None], **kw: Any) -> None:
        index = next((i for i, (v, _, _) in enumerate(items) if v == current), 0)
        self.st.picker = Picker(title, items, current, on_pick, index, **kw)
        self.dirty = True

    def move_in_picker(self, step: int) -> None:
        p = self.st.picker
        if p is not None:
            count = len(self.picker_items(p))
            p.index = around(min(p.index, count - 1) + step, count)

    def choose(self) -> None:
        p, self.st.picker = self.st.picker, None
        if p is None:
            return
        typed = self.buf.strip()
        if (p.question or p.typed) and typed:
            self.set_buf("")
            p.on_pick(typed)
            return
        items = self.picker_items(p)
        if not items:
            if p.search and typed:  # nothing matches: what was typed is the value (a model id the list lacks)
                self.set_buf("")
                p.on_pick(typed)
                return
            self.st.picker = p
            return
        if p.search:
            self.set_buf("")
        p.on_pick(items[min(p.index, len(items) - 1)][0])

    # -- settings --------------------------------------------------------------------------------------
    def shown(self, key: str) -> str:
        v = self.cfg.get(key)
        if key == "api_key":
            return "set" if v else "not set (environment variables are used)"
        if key == "effort":
            return v or "auto"
        if key == "output_dir":
            return v or "default · ~/.revapi/runs"
        if key == "model":
            return v or "provider default"
        if isinstance(v, bool):
            return "on" if v else "off"
        return "default" if v is None else str(v)

    def open_settings(self, at: str | None = None) -> None:
        items = [(k, label, self.shown(k)) for k, label, _ in SETTINGS]
        self.open_picker("settings · enter changes · esc closes", items, at, lambda key: self.change_setting(key) if key else None)
        if at:
            self.st.picker.current = None  # type: ignore[union-attr]

    def change_setting(self, key: str) -> None:
        kind = KINDS[key]
        label = LABELS[key]
        if kind == "toggle":
            self.save_setting(key, not bool(self.cfg.get(key)))
            self.open_settings(at=key)
        elif kind == "choice":
            current = self.cfg.get(key) or ("" if key == "effort" else None)
            self.open_picker(f"{label}", CHOICES[key], current, lambda v: self._picked_setting(key, v))
        elif kind == "model":
            self.pick_model(then=lambda: self.open_settings(at=key))
        else:
            hint = "type a value · enter saves · esc keeps it"
            sugg = SUGGESTIONS.get(key, [])
            self.open_picker(f"{label} · now {self.shown(key)}", sugg, None, lambda v: self._picked_setting(key, v), typed=True, hint=hint)

    def _picked_setting(self, key: str, value: str | None) -> None:
        if value is not None:
            self.save_setting(key, value)
        self.open_settings(at=key)

    def save_setting(self, key: str, value: Any) -> None:
        old = self.shown(key)
        if isinstance(value, str) and key == "compact_at":
            with contextlib.suppress(ValueError):
                value = float(value)
        try:
            self.cfg.set(key, value if value != "" else None)
        except ValueError as e:
            self.flash(str(e))
            self.notice(str(e))
            return
        new = self.shown(key)
        if old != new:
            self.event(LABELS.get(key, key), old if key != "api_key" else "…", new, "from the next run" if self.st.running or self.st.waiting else "")
        if key in ("provider", "model", "engine", "hermes_model", "collector_model", "api_key", "base_url"):
            self._refresh_model()

    def pick_model(self, then: Callable[[], None] | None = None) -> None:
        loading = Picker("loading the provider's models…", [], None, lambda v: None, hint="esc closes")
        self.st.picker = loading
        collector = self.st.mode == "collector"

        def load() -> None:
            try:
                llm = modes.make_llm(self.cfg, collector=collector)
                ids = llm.list_models()
                from .cli import SKIP_MODELS

                ids = [i for i in ids if not any(k in i.lower() for k in SKIP_MODELS)]
                err = None
            except Exception as e:  # noqa: BLE001
                ids, llm, err = [], None, str(e)

            def show() -> None:
                if self.st.picker is not loading:
                    return  # closed while it loaded
                key = "collector_model" if collector else "model"
                current = self.cfg.get(key) or (llm.model if llm else None)

                def picked(v: str | None) -> None:
                    if v:
                        self.set_model(v, collector)
                    if then:
                        then()

                provider = llm.provider if llm else self.cfg.get("provider")
                if err or not ids:
                    note = f"could not list models ({T.truncate(err or 'none returned', 60)})" if err or not ids else ""
                    self.open_picker(f"type a model id · provider:model switches provider · {note}", [], None, picked, typed=True, hint="e.g. nvidia:moonshotai/kimi-k2.6")
                    return
                items = [(i, i, "current" if i == current else "") for i in ids]
                if current and current not in ids:
                    items.insert(0, (current, current, "current"))
                self.open_picker(f"choose a model on {provider} · type to search · provider:model switches", items, current, picked, search=True, hint="type to search the models")

            self.call_soon(show)

        threading.Thread(target=load, name="revapi-models", daemon=True).start()

    def set_model(self, value: str, collector: bool = False) -> None:
        old = self.model_desc
        if ":" in value and value.split(":", 1)[0] in PROVIDERS:
            prov, _, m = value.partition(":")
            self.cfg.set("provider", prov)
            self.cfg.set("model", m or None)
        else:
            self.cfg.set("collector_model" if collector else "model", value)
        self._refresh_model()
        self.event("collector model" if collector else "model", old, self.model_desc, "from the next run" if self.st.running or self.st.waiting else "")

    def set_effort(self, value: str | None) -> None:
        if value is None:
            return
        old = self.effort
        self.cfg.set("effort", value or None)
        if old != self.effort:
            self.event("effort", old, self.effort, "from the next run" if self.st.running else "")

    def cycle_effort(self) -> None:
        i = EFFORTS.index(self.effort) if self.effort in EFFORTS else 0
        nxt = EFFORTS[(i + 1) % len(EFFORTS)]
        self.cfg.set("effort", None if nxt == "auto" else nxt)
        self.dirty = True

    def set_mode(self, value: str | None, quiet: bool = False) -> None:
        if value and value in MODES and value != self.st.mode:
            old, self.st.mode = self.st.mode, value
            self.cfg.set("repl_mode", value)
            self._refresh_model()
            if not quiet:
                self.event("mode", old, value)

    # -- commands --------------------------------------------------------------------------------------
    def run_command(self, text: str) -> None:  # noqa: C901  (one branch per command)
        cmd, _, arg = text.partition(" ")
        cmd, arg = ALIASES.get(cmd.lower(), cmd.lower()), arg.strip()
        st = self.st
        if cmd not in PICKER_COMMANDS | {"/exit", "/clear"} or arg:
            self.command(cmd, arg)
        if cmd == "/exit":
            self.quit()
        elif cmd == "/new":
            self.after_run = None
            if st.running:
                self.notice("a run is in progress · esc stops it")
            else:
                self.end_run()
                self.notice("ready · describe the API you want")
        elif cmd == "/help":
            self.help()
        elif cmd == "/model":
            if arg:
                self.set_model(arg, st.mode == "collector")
            else:
                self.pick_model()
        elif cmd == "/models":
            self.list_models(arg)
        elif cmd == "/effort":
            if arg:
                try:
                    self.set_effort("" if arg in ("auto", "none", "default") else arg)
                except ValueError as e:
                    self.notice(str(e))
            else:
                self.open_picker("how hard the agent thinks", [(("" if e == "auto" else e), e, EFFORT_NOTES[e]) for e in EFFORTS], self.cfg.get("effort") or "", self.set_effort)
        elif cmd == "/mode":
            if arg:
                self.set_mode(arg) if arg in MODES else self.notice(f"modes: {', '.join(MODES)}")
            else:
                self.open_picker("choose a mode", [(m, m, d) for m, d in MODES.items()], st.mode, self.set_mode)
        elif cmd == "/language":
            pick = lambda v: v and self.save_setting("output_language", v)  # noqa: E731
            if arg:
                pick(arg)
            else:
                self.open_picker("the client's language", CHOICES["output_language"], self.cfg.get("output_language"), pick)
        elif cmd == "/docs":
            self.save_setting("output_mode", "client" if self.cfg.get("output_mode") == "docs" else "docs")
        elif cmd == "/mcp":
            self.save_setting("output_mode", "client" if self.cfg.get("output_mode") == "mcp" else "mcp")
        elif cmd == "/settings":
            if arg:
                k, _, v = arg.partition(" ")
                if k in KINDS or k in store.DEFAULTS:
                    self.save_setting(k, v)
                else:
                    self.notice(f"unknown setting {k} · /settings lists them")
            else:
                self.open_settings()
        elif cmd == "/history":
            self.pick_run("/history", "past runs", self.show_run)
        elif cmd == "/resume":
            if arg:
                self.resume(arg)
            else:
                self.pick_run("/resume", "resume a run", self.resume, keep=self._resumable, empty="no run can be resumed yet")
        elif cmd == "/engineer":
            rid, _, extra = arg.partition(" ")
            if rid:
                self.reengineer(rid, extra or None)
            else:
                self.pick_run("/engineer", "re-engineer a run", lambda r: self.reengineer(r, None), keep=self._has_har, empty="no capture to engineer yet")
        elif cmd == "/run":
            parts = arg.split()
            self.run_client(parts[0] if parts else None, parts[1:])
        elif cmd == "/thinking":
            text = LAST_THINKING[0]
            if text:
                bar = sty("┊", HAIRLINE)
                self.add([f"      {bar} {sty(T.strip_ansi(ln), DIM, italic=True)}" for ln in T.render_markdown(text, T.cols() - 10, "", "")] + [""])
            else:
                self.notice("no reasoning yet — models that think show it here (try /effort high)")
        elif cmd in ("/context", "/compact"):
            self.context(cmd == "/compact")
        elif cmd == "/output":
            self.show_output()
        elif cmd == "/open":
            path = st.last_client or ((store.History().get("latest") or {}).get("paths") or {}).get("script_path")
            if path and Path(path).exists():
                open_target(Path(path).resolve().as_uri())
                self.notice(f"opening {path}")
            else:
                self.notice("no client yet")
        elif cmd == "/clear":
            if st.running:
                self.notice("a run is in progress · esc stops it")
            else:
                self.end_run()
                self.transcript.clear()
                self.selection = None
                st.scroll, st.cost, st.run_id, st.last_client, st.host, st.client = 0, 0.0, None, None, "", ""
        else:
            self.notice(f"unknown command {cmd} · / lists them")
        self.dirty = True

    def list_models(self, flt: str) -> None:
        def load() -> None:
            try:
                llm = modes.make_llm(self.cfg)
                ids = llm.list_models()
                from .cli import SKIP_MODELS

                ids = [i for i in ids if not any(k in i.lower() for k in SKIP_MODELS) and flt.lower() in i.lower()]
                lines = [f"   {sty('●', self.accent) if i == llm.model else ' '} {sty(i, INK_SOFT)}" for i in ids]
                lines.append(f"   {sty(f'{len(ids)} models on {llm.provider} · /model picks one', DIM)}")
                self.call_soon(lambda: self.add(lines + [""]))
            except Exception as e:  # noqa: BLE001
                msg = f"could not list models: {e}"
                self.call_soon(lambda: self.notice(msg))

        threading.Thread(target=load, daemon=True).start()

    def context(self, compact: bool) -> None:
        ag = self.keep.get("agent")
        if ag is None:
            self.notice("no conversation yet")
            return
        if compact:
            if self.st.running:
                self.notice("a run is in progress · esc stops it")
                return
            self.launch("compacting", lambda rep: (setattr(ag, "on_event", rep), ag.maybe_compact([t.schema() for t in ag.tools.values()], force=True), {})[-1], echo=False, verb="compacting context")
            return
        used, win = ag.context_tokens([t.schema() for t in ag.tools.values()]), ag._window()
        pct = round(100 * used / win) if win else 0
        fill = min(24, round(24 * used / win)) if win else 0
        color = MINT if pct < 60 else AMBER if pct < 80 else PINK
        self.add(
            [
                f"   {sty('▰' * fill, color)}{sty('▱' * (24 - fill), HAIRLINE)}  {sty(f'{T.k(used)} / {T.k(win)}', INK)} {sty(f'tokens · {pct}%', DIM)}",
                f"   {sty(f'compacts at {int(ag.compact_at * 100)}% · {ag.compactions} compacted · {len(ag.messages)} messages live · full history recallable', DIM)}",
            ]
        )

    # -- past runs -------------------------------------------------------------------------------------
    @staticmethod
    def _messages_path(run: dict[str, Any]) -> Path:
        return Path((run.get("paths") or {}).get("messages_path") or store.messages_path(run["run_id"], run.get("output_dir")))

    def _resumable(self, run: dict[str, Any]) -> bool:
        return self._messages_path(run).exists()

    def _has_har(self, run: dict[str, Any]) -> bool:
        hp = (run.get("paths") or {}).get("har_path")
        return bool(hp and Path(hp).exists()) or store.har_path(run["run_id"], run.get("output_dir")).exists()

    def pick_run(self, cmd: str, title: str, then: Callable[[str], None], keep: Callable[[dict[str, Any]], bool] | None = None, empty: str = "no runs yet") -> None:
        runs = [r for r in store.History().all() if keep is None or keep(r)][:HISTORY_RUNS]
        if not runs:
            self.command(cmd)
            self.notice(empty)
            return
        items = [(r["run_id"], r["run_id"], run_summary(r)) for r in runs]
        self.open_picker(title, items, None, lambda v: then(v) if v else None)

    def show_run(self, run_id: str) -> None:
        run = store.History().get(run_id)
        self.command("/history", run_id)
        if run is None:
            self.notice(f"no run {run_id}")
            return
        self.run_details(run)
        actions = [("resume", "resume", "continue its conversation")] if self._resumable(run) else []
        if self._has_har(run):
            actions.append(("engineer", "re-engineer", "write its client again from the capture"))
        if (run.get("paths") or {}).get("script_path"):
            actions.append(("run", "run", "run its client"))
        actions.append(("", "back", ""))

        def act(v: str | None) -> None:
            if v == "resume":
                self.resume(run_id)
            elif v == "engineer":
                self.reengineer(run_id, None)
            elif v == "run":
                self.run_client(run_id, [])

        self.open_picker(f"run {run_id}", actions, None, act)

    def resume(self, run_id: str) -> None:
        st = self.st
        run = store.History().get(run_id)
        self.command("/resume", run_id)
        if st.running:
            self.notice("a run is in progress · esc stops it")
            return
        if run is None:
            self.notice(f"no run {run_id}")
            return
        self.end_run()
        if not self.replay(run):
            self.run_details(run)
        try:
            rep = self.new_reporter(run.get("mode", "agent"))
            self.keep = modes.resume(run["run_id"], self.cfg, rep, interactive=True)
        except (ValueError, LLMError) as e:
            self.notice(f"run {run_id} cannot continue: {e} · /history re-engineers it")
            self.keep = {}
            return
        mode = self.keep.get("mode", "agent")
        if mode in MODES:
            st.mode = mode
        st.waiting, st.run_id = True, run["run_id"]
        st.last_client = (run.get("paths") or {}).get("script_path")
        self.gap()
        n = len(self.keep["agent"].messages) if self.keep.get("agent") is not None else 0
        self.notice(f"resumed {run['run_id']}" + (f" · {n} messages in context" if n else "") + " · type a follow-up")

    def replay(self, run: dict[str, Any]) -> bool:
        """Draw a past run's saved conversation the way it looked live, so a resumed one picks up in view."""
        path = self._messages_path(run)
        try:
            recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except (OSError, ValueError):
            return False
        if not recs:
            return False
        rep = self.new_reporter(run.get("mode", "agent"))
        rep.host = host_in(str(run.get("prompt") or ""))
        rep.run_header(run.get("mode", "agent"), run["run_id"], run.get("model") or "", "replay")
        names: dict[str, str] = {}
        first = True
        try:
            for m in recs:
                if m.get("type") == "compaction":
                    rep("compaction", {"before": m.get("before", 0), "after": m.get("after", 0), "window": 0, "method": m.get("method"), "count": rep.compactions + 1})
                    continue
                role = m.get("role")
                if role == "user":
                    if m.get("_summary"):
                        continue
                    self.user(str(run.get("prompt") or user_words(m.get("content"))) if first else user_words(m.get("content")))
                    first = False
                elif role == "assistant":
                    if m.get("reasoning"):
                        rep("thinking", {"text": m["reasoning"]})
                    if (m.get("content") or "").strip():
                        rep("text", {"text": m["content"]})
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        names[tc.get("id", "")] = fn.get("name", "")
                        rep("tool_start", {"name": fn.get("name", ""), "input": args if isinstance(args, dict) else {}})
                elif role == "tool":
                    name = m.get("name") or names.get(m.get("tool_call_id", ""), "")
                    rep("tool_end", {"name": name, "is_error": bool(m.get("_error") or m.get("is_error")), "output": str(m.get("content") or ""), "duration_ms": 0})
            script = (run.get("paths") or {}).get("script_path")
            if script and Path(script).exists() and run.get("mode") != "collector":
                rep.usage = run.get("usage") or {}
                rep.card(script, run.get("model") or "")
        finally:
            rep.stop()
            self.rep = None
        return True

    def reengineer(self, run_id: str, extra: str | None) -> None:
        if self.st.running:
            self.notice("a run is in progress · esc stops it")
            return
        self.end_run()
        self.command("re-engineer", run_id + (f" {extra}" if extra else ""))
        self.launch(run_id, lambda rep: modes.run_engineer(run_id, self.cfg, rep, prompt=extra, interactive=True, keep=self.keep), mode="engineer", echo=False)

    def run_client(self, run_id: str | None, args: list[str]) -> None:
        if self.st.running:
            self.notice("a run is in progress · esc stops it")
            return
        rid = run_id or self.st.run_id or "latest"
        self.launch(f"run {rid}", lambda rep: modes.run_script(rid, self.cfg, rep, args=args, interactive=True) or {}, echo=False, verb="running the client", kind="run")

    # -- runs ------------------------------------------------------------------------------------------
    def new_reporter(self, mode: str) -> Reporter:
        rep = Reporter(mode=mode if mode in MODES else "agent", show_thinking=self.cfg.get("show_thinking", True) is not False and not os.environ.get("REVAPI_HIDE_THINKING"))
        rep.live = self.live  # type: ignore[assignment]
        rep.tty = True
        self.rep = rep
        return rep

    def submit(self, text: str) -> None:
        st = self.st
        if isinstance(st.question, Future):
            fut, st.question = st.question, None
            self.add([f"      {sty('┊', HAIRLINE)} {sty(text or '(no answer)', INK_SOFT)}", ""])
            if not fut.done():
                fut.set_result(text)
            return
        if not text:
            return
        if text.startswith("/"):
            self.run_command(text)
            return
        st.scroll = 0
        if st.running:
            st.queue.append(text)
        else:
            self.start(text)

    def start(self, prompt: str) -> None:
        st = self.st
        mode = st.mode
        keep = self.keep
        if st.waiting and (keep.get("agent") is not None or keep.get("hermes") is not None):
            self.launch(prompt, lambda rep: modes.follow_up(keep, prompt, rep), mode=keep.get("mode") or mode)
            return
        self.end_run()
        st.run_id = None
        keep = self.keep
        url_m = URL_RE.search(prompt)
        url = url_m.group(0) if url_m else None
        runner = self.runners.get(mode)
        fn: Callable[[Reporter], dict]
        if runner is not None:
            fn = lambda rep: runner(prompt, rep, keep)  # noqa: E731
        elif mode == "manual":
            fn = lambda rep: modes.run_manual(self.cfg, rep, None if (url and url == prompt) else prompt, url, interactive=True, keep=keep)  # noqa: E731
        elif mode == "engineer":
            hist = store.History()
            rid = prompt if " " not in prompt and hist.get(prompt) else (hist.get("latest") or {}).get("run_id")
            extra = None if rid == prompt else prompt
            if not rid:
                self.user(prompt)
                self.error("no runs to engineer yet · capture one in agent or manual mode")
                return
            fn = lambda rep: modes.run_engineer(rid, self.cfg, rep, prompt=extra, interactive=True, keep=keep)  # noqa: E731
        elif mode == "collector":
            fn = lambda rep: modes.run_collector(prompt, self.cfg, rep, interactive=True, keep=keep)  # noqa: E731
        else:
            fn = lambda rep: modes.run_agent(prompt, url, self.cfg, rep, interactive=True, keep=keep)  # noqa: E731
        self.launch(prompt, fn, mode=mode)

    def launch(self, prompt: str, fn: Callable[[Reporter], dict], mode: str | None = None, echo: bool = True, verb: str | None = None, kind: str = "other") -> None:
        """Start a run in the session; its Reporter streams into the conversation."""
        st = self.st
        mode = mode or st.mode
        if echo:
            self.user(prompt)
        st.host, st.client = host_in(prompt), ""
        st.started, st.running, st.stopping, st.kind = time.time(), True, False, kind
        st.verb = verb or ("recording" if mode == "manual" else "starting")
        rep = self.new_reporter(mode)
        self._interrupted = False

        def target() -> None:
            result: dict[str, Any] = {}
            try:
                result = fn(rep) or {}
            except LLMError as e:
                result = {"status": "error", "error": str(e), "error_kind": e.kind}
            except KeyboardInterrupt:
                result = {"status": "error", "error": "interrupted", "error_kind": "interrupted"}
            except BaseException as e:  # noqa: BLE001  (a run must never take the session down)
                result = {"status": "error", "error": f"{type(e).__name__}: {e}", "error_kind": "unknown"}
                if os.environ.get("REVAPI_DEBUG"):
                    result["traceback"] = traceback.format_exc()
            finally:
                self.call_soon(lambda: self.finish(result, rep))

        self.worker = threading.Thread(target=target, name="revapi-run", daemon=True)
        self.worker.start()

    def finish(self, result: dict[str, Any], rep: Reporter) -> None:
        st = self.st
        with contextlib.suppress(Exception):
            rep._close_stream()
        rep.stop()
        if st.picker is not None and st.picker.question:
            st.picker = None
        if isinstance(st.question, Future) and not st.question.done():
            st.question.set_result("")
        st.question = None
        self.results.append(result)
        err = result.get("error")
        if err:
            if result.get("error_kind") == "interrupted" or err == "interrupted":
                if not any("interrupted" in T.strip_ansi(x) for it in self.transcript.items[-3:] for x in it.lines):
                    self.add([f"      {sty('┊', HAIRLINE)} {sty('interrupted', PINK)} {sty('· everything so far is saved; /resume continues', DIM)}", ""])
            else:
                self.error(str(err))
                if result.get("traceback"):
                    self.add([f"    {sty(ln, DIM)}" for ln in result["traceback"].splitlines()])
                if result.get("error_kind") in ("network", "rate_limit", "engine_failure", "max_turns") and result.get("run_id"):
                    self.notice(f"nothing is lost · type a follow-up, or /resume {result['run_id']} later")
        if result.get("summary") and not rep.summary_shown and not err:
            self.add(T.render_markdown(str(result["summary"]).strip(), T.cols() - 1, "   ", f" {sty('¶', self.accent)} ") + [""])
        if result.get("har_path"):
            p = result["har_path"]
            self.add([f"   {sty('└', HAIRLINE)} {sty('capture', DIM)} {sty(T.link(str(p), Path(str(p)).resolve().as_uri()), INK_SOFT)}"])
        st.cost += float((result.get("usage") or {}).get("estimated_cost_usd") or 0) if not result.get("usage_turn") else float(result["usage_turn"].get("estimated_cost_usd") or 0)
        if result.get("script_path"):
            st.last_client = result["script_path"]
        if result.get("run_id"):
            st.run_id = result["run_id"]
        st.waiting = self.keep.get("agent") is not None or self.keep.get("hermes") is not None
        st.running, st.stopping = False, False
        self.rep = None
        self.dirty = True
        after, self.after_run = self.after_run, None
        if after is not None:
            after()
        elif st.queue:
            self.start(st.queue.pop(0))

    def end_run(self) -> None:
        """Close a run that waits for a follow-up."""
        ctx = self.keep.get("ctx")
        if ctx is not None:
            with contextlib.suppress(Exception):
                ctx.close_browser()
        self.keep.clear()
        self.st.waiting = False
        self.dirty = True

    def stop(self) -> None:
        st = self.st
        if st.running and st.stopping:
            st.queue.clear()
            self.after_run = None
            self.notice("still wrapping up · it ends as soon as its cleanup does")
            return
        if st.running:
            st.queue.clear()
            st.stopping = True
            if isinstance(st.question, Future) and not st.question.done():
                st.question.set_result("")
                st.question = None
            st.picker = None if (st.picker and st.picker.question) else st.picker
            if not self._interrupt_worker():
                self.notice("this run cannot be stopped from here")
        elif st.waiting:
            self.end_run()
            self.notice("run closed · describe the next API")

    def _interrupt_worker(self) -> bool:
        w = self.worker
        if w is None or not w.is_alive() or w.ident is None:
            return False
        if self._interrupted:
            return True
        try:
            import ctypes

            n = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(w.ident), ctypes.py_object(KeyboardInterrupt))
        except Exception:  # noqa: BLE001
            return False
        self._interrupted = n == 1
        return n == 1

    def show_output(self) -> None:
        lines = self.transcript.full_lines()
        header = self.header_rows(0.0, T.cols() - 1, 0)
        text = "\n".join(header + lines) + "\n"
        with self.suspended():
            pager = shutil.which("less")
            if pager and T.COLOR:
                with contextlib.suppress(OSError, KeyboardInterrupt):
                    subprocess.run([pager, "-R", "+G"], input=text.encode("utf-8", errors="replace"), env={**os.environ, "LESSCHARSET": "utf-8"}, check=False)
            else:
                sys.stdout.write(text)
                sys.stdout.write(T.sty("\n  press enter to go back ", DIM))
                sys.stdout.flush()
                with contextlib.suppress(EOFError, KeyboardInterrupt):
                    _REAL_INPUT()

    # -- questions from a run (ask_user, any input()) --------------------------------------------------
    def _wait(self, fut: Future) -> str:
        while True:  # short waits, so a stop (an async KeyboardInterrupt) lands at once
            try:
                return fut.result(timeout=0.2)
            except FutureTimeout:
                continue

    def ask(self, question: str, options: list[str] | None) -> str:
        fut: Future = Future()

        def show() -> None:
            self.gap()
            self.add([f" {sty('?', AMBER, bold=True)} {sty(question, INK)}"])
            if options:
                items = [(o, o, "") for o in options] + [("", "something else", "type your own answer")]

                def picked(v: str | None) -> None:
                    if v == "" or v is None:
                        if v == "":
                            self.st.question = fut
                        elif not fut.done():
                            fut.set_result("")
                    elif not fut.done():
                        self.add([f"      {sty('┊', HAIRLINE)} {sty(v, INK_SOFT)}", ""])
                        fut.set_result(v)

                self.open_picker("choose an answer · or type your own", items, None, picked, question=True, hint="↑↓ and enter, or type an answer")
            else:
                self.st.question = fut
            self.dirty = True

        self.call_soon(show)
        return self._wait(fut)

    def input(self, prompt: Any = "") -> str:
        text = T.strip_ansi(str(prompt or "")).strip()
        fut: Future = Future()

        def show() -> None:
            if text:
                self.gap()
                self.add([f" {sty('?', AMBER, bold=True)} {sty(text, INK)}"])
            self.st.question = fut
            self.dirty = True

        self.call_soon(show)
        return self._wait(fut)

    # -- lifecycle -------------------------------------------------------------------------------------
    def quit(self) -> None:
        self.st.queue.clear()
        self.after_run = None
        if self.st.running:
            self.st.stopping = True
            self._interrupt_worker()
        if isinstance(self.st.question, Future) and not self.st.question.done():
            self.st.question.set_result("")
        self.quitting = True

    def animating(self) -> bool:
        st = self.st
        if st.running or (st.flash and time.time() < st.flash[1]):
            return True
        logo_visible = self.view_start < TOP_PADDING + LOGO_ROWS and self.size[0] >= 80 and self.size[1] >= 24
        return logo_visible

    def loop(self) -> None:
        last_draw = 0.0
        pending_esc = 0.0
        while not self.quitting:
            while self.calls:
                fn = self.calls.popleft()
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    self.error(f"{type(e).__name__}: {e}")
                self.dirty = True
            now = time.time()
            interval = 1 / 20 if self.animating() else 0.5
            if self.dirty or now - last_draw >= interval:
                self.draw()
                self.dirty = False
                last_draw = now
            timeout = 0.03 if pending_esc else max(0.0, min(interval, interval - (time.time() - last_draw)))
            try:
                ready, _, _ = select.select([self.in_fd], [], [], timeout if not self.calls else 0)
            except InterruptedError:
                continue
            if ready:
                try:
                    data = os.read(self.in_fd, 65536)
                except OSError:
                    data = b""
                if not data:
                    self.quit()
                    break
                events = self.keys.feed(data)
                pending_esc = time.time() if self.keys.buf else 0.0
            elif self.keys.buf and pending_esc and time.time() - pending_esc > 0.025:
                events = self.keys.parse(final=True)
                pending_esc = 0.0
            else:
                continue
            for ev in events:
                try:
                    self.on_event(ev)
                except Exception as e:  # noqa: BLE001
                    self.error(f"{type(e).__name__}: {e}")

    def print_transcript(self) -> None:
        """Leave the session in the terminal's history, like a normal command's output."""
        width = T.cols() - 1
        out = self.header_rows(0.0, width, T.rows()) + self.transcript.rows(width)
        while out and not out[-1].strip():
            out.pop()
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()

    def run(self) -> int:
        global _ACTIVE
        _ACTIVE = self
        self.capture = Capture(self)
        orig_input, orig_ask = builtins.input, modes.terminal_ask
        builtins.input = self.input  # type: ignore[assignment]
        modes.terminal_ask = self.ask  # type: ignore[assignment]
        handlers = {}
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                with contextlib.suppress(ValueError, OSError):
                    handlers[name] = signal.signal(getattr(signal, name), lambda *_: self.quit())
        if hasattr(signal, "SIGWINCH"):
            with contextlib.suppress(ValueError, OSError):
                handlers["SIGWINCH"] = signal.signal(signal.SIGWINCH, lambda *_: setattr(self, "dirty", True))
        self.enter_screen()
        sys.stdout = sys.stderr = self.capture  # type: ignore[assignment]
        threading.Thread(target=self._check_update, name="revapi-update", daemon=True).start()
        if self.first is not None:
            prompt, mode = self.first
            self.first = None
            if mode:
                self.set_mode(mode, quiet=True)
            self.call_soon(lambda: self.start(prompt))
        try:
            self.loop()
        finally:
            sys.stdout, sys.stderr = self.real_out, self.real_err
            self.leave_screen()
            builtins.input, modes.terminal_ask = orig_input, orig_ask  # type: ignore[assignment]
            for name, h in handlers.items():
                with contextlib.suppress(ValueError, OSError, TypeError):
                    signal.signal(getattr(signal, name), h)
            _ACTIVE = None
        w = self.worker
        if w is not None and w.is_alive():
            print(sty(" stopping the run…", DIM), file=sys.stderr, flush=True)
            deadline = time.time() + 60
            while w.is_alive() and time.time() < deadline:
                w.join(3)  # a run that caught the first stop (manual capture ends on ctrl+c) gets another
                self._interrupted = False
                if w.is_alive():
                    self._interrupt_worker()
        with contextlib.suppress(Exception):
            self.end_run()
        while self.calls:  # the finished run's last lines
            with contextlib.suppress(Exception):
                self.calls.popleft()()
        self.print_transcript()
        return 0

    def _check_update(self) -> None:
        note = modes.update_notice()
        if note:
            self.call_soon(lambda: self.notice(note))


_ACTIVE: Session | None = None
_REAL_INPUT = builtins.input


def run(cfg: store.Config, mode: str | None = None, first: tuple[str, str | None] | None = None, runners: dict[str, Callable[[str, Reporter, dict], dict]] | None = None) -> int:
    """Open the full-screen session (optionally starting a run a subcommand handed over)."""
    s = Session(cfg, mode)
    s.first = first
    s.runners = runners or {}
    return s.run()
