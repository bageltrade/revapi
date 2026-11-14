"""Terminal rendering with nothing but the standard library.

Colors follow reverse-api-engineer's design tokens (dark terminal palette). Provides ANSI-aware width,
wrapping and truncation, a streaming Markdown renderer, a small syntax highlighter, and a live region
(spinner + streaming text) redrawn in place below normal output.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from collections.abc import Callable

# ----------------------------------------------------------------------------- palette (from rae's theme)

INK = "#fff7f0"
INK_SOFT = "#c9bdb3"
DIM = "#8a7d72"
PINK = "#ff3d8b"
MINT = "#7ec99a"
SKY = "#7eb8ff"
AMBER = "#e8b86d"
HAIRLINE = "#3a332d"
PASTEL_READ = "#bfe0ff"
PASTEL_WRITE = "#c3ecd2"
PASTEL_RUN = "#ffd6b8"
PASTEL_OTHER = "#d9cff0"
CHIP_LAVENDER = "#e8e3f0"
CHIP_YELLOW = "#fdf08a"
CHIP_INK = "#1f1f1f"
MODE_COLORS = {"agent": PINK, "manual": MINT, "engineer": SKY, "collector": AMBER}
KIND_TINT = {"read": PASTEL_READ, "write": PASTEL_WRITE, "run": PASTEL_RUN, "other": PASTEL_OTHER}

RESET = "\033[0m"


def _detect(stream: object = None) -> int:
    """0 = no color, 256 = xterm-256, 24 = truecolor."""
    stream = stream or sys.stderr
    if os.environ.get("NO_COLOR") or os.environ.get("REVAPI_NO_COLOR"):
        return 0
    forced = os.environ.get("FORCE_COLOR") or os.environ.get("REVAPI_FORCE_COLOR")
    if not forced and not (hasattr(stream, "isatty") and stream.isatty()):
        return 0
    if os.environ.get("TERM") == "dumb" and not forced:
        return 0
    ct = os.environ.get("COLORTERM", "").lower()
    termux = bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in os.environ.get("PREFIX", "")
    tp = os.environ.get("TERM_PROGRAM", "")
    if ct in ("truecolor", "24bit") or termux or tp in ("iTerm.app", "WezTerm", "vscode", "ghostty", "Hyper", "Tabby") or os.environ.get("WT_SESSION") or os.environ.get("KITTY_WINDOW_ID"):
        return 24
    return 256


COLOR = _detect()


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _x256(r: int, g: int, b: int) -> int:
    if abs(r - g) < 10 and abs(g - b) < 10:  # grey ramp
        if r < 8:
            return 16
        if r > 238:
            return 231
        return 232 + round((r - 8) / 247 * 24)

    def q(v: int) -> int:
        return 0 if v < 48 else 1 if v < 115 else (v - 35) // 40

    return 16 + 36 * q(r) + 6 * q(g) + q(b)


def _rgb(h: str, bg: bool = False) -> str:
    r, g, b = _hex(h)
    if COLOR == 24:
        return f"{48 if bg else 38};2;{r};{g};{b}"
    return f"{48 if bg else 38};5;{_x256(r, g, b)}"


def seq(fg: str | None = None, bg: str | None = None, bold: bool = False, italic: bool = False, dim: bool = False, underline: bool = False, strike: bool = False) -> str:
    if not COLOR:
        return ""
    parts = []
    if bold:
        parts.append("1")
    if dim:
        parts.append("2")
    if italic:
        parts.append("3")
    if underline:
        parts.append("4")
    if strike:
        parts.append("9")
    if fg:
        parts.append(_rgb(fg))
    if bg:
        parts.append(_rgb(bg, True))
    return f"\033[{';'.join(parts)}m" if parts else ""


def sty(text: str, fg: str | None = None, **kw: object) -> str:
    s = seq(fg, **kw)  # type: ignore[arg-type]
    return f"{s}{text}{RESET}" if s and text else text


def paint(text: str, base: str) -> str:
    """Give the whole text a base style that survives inner resets."""
    if not base or not COLOR:
        return text
    return base + text.replace(RESET, RESET + base) + RESET


def link(text: str, url: str) -> str:
    """OSC 8 hyperlink (clickable in Termux, iTerm, WezTerm, kitty, VS Code…)."""
    if not COLOR or os.environ.get("REVAPI_NO_LINKS"):
        return text
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


# ----------------------------------------------------------------------------- width, wrap, truncate

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\]8;[^\x1b]*\x1b\\|\x1b\][^\x07]*\x07")
_ZERO = {"\u200d", "\ufe0f", "\ufe0e", "\u200b"}


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def char_width(ch: str) -> int:
    if ch in _ZERO or unicodedata.combining(ch):
        return 0
    if ord(ch) < 32:
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def vis_len(s: str) -> int:
    return sum(char_width(ch) for ch in strip_ansi(s))


def cols() -> int:
    for fd in (2, 1, 0):
        try:
            return max(20, os.get_terminal_size(fd).columns)
        except OSError:
            continue
    return max(20, shutil.get_terminal_size((80, 24)).columns)


def rows() -> int:
    for fd in (2, 1, 0):
        try:
            return max(8, os.get_terminal_size(fd).lines)
        except OSError:
            continue
    return 24


_TOKEN_RE = re.compile(r"(\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\]8;[^\x1b]*\x1b\\)|(\s+)|([^\s\x1b]+)")


def wrap(text: str, width: int, first: str = "", rest: str | None = None) -> list[str]:
    """Word-wrap ANSI text to ``width`` visible columns; styles carry across the line breaks."""
    rest = first if rest is None else rest
    width = max(8, width)
    lines: list[str] = []
    active: list[str] = []
    link_open = ""
    cur, cur_w = first, vis_len(first)
    pending_space = ""
    start_w = cur_w

    def newline() -> None:
        nonlocal cur, cur_w, pending_space, start_w
        lines.append(cur + (RESET if active and COLOR else "") + ("\033]8;;\033\\" if link_open else ""))
        cur = rest + "".join(active) + link_open
        cur_w = vis_len(rest)
        start_w = cur_w
        pending_space = ""

    for m in _TOKEN_RE.finditer(text):
        esc, space, word = m.groups()
        if esc:
            if esc.startswith("\x1b]8;"):
                link_open = "" if esc == "\x1b]8;;\x1b\\" else esc
            elif esc == RESET or esc == "\x1b[m":
                active = []
            else:
                active.append(esc)
            cur += esc
            continue
        if space:
            if "\n" in space:
                for _ in range(space.count("\n")):
                    newline()
                continue
            pending_space = " " if cur_w > start_w else ""
            continue
        w = vis_len(word)
        if cur_w + len(pending_space) + w > width and cur_w > start_w:
            newline()
        if cur_w + len(pending_space) + w <= width:
            cur += pending_space + word
            cur_w += len(pending_space) + w
            pending_space = ""
            continue
        # a word longer than the line: break it by characters
        cur += pending_space
        cur_w += len(pending_space)
        pending_space = ""
        for ch in word:
            cw = char_width(ch)
            if cur_w + cw > width and cur_w > start_w:
                newline()
            cur += ch
            cur_w += cw
    lines.append(cur + (RESET if active and COLOR else ""))
    return lines


def truncate(s: str, width: int, ellipsis: str = "…") -> str:
    """Cut ANSI text to ``width`` visible columns (never wraps, so live redraws stay aligned)."""
    if vis_len(s) <= width:
        return s
    out, w = [], 0
    limit = max(0, width - len(ellipsis))
    for m in re.finditer(r"(\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\]8;[^\x1b]*\x1b\\)|(.)", s, re.S):
        esc, ch = m.groups()
        if esc:
            out.append(esc)
            continue
        cw = char_width(ch)
        if w + cw > limit:
            break
        out.append(ch)
        w += cw
    return "".join(out) + ellipsis + (RESET if COLOR else "")


def tail_fit(s: str, width: int) -> str:
    """The end of a plain string that fits in ``width`` columns (for the line being streamed)."""
    w, i = 0, len(s)
    while i > 0 and w + char_width(s[i - 1]) <= width:
        i -= 1
        w += char_width(s[i])
    return ("…" + s[i + 1 :]) if i > 0 else s


# ----------------------------------------------------------------------------- syntax highlighting

_KW = {
    "python": "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case",
    "js": "async await break case catch class const continue debugger default delete do else export extends finally for from function if import in instanceof let new of return static super switch this throw try typeof var void while with yield interface type enum implements private public protected readonly as",
    "go": "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var",
    "java": "abstract assert break case catch class const continue default do else enum extends final finally for if implements import instanceof interface native new package private protected public return static super switch synchronized this throw throws try void volatile while var record",
    "csharp": "abstract as async await base break case catch class const continue default delegate do else enum event explicit extern finally fixed for foreach if implicit in interface internal is lock namespace new operator out override params private protected public readonly ref return sealed sizeof static struct switch this throw try typeof unchecked unsafe using var virtual void volatile while",
    "c": "auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while include define ifdef ifndef endif",
    "php": "abstract and array as break callable case catch class clone const continue declare default do echo else elseif empty enddeclare endfor endforeach endif endswitch endwhile extends final finally fn for foreach function global goto if implements include instanceof insteadof interface isset list match namespace new or print private protected public require return static switch throw trait try unset use var while xor yield",
    "ruby": "alias and begin break case class def defined do else elsif end ensure for if in module next not or redo rescue retry return self super then undef unless until when while yield require",
    "bash": "if then else elif fi case esac for while until do done in function select time export local return exit set unset source alias echo cd pip pkg apt npm curl",
    "powershell": "begin break catch class continue data define do dynamicparam else elseif end exit filter finally for foreach from function if in param process return switch throw trap try until using var while",
}
_CONST = {"True", "False", "None", "true", "false", "null", "nil", "undefined", "NaN", "self", "this"}
_ALIASES = {
    "py": "python",
    "python3": "python",
    "javascript": "js",
    "typescript": "js",
    "ts": "js",
    "jsx": "js",
    "tsx": "js",
    "mjs": "js",
    "node": "js",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "console": "bash",
    "golang": "go",
    "cs": "csharp",
    "c#": "csharp",
    "cpp": "c",
    "c++": "c",
    "h": "c",
    "rb": "ruby",
    "ps1": "powershell",
    "psm1": "powershell",
    "pwsh": "powershell",
    "kotlin": "java",
    "kt": "java",
    "swift": "js",
    "rust": "js",
    "rs": "js",
}
_HASH_COMMENT = {"python", "ruby", "bash", "powershell", "yaml", "toml"}


def lang_of(name: str) -> str:
    n = (name or "").lower().strip()
    return _ALIASES.get(n, n)


def highlight(line: str, lang: str) -> str:
    """One line of code in rae's code colors (keywords pink, strings mint, numbers sky, defs amber)."""
    lang = lang_of(lang)
    if not COLOR:
        return line
    if lang == "json":
        return re.sub(
            r'("(?:\\.|[^"\\])*")(\s*:)?|(-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|\b(true|false|null)\b',
            lambda m: sty(m.group(1), AMBER) + m.group(2) if m.group(1) and m.group(2) else sty(m.group(1), MINT) if m.group(1) else sty(m.group(0), SKY),
            line,
        )
    kws = set(_KW.get(lang, "").split())
    comment = r"#.*$" if lang in _HASH_COMMENT else r"//.*$|/\*.*?\*/" + (r"|#.*$" if lang == "php" else "")
    if not lang or lang not in _KW:
        comment = r"(?<![:\w])//.*$|#.*$"
    rx = re.compile(
        rf"(?P<com>{comment})|(?P<str>\"\"\".*?\"\"\"|'''.*?'''|[fbrFBR]?\"(?:\\.|[^\"\\])*\"|[fbrFBR]?'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)"
        r"|(?P<num>\b0[xX][0-9a-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(?P<dec>@[\w.]+)|(?P<word>[A-Za-z_$][\w$]*)|(?P<pun>[^\w\s])"
    )
    out: list[str] = []
    pos = 0
    prev_word = ""
    for m in rx.finditer(line):
        out.append(line[pos : m.start()])
        pos = m.end()
        tok = m.group(0)
        kind = m.lastgroup
        if kind == "com":
            out.append(sty(tok, DIM, italic=True))
        elif kind == "str":
            out.append(sty(tok, MINT))
        elif kind == "num":
            out.append(sty(tok, SKY))
        elif kind == "dec":
            out.append(sty(tok, AMBER))
        elif kind == "word":
            nxt = line[pos : pos + 1]
            if tok in kws:
                out.append(sty(tok, PINK, bold=True))
            elif tok in _CONST:
                out.append(sty(tok, SKY, italic=tok in ("self", "this")))
            elif prev_word in ("def", "class", "function", "func", "fn", "struct", "interface", "type") or nxt == "(":
                out.append(sty(tok, AMBER, bold=prev_word in ("def", "class", "function", "func", "fn")))
            else:
                out.append(sty(tok, INK))
            prev_word = tok
            continue
        else:
            out.append(sty(tok, INK_SOFT))
        if kind != "pun" or tok not in "().,":
            prev_word = "" if kind != "word" else prev_word
    out.append(line[pos:])
    return "".join(out)


# ----------------------------------------------------------------------------- receipts (tool output lines)

_RECEIPT_RE = re.compile(
    r"(?P<ok>^\s*[23]\d\d\b)|(?P<fail>^\s*[45]\d\d\b)|(?P<method>\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b)"
    r"|(?P<url>https?://[^\s·\"'<>)]+|(?<![\w/.])/(?:[\w\-.%{}]+/?)+(?:\?[\w\-.=&%{}]*)?)|(?P<string>\"[^\"]{0,200}\")"
    r"|(?P<key>\b[\w-]+(?=: ))|(?P<file>\b[\w\-]+\.(?:py|json|md|ts|js|mjs|csv|go|java|rb|php|cs|c|h|psm1|ps1|har|jsonl|sh|txt|html)\b)"
    r"|(?P<number>(?<![\w/.#])\d[\d,.]*(?:\s?(?:kB|KB|MB|ms|s|B))?\b)"
)
_RECEIPT_STYLE = {
    "ok": (MINT, {"bold": True}),
    "fail": (PINK, {"bold": True}),
    "method": (SKY, {"bold": True}),
    "url": (INK_SOFT, {"underline": True}),
    "string": (MINT, {}),
    "key": (AMBER, {}),
    "file": (INK, {"underline": True}),
    "number": (SKY, {}),
}


def receipt_line(text: str, base: str = DIM) -> str:
    if not COLOR:
        return text
    b = seq(base)

    def rep(m: re.Match) -> str:
        fg, kw = _RECEIPT_STYLE[m.lastgroup or "number"]
        return seq(fg, **kw) + m.group(0) + RESET + b  # type: ignore[arg-type]

    return b + _RECEIPT_RE.sub(rep, text) + RESET


# ----------------------------------------------------------------------------- markdown

_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BOLD = re.compile(r"(?<![\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])|(?<![\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])")
_ITAL = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])|(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_BARE_URL = re.compile(r"(?<![\w(\x00\"'=])(https?://[^\s<>)\]]+[^\s<>)\].,;:!?'\"])")


def inline(text: str, base: str = INK) -> str:
    if not COLOR:
        text = _LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})" if m.group(1) != m.group(2) else m.group(1), text)
        return _INLINE_CODE.sub(lambda m: m.group(2), _BOLD.sub(lambda m: m.group(1) or m.group(2), text))
    keep: list[str] = []

    def hold(s: str) -> str:
        keep.append(s)
        return f"\x00{len(keep) - 1}\x00"

    text = _INLINE_CODE.sub(lambda m: hold(sty(m.group(2).strip() if len(m.group(1)) > 1 else m.group(2), MINT)), text)
    text = _LINK.sub(lambda m: hold(link(sty(m.group(1), SKY, underline=True), m.group(2))), text)
    text = _BARE_URL.sub(lambda m: hold(link(sty(m.group(1), SKY, underline=True), m.group(1))), text)
    text = _BOLD.sub(lambda m: hold(sty(m.group(1) or m.group(2), INK, bold=True)), text)
    text = _STRIKE.sub(lambda m: hold(sty(m.group(1), DIM, strike=True)), text)
    text = _ITAL.sub(lambda m: hold(sty(m.group(1) or m.group(2), PINK, italic=True)), text)
    for _ in range(4):
        if "\x00" not in text:
            break
        text = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], text)
    return paint(text, seq(base))


class Markdown:
    """Renders Markdown line by line, so it can follow a stream. Each call returns finished screen lines."""

    def __init__(self, width: Callable[[], int] | int, margin: str = "   ", first_margin: str | None = None) -> None:
        self._width = width
        self.margin = margin
        self.first_margin = first_margin
        self.code: str | None = None  # language while inside a fence
        self.fence = ""
        self.table: list[str] = []
        self.emitted = 0
        self.blank = False

    @property
    def width(self) -> int:
        w = self._width() if callable(self._width) else self._width
        return max(20, min(w, 110))

    def _m(self) -> str:
        if self.emitted == 0 and self.first_margin is not None:
            return self.first_margin
        return self.margin

    def _out(self, lines: list[str]) -> list[str]:
        self.emitted += len(lines)
        return lines

    def feed(self, line: str) -> list[str]:
        out: list[str] = []
        if self.table and not (line.strip().startswith("|") and self.code is None):
            out += self._flush_table()
        s = line.rstrip("\n")
        stripped = s.strip()
        # code fences
        fm = re.match(r"^\s*(`{3,}|~{3,})\s*([\w#+.-]*)", s)
        if fm and (self.code is None or fm.group(1)[0] == self.fence[0] and len(fm.group(1)) >= len(self.fence) and not fm.group(2)):
            if self.code is None:
                self.code, self.fence = fm.group(2) or "", fm.group(1)
                label = sty(self.code or "code", DIM, italic=True)
                return out + self._out([self._m() + sty("╭─ ", HAIRLINE) + label])
            self.code = None
            return out + self._out([self._m() + sty("╰─", HAIRLINE)])
        if self.code is not None:
            gutter = sty("│ ", HAIRLINE)
            hl = highlight(s.replace("\t", "    "), self.code)
            return out + self._out(wrap(hl, self.width, self._m() + gutter, self.margin + gutter + "  "))
        if not stripped:
            if self.blank or self.emitted == 0:
                return out
            self.blank = True
            return out + self._out([""])
        self.blank = False
        if stripped.startswith("|"):
            self.table.append(stripped)
            return out
        m = self._m()
        h = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", s)
        if h:
            level = len(h.group(1))
            body = inline(h.group(2), INK if level <= 2 else INK_SOFT)
            styled = sty(strip_ansi(body), INK if level <= 2 else INK_SOFT, bold=True, underline=level == 1) if level <= 3 else body
            return out + self._out(wrap(styled, self.width, m))
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", s):
            return out + self._out([m + sty("─" * max(4, min(self.width - len(m), 48)), HAIRLINE)])
        q = re.match(r"^\s*>\s?(.*)$", s)
        if q:
            bar = sty("│ ", HAIRLINE)
            return out + self._out(wrap(sty(strip_ansi(inline(q.group(1))), INK_SOFT, italic=True), self.width, m + bar, self.margin + bar))
        li = re.match(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(\[[ xX]\]\s+)?(.*)$", s)
        if li:
            depth = min(len(li.group(1).replace("\t", "    ")) // 2, 4)
            marker = li.group(2)
            box = li.group(3)
            if box:
                bullet = sty("✓", MINT) if box.strip("[] ").lower() == "x" else sty("○", DIM)
            elif marker[0].isdigit():
                bullet = sty(marker.rstrip(".)") + ".", DIM)
            else:
                bullet = sty("•◦▪▫"[depth % 4], DIM)
            pad = "  " * depth
            first = m + pad + bullet + " "
            rest = self.margin + pad + " " * (vis_len(bullet) + 1)
            return out + self._out(wrap(inline(li.group(4)), self.width, first, rest))
        return out + self._out(wrap(inline(stripped), self.width, m, self.margin))

    def _flush_table(self) -> list[str]:
        rows_ = [r.strip().strip("|").split("|") for r in self.table]
        self.table = []
        rows_ = [[c.strip() for c in r] for r in rows_ if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in r if c.strip())]
        if not rows_:
            return []
        n = max(len(r) for r in rows_)
        rows_ = [r + [""] * (n - len(r)) for r in rows_]
        cells = [[inline(c) for c in r] for r in rows_]
        widths = [max(vis_len(r[i]) for r in cells) for i in range(n)]
        m = self._m()
        avail = self.width - len(self.margin) - (3 * n + 1)
        if sum(widths) <= avail:
            hl = lambda left, mid, right: m + sty(left + mid.join("─" * (w + 2) for w in widths) + right, HAIRLINE)  # noqa: E731
            out = [hl("╭", "┬", "╮")]
            for i, r in enumerate(cells):
                body = sty("│", HAIRLINE).join(" " + (sty(strip_ansi(c), INK, bold=True) if i == 0 else c) + " " * (widths[j] - vis_len(c) + 1) for j, c in enumerate(r))
                out.append((m if i == 0 else self.margin) + sty("│", HAIRLINE) + body + sty("│", HAIRLINE))
                if i == 0 and len(cells) > 1:
                    out.append(self.margin + sty("├" + "┼".join("─" * (w + 2) for w in widths) + "┤", HAIRLINE))
            out.append(self.margin + sty("╰" + "┴".join("─" * (w + 2) for w in widths) + "╯", HAIRLINE))
            return self._out(out)
        # too wide for this screen (phones): one block per row, header: value
        head, body = rows_[0], cells[1:] or cells
        out = []
        for r in body:
            first = True
            for j, c in enumerate(r):
                if not strip_ansi(c):
                    continue
                label = sty(head[j] + ":", AMBER) + " " if len(cells) > 1 else ""
                out += wrap(label + c, self.width, (m if first else self.margin) + (sty("• ", DIM) if first else "  "), self.margin + "    ")
                first = False
        return self._out(out)

    def finish(self) -> list[str]:
        out = self._flush_table() if self.table else []
        if self.code is not None:
            self.code = None
            out += self._out([self.margin + sty("╰─", HAIRLINE)])
        return out


def render_markdown(text: str, width: int | None = None, margin: str = "   ", first_margin: str | None = None) -> list[str]:
    md = Markdown(width or cols(), margin, first_margin)
    out: list[str] = []
    for line in text.splitlines():
        out += md.feed(line)
    out += md.finish()
    while out and not out[-1].strip():
        out.pop()
    return out


# ----------------------------------------------------------------------------- live region


class Live:
    """A few lines (spinner, streaming text) kept at the bottom and redrawn in place; normal output prints above."""

    def __init__(self, stream: object = None, fps: float = 12.0) -> None:
        self.stream = stream or sys.stderr
        self.enabled = bool(COLOR) and hasattr(self.stream, "isatty") and self.stream.isatty() and not os.environ.get("REVAPI_NO_LIVE")
        self.lock = threading.RLock()
        self.render: Callable[[], list[str]] | None = None
        self.drawn = 0
        self.fps = fps
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = 0

    def _write(self, s: str) -> None:
        try:
            self.stream.write(s)  # type: ignore[attr-defined]
            self.stream.flush()  # type: ignore[attr-defined]
        except (OSError, ValueError):
            pass

    def _clear_seq(self) -> str:
        if not self.drawn:
            return ""
        s = "\r" + (f"\033[{self.drawn - 1}A" if self.drawn > 1 else "") + "\033[J"
        self.drawn = 0
        return s

    def _draw_seq(self) -> str:
        if not self.render or self._paused:
            return ""
        try:
            lines = self.render()
        except Exception:  # noqa: BLE001  (a render bug must never kill the run)
            return ""
        if not lines:
            return ""
        w = cols() - 1
        lines = [truncate(x, w) for x in lines[-max(1, rows() - 2) :]]
        self.drawn = len(lines)
        return "\n".join(lines)

    def start(self, render: Callable[[], list[str]]) -> None:
        self.render = render
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._write("\033[?25l")
        self._thread = threading.Thread(target=self._loop, name="revapi-live", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(1 / self.fps):
            with self.lock:
                if self._paused:
                    continue
                self._write(self._clear_seq() + self._draw_seq())

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._thread = None
        with self.lock:
            self._write(self._clear_seq() + ("\033[?25h" if self.enabled else ""))
        self.render = None

    def pause(self) -> None:
        with self.lock:
            self._paused += 1
            self._write(self._clear_seq() + ("\033[?25h" if self.enabled else ""))

    def resume(self) -> None:
        with self.lock:
            self._paused = max(0, self._paused - 1)
            if not self._paused and self._thread:
                self._write("\033[?25l")

    def print(self, *lines: str) -> None:
        with self.lock:
            text = "\n".join(lines)
            if self.enabled and self._thread:
                self._write(self._clear_seq() + text + "\n" + self._draw_seq())
            else:
                self._write(text + "\n")


@atexit.register
def _restore_cursor() -> None:
    if COLOR and sys.stderr.isatty():
        try:
            sys.stderr.write("\033[?25h")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass


SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FLAP = "abcdefghijklmnopqrstuvwxyz"


def spinner_frame() -> str:
    return SPIN[int(time.time() * 12) % len(SPIN)]


def k(n: float) -> str:
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


def money(cost: float) -> str:
    return "<$0.01" if 0 < cost < 0.005 else f"${cost:.2f}"


def elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"
