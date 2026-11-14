"""Tools exposed to the agent. Each tool: JSON-schema params + a handler returning text."""

from __future__ import annotations

import html
import html.parser
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import har as harlib
from .browser import Browser, CDPError

MAX_TOOL_OUTPUT = int(os.environ.get("REVAPI_MAX_TOOL_OUTPUT", "14000"))


def _clip(s: str, n: int = MAX_TOOL_OUTPUT) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[: n // 2] + f"\n\n... [{len(s) - n} chars omitted] ...\n\n" + s[-n // 2 :]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props or {}, "required": required or []}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}


@dataclass
class Context:
    """Shared state for one run."""

    workdir: Path  # scripts dir; relative paths resolve here
    har_path: Path | None = None
    headless: bool = True
    interactive: bool = False
    shell_timeout: int = 180
    proxy: str | None = None
    user_data_dir: str | None = None
    items_path: Path | None = None
    ask: Callable[[str, list[str] | None], str] | None = None
    browser: Browser | None = None
    browser_closed: bool = False
    finished: dict[str, Any] | None = None
    screenshots: int = 0
    notes: list[str] = field(default_factory=list)
    prior_entries: list[dict[str, Any]] = field(default_factory=list)

    # -- browser lifecycle ----------------------------------------------------
    def get_browser(self) -> Browser:
        if self.browser is None:
            # relaunch within the same run: keep earlier traffic and append to it
            if self.har_path and self.har_path.exists():
                try:
                    self.prior_entries = harlib.load(self.har_path)
                except (OSError, ValueError):
                    self.prior_entries = []
            self.browser = Browser(headless=self.headless, user_data_dir=self.user_data_dir, proxy=self.proxy)
            self.browser_closed = False
        return self.browser

    def save_har(self) -> str | None:
        if self.browser and self.har_path:
            path = self.browser.recorder.save(self.har_path)
            if self.prior_entries:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                data["log"]["entries"] = self.prior_entries + data["log"]["entries"]
                Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return path
        return None

    def close_browser(self) -> str | None:
        path = None
        if self.browser:
            try:
                path = self.save_har()
            finally:
                self.browser.close()
                self.browser = None
                self.browser_closed = True
        return path

    def resolve(self, p: str) -> Path:
        path = Path(os.path.expanduser(p))
        return path if path.is_absolute() else (self.workdir / path)


# ============================================================================ browser


def _with_notices(ctx: Context, text: str) -> str:
    notes = ctx.browser.drain_notices() if ctx.browser else []
    return text + ("\n[notice] " + "\n[notice] ".join(notes) if notes else "")


def browser_tools(ctx: Context) -> list[Tool]:
    def b() -> Browser:
        return ctx.get_browser()

    def navigate(url: str) -> str:
        r = b().navigate(url)
        return _with_notices(ctx, r + "\n\n" + _clip(b().snapshot(6000), 6500))

    def snapshot(max_chars: int = 12000) -> str:
        return _with_notices(ctx, b().snapshot(max_chars))

    def click(ref: str) -> str:
        return _with_notices(ctx, b().click(ref.strip("[]")))

    def type_(ref: str, text: str, submit: bool = False, clear: bool = True) -> str:
        return _with_notices(ctx, b().type(ref.strip("[]"), text, clear=clear, submit=submit))

    def press(key: str) -> str:
        return _with_notices(ctx, b().press(key))

    def select(ref: str, value: str) -> str:
        return b().select(ref.strip("[]"), value)

    def scroll(direction: str = "down", amount: int = 800) -> str:
        return b().scroll(direction, amount)

    def back() -> str:
        return b().back()

    def evaluate(expression: str) -> str:
        v = b().evaluate(expression)
        return _clip(json.dumps(v, ensure_ascii=False, default=str, indent=1) if not isinstance(v, str) else v)

    def wait_for(text: str = "", seconds: float = 0) -> str:
        if text:
            return b().wait_for_text(text, timeout=seconds or 15)
        time.sleep(min(seconds or 2, 60))
        b().wait_idle()
        return f"waited {seconds or 2}s"

    def find_text(text: str) -> str:
        return b().find_text(text)

    def screenshot(full_page: bool = False) -> str:
        ctx.screenshots += 1
        path = ctx.workdir.parent.parent / "screenshots" / ctx.workdir.name / f"shot_{ctx.screenshots}.png"
        return f"saved screenshot to {b().screenshot(path, full_page)} (you cannot see images; use browser_snapshot for structure)"

    def tabs() -> str:
        return json.dumps(b().tabs(), indent=1)

    def switch_tab(tab: str) -> str:
        return b().switch_tab(tab)

    def cookies() -> str:
        return _clip(json.dumps([{k: c[k] for k in ("name", "value", "domain", "path", "expires", "httpOnly", "secure") if k in c} for c in b().cookies()], indent=1))

    def storage() -> str:
        return _clip(json.dumps(b().storage(), indent=1, ensure_ascii=False))

    def network_requests(all: bool = False, filter: str = "", last: int = 80) -> str:  # noqa: A002
        if not ctx.browser:
            return "browser not started"
        ctx.save_har()
        entries = harlib.load(ctx.har_path)
        idx = list(range(len(entries))) if all else harlib.api_indices(entries)
        if filter:
            rx = re.compile(re.escape(filter), re.I)
            idx = [i for i in idx if rx.search(entries[i]["request"]["url"])]
        lines = [harlib.line(i, entries[i]) for i in idx[-last:]]
        head = f"{len(idx)} {'requests' if all else 'API-like requests'} (of {len(entries)} total). Details: browser_network_request(index)."
        return head + "\n" + "\n".join(lines)

    def network_request(index: int, max_body: int = 6000, offset: int = 0) -> str:
        if not ctx.browser and not (ctx.har_path and ctx.har_path.exists()):
            return "no traffic recorded yet"
        if ctx.browser:
            ctx.save_har()
        return _clip(harlib.entry(ctx.har_path, index, max_body=max_body, offset=offset))

    def fetch(url: str, method: str = "GET", headers: dict | None = None, body: str | None = None) -> str:
        """Run fetch() inside the page (same cookies/origin); recorded in the HAR."""
        js = (
            "(async () => { const r = await fetch(%s, {method: %s, headers: %s, body: %s, credentials: 'include'});"
            " const t = await r.text(); return {status: r.status, headers: Object.fromEntries(r.headers), body: t.slice(0, 20000)}; })()"
        ) % (json.dumps(url), json.dumps(method), json.dumps(headers or {}), json.dumps(body) if body is not None else "undefined")
        return _clip(json.dumps(b().evaluate(js), indent=1, ensure_ascii=False))

    def close() -> str:
        if not ctx.browser:
            return "browser already closed" + (f"; HAR at {ctx.har_path}" if ctx.har_path and ctx.har_path.exists() else "")
        path = ctx.close_browser()
        n = len(harlib.load(path)) if path else 0
        return f"browser closed; HAR saved to {path} ({n} entries). Use har_overview / har_search / har_entry to analyze."

    ref = {"ref": {"type": "string", "description": "element ref from browser_snapshot, e.g. e12"}}
    return [
        Tool("browser_navigate", "Open a URL in the headless browser. Returns the page snapshot with element refs.", _obj({"url": S}, ["url"]), navigate),
        Tool("browser_snapshot", "Get a text snapshot of the current page: headings, text, and interactive elements as [ref] lines.", _obj({"max_chars": I}), snapshot),
        Tool("browser_click", "Click an element by ref.", _obj(ref, ["ref"]), click),
        Tool(
            "browser_type",
            "Click an input/textarea by ref and type text (clears first unless clear=false). submit=true presses Enter.",
            _obj({**ref, "text": S, "submit": B, "clear": B}, ["ref", "text"]),
            type_,
        ),
        Tool("browser_press_key", "Press a key: Enter, Tab, Escape, Backspace, ArrowDown, PageDown, Space, or a single character.", _obj({"key": S}, ["key"]), press),
        Tool("browser_select", "Choose an option (by value or visible text) in a <select> by ref.", _obj({**ref, "value": S}, ["ref", "value"]), select),
        Tool("browser_scroll", "Scroll the page up or down (loads lazy content / infinite scroll).", _obj({"direction": {"type": "string", "enum": ["down", "up"]}, "amount": I}), scroll),
        Tool("browser_back", "Go back in history.", _obj(), back),
        Tool("browser_evaluate", "Evaluate a JavaScript expression in the page and return the JSON result (await is supported via async IIFE).", _obj({"expression": S}, ["expression"]), evaluate),
        Tool("browser_wait_for", "Wait for text to appear on the page, or for a number of seconds.", _obj({"text": S, "seconds": {"type": "number"}}), wait_for),
        Tool("browser_find_text", "Find text snippets on the page containing a string.", _obj({"text": S}, ["text"]), find_text),
        Tool("browser_take_screenshot", "Save a PNG screenshot to disk (for the user's records).", _obj({"full_page": B}), screenshot),
        Tool("browser_tabs", "List open tabs (popups/new windows are tracked automatically).", _obj(), tabs),
        Tool("browser_switch_tab", "Make another tab active by its id from browser_tabs.", _obj({"tab": S}, ["tab"]), switch_tab),
        Tool("browser_cookies", "List all browser cookies (including HttpOnly).", _obj(), cookies),
        Tool("browser_storage", "Dump localStorage and sessionStorage of the current page (tokens often live here).", _obj(), storage),
        Tool(
            "browser_fetch",
            "Run fetch() from inside the page with its cookies/origin, to probe an endpoint. Recorded in the HAR.",
            _obj({"url": S, "method": S, "headers": {"type": "object"}, "body": S}, ["url"]),
            fetch,
        ),
        Tool(
            "browser_network_requests",
            "List captured network requests (API-like by default; all=true for everything; filter=substring of URL).",
            _obj({"all": B, "filter": S, "last": I}),
            network_requests,
        ),
        Tool(
            "browser_network_request",
            "Full details of one captured request: headers, cookies, request body, response headers and body.",
            _obj({"index": I, "max_body": I, "offset": I}, ["index"]),
            network_request,
        ),
        Tool("browser_close", "Close the browser and write the HAR recording. Do this when browsing is done.", _obj(), close),
    ]


# ============================================================================ HAR


def har_tools(ctx: Context) -> list[Tool]:
    def path() -> Path:
        if ctx.browser:
            ctx.save_har()
        if not ctx.har_path or not ctx.har_path.exists():
            raise FileNotFoundError("no HAR recorded yet")
        return ctx.har_path

    return [
        Tool(
            "har_overview",
            "Summarize the HAR: hosts, auth headers, endpoint shapes, and a numbered list of API requests.",
            _obj({"include_all": B, "limit": I}),
            lambda include_all=False, limit=300: _clip(harlib.overview(path(), include_all, limit)),
        ),
        Tool(
            "har_search",
            "Regex search across the HAR (where: all|url|request_headers|request_body|response_headers|response_body|websocket). Great for finding where a token/ID comes from.",
            _obj({"query": S, "where": S, "limit": I}, ["query"]),
            lambda query, where="all", limit=40: _clip(harlib.search(path(), query, where, limit)),
        ),
        Tool(
            "har_entry",
            "Full details of HAR entry #index (part: all|request|response; offset pages through long bodies).",
            _obj({"index": I, "max_body": I, "offset": I, "part": S}, ["index"]),
            lambda index, max_body=6000, offset=0, part="all": _clip(harlib.entry(path(), index, max_body, offset, part)),
        ),
    ]


# ============================================================================ files + shell


def file_tools(ctx: Context) -> list[Tool]:
    def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
        p = ctx.resolve(path)
        if p.is_dir():
            return list_dir(path)
        if p.suffix == ".har" or p.stat().st_size > 2_000_000:
            return f"{p} is large ({p.stat().st_size} bytes); use har_overview/har_search/har_entry or the shell (grep/jq/python) instead"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = lines[offset - 1 : offset - 1 + limit]
        out = "\n".join(f"{offset + i:5}| {ln}" for i, ln in enumerate(chunk))
        if offset - 1 + limit < len(lines):
            out += f"\n... ({len(lines)} lines total; continue with offset={offset + limit})"
        return _clip(out)

    def write_file(path: str, content: str) -> str:
        p = ctx.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {p} ({len(content)} chars, {content.count(chr(10)) + 1} lines)"

    def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        p = ctx.resolve(path)
        s = p.read_text(encoding="utf-8")
        n = s.count(old_text)
        if n == 0:
            return "old_text not found; read the file and copy the exact text"
        if n > 1 and not replace_all:
            return f"old_text matches {n} places; add more context or set replace_all=true"
        p.write_text(s.replace(old_text, new_text) if replace_all else s.replace(old_text, new_text, 1), encoding="utf-8")
        return f"edited {p} ({n if replace_all else 1} replacement)"

    def list_dir(path: str = ".") -> str:
        p = ctx.resolve(path)
        if not p.exists():
            return f"{p} does not exist"
        out = []
        for c in sorted(p.iterdir())[:300]:
            out.append(f"{c.name}/" if c.is_dir() else f"{c.name}  ({c.stat().st_size}B)")
        return f"{p}:\n" + ("\n".join(out) or "(empty)")

    def shell(command: str, timeout: int = 0) -> str:
        sh = shutil.which("bash") or shutil.which("sh") or "/system/bin/sh"
        t = min(timeout or ctx.shell_timeout, 1800)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        try:
            r = subprocess.run([sh, "-c", command], cwd=ctx.workdir, capture_output=True, text=True, timeout=t, env=env, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return _clip(f"[timed out after {t}s]\n{out}")
        out = r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
        return _clip(f"[exit {r.returncode}]\n{out}".rstrip())

    return [
        Tool("read_file", "Read a text file (relative paths are in the output directory). Returns numbered lines.", _obj({"path": S, "offset": I, "limit": I}, ["path"]), read_file),
        Tool("write_file", "Create or overwrite a file with full content.", _obj({"path": S, "content": S}, ["path", "content"]), write_file),
        Tool(
            "edit_file",
            "Replace exact text in a file (must match once unless replace_all).",
            _obj({"path": S, "old_text": S, "new_text": S, "replace_all": B}, ["path", "old_text", "new_text"]),
            edit_file,
        ),
        Tool("list_dir", "List a directory.", _obj({"path": S}), list_dir),
        Tool(
            "shell",
            "Run a shell command in the output directory (install deps, run/test the client, grep, jq, python...). Returns exit code and output.",
            _obj({"command": S, "timeout": I}, ["command"]),
            shell,
        ),
    ]


# ============================================================================ web


class _Text(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript", "svg", "template"):
            self.skip += 1
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"):
            self.out.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href and href.startswith("http") and len(self.links) < 60:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "template") and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.out.append(data)


UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Mobile Safari/537.36"


def _get(url: str, timeout: float = 30) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        raw = r.read(5_000_000)
    charset = re.search(r"charset=([\w-]+)", ctype)
    return raw.decode(charset.group(1) if charset else "utf-8", errors="replace"), ctype


def web_fetch(url: str, max_chars: int = 12000) -> str:
    try:
        body, ctype = _get(url)
    except Exception as e:  # noqa: BLE001
        return f"fetch failed: {e}"
    if "html" not in ctype:
        return _clip(body, max_chars)
    p = _Text()
    p.feed(body)
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", "".join(p.out))).strip()
    links = "\n".join(p.links[:40])
    return _clip(f"{text}\n\nLINKS:\n{links}", max_chars)


def web_search(query: str, max_results: int = 10) -> str:
    try:
        body, _ = _get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
    except Exception as e:  # noqa: BLE001
        return f"search failed: {e}"
    results = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', body, re.S):
        href = html.unescape(m.group(1))
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0])
        title = html.unescape(re.sub("<.*?>", "", m.group(2))).strip()
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', m.group(3), re.S)
        snippet = html.unescape(re.sub("<.*?>", "", snip.group(1))).strip() if snip else ""
        if "duckduckgo.com/y.js" in href:
            continue  # ads
        results.append(f"{len(results) + 1}. {title}\n   {href}\n   {snippet}")
        if len(results) >= max_results:
            break
    return "\n".join(results) or "no results"


def web_tools(ctx: Context) -> list[Tool]:
    return [
        Tool("web_search", "Search the web (DuckDuckGo). Returns titles, URLs, snippets.", _obj({"query": S, "max_results": I}, ["query"]), web_search),
        Tool("web_fetch", "Fetch a URL over plain HTTP and return readable text + links (use the browser for JS-heavy pages).", _obj({"url": S, "max_chars": I}, ["url"]), web_fetch),
    ]


def collector_tools(ctx: Context) -> list[Tool]:
    def save_items(items: list) -> str:
        assert ctx.items_path
        ctx.items_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with ctx.items_path.open("a", encoding="utf-8") as f:
            for it in items if isinstance(items, list) else [items]:
                if isinstance(it, dict):
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    n += 1
        with ctx.items_path.open(encoding="utf-8") as fh:
            total = sum(1 for _ in fh)
        return f"saved {n} items ({total} total)"

    return [Tool("save_items", "Append collected items (array of JSON objects) to the output JSONL.", _obj({"items": {"type": "array", "items": {"type": "object"}}}, ["items"]), save_items)]


# ============================================================================ control


def control_tools(ctx: Context) -> list[Tool]:
    def ask_user(question: str, options: list | None = None) -> str:
        if not ctx.interactive or not ctx.ask:
            return "ask_user is unavailable in non-interactive mode — make a reasonable assumption, note it in your summary, and continue."
        return ctx.ask(question, options) or "(no answer — use your best judgment)"

    def finish(summary: str, script_path: str = "") -> str:
        ctx.finished = {"summary": summary, "script_path": script_path}
        return "ok — run complete."

    tools = [Tool("finish", "Call when the task is complete. summary = final report for the user; script_path = main generated file.", _obj({"summary": S, "script_path": S}, ["summary"]), finish)]
    tools.append(
        Tool(
            "ask_user",
            "Ask the user a clarifying question (interactive runs only). options = optional choices.",
            _obj({"question": S, "options": {"type": "array", "items": S}}, ["question"]),
            ask_user,
        )
    )
    return tools


def coerce_args(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Weaker models send "false" / "3" / '["a"]' as strings; coerce to the declared types and drop unknown keys."""
    props = schema.get("properties", {})
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k not in props:
            continue
        typ = props[k].get("type")
        try:
            if typ == "boolean" and isinstance(v, str):
                v = v.strip().lower() in ("true", "1", "yes", "y")
            elif typ == "integer" and isinstance(v, (str, float)) and str(v).strip().lstrip("-").replace(".", "", 1).isdigit():
                v = int(float(v))
            elif typ == "number" and isinstance(v, str):
                v = float(v)
            elif typ in ("array", "object") and isinstance(v, str) and v.strip()[:1] in "[{":
                v = json.loads(v)
            elif typ == "string" and v is not None and not isinstance(v, str):
                v = v if isinstance(v, (dict, list)) and k in ("content",) else (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        except (ValueError, json.JSONDecodeError):
            pass
        out[k] = v
    return out


def run_tool(tools: dict[str, Tool], name: str, args: dict[str, Any]) -> tuple[str, bool]:
    t = tools.get(name)
    if not t:
        return f"unknown tool {name!r}. Available: {', '.join(tools)}", True
    if "_raw" in args:
        return f"could not parse your arguments as JSON: {args['_raw'][:300]}", True
    args = coerce_args(t.parameters, args)
    try:
        return str(t.handler(**args)), False
    except TypeError as e:
        return f"bad arguments for {name}: {e}. Schema: {json.dumps(t.parameters)}", True
    except (CDPError, OSError, ValueError, KeyError, RuntimeError, AssertionError) as e:
        return f"{type(e).__name__}: {e}", True
