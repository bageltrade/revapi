"""Headless Chromium driven over CDP, with a full-fidelity HAR recorder.

Works with any Chromium-family binary: Termux `chromium-browser`, Debian `chromium`,
Chrome, or a downloaded chrome-headless-shell. Set REVAPI_CHROMIUM to override.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from . import __version__
from .cdp import CDPConnection, CDPError, wait_port_file

IS_TERMUX = "com.termux" in os.environ.get("PREFIX", "") or os.path.isdir("/data/data/com.termux")

CANDIDATES = [
    "chromium-browser",  # Termux (x11-repo), Ubuntu
    "chromium",  # Debian, Arch, Alpine
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "chrome-headless-shell",
    "headless_shell",
    "microsoft-edge",
    "brave-browser",
]

STATIC_EXT = (".js", ".mjs", ".css", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp4", ".webm", ".mp3", ".wasm", ".map", ".avif")
NOISE_HOSTS = (
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "sentry.io",
    "hotjar",
    "segment.io",
    "segment.com",
    "facebook.net",
    "clarity.ms",
    "simpleanalytics",
    "intercom",
    "datadoghq",
    "newrelic",
    "mixpanel",
    "amplitude",
    "fullstory",
)
MAX_BODY = int(os.environ.get("REVAPI_MAX_BODY_BYTES", str(3 * 1024 * 1024)))


def find_chromium() -> str | None:
    env = os.environ.get("REVAPI_CHROMIUM")
    if env:
        return env if (os.path.exists(env) or shutil.which(env)) else None
    for name in CANDIDATES:
        p = shutil.which(name)
        if p:
            return p
    prefix = os.environ.get("PREFIX", "")
    for p in (f"{prefix}/lib/chromium/chrome", "/usr/lib/chromium/chromium", "/usr/lib/chromium-browser/chromium-browser", "/opt/google/chrome/chrome"):
        if p and os.path.exists(p):
            return p
    cache = Path.home() / ".cache" / "ms-playwright"
    for p in sorted(cache.glob("chromium*/chrome-*/chrome*"), reverse=True) if cache.exists() else []:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def is_api_like(url: str, mime: str = "", rtype: str = "") -> bool:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https", "ws", "wss"):
        return False
    if any(h in u.netloc for h in NOISE_HOSTS):
        return False
    path = u.path.lower()
    if rtype in ("Image", "Font", "Stylesheet", "Media", "Manifest", "Script"):
        return False
    if path.endswith(STATIC_EXT):
        return False
    if rtype in ("XHR", "Fetch", "WebSocket", "EventSource", "Other", "Preflight"):
        return True
    m = mime.lower()
    return any(t in m for t in ("json", "graphql", "xml", "protobuf", "grpc", "x-www-form", "event-stream", "msgpack", "javascript+json"))


def _iso(wall: float) -> str:
    return dt.datetime.fromtimestamp(wall, dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _hdr_list(headers: dict | None) -> list[dict[str, str]]:
    out = []
    for k, v in (headers or {}).items():
        for line in str(v).split("\n"):
            out.append({"name": k, "value": line})
    return out


def _cookies_from_header(value: str) -> list[dict[str, str]]:
    out = []
    for part in value.split(";"):
        if "=" in part:
            n, v = part.strip().split("=", 1)
            out.append({"name": n, "value": v})
    return out


def _query(url: str) -> list[dict[str, str]]:
    return [{"name": k, "value": v} for k, v in urllib.parse.parse_qsl(urllib.parse.urlparse(url).query, keep_blank_values=True)]


class NetworkRecorder:
    """Turns CDP Network.* events into HAR 1.2 entries (incl. bodies, cookies, redirects, websockets)."""

    def __init__(self, browser: Browser) -> None:
        self.b = browser
        self.entries: list[dict[str, Any]] = []
        self._live: dict[tuple, dict[str, Any]] = {}  # (sessionId, requestId) -> entry being built
        self._extra_req: dict[tuple, dict] = {}
        self._extra_resp: dict[tuple, dict] = {}
        self._ws: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._bodies: queue.Queue = queue.Queue()
        self.inflight: set[tuple] = set()
        self.last_activity = time.time()
        threading.Thread(target=self._body_worker, daemon=True).start()
        c = browser.conn
        c.on("Network.requestWillBeSent", self._on_request)
        c.on("Network.requestWillBeSentExtraInfo", self._on_request_extra)
        c.on("Network.responseReceived", self._on_response)
        c.on("Network.responseReceivedExtraInfo", self._on_response_extra)
        c.on("Network.loadingFinished", self._on_finished)
        c.on("Network.loadingFailed", self._on_failed)
        c.on("Network.webSocketCreated", self._on_ws_created)
        c.on("Network.webSocketFrameSent", lambda p, s: self._on_ws_frame(p, "send"))
        c.on("Network.webSocketFrameReceived", lambda p, s: self._on_ws_frame(p, "receive"))
        c.on("Network.webSocketHandshakeResponseReceived", self._on_ws_handshake)

    # -- events ---------------------------------------------------------------
    def _on_request(self, p: dict, sid: str | None) -> None:
        req = p["request"]
        if req["url"].startswith(("data:", "blob:", "chrome-extension:")):
            return
        rid = (sid, p["requestId"])
        with self._lock:
            if p.get("redirectResponse") and rid in self._live:
                prev = self._live.pop(rid)
                self._apply_response(prev, p["redirectResponse"], rid)
                prev["response"]["redirectURL"] = req["url"]
                self._finalize(prev, body=False)
            entry = {
                "_requestId": p["requestId"],
                "_sessionId": sid,
                "_resourceType": p.get("type", "Other"),
                "_ts": p["timestamp"],
                "startedDateTime": _iso(p.get("wallTime", time.time())),
                "time": 0,
                "request": {
                    "method": req["method"],
                    "url": req["url"] + (req.get("urlFragment") or ""),
                    "httpVersion": "HTTP/1.1",
                    "cookies": [],
                    "headers": _hdr_list(req.get("headers")),
                    "queryString": _query(req["url"]),
                    "headersSize": -1,
                    "bodySize": 0,
                },
                "response": {
                    "status": 0,
                    "statusText": "",
                    "httpVersion": "",
                    "cookies": [],
                    "headers": [],
                    "content": {"size": 0, "mimeType": ""},
                    "redirectURL": "",
                    "headersSize": -1,
                    "bodySize": -1,
                },
                "cache": {},
                "timings": {"send": 0, "wait": 0, "receive": 0},
            }
            if req.get("postData") is not None or req.get("hasPostData"):
                entry["request"]["postData"] = {"mimeType": req.get("headers", {}).get("Content-Type", req.get("headers", {}).get("content-type", "")), "text": req.get("postData", "")}
                entry["request"]["bodySize"] = len((req.get("postData") or "").encode())
                if req.get("hasPostData") and not req.get("postData"):
                    self._bodies.put(("post", p["requestId"], sid, entry))
            if rid in self._extra_req:
                self._merge_req_extra(entry, self._extra_req.pop(rid))
            self._live[rid] = entry
            self.inflight.add(rid)
            self.last_activity = time.time()

    def _merge_req_extra(self, entry: dict, extra: dict) -> None:
        hdrs = extra.get("headers") or {}
        if hdrs:
            entry["request"]["headers"] = _hdr_list(hdrs)
        cookie = next((v for k, v in hdrs.items() if k.lower() == "cookie"), "")
        if cookie:
            entry["request"]["cookies"] = _cookies_from_header(cookie)
        elif extra.get("associatedCookies"):
            entry["request"]["cookies"] = [{"name": c["cookie"]["name"], "value": c["cookie"]["value"]} for c in extra["associatedCookies"] if not c.get("blockedReasons")]

    def _on_request_extra(self, p: dict, sid: str | None) -> None:
        with self._lock:
            e = self._live.get((sid, p["requestId"]))
            if e:
                self._merge_req_extra(e, p)
            else:
                self._extra_req[(sid, p["requestId"])] = p

    def _apply_response(self, entry: dict, r: dict, rid: str) -> None:
        resp = entry["response"]
        resp["status"] = r.get("status", 0)
        resp["statusText"] = r.get("statusText", "")
        proto = (r.get("protocol") or "http/1.1").upper()
        resp["httpVersion"] = {"H2": "HTTP/2", "H3": "HTTP/3"}.get(proto, proto)
        entry["request"]["httpVersion"] = resp["httpVersion"]
        resp["headers"] = _hdr_list(r.get("headers"))
        resp["content"]["mimeType"] = r.get("mimeType", "")
        if r.get("remoteIPAddress"):
            entry["serverIPAddress"] = r["remoteIPAddress"]
        extra = self._extra_resp.pop(rid, None)
        if extra:
            self._merge_resp_extra(entry, extra)
        t = r.get("timing")
        if t:
            send = max(0.0, t.get("sendEnd", 0) - t.get("sendStart", 0))
            wait = max(0.0, t.get("receiveHeadersEnd", 0) - t.get("sendEnd", 0))
            entry["timings"].update(send=round(send, 3), wait=round(wait, 3))

    def _merge_resp_extra(self, entry: dict, extra: dict) -> None:
        hdrs = extra.get("headers") or {}
        if hdrs:
            entry["response"]["headers"] = _hdr_list(hdrs)
        cookies = []
        for k, v in hdrs.items():
            if k.lower() == "set-cookie":
                for line in str(v).split("\n"):
                    first = line.split(";", 1)[0]
                    if "=" in first:
                        n, val = first.split("=", 1)
                        cookies.append({"name": n.strip(), "value": val.strip()})
        if cookies:
            entry["response"]["cookies"] = cookies

    def _on_response(self, p: dict, sid: str | None) -> None:
        with self._lock:
            e = self._live.get((sid, p["requestId"]))
            if e:
                self._apply_response(e, p["response"], (sid, p["requestId"]))
                e["_resourceType"] = p.get("type", e["_resourceType"])

    def _on_response_extra(self, p: dict, sid: str | None) -> None:
        with self._lock:
            e = self._live.get((sid, p["requestId"]))
            if e and e["response"]["status"] and p.get("statusCode", e["response"]["status"]) == e["response"]["status"]:
                self._merge_resp_extra(e, p)
            else:
                self._extra_resp[(sid, p["requestId"])] = p

    def _on_finished(self, p: dict, sid: str | None) -> None:
        with self._lock:
            e = self._live.pop((sid, p["requestId"]), None)
            self.inflight.discard((sid, p["requestId"]))
            self.last_activity = time.time()
        if not e:
            return
        e["time"] = round(max(0.0, (p["timestamp"] - e["_ts"]) * 1000), 3)
        e["response"]["bodySize"] = int(p.get("encodedDataLength", -1))
        e["timings"]["receive"] = max(0, round(e["time"] - e["timings"]["send"] - e["timings"]["wait"], 3))
        self._bodies.put(("body", p["requestId"], sid, e))

    def _on_failed(self, p: dict, sid: str | None) -> None:
        with self._lock:
            e = self._live.pop((sid, p["requestId"]), None)
            self.inflight.discard((sid, p["requestId"]))
        if e:
            e["response"]["_error"] = p.get("errorText", "failed")
            e["time"] = round(max(0.0, (p["timestamp"] - e["_ts"]) * 1000), 3)
            self._finalize(e, body=False)

    def _on_ws_created(self, p: dict, sid: str | None) -> None:
        self._ws[p["requestId"]] = {
            "_requestId": p["requestId"],
            "_resourceType": "WebSocket",
            "_ts": time.time(),
            "startedDateTime": _iso(time.time()),
            "time": 0,
            "request": {"method": "GET", "url": p["url"], "httpVersion": "HTTP/1.1", "cookies": [], "headers": [], "queryString": _query(p["url"]), "headersSize": -1, "bodySize": 0},
            "response": {
                "status": 101,
                "statusText": "Switching Protocols",
                "httpVersion": "HTTP/1.1",
                "cookies": [],
                "headers": [],
                "content": {"size": 0, "mimeType": ""},
                "redirectURL": "",
                "headersSize": -1,
                "bodySize": -1,
            },
            "cache": {},
            "timings": {"send": 0, "wait": 0, "receive": 0},
            "_webSocketMessages": [],
        }
        with self._lock:
            self.entries.append(self._ws[p["requestId"]])

    def _on_ws_handshake(self, p: dict, sid: str | None) -> None:
        e = self._ws.get(p["requestId"])
        if e:
            e["response"]["headers"] = _hdr_list(p["response"].get("headers"))

    def _on_ws_frame(self, p: dict, direction: str) -> None:
        e = self._ws.get(p["requestId"])
        if e is not None and len(e["_webSocketMessages"]) < 2000:
            r = p["response"]
            e["_webSocketMessages"].append({"type": direction, "time": p["timestamp"], "opcode": r.get("opcode", 1), "data": r.get("payloadData", "")[:20000]})

    # -- bodies ---------------------------------------------------------------
    def _body_worker(self) -> None:
        while True:
            kind, rid, sid, e = self._bodies.get()
            try:
                if kind == "post":
                    r = self.b.conn.send("Network.getRequestPostData", {"requestId": rid}, session_id=sid, timeout=10)
                    e["request"]["postData"]["text"] = r.get("postData", "")
                    e["request"]["bodySize"] = len(e["request"]["postData"]["text"].encode())
                    continue
                if e["response"]["status"] in (204, 304) or 300 <= e["response"]["status"] < 400:
                    self._finalize(e, body=False)
                    continue
                if e["_resourceType"] in ("Image", "Font", "Media") and not os.environ.get("REVAPI_RECORD_MEDIA"):
                    self._finalize(e, body=False)
                    continue
                r = self.b.conn.send("Network.getResponseBody", {"requestId": rid}, session_id=sid, timeout=15)
                body = r.get("body", "")
                content = e["response"]["content"]
                if r.get("base64Encoded"):
                    raw = base64.b64decode(body)
                    content["size"] = len(raw)
                    if len(raw) <= MAX_BODY:
                        try:
                            content["text"] = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            content["text"], content["encoding"] = body, "base64"
                    else:
                        content["comment"] = f"body omitted ({len(raw)} bytes > REVAPI_MAX_BODY_BYTES)"
                else:
                    content["size"] = len(body.encode())
                    if len(body) <= MAX_BODY:
                        content["text"] = body
                    else:
                        content["comment"] = f"body omitted ({len(body)} chars > REVAPI_MAX_BODY_BYTES)"
                self._finalize(e, body=True)
            except (CDPError, KeyError, ValueError):
                if kind == "body":
                    self._finalize(e, body=False)

    def _finalize(self, e: dict, body: bool) -> None:
        with self._lock:
            if not e.get("_done"):
                e["_done"] = True
                self.entries.append(e)

    # -- export ---------------------------------------------------------------
    def har(self) -> dict[str, Any]:
        with self._lock:
            done = list(self.entries)
            pending = list(self._live.values())
        entries = sorted(done + pending, key=lambda x: x["_ts"])
        clean = []
        for e in entries:
            c = {k: v for k, v in e.items() if not k.startswith("_") or k in ("_resourceType", "_webSocketMessages")}
            clean.append(c)
        pages = [{"startedDateTime": entries[0]["startedDateTime"] if entries else _iso(time.time()), "id": "page_1", "title": self.b.last_url or "", "pageTimings": {}}]
        for c in clean:
            c["pageref"] = "page_1"
        return {"log": {"version": "1.2", "creator": {"name": "revapi", "version": __version__}, "browser": {"name": "Chromium", "version": self.b.version}, "pages": pages, "entries": clean}}

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 5
        while not self._bodies.empty() and time.time() < deadline:
            time.sleep(0.1)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.har(), ensure_ascii=False))
        tmp.replace(path)
        return str(path)


# --------------------------------------------------------------------------- page JS

SNAPSHOT_JS = r"""
(() => {
  const MAX = %d;
  const out = [];
  let n = window.__revapiRef || 0;
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; };
  const txt = el => (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
  const label = el => el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('alt') || el.getAttribute('name') || '';
  const interactive = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=tab],[role=menuitem],[role=checkbox],[role=option],[role=switch],[role=combobox],[onclick],[contenteditable=true],[tabindex]:not([tabindex="-1"])';
  out.push(`URL: ${location.href}`); out.push(`TITLE: ${document.title}`);
  const seen = new Set();
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_ELEMENT);
  let el; let chars = 0;
  while ((el = walker.nextNode())) {
    if (chars > MAX) { out.push('... (snapshot truncated; scroll or use find_text)'); break; }
    const tag = el.tagName.toLowerCase();
    if (['script','style','noscript','svg','path','template','meta','link'].includes(tag)) continue;
    let line = null;
    if (el.matches(interactive)) {
      if (!vis(el) || seen.has(el)) continue;
      let ref = el.getAttribute('data-revapi-ref');
      if (!ref) { ref = 'e' + (++n); el.setAttribute('data-revapi-ref', ref); }
      seen.add(el);
      const role = el.getAttribute('role') || tag;
      let d = `[${ref}] ${role}`;
      if (tag === 'input') d += `[type=${el.type}]` + (el.type === 'checkbox' || el.type === 'radio' ? (el.checked ? ' checked' : ' unchecked') : ` value="${(el.value||'').slice(0,60)}"`);
      if (tag === 'select') d += ` value="${el.value}" options=[${[...el.options].slice(0,15).map(o => o.text.trim()).join(' | ')}]`;
      const t = txt(el).slice(0, 80); const l = label(el);
      if (t) d += ` "${t}"`; if (l && l !== t) d += ` (${l.slice(0,60)})`;
      if (tag === 'a') { const h = el.getAttribute('href') || ''; if (h && !h.startsWith('javascript')) d += ` -> ${h.slice(0,100)}`; }
      line = d;
    } else if (/^h[1-6]$/.test(tag) && vis(el)) {
      line = `${'#'.repeat(+tag[1])} ${txt(el).slice(0,120)}`;
    } else if (['p','li','td','th','label','span','div','dd','dt','pre','code'].includes(tag) && el.children.length === 0 && vis(el)) {
      const t = txt(el); if (t.length > 2) line = `  ${t.slice(0,160)}`;
    }
    if (line) { out.push(line); chars += line.length; }
  }
  window.__revapiRef = n;
  return out.join('\n');
})()
"""

ELEMENT_BOX_JS = r"""
(ref) => { const el = document.querySelector(`[data-revapi-ref="${ref}"]`); if (!el) return null;
  el.scrollIntoView({block:'center', inline:'center'}); const r = el.getBoundingClientRect();
  return {x: r.left + r.width/2, y: r.top + r.height/2, w: r.width, h: r.height, tag: el.tagName.toLowerCase()}; }
"""

KEYS = {
    "Enter": ("Enter", "Enter", 13, "\r"),
    "Tab": ("Tab", "Tab", 9, ""),
    "Escape": ("Escape", "Escape", 27, ""),
    "Backspace": ("Backspace", "Backspace", 8, ""),
    "Delete": ("Delete", "Delete", 46, ""),
    "ArrowDown": ("ArrowDown", "ArrowDown", 40, ""),
    "ArrowUp": ("ArrowUp", "ArrowUp", 38, ""),
    "ArrowLeft": ("ArrowLeft", "ArrowLeft", 37, ""),
    "ArrowRight": ("ArrowRight", "ArrowRight", 39, ""),
    "PageDown": ("PageDown", "PageDown", 34, ""),
    "PageUp": ("PageUp", "PageUp", 33, ""),
    "Home": ("Home", "Home", 36, ""),
    "End": ("End", "End", 35, ""),
    "Space": (" ", "Space", 32, " "),
}


class Browser:
    """One Chromium process, one active tab (others tracked), network recorded for every tab."""

    def __init__(self, headless: bool = True, user_data_dir: str | None = None, proxy: str | None = None, extra_args: list[str] | None = None) -> None:
        exe = find_chromium()
        if not exe:
            raise CDPError("No Chromium found. Termux: `pkg install x11-repo && pkg install chromium`; Debian/Ubuntu: `apt install chromium`; or set REVAPI_CHROMIUM=/path/to/chrome")
        self.exe = exe
        self._tmp = None
        if user_data_dir is None:
            self._tmp = tempfile.mkdtemp(prefix="revapi-chrome-")
            user_data_dir = self._tmp
        Path(user_data_dir).mkdir(parents=True, exist_ok=True)
        port_file = Path(user_data_dir) / "DevToolsActivePort"
        port_file.unlink(missing_ok=True)
        args = [
            exe,
            "--remote-debugging-port=0",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-dev-shm-usage",
            "--disable-features=Translate,OptimizationHints,MediaRouter,DialMediaRouteProvider",
            "--metrics-recording-only",
            "--password-store=basic",
            "--use-mock-keychain",
            "--mute-audio",
            "--window-size=1280,900",
            "--lang=en-US",
        ]
        if headless:
            args += ["--headless=new", "--hide-scrollbars", "--disable-gpu"]
        if IS_TERMUX or os.geteuid() == 0 or os.environ.get("REVAPI_NO_SANDBOX"):
            args.append("--no-sandbox")
        if proxy:
            args.append(f"--proxy-server={proxy}")
        args += extra_args or []
        args.append("about:blank")
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        try:
            port, path = wait_port_file(str(port_file), timeout=40)
        except CDPError:
            err = self.proc.stderr.read(2000).decode(errors="replace") if self.proc.poll() is not None else ""
            self.close()
            raise CDPError(f"Chromium failed to start: {err.strip()[:500] or 'timeout'}") from None
        self.conn = CDPConnection(f"ws://127.0.0.1:{port}{path}")
        self.version = self.conn.send("Browser.getVersion").get("product", "Chromium")
        self.recorder = NetworkRecorder(self)
        self.sessions: dict[str, dict[str, Any]] = {}  # sessionId -> {targetId, url}
        self._sess_lock = threading.RLock()
        self.active: str | None = None
        self.last_url = ""
        self.dialogs: list[str] = []
        self.console: list[str] = []
        self.conn.on("Target.attachedToTarget", self._on_attached)
        self.conn.on("Target.detachedFromTarget", lambda p, s: self.sessions.pop(p.get("sessionId", ""), None))
        self.conn.on("Page.javascriptDialogOpening", self._on_dialog)
        self.conn.on("Runtime.consoleAPICalled", self._on_console)
        self.conn.on("Page.frameNavigated", self._on_nav)
        self.conn.send("Target.setDiscoverTargets", {"discover": True})
        self.conn.send("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True})
        # attach to the initial page
        targets = self.conn.send("Target.getTargets")["targetInfos"]
        page = next((t for t in targets if t["type"] == "page"), None)
        if page is None:
            page = {"targetId": self.conn.send("Target.createTarget", {"url": "about:blank"})["targetId"]}
        deadline = time.time() + 5
        while time.time() < deadline and not any(v["targetId"] == page["targetId"] for v in self.sessions.values()):
            time.sleep(0.05)
        sid = next((k for k, v in self.sessions.items() if v["targetId"] == page["targetId"]), None)
        if sid is None:
            sid = self.conn.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True})["sessionId"]
            self._setup_session(sid, page["targetId"])
        self.sessions[sid]["ready"].wait(15)
        self.active = sid

    # -- sessions -------------------------------------------------------------
    def _setup_session(self, sid: str, target_id: str) -> None:
        with self._sess_lock:
            if sid in self.sessions:
                return
            if any(v["targetId"] == target_id for v in self.sessions.values()):
                # already attached to this target via another session -> drop the duplicate
                try:
                    self.conn.send("Target.detachFromTarget", {"sessionId": sid}, timeout=5)
                except CDPError:
                    pass
                return
            self.sessions[sid] = {"targetId": target_id, "url": "", "ready": threading.Event()}
        for m, p in (
            ("Network.enable", {"maxPostDataSize": 1024 * 1024, "maxTotalBufferSize": 200 * 1024 * 1024, "maxResourceBufferSize": 50 * 1024 * 1024}),
            ("Page.enable", {}),
            ("Runtime.enable", {}),
            ("Network.setCacheDisabled", {"cacheDisabled": True}),
        ):
            try:
                self.conn.send(m, p, session_id=sid, timeout=10)
            except CDPError:
                pass
        ua = os.environ.get("REVAPI_USER_AGENT")
        if ua:
            try:
                self.conn.send("Network.setUserAgentOverride", {"userAgent": ua}, session_id=sid)
            except CDPError:
                pass
        try:
            self.conn.send("Runtime.runIfWaitingForDebugger", session_id=sid, timeout=5)
        except CDPError:
            pass
        self.sessions.get(sid, {}).get("ready", threading.Event()).set()

    def _on_attached(self, p: dict, _sid: str | None) -> None:
        info = p.get("targetInfo", {})
        if info.get("type") in ("page", "iframe", "worker", "service_worker", "shared_worker"):
            threading.Thread(target=self._setup_session, args=(p["sessionId"], info.get("targetId", "")), daemon=True).start()
            if info.get("type") == "page" and self.active and p.get("sessionId") != self.active:
                self.dialogs.append(f"new tab opened: {info.get('url')} (use switch_tab)")

    def _on_dialog(self, p: dict, sid: str | None) -> None:
        self.dialogs.append(f"{p.get('type')} dialog: {p.get('message')}")
        try:
            self.conn.send("Page.handleJavaScriptDialog", {"accept": True}, session_id=sid, timeout=5)
        except CDPError:
            pass

    def _on_console(self, p: dict, sid: str | None) -> None:
        if len(self.console) < 200:
            args = " ".join(str(a.get("value", a.get("description", "")))[:200] for a in p.get("args", []))
            self.console.append(f"[{p.get('type')}] {args}")

    def _on_nav(self, p: dict, sid: str | None) -> None:
        f = p.get("frame", {})
        if not f.get("parentId") and sid in self.sessions:
            self.sessions[sid]["url"] = f.get("url", "")
            if sid == self.active:
                self.last_url = f.get("url", "")

    def _send(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        return self.conn.send(method, params, session_id=self.active, timeout=timeout)

    # -- actions --------------------------------------------------------------
    def wait_idle(self, idle_ms: int = 600, timeout: float = 15) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            quiet = (time.time() - self.recorder.last_activity) * 1000
            if not self.recorder.inflight and quiet > idle_ms:
                return
            if quiet > idle_ms * 4:  # long-polling / streaming connections never finish
                return
            time.sleep(0.1)

    def navigate(self, url: str, timeout: float = 45) -> str:
        if not urllib.parse.urlparse(url).scheme:
            url = "https://" + url
        sid = self.active
        waiter = self.conn.expect("Page.loadEventFired", lambda p, s: s == sid)
        try:
            res = self._send("Page.navigate", {"url": url}, timeout=timeout)
            if res.get("errorText"):
                return f"navigation error: {res['errorText']}"
            deadline = time.time() + timeout
            while time.time() < deadline:
                if waiter.wait(1.0) is not None:
                    break
                try:  # fallback if the load event was missed
                    if self.evaluate("document.readyState", timeout=5) == "complete":
                        break
                except CDPError:
                    pass
        finally:
            waiter.cancel()
        self.wait_idle()
        self.last_url = self.evaluate("location.href") or url
        return f"loaded {self.last_url} (title: {self.evaluate('document.title')!s})"

    def evaluate(self, expression: str, await_promise: bool = True, timeout: float = 30) -> Any:
        r = self._send("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": await_promise, "userGesture": True}, timeout=timeout)
        if r.get("exceptionDetails"):
            d = r["exceptionDetails"]
            msg = d.get("exception", {}).get("description") or d.get("text")
            raise CDPError(f"JS error: {msg}")
        return r.get("result", {}).get("value")

    def snapshot(self, max_chars: int = 12000) -> str:
        return self.evaluate(SNAPSHOT_JS % max_chars) or ""

    def _box(self, ref: str) -> dict:
        box = self.evaluate(f"({ELEMENT_BOX_JS})({json.dumps(ref)})")
        if not box:
            raise CDPError(f"element {ref} not found; take a new snapshot (refs change after navigation)")
        return box

    def click(self, ref: str) -> str:
        box = self._box(ref)
        x, y = box["x"], box["y"]
        for t in ("mouseMoved", "mousePressed", "mouseReleased"):
            self._send("Input.dispatchMouseEvent", {"type": t, "x": x, "y": y, "button": "left", "clickCount": 1 if t != "mouseMoved" else 0})
        time.sleep(0.3)
        self.wait_idle()
        return f"clicked {ref} ({box['tag']}) — now at {self.evaluate('location.href')}"

    def type(self, ref: str, text: str, clear: bool = True, submit: bool = False) -> str:
        self.click(ref)
        if clear:
            self.evaluate(
                f"(() => {{ const el = document.querySelector('[data-revapi-ref=\"{ref}\"]'); if (!el) return;"
                " if ('value' in el) { el.value=''; el.dispatchEvent(new Event('input',{bubbles:true})); } else el.textContent=''; })()"
            )
        self._send("Input.insertText", {"text": text})
        if submit:
            self.press("Enter")
        self.wait_idle()
        return f"typed into {ref}" + (" and pressed Enter" if submit else "")

    def press(self, key: str) -> str:
        k = KEYS.get(key) or (key, f"Key{key.upper()}" if len(key) == 1 else key, ord(key.upper()) if len(key) == 1 else 0, key if len(key) == 1 else "")
        name, code, vk, text = k
        down = {"type": "keyDown" if text else "rawKeyDown", "key": name, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
        if text:
            down["text"] = text
        self._send("Input.dispatchKeyEvent", down)
        self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": name, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk})
        time.sleep(0.2)
        self.wait_idle()
        return f"pressed {key}"

    def select(self, ref: str, value: str) -> str:
        ok = self.evaluate(
            f"(() => {{ const el = document.querySelector('[data-revapi-ref=\"{ref}\"]'); if (!el) return false;"
            f" const v = {json.dumps(value)}; const o = [...el.options].find(o => o.value === v || o.text.trim() === v);"
            " if (!o) return false; el.value = o.value; el.dispatchEvent(new Event('input',{bubbles:true}));"
            " el.dispatchEvent(new Event('change',{bubbles:true})); return true; })()"
        )
        self.wait_idle()
        return f"selected {value!r} in {ref}" if ok else f"option {value!r} not found in {ref}"

    def scroll(self, direction: str = "down", amount: int = 800) -> str:
        dy = amount if direction == "down" else -amount
        self.evaluate(f"window.scrollBy(0, {dy})")
        time.sleep(0.4)
        self.wait_idle()
        return f"scrolled {direction} — at y={self.evaluate('Math.round(window.scrollY)')} of {self.evaluate('document.documentElement.scrollHeight')}"

    def back(self) -> str:
        self.evaluate("history.back()")
        time.sleep(1)
        self.wait_idle()
        return f"went back to {self.evaluate('location.href')}"

    def find_text(self, text: str) -> str:
        r = self.evaluate(
            f"(() => {{ const q = {json.dumps(text.lower())}; const hits = [];"
            " const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;"
            " while ((n = w.nextNode()) && hits.length < 20) { const t = n.textContent.replace(/\\s+/g,' ').trim();"
            " if (t.toLowerCase().includes(q)) hits.push(t.slice(0, 200)); } return hits; })()"
        )
        return "\n".join(r or []) or f"'{text}' not found on page"

    def wait_for_text(self, text: str, timeout: float = 15) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.evaluate(f"document.body && document.body.innerText.includes({json.dumps(text)})"):
                return f"'{text}' appeared"
            time.sleep(0.5)
        return f"'{text}' did not appear within {timeout}s"

    def screenshot(self, path: str | Path, full_page: bool = False) -> str:
        params: dict[str, Any] = {"format": "png"}
        if full_page:
            m = self._send("Page.getLayoutMetrics")
            cs = m.get("cssContentSize") or m.get("contentSize")
            params.update(captureBeyondViewport=True, clip={"x": 0, "y": 0, "width": cs["width"], "height": min(cs["height"], 8000), "scale": 1})
        data = self._send("Page.captureScreenshot", params)["data"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(data))
        return str(path)

    def cookies(self) -> list[dict]:
        return self._send("Network.getAllCookies").get("cookies", [])

    def storage(self) -> dict:
        return self.evaluate("({local: Object.fromEntries(Object.entries(localStorage)), session: Object.fromEntries(Object.entries(sessionStorage))})") or {}

    def tabs(self) -> list[dict]:
        out = []
        for sid, s in list(self.sessions.items()):
            try:
                info = self.conn.send("Target.getTargetInfo", {"targetId": s["targetId"]}, timeout=5)["targetInfo"]
            except CDPError:
                continue
            if info["type"] == "page":
                out.append({"tab": sid[:8], "url": info["url"], "title": info["title"], "active": sid == self.active})
        return out

    def switch_tab(self, tab: str) -> str:
        for sid in self.sessions:
            if sid.startswith(tab):
                self.sessions[sid]["ready"].wait(10)
                self.active = sid
                self.conn.send("Target.activateTarget", {"targetId": self.sessions[sid]["targetId"]})
                return f"switched to tab {tab}: {self.evaluate('location.href')}"
        return f"no tab {tab}"

    def drain_notices(self) -> list[str]:
        out, self.dialogs = self.dialogs, []
        return out

    def close(self) -> None:
        try:
            if getattr(self, "conn", None):
                try:
                    self.conn.send("Browser.close", timeout=5)
                except CDPError:
                    pass
                self.conn.close()
        finally:
            if getattr(self, "proc", None) and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            if self._tmp:
                shutil.rmtree(self._tmp, ignore_errors=True)
