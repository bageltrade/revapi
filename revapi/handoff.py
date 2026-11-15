"""Human check hand-off: when a page shows a Cloudflare / CAPTCHA check, a person passes it.

revapi never solves or works around these checks. It notices them, waits a few seconds (most
clear on their own in a real browser), and if one is still there it opens a *live view*: a tiny
local web page that mirrors the headless browser's screen and forwards your taps, scrolls and
typing to it. You pass the check yourself; revapi notices the page has moved on and carries on.

After a pass, the session (cookies + the exact user agent) is saved to
``~/.revapi/sessions/<host>.json`` so generated clients and later runs can reuse it until it
expires. ``revapi refresh <url>`` renews it; ``revapi cookies import <host>`` takes cookies from a
browser where you already passed.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cdp import CDPError

DETECT_JS = r"""(() => {
  const d = document, t = (d.title || '').toLowerCase();
  const q = (s) => d.querySelector(s);
  const empty = (s) => { const el = q(s); return !!el && !el.value; };
  const hasCf = !!q('iframe[src*="challenges.cloudflare.com"], script[src*="challenges.cloudflare.com"], .cf-turnstile');
  if (/^(just a moment|attention required|one more step|checking your browser|please wait\.\.\.)/.test(t)
      || q('#challenge-form, #challenge-stage, #cf-challenge-running, #cf-please-wait, #challenge-running')
      || typeof window._cf_chl_opt !== 'undefined') return 'cloudflare';
  if (hasCf && (empty('input[name="cf-turnstile-response"]') || !q('input[name="cf-turnstile-response"]'))) return 'turnstile';
  if (q('iframe[src*="hcaptcha.com"]') && empty('textarea[name="h-captcha-response"]')) return 'hcaptcha';
  if (q('iframe[src*="/recaptcha/"]') && empty('textarea[name="g-recaptcha-response"]')) return 'recaptcha';
  return '';
})()"""

FOCUS_JS = r"""(() => {
  const sels = ['iframe[src*="challenges.cloudflare.com"]', '.cf-turnstile', 'iframe[src*="hcaptcha.com"]', 'iframe[src*="/recaptcha/"]',
    '#challenge-stage iframe, #challenge-stage input, #challenge-stage button, #challenge-form iframe, #challenge-form input, #challenge-form button',
    '#challenge-stage', '#challenge-form', '#cf-challenge-running'];
  for (const s of sels) for (const el of document.querySelectorAll(s)) {
    const r = el.getBoundingClientRect();
    if (r.width > 4 && r.height > 4) return {x: r.left, y: r.top, w: r.width, h: r.height};
  }
  return null;
})()"""

LABELS = {
    "cloudflare": "Cloudflare check",
    "turnstile": "Cloudflare Turnstile check",
    "hcaptcha": "hCaptcha check",
    "recaptcha": "reCAPTCHA check",
}

GRACE_S = float(os.environ.get("REVAPI_HANDOFF_GRACE", "8"))


def detect(browser: Any) -> str:
    """'' when the page is clear, else the kind of human check showing."""
    try:
        return browser.evaluate(DETECT_JS, timeout=8) or ""
    except CDPError:
        return ""  # page mid-navigation: unknown, treat as clear for now


def sessions_dir() -> Path:
    from .store import app_dir

    d = app_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def host_of(url: str) -> str:
    if "://" not in url:
        url = "https://" + url
    return (urllib.parse.urlparse(url).hostname or "").lower()


def session_path(host: str) -> Path:
    safe = "".join(c for c in host_of(host) if c.isalnum() or c in ".-") or "unknown"
    return sessions_dir() / f"{safe}.json"


def _domain_match(host: str, domain: str) -> bool:
    d = domain.lstrip(".").lower()
    return host == d or host.endswith("." + d)


def save_session(browser: Any, url: str, source: str = "live view") -> Path:
    """Write the host's cookies and the browser's exact user agent for clients to reuse."""
    host = host_of(url)
    cookies = [c for c in browser.cookies() if _domain_match(host, c.get("domain", ""))]
    try:
        ua = browser.evaluate("navigator.userAgent", timeout=5) or ""
    except CDPError:
        ua = ""
    return write_session(host, cookies, ua, source)


def write_session(host: str, cookies: list[dict], user_agent: str, source: str) -> Path:
    keep = ("name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite")
    data = {
        "host": host_of(host),
        "user_agent": user_agent,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "cookie_header": "; ".join(f"{c['name']}={c['value']}" for c in cookies),
        "cookies": [{k: c[k] for k in keep if k in c} for c in cookies],
    }
    p = session_path(host)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return p


def load_session(host: str) -> dict | None:
    p = session_path(host)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_cookies(text: str, host: str) -> list[dict]:
    """Cookies from a pasted ``Cookie:`` header, a JSON export (Cookie-Editor / EditThisCookie /
    CDP), or a Netscape cookies.txt."""
    text = text.strip()
    h = host_of(host)
    if text.startswith(("[", "{")):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("cookies", [data])
        out = []
        for c in data:
            if not isinstance(c, dict) or "name" not in c:
                continue
            exp = c.get("expires", c.get("expirationDate", -1))
            out.append(
                {
                    "name": str(c["name"]),
                    "value": str(c.get("value", "")),
                    "domain": c.get("domain") or h,
                    "path": c.get("path") or "/",
                    "expires": float(exp) if isinstance(exp, (int, float)) else -1,
                    "httpOnly": bool(c.get("httpOnly", False)),
                    "secure": bool(c.get("secure", True)),
                }
            )
        return out
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#") or ln.startswith("#HttpOnly_")]
    if lines and all(len(ln.split("\t")) >= 7 for ln in lines):
        out = []
        for ln in lines:
            http_only = ln.startswith("#HttpOnly_")
            dom, _sub, path, sec, exp, name, value = ln.removeprefix("#HttpOnly_").split("\t")[:7]
            out.append({"name": name, "value": value, "domain": dom, "path": path, "expires": float(exp or -1), "httpOnly": http_only, "secure": sec.upper() == "TRUE"})
        return out
    if text.lower().startswith("cookie:"):
        text = text[7:]
    out = []
    for part in text.replace("\n", ";").split(";"):
        name, eq, value = part.strip().partition("=")
        if name and eq:
            out.append({"name": name.strip(), "value": value.strip(), "domain": h, "path": "/", "expires": -1, "httpOnly": False, "secure": True})
    return out


# ---------------------------------------------------------------------------- live view


def _fmt_left(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class LiveView:
    """A loopback web page mirroring the browser's active tab; taps/keys go back into it."""

    def __init__(self, browser: Any, label: str, port: int = 8790, host: str = "127.0.0.1") -> None:
        self.b = browser
        self.label = label
        self.key = secrets.token_urlsafe(9)
        self.done = threading.Event()
        self.challenge = True
        self.deadline = time.time() + 600
        self.finished_message = ""
        self._shot: tuple[float, bytes] = (0.0, b"")
        self._shot_lock = threading.Lock()
        self.viewport = {"w": 1280, "h": 900}
        self._focus: tuple[float, dict | None] = (0.0, None)
        self.server = self._bind(host, port)
        self.port = self.server.server_address[1]
        self.url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{self.port}/?k={self.key}"
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)

    def _bind(self, host: str, port: int) -> http.server.ThreadingHTTPServer:
        view = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:  # keep the terminal clean
                pass

            def _ok(self) -> bool:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if q.get("k", [""])[0] != view.key:
                    self._send(403, b"forbidden", "text/plain")
                    return False
                return True

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                try:
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):  # the viewer went away mid-frame
                    pass

            def _json(self, obj: Any) -> None:
                self._send(200, json.dumps(obj).encode(), "application/json")

            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                if not self._ok():
                    return
                if path == "/":
                    self._send(200, view.page().encode(), "text/html; charset=utf-8")
                elif path == "/frame.jpg":
                    img = view.frame()
                    self._send(200 if img else 503, img or b"", "image/jpeg")
                elif path == "/state":
                    self._json(view.state())
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                if not self._ok():
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0) or 0) or b"{}")
                except ValueError:
                    body = {}
                try:
                    view.act(path, body)
                    self._json({"ok": True})
                except (CDPError, KeyError, TypeError, ValueError) as e:
                    self._json({"ok": False, "error": str(e)})

        last: OSError | None = None
        for p in [port + i for i in range(20)] + [0]:
            try:
                srv = http.server.ThreadingHTTPServer((host, p), H)
                srv.daemon_threads = True
                srv.handle_error = lambda request, client_address: None  # never print into the session UI
                return srv
            except OSError as e:
                last = e
        raise last or OSError("no free port for the live view")

    # -- page/frames ----------------------------------------------------------
    def page(self) -> str:
        html = (Path(__file__).with_name("liveview.html")).read_text(encoding="utf-8")
        return html.replace("__LABEL__", self.label)

    def frame(self) -> bytes:
        with self._shot_lock:
            t, img = self._shot
            if img and time.time() - t < 0.3:
                return img
            try:
                m = self.b._send("Page.getLayoutMetrics", timeout=5)
                vv = m.get("cssVisualViewport") or m.get("visualViewport") or {}
                if vv.get("clientWidth"):
                    self.viewport = {"w": vv["clientWidth"], "h": vv["clientHeight"]}
                data = self.b._send("Page.captureScreenshot", {"format": "jpeg", "quality": 60}, timeout=10)["data"]
                img = base64.b64decode(data)
                self._shot = (time.time(), img)
            except (CDPError, KeyError):
                pass
            return img

    def focus(self) -> dict | None:
        """Where the check sits on screen (CSS px), so the live view can zoom onto it."""
        t, rect = self._focus
        if time.time() - t > 2:
            try:
                rect = self.b.evaluate(FOCUS_JS, timeout=5)
            except CDPError:
                pass
            self._focus = (time.time(), rect)
        return rect

    def state(self) -> dict:
        return {
            **self.viewport,
            "focus": self.focus(),
            "challenge": self.challenge,
            "left": _fmt_left(self.deadline - time.time()),
            "finished": bool(self.finished_message),
            "message": self.finished_message,
        }

    # -- input ----------------------------------------------------------------
    def act(self, path: str, body: dict) -> None:
        send = self.b._send
        if path == "/tap":
            x = float(body["fx"]) * self.viewport["w"]
            y = float(body["fy"]) * self.viewport["h"]
            send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
            send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
            send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
        elif path == "/scroll":
            send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": self.viewport["w"] / 2, "y": self.viewport["h"] / 2, "deltaX": 0, "deltaY": float(body.get("dy", 600))})
        elif path == "/type":
            send("Input.insertText", {"text": str(body.get("text", ""))})
        elif path == "/key":
            self.b.press(str(body.get("key", "Enter")))
        elif path == "/done":
            self.done.set()
        else:
            raise ValueError(f"unknown action {path}")
        with self._shot_lock:
            self._shot = (0.0, self._shot[1])  # next frame is fresh

    def start(self) -> LiveView:
        self._thread.start()
        return self

    def stop(self, message: str = "") -> None:
        self.finished_message = message or "done — back to revapi"
        time.sleep(1.2)  # let an open page read the final state
        self.server.shutdown()
        self.server.server_close()


def _open_on_phone(url: str) -> None:
    for exe in ("termux-open-url", "xdg-open"):
        if shutil.which(exe):
            try:
                subprocess.Popen([exe, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            except OSError:
                pass
            return


def hand_off(
    browser: Any,
    notify: Callable[[str, dict], None] | None = None,
    *,
    timeout: float = 600,
    port: int = 8790,
    host: str = "127.0.0.1",
    open_url: bool = True,
    live: bool = True,
) -> dict | None:
    """None when no check is showing. Otherwise a dict: kind, label, url, passed, seconds, how,
    session (path of the saved session when passed)."""
    kind = detect(browser)
    if not kind:
        return None
    say = notify or (lambda msg, data: None)
    page_url = getattr(browser, "last_url", "") or ""
    try:
        page_url = browser.evaluate("location.href", timeout=5) or page_url
    except CDPError:
        pass
    label = LABELS.get(kind, "human check")
    t0 = time.time()
    res: dict[str, Any] = {"kind": kind, "label": label, "url": page_url, "passed": False, "seconds": 0, "how": "", "session": ""}
    # most checks clear on their own in a real browser within a few seconds
    say(f"{label} on {host_of(page_url)} · giving it a few seconds", {"stage": "grace", **res})
    end, tick = time.time() + GRACE_S, 0
    while time.time() < end:
        time.sleep(0.25)
        tick += 1
        if tick % 4 == 0 and not detect(browser):
            res.update(passed=True, how="cleared on its own")
            break
    if not res["passed"] and live:
        view = LiveView(browser, label, port=port, host=host).start()
        view.deadline = time.time() + timeout
        say(f"{label} on {host_of(page_url)} needs you · open {view.url}", {"stage": "waiting", "live_url": view.url, "timeout": timeout, **res})
        if open_url:
            _open_on_phone(view.url)
        clear_streak = 0
        try:
            while time.time() < view.deadline and not view.done.is_set():
                for _ in range(4):
                    time.sleep(0.25)  # short slices: esc lands at once
                    if view.done.is_set():
                        break
                view.challenge = bool(detect(browser))
                clear_streak = 0 if view.challenge else clear_streak + 1
                if clear_streak >= 2:
                    break
            if not view.challenge or (view.done.is_set() and not detect(browser)):
                res.update(passed=True, how="you passed it in the live view")
            elif view.done.is_set():
                res["how"] = "you pressed done but the check is still showing"
            else:
                res["how"] = f"not passed within {_fmt_left(timeout)}"
        finally:
            view.stop("passed — revapi carries on" if res["passed"] else "stopped — " + (res["how"] or "interrupted"))
    elif not res["passed"]:
        res["how"] = "live view is off (handoff=off)"
    res["seconds"] = int(time.time() - t0)
    if res["passed"]:
        try:
            browser.wait_idle()
        except CDPError:
            pass
        try:
            res["session"] = str(save_session(browser, page_url, source=res["how"]))
        except (CDPError, OSError):
            pass
    say("", {"stage": "end", **res})
    return res


def result_note(res: dict) -> str:
    """What the agent reads after a hand-off."""
    host = host_of(res["url"])
    if res["passed"]:
        return (
            f"[human check] {res['label']} on {host}: {res['how']} ({res['seconds']}s). The page has moved on; continue."
            f" Session saved to {res['session']} (cookies + the exact user agent). Generated clients for {host} should load it"
            " at startup (see the session file note in your instructions)."
        )
    return (
        f"[human check] {res['label']} on {host} is still showing: {res['how']}. revapi does not get past these checks itself."
        " Do NOT treat the check page or its challenge requests as the site's API. Tell the user to run"
        f" `revapi refresh {res['url'] or host}` (passes it once in the live view) or `revapi cookies import {host}`,"
        " then finish with what you have."
    )
