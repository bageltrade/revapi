"""HAR utilities: load, filter to API traffic, summarize, search, and inspect entries.

These power the engineer's har_* tools so large recordings never need to be
dumped wholesale into the model context.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

from .browser import is_api_like


def load(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    return data.get("log", {}).get("entries", [])


def _mime(e: dict) -> str:
    return e.get("response", {}).get("content", {}).get("mimeType", "") or ""


def _rtype(e: dict) -> str:
    return e.get("_resourceType", "") or ""


def api_indices(entries: list[dict]) -> list[int]:
    return [i for i, e in enumerate(entries) if is_api_like(e["request"]["url"], _mime(e), _rtype(e))]


def _short(url: str, n: int = 140) -> str:
    return url if len(url) <= n else url[: n - 3] + "..."


def _hdr(headers: list[dict], name: str) -> str:
    name = name.lower()
    return next((h["value"] for h in headers if h["name"].lower() == name), "")


AUTH_HEADERS = ("authorization", "cookie", "x-csrf-token", "x-xsrf-token", "x-api-key", "api-key", "x-auth-token", "x-access-token", "x-requested-with", "x-client-id", "apikey")


def line(i: int, e: dict) -> str:
    r, s = e["request"], e["response"]
    size = s.get("content", {}).get("size", 0)
    extra = ""
    if e.get("_webSocketMessages"):
        extra = f" ws_frames={len(e['_webSocketMessages'])}"
    err = s.get("_error")
    return f"#{i} {r['method']} {_short(r['url'])} -> {s.get('status')}{' ' + err if err else ''} {_mime(e).split(';')[0]} {size}B{extra}"


def overview(path: str | Path, include_all: bool = False, limit: int = 300) -> str:
    entries = load(path)
    idx = list(range(len(entries))) if include_all else api_indices(entries)
    hosts = Counter(urllib.parse.urlparse(entries[i]["request"]["url"]).netloc for i in idx)
    auth = Counter()
    for i in idx:
        for h in entries[i]["request"].get("headers", []):
            if h["name"].lower() in AUTH_HEADERS or h["name"].lower().startswith("x-"):
                auth[h["name"].lower()] += 1
    # group by endpoint shape (numbers / ids collapsed)
    shapes = Counter()
    for i in idx:
        r = entries[i]["request"]
        u = urllib.parse.urlparse(r["url"])
        p = re.sub(r"/[0-9a-fA-F-]{8,}|/\d+", "/{id}", u.path)
        shapes[f"{r['method']} {u.netloc}{p}"] += 1
    out = [f"HAR: {path}", f"total entries: {len(entries)}; {'all' if include_all else 'API-like'}: {len(idx)}", "", "hosts:"]
    out += [f"  {h} ({n})" for h, n in hosts.most_common(20)]
    out += ["", "auth/custom request headers seen:"] + [f"  {h} ({n})" for h, n in auth.most_common(25)]
    out += ["", "endpoint shapes:"] + [f"  {s} x{n}" for s, n in shapes.most_common(80)]
    out += ["", "requests (use har_entry <#> for details):"] + [line(i, entries[i]) for i in idx[:limit]]
    if len(idx) > limit:
        out.append(f"... {len(idx) - limit} more (use har_search)")
    return "\n".join(out)


def search(path: str | Path, query: str, where: str = "all", limit: int = 40) -> str:
    entries = load(path)
    try:
        rx = re.compile(query, re.I)
    except re.error:
        rx = re.compile(re.escape(query), re.I)
    hits = []
    for i, e in enumerate(entries):
        r, s = e["request"], e["response"]
        fields = {
            "url": r["url"],
            "method": r["method"],
            "request_headers": "\n".join(f"{h['name']}: {h['value']}" for h in r.get("headers", [])),
            "request_body": (r.get("postData") or {}).get("text", ""),
            "response_headers": "\n".join(f"{h['name']}: {h['value']}" for h in s.get("headers", [])),
            "response_body": s.get("content", {}).get("text", "") or "",
            "websocket": "\n".join(m.get("data", "") for m in e.get("_webSocketMessages", [])),
        }
        for name, val in fields.items():
            if where not in ("all", name):
                continue
            m = rx.search(val or "")
            if m:
                a = max(0, m.start() - 80)
                hits.append(f"{line(i, e)}\n    [{name}] ...{val[a : m.end() + 80]!r}...")
                break
        if len(hits) >= limit:
            break
    return "\n".join(hits) or f"no matches for {query!r}"


def _body(text: str, max_chars: int) -> str:
    if not text:
        return ""
    try:
        text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [{len(text) - max_chars} more chars; use har_entry with offset]"
    return text


def entry(path: str | Path, index: int, max_body: int = 6000, offset: int = 0, part: str = "all") -> str:
    entries = load(path)
    if not 0 <= index < len(entries):
        return f"no entry #{index} (have {len(entries)})"
    e = entries[index]
    r, s = e["request"], e["response"]
    out = [line(index, e), f"started: {e.get('startedDateTime')}  time: {e.get('time')}ms  type: {_rtype(e)}  http: {r.get('httpVersion')}"]
    if part in ("all", "request"):
        out += ["", "REQUEST HEADERS:"] + [f"  {h['name']}: {h['value']}" for h in r.get("headers", [])]
        if r.get("cookies"):
            out += ["REQUEST COOKIES:"] + [f"  {c['name']}={c['value']}" for c in r["cookies"]]
        if r.get("postData"):
            pd = r["postData"]
            out += ["", f"REQUEST BODY ({pd.get('mimeType', '')}):", _body(pd.get("text", ""), max_body)]
    if part in ("all", "response"):
        out += ["", "RESPONSE HEADERS:"] + [f"  {h['name']}: {h['value']}" for h in s.get("headers", [])]
        c = s.get("content", {})
        txt = c.get("text", "") or ""
        if c.get("encoding") == "base64":
            out += ["", f"RESPONSE BODY: <binary base64, {c.get('size')} bytes>"]
        elif txt:
            out += ["", f"RESPONSE BODY ({c.get('mimeType', '')}, {c.get('size')}B):", _body(txt[offset:], max_body)]
        elif c.get("comment"):
            out += ["", f"RESPONSE BODY: {c['comment']}"]
    if e.get("_webSocketMessages"):
        out += ["", "WEBSOCKET FRAMES:"] + [f"  {m['type']}: {m.get('data', '')[:400]}" for m in e["_webSocketMessages"][:60]]
    return "\n".join(out)


def filter_api(src: str | Path, dst: str | Path) -> int:
    """Write a smaller HAR containing only API-like entries; returns count."""
    data = json.loads(Path(src).read_text(encoding="utf-8", errors="replace"))
    entries = data["log"]["entries"]
    keep = [entries[i] for i in api_indices(entries)]
    data["log"]["entries"] = keep
    Path(dst).write_text(json.dumps(data, ensure_ascii=False))
    return len(keep)
