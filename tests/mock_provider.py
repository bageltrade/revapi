"""A scripted OpenAI-compatible provider for end-to-end runs without an API key.

Streams reasoning, text and tool calls over SSE, and misbehaves on purpose: the first request gets a
503, the second stream is cut mid-reply, so retries and stream resets are exercised for real.
Requests without tools are compaction requests and get a summary.

    python tests/mock_provider.py 8799 http://127.0.0.1:8765
    REVAPI_BASE_URL=http://127.0.0.1:8799/v1 revapi agent --provider custom ...
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SITE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8765"
STATE = {"requests": 0}

CLIENT = '''"""Client for the fixture todo API (reverse engineered from the captured traffic)."""

import requests

BASE = "{site}"
HEADERS = {{"X-Client": "web"}}
COOKIES = {{"sid": "abc123"}}


def list_todos() -> list[dict]:
    r = requests.get(f"{{BASE}}/api/todos", headers=HEADERS, cookies=COOKIES, timeout=30)
    r.raise_for_status()
    return r.json()["items"]


def add_todo(title: str) -> dict:
    r = requests.post(f"{{BASE}}/api/todos", json={{"title": title}}, headers=HEADERS, cookies=COOKIES, timeout=30)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print(add_todo("from revapi"))
    for t in list_todos():
        print(t["id"], t["title"])
'''

SUMMARY = """## Done — todo API reverse engineered

| endpoint | method | auth |
|---|---|---|
| `/api/todos` | GET | cookie `sid` + `X-Client` |
| `/api/todos` | POST | same, JSON `{"title"}` → 201 |

- **Client:** `api_client.py` with `list_todos()` and `add_todo(title)`
- Tested against the live site: exit code 0
"""


def turns(site: str) -> list[dict]:
    return [
        {
            "think": "The user wants the todo API. I should open the start page, interact with the form so the app fires its requests, then read the captured traffic.",
            "text": "I'll open the site and **capture** its traffic.",
            "call": ("browser_navigate", {"url": site + "/"}),
        },
        {"think": "Let me list what the page requested so far.", "call": ("browser_network_requests", {})},
        {
            "think": "GET /api/todos with cookie sid and header X-Client. I have enough to write the client: list + add.",
            "call": ("write_file", {"path": "api_client.py", "content": CLIENT.format(site=site)}),
        },
        {"think": "Run it against the live site to verify.", "call": ("shell", {"command": "python3 api_client.py"})},
        {"think": "It works. Wrap up.", "text": "Verified: the client ran against the live site.", "call": ("finish", {"summary": SUMMARY, "script_path": "api_client.py"})},
    ]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:  # quiet
        pass

    def _sse(self, events: list[dict], cut_after: int | None = None, delay: float = 0.01) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for i, e in enumerate(events):
            if cut_after is not None and i >= cut_after:
                self.wfile.flush()
                self.connection.shutdown(2)  # the network drops mid-reply
                return
            self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
            self.wfile.flush()
            time.sleep(delay)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        body = json.dumps({"data": [{"id": "mock-thinker"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        STATE["requests"] += 1
        n = STATE["requests"]
        if n == 1:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"upstream busy"}')
            return
        msgs = req.get("messages", [])
        if not req.get("tools"):  # a compaction request
            text = "<analysis>chronological review</analysis><summary>1. Primary Request: map the todo API.\n3. Discovered API Surface: GET/POST /api/todos [#4], cookie sid=abc123, header X-Client: web.\n9. Current Work: verifying the client.\n10. Next Step: run api_client.py.</summary>"
            self._sse(
                [{"choices": [{"delta": {"content": text[i : i + 40]}}]} for i in range(0, len(text), 40)]
                + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 120}}]
            )
            return
        done_calls = sum(1 for m in msgs if m.get("role") == "tool")
        script = turns(SITE)
        t = script[min(done_calls, len(script) - 1)]
        events: list[dict] = []
        for i in range(0, len(t["think"]), 12):
            events.append({"choices": [{"delta": {"reasoning_content": t["think"][i : i + 12]}}]})
        for i in range(0, len(t.get("text", "")), 6):
            events.append({"choices": [{"delta": {"content": t["text"][i : i + 6]}}]})
        name, args = t["call"]
        raw = json.dumps(args)
        events.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": ""}}]}}]})
        for i in range(0, len(raw), 24):
            events.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[i : i + 24]}}]}}]})
        events.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        chars = sum(len(json.dumps(m)) for m in msgs)
        events.append({"choices": [], "usage": {"prompt_tokens": chars // 4, "completion_tokens": len(raw) // 4 + 30}})
        self._sse(events, cut_after=6 if n == 2 else None, delay=0.012)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
