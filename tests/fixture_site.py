"""Tiny local web app used by tests: page + JSON API with cookie auth, POST, redirect."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><html><head><title>Todo Fixture</title></head><body>
<h1>Todos</h1><ul id=list></ul>
<input id=new placeholder="New todo"><button id=add onclick="add()">Add</button>
<a href="/about">About</a>
<select id=filter><option value=all>All</option><option value=done>Done</option></select>
<script>
async function load(){ const r = await fetch('/api/todos', {headers:{'X-Client':'fixture'}}); const j = await r.json();
  document.getElementById('list').innerHTML = j.items.map(t=>'<li>'+t.title+'</li>').join(''); }
async function add(){ const v = document.getElementById('new').value;
  await fetch('/api/todos', {method:'POST', headers:{'Content-Type':'application/json','X-Client':'fixture'}, body: JSON.stringify({title:v})}); load(); }
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    todos = [{"id": 1, "title": "first"}]

    def log_message(self, *a):
        pass

    def _json(self, code, obj, extra=()):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGE
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "sid=abc123; Path=/; HttpOnly")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/old":
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
        elif self.path == "/about":
            b = b"<html><head><title>About</title></head><body><h1>About page</h1><p>hello</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path.startswith("/api/todos"):
            if "sid=abc123" not in (self.headers.get("Cookie") or ""):
                return self._json(401, {"error": "no session"})
            self._json(200, {"items": H.todos})
        else:
            self._json(404, {"error": "nf"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if "sid=abc123" not in (self.headers.get("Cookie") or ""):
            return self._json(401, {"error": "no session"})
        t = {"id": len(H.todos) + 1, "title": body.get("title", "")}
        H.todos.append(t)
        self._json(201, t)


def serve(port=0):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if __name__ == "__main__":
    import sys

    s = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
    print("serving", s.server_address)
    threading.Event().wait()
