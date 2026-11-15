"""Generated MCP servers: install them under ~/.revapi/mcp/<name>, test them, and hook them into MCP clients.

Every server revapi writes runs on the bundled stdlib runtime (``revapi_mcp.py``), so the only
thing a client needs is ``python /path/server.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from . import store

RUNTIME = Path(__file__).with_name("mcp_runtime.py")
SKIP = {"__pycache__", "node_modules", ".venv", "venv", ".git"}


def root() -> Path:
    d = store.app_dir() / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(text: str, fallback: str = "site") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40].strip("-") or fallback


def name_for(url: str | None, prompt: str | None = None) -> str:
    """shop.example.com -> shop-example; else a slug of the prompt."""
    host = (urllib.parse.urlparse(url if url and "://" in url else f"https://{url}").hostname or "") if url else ""
    host = host.removeprefix("www.").removeprefix("m.")
    parts = [p for p in host.split(".") if p]
    if re.fullmatch(r"[\d.]+|[\da-f:]+", host or "x") and host:  # an IP address: keep all of it
        return slug(host)
    if len(parts) >= 2:
        return slug("-".join(parts[:-1]))
    if parts:
        return slug(parts[0])
    words = re.findall(r"[a-z0-9]+", (prompt or "").lower())
    return slug("-".join(w for w in words if w not in {"a", "an", "the", "for", "of", "to", "mcp", "server", "make", "create", "build", "me"})[:30])


def server_dir(name: str) -> Path:
    return root() / slug(name)


def python_cmd() -> str:
    return sys.executable or shutil.which("python3") or shutil.which("python") or "python"


def list_tools(server: Path, timeout: float = 30) -> list[dict]:
    """Tool specs, by asking the server itself (``server.py --list``)."""
    r = subprocess.run([python_cmd(), str(server), "--list"], capture_output=True, text=True, timeout=timeout, cwd=str(server.parent))
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-800:] or f"server exited with {r.returncode}")
    out, dec, text, i = [], json.JSONDecoder(), r.stdout, 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
        out.append(obj)
    return out


def handshake(server: Path, timeout: float = 30) -> dict:
    """Speak MCP over stdio like a client: initialize, initialized, tools/list."""
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "revapi", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    r = subprocess.run([python_cmd(), str(server)], input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=timeout, cwd=str(server.parent))
    replies = {}
    for line in r.stdout.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            raise RuntimeError(f"server printed non-protocol output on stdout: {line[:200]!r}") from None
        replies[m.get("id")] = m
    if 1 not in replies or "result" not in replies[1]:
        raise RuntimeError(f"no initialize reply: {(r.stderr or r.stdout).strip()[-500:]}")
    if 2 not in replies or "result" not in replies[2]:
        raise RuntimeError(f"no tools/list reply: {replies.get(2) or (r.stderr or '').strip()[-500:]}")
    init = replies[1]["result"]
    return {"server": init.get("serverInfo", {}), "protocol": init.get("protocolVersion"), "tools": replies[2]["result"].get("tools", [])}


def install(src: Path, name: str, run_id: str = "", url: str | None = None, prompt: str = "") -> dict:
    """Copy a generated server into ~/.revapi/mcp/<name> (with a fresh runtime) and record what it is."""
    if not (src / "server.py").exists():
        raise FileNotFoundError(f"no server.py in {src}")
    dst = server_dir(name)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in SKIP or p.name == "revapi-mcp.json":
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*SKIP))
        else:
            shutil.copy2(p, dst / p.name)
    shutil.copy2(RUNTIME, dst / "revapi_mcp.py")
    try:
        tools = [t["name"] for t in list_tools(dst / "server.py")]
        error = ""
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
        tools, error = [], str(e)
    meta = {"name": dst.name, "run_id": run_id, "url": url, "prompt": prompt, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tools": tools, "path": str(dst / "server.py")}
    (dst / "revapi-mcp.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return {**meta, "error": error}


def installed() -> list[dict]:
    out = []
    for d in sorted(root().iterdir()):
        if not (d / "server.py").exists():
            continue
        try:
            meta = json.loads((d / "revapi-mcp.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {"name": d.name, "tools": []}
        meta["path"] = str(d / "server.py")
        out.append(meta)
    return out


def resolve(target: str) -> Path:
    """An installed name, a run id, a directory or a server.py path -> server.py."""
    p = Path(os.path.expanduser(target))
    if p.is_file():
        return p
    if (p / "server.py").exists():
        return p / "server.py"
    if (server_dir(target) / "server.py").exists():
        return server_dir(target) / "server.py"
    rec = store.History().get(target)
    if rec:
        s = store.scripts_dir(rec["run_id"], rec.get("output_dir")) / "server.py"
        if s.exists():
            return s
    raise FileNotFoundError(f"no MCP server called {target!r} · see: revapi mcp list")


# ---------------------------------------------------------------------------- MCP clients

CLIENTS = ("claude", "codex", "gemini", "opencode", "hermes")


def _toml_str(s: str) -> str:
    return json.dumps(s)  # TOML basic strings share JSON's escapes for these characters


def config(server: Path, name: str) -> dict[str, str]:
    """Ready-to-paste configuration for common MCP clients."""
    py, path, n = python_cmd(), str(server), slug(name)
    std = {"mcpServers": {n: {"command": py, "args": [path]}}}
    return {
        "claude": f"claude mcp add {n} -- {py} {path}",
        "codex": f"# ~/.codex/config.toml\n[mcp_servers.{n}]\ncommand = {_toml_str(py)}\nargs = [{_toml_str(path)}]",
        "gemini": "// ~/.gemini/settings.json\n" + json.dumps(std, indent=2),
        "opencode": "// ~/.config/opencode/opencode.json\n" + json.dumps({"mcp": {n: {"type": "local", "command": [py, path], "enabled": True}}}, indent=2),
        "hermes": f"# ~/.hermes/config.yaml\nmcp_servers:\n  {n}:\n    command: {json.dumps(py)}\n    args: [{json.dumps(path)}]",
        "json": json.dumps(std, indent=2),
        "http": f"{py} {path} --http 8811     # then use http://127.0.0.1:8811/mcp",
    }


def _merge_json(file: Path, key: str, n: str, entry: dict) -> str:
    data: dict[str, Any] = {}
    if file.exists():
        raw = file.read_text(encoding="utf-8")
        try:
            data = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raise RuntimeError(f"{file} is not plain JSON (comments?); add this by hand:\n{json.dumps({key: {n: entry}}, indent=2)}") from None
        shutil.copy2(file, file.with_suffix(file.suffix + ".bak"))
    data.setdefault(key, {})[n] = entry
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"added {n} to {file}"


def add_to(client: str, server: Path, name: str) -> str:
    """Register the server with an MCP client on this device."""
    py, path, n = python_cmd(), str(server), slug(name)
    home = Path.home()
    if client == "claude":
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("the claude CLI is not installed (npm i -g @anthropic-ai/claude-code); or paste: " + config(server, name)["claude"])
        r = subprocess.run([exe, "mcp", "add", "--scope", "user", n, "--", py, path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip() or "claude mcp add failed")
        return f"added {n} to Claude Code (user scope)"
    if client == "codex":
        f = home / ".codex" / "config.toml"
        text = f.read_text(encoding="utf-8") if f.exists() else ""
        if re.search(rf"^\[mcp_servers\.{re.escape(n)}\]", text, re.M):
            return f"{n} is already in {f}"
        f.parent.mkdir(parents=True, exist_ok=True)
        if text:
            shutil.copy2(f, f.with_suffix(".toml.bak"))
        block = f"\n[mcp_servers.{n}]\ncommand = {_toml_str(py)}\nargs = [{_toml_str(path)}]\n"
        f.write_text(text.rstrip("\n") + ("\n" if text else "") + block, encoding="utf-8")
        return f"added {n} to {f}"
    if client == "gemini":
        return _merge_json(home / ".gemini" / "settings.json", "mcpServers", n, {"command": py, "args": [path]})
    if client == "opencode":
        return _merge_json(home / ".config" / "opencode" / "opencode.json", "mcp", n, {"type": "local", "command": [py, path], "enabled": True})
    if client == "hermes":
        raise RuntimeError("add it to ~/.hermes/config.yaml by hand (revapi does not rewrite YAML):\n" + config(server, name)["hermes"])
    raise RuntimeError(f"unknown client {client!r}; choose from {', '.join(CLIENTS)}")


def name_for_run(run_id: str, url: str | None, prompt: str | None) -> str:
    """Re-engineering a run updates its installed server; a new run never overwrites another one."""
    for meta in installed():
        if run_id and meta.get("run_id") == run_id:
            return meta["name"]
    base = name_for(url, prompt)
    name, n = base, 2
    while (server_dir(name) / "server.py").exists():
        name, n = f"{base}-{n}", n + 1
    return name
