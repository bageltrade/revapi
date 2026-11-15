"""Config, run storage, history, and output sync (mirrors reverse-api-engineer's layout).

~/.revapi/
  config.json
  history.json
  runs/har/<run_id>/recording.har
  runs/scripts/<run_id>/...        (generated client / docs)
  runs/messages/<run_id>.jsonl     (full agent transcript)
  runs/collected/<run_id>/...      (collector output)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

EFFORTS = ("low", "medium", "high", "xhigh", "max")
LANGUAGES = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "java": ".java",
    "csharp": ".cs",
    "php": ".php",
    "ruby": ".rb",
    "c": ".c",
    "powershell": ".psm1",
}
LANGUAGE_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "java": "Java",
    "csharp": "C#",
    "php": "PHP",
    "ruby": "Ruby",
    "c": "C",
    "powershell": "PowerShell",
}
SCRIPT_EXTS = set(LANGUAGES.values()) | {".sh", ".mjs", ".ps1"}

DEFAULTS: dict[str, Any] = {
    "provider": "auto",  # auto | anthropic | openai | openrouter | gemini | groq | deepseek | ... | ollama | pollinations | custom
    "model": None,  # None -> provider default
    "api_key": None,  # optional; env vars are preferred
    "base_url": None,  # for provider=custom / self-hosted
    "effort": None,  # low | medium | high | xhigh | max
    "collector_model": None,
    "output_language": "python",
    "output_mode": "client",  # client | docs | mcp
    "output_dir": None,  # None -> ~/.revapi/runs
    "real_time_sync": True,  # mirror generated files into ./scripts/<name> while engineering
    "headless": True,
    "max_turns": 120,
    "shell_timeout": 180,
    "proxy": None,
    "user_data_dir": None,  # custom Chromium profile directory (overrides keep_session)
    "keep_session": True,  # reuse ~/.revapi/profile so logins and passed checks survive between runs
    "handoff": "auto",  # human checks (Cloudflare/CAPTCHA): live (open the live view) | off (just report) | auto (live when a terminal is attached)
    "handoff_timeout": 600,  # seconds to wait for you in the live view
    "handoff_port": 8790,  # live view port (next free one is used if taken)
    "handoff_host": "127.0.0.1",  # 0.0.0.0 lets another device on your network open the live view
    "handoff_open": True,  # open the live view in the phone's browser automatically (termux-open-url)
    "stream": True,  # stream replies token by token
    "retries": 40,  # attempts per model request before giving up (network drops, 429, 5xx, overloads…)
    "context_window": None,  # None -> known size for the model (128k if unknown); set to override
    "compact_at": 0.8,  # compact the conversation when it fills this share of the context window
    "show_thinking": True,  # show the model's reasoning (folded) in the margin
    "max_tokens": 16000,  # output tokens per reply (lowered automatically if a model caps it)
    "engine": "revapi",  # revapi (built-in agent) | hermes (the local Hermes CLI engineers saved captures)
    "hermes_model": None,  # None -> the model your Hermes config selects
    "hermes_provider": None,  # None -> the provider your Hermes config selects
    "hermes_max_turns": None,  # None -> the turn cap from your Hermes config
}


OUTPUT_MODES = ("client", "docs", "mcp")


def app_dir() -> Path:
    d = Path(os.environ.get("REVAPI_HOME", Path.home() / ".revapi"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class Config:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or app_dir() / "config.json"
        self.data = dict(DEFAULTS)
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text()))
            except json.JSONDecodeError:
                pass

    def get(self, key: str, default: Any = None) -> Any:
        v = self.data.get(key)
        return default if v is None else v

    def set(self, key: str, value: Any) -> None:
        if isinstance(value, str):
            low = value.lower()
            if low in ("true", "false"):
                value = low == "true"
            elif low in ("none", "null", ""):
                value = None
            elif re.fullmatch(r"-?\d+", value):
                value = int(value)
        if key == "output_language" and value not in LANGUAGES:
            raise ValueError(f"output_language must be one of {', '.join(LANGUAGES)}")
        if key == "engine" and value not in ("revapi", "hermes"):
            raise ValueError("engine must be revapi or hermes")
        if key == "hermes_max_turns" and value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError("hermes_max_turns must be a positive integer (or none)")
        if key == "output_mode" and value not in OUTPUT_MODES:
            raise ValueError("output_mode must be client, docs or mcp")
        if key == "compact_at" and value is not None:
            value = float(value)
            if not 0.3 <= value <= 0.95:
                raise ValueError("compact_at must be between 0.3 and 0.95")
        if key in ("retries", "context_window", "max_tokens") and value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} must be a positive integer")
        if key == "handoff" and value not in ("live", "off", "auto"):
            raise ValueError("handoff must be live, off or auto")
        if key in ("handoff_timeout", "handoff_port") and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} must be a positive integer")
        if key == "effort" and value not in (None, *EFFORTS):
            raise ValueError("effort must be " + ", ".join(EFFORTS) + " or none")
        self.data[key] = value
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


# ----------------------------------------------------------------------------- runs


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _check_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id or ""):
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


def runs_root(output_dir: str | None = None) -> Path:
    return Path(output_dir).expanduser() if output_dir else app_dir() / "runs"


def har_dir(run_id: str, output_dir: str | None = None) -> Path:
    d = runs_root(output_dir) / "har" / _check_id(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def har_path(run_id: str, output_dir: str | None = None) -> Path:
    return har_dir(run_id, output_dir) / "recording.har"


def scripts_dir(run_id: str, output_dir: str | None = None) -> Path:
    d = runs_root(output_dir) / "scripts" / _check_id(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def collected_dir(run_id: str, output_dir: str | None = None) -> Path:
    d = runs_root(output_dir) / "collected" / _check_id(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def messages_path(run_id: str, output_dir: str | None = None) -> Path:
    d = runs_root(output_dir) / "messages"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_check_id(run_id)}.jsonl"


class History:
    def __init__(self) -> None:
        self.path = app_dir() / "history.json"
        self._lock = threading.Lock()

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return []

    def get(self, run_id: str) -> dict[str, Any] | None:
        if run_id in ("latest", "last"):
            runs = self.all()
            return runs[0] if runs else None
        return next((r for r in self.all() if r["run_id"] == run_id or (len(run_id) >= 4 and r["run_id"].startswith(run_id))), None)

    def upsert(self, record: dict[str, Any]) -> None:
        with self._lock:
            runs = [r for r in self.all() if r["run_id"] != record["run_id"]]
            runs.insert(0, record)
            self.path.write_text(json.dumps(runs[:1000], indent=2, default=str))

    def update(self, run_id: str, **fields: Any) -> None:
        rec = self.get(run_id) or {"run_id": run_id}
        for k, v in fields.items():
            if isinstance(v, dict) and isinstance(rec.get(k), dict):
                rec[k] = {**rec[k], **v}
            else:
                rec[k] = v
        self.upsert(rec)


def make_record(run_id: str, mode: str, prompt: str, model: str, url: str | None = None, output_dir: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "model": model,
        "mode": mode,
        "output_mode": extra.pop("output_mode", "client"),
        "output_language": extra.pop("output_language", "python"),
        "status": "running",
        "usage": {},
        "paths": {
            "har_dir": str(runs_root(output_dir) / "har" / run_id),
            "script_path": str(runs_root(output_dir) / "scripts" / run_id),
            "messages_path": str(runs_root(output_dir) / "messages" / f"{run_id}.jsonl"),
        },
        **extra,
    }


def discover_scripts(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in SCRIPT_EXTS and "node_modules" not in p.parts and "__pycache__" not in p.parts]
    order = {".py": 0, ".js": 1, ".ts": 2, ".go": 3}
    return sorted(files, key=lambda p: (not p.stem.startswith("api_client"), order.get(p.suffix, 9), str(p)))


def run_command_for(path: Path) -> list[str]:
    ext = path.suffix
    if ext == ".py":
        import sys

        return [sys.executable, path.name]
    if ext in (".js", ".mjs"):
        return ["node", path.name]
    if ext == ".ts":
        return ["npx", "--yes", "tsx", path.name]
    if ext == ".go":
        return ["go", "run", path.name]
    if ext == ".php":
        return ["php", path.name]
    if ext == ".rb":
        return ["ruby", path.name]
    if ext == ".sh":
        return ["bash", path.name]
    if ext == ".java":
        return ["mvn", "-q", "compile", "exec:exec"] if (path.parent / "pom.xml").exists() else ["java", path.name]
    if ext == ".cs":
        return ["dotnet", "run"]
    if ext == ".c":
        return ["bash", "-c", f"cc {path.name} $(ls cJSON.c 2>/dev/null) -lcurl -o api_client && ./api_client"]
    if ext in (".psm1", ".ps1"):
        return ["pwsh", "-NoProfile", "-File", "Example.ps1" if (path.parent / "Example.ps1").exists() else path.name]
    return [str(path)]


# ----------------------------------------------------------------------------- sync


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:n].rstrip("_") or "run") if s else "run"


def available_dir(base: Path, name: str) -> Path:
    d = base / name
    if not d.exists() or (d.is_dir() and not any(d.iterdir())):
        return d
    return base / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class Syncer:
    """Polling mirror of a directory (stdlib replacement for watchdog)."""

    def __init__(self, src: Path, dst: Path, interval: float = 1.0) -> None:
        self.src, self.dst, self.interval = src, dst, interval
        self._stop = threading.Event()
        self._mtimes: dict[str, float] = {}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> Syncer:
        self.thread.start()
        return self

    def sync_once(self) -> int:
        n = 0
        for p in self.src.rglob("*"):
            if not p.is_file() or any(part in ("node_modules", "__pycache__", ".venv") for part in p.parts):
                continue
            rel = p.relative_to(self.src)
            m = p.stat().st_mtime
            if self._mtimes.get(str(rel)) != m:
                out = self.dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
                self._mtimes[str(rel)] = m
                n += 1
        return n

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync_once()
            except OSError:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.sync_once()
        except OSError:
            pass


def wait_until(pred, timeout: float, step: float = 0.2) -> bool:  # noqa: ANN001
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return False
