"""Hermes engine: the local `hermes` CLI performs the engineering turn (ported from RAE's Hermes backend).

revapi's own engine talks to a model over HTTP. Hermes is a CLI that is already installed and already
authenticated on the user's machine, so a run here is a subprocess: `hermes chat --oneshot --format
stream-json` reads the prompt from stdin and writes one JSON event per line, which this module turns into
the same events (thinking, tool_start, tool_end, usage) the built-in agent feeds the Reporter.

- No credential passes through revapi: the model, the provider and the authentication are the user's own
  Hermes configuration. ``hermes_model`` / ``hermes_provider`` only override which model it runs.
- ``--ignore-rules`` keeps an AGENTS.md or CLAUDE.md in the workspace from steering the run.
- Cost is estimated from the token counts the CLI reports. A model with no known price stays at 0.
- It covers engineering only (engineer mode, and manual mode's engineering step): capture with the
  built-in engine, then engineer with Hermes.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .llm import PRICING, Usage, _price

BIN_ENVS = ("REVAPI_HERMES_BIN", "RAE_HERMES_BIN")  # a hermes outside PATH (a virtualenv, a wrapper script)
STREAM_TIMEOUT_S = 900.0  # a single event arriving later than this means the CLI is stuck, not busy
MIN_VERSION = "v2026.9.21"  # `hermes chat --format stream-json` first shipped here; the PyPI package is older


def install_command() -> str:
    """The official Hermes installer for this OS."""
    if sys.platform == "win32":
        return "iex (irm https://hermes-agent.nousresearch.com/install.ps1)"
    return "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"


def missing_message() -> str:
    return f"The hermes CLI was not found. Install Hermes Agent {MIN_VERSION} or later with `{install_command()}`, or point REVAPI_HERMES_BIN at the binary."


def outdated_message() -> str:
    return (
        "The installed hermes CLI does not support `--format stream-json`, which this engine needs. "
        f"Install Hermes Agent {MIN_VERSION} or later with `{install_command()}`; the hermes-agent package on PyPI is older."
    )


def hermes_binary() -> str | None:
    """The `hermes` CLI to run: REVAPI_HERMES_BIN (or RAE_HERMES_BIN) when set, otherwise PATH."""
    override = next((os.environ[k] for k in BIN_ENVS if os.environ.get(k)), None)
    if override:
        found = shutil.which(override)
        if found:
            return found
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return str(Path(override).resolve())  # the run changes directory, so a relative path would stop resolving
        return None
    return shutil.which("hermes")


def parse_event(line: str) -> dict[str, Any] | None:
    """Decode one NDJSON event; anything else (a banner, a stray log line) is ignored."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _as_int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


class HermesRun:
    """One engineering conversation through the Hermes CLI; each turn is one subprocess."""

    def __init__(
        self,
        scripts_dir: Path,
        on_event: Callable[[str, dict[str, Any]], None],
        model: str | None = None,
        provider: str | None = None,
        max_turns: int | None = None,
        log_path: Path | None = None,
        session_id: str | None = None,
    ) -> None:
        # The CLI runs with its cwd in the workspace, so every path it gets must be absolute.
        self.scripts_dir = Path(scripts_dir).expanduser().resolve()
        self.on_event = on_event
        self.model = model or None
        self.provider = provider or None
        self.max_turns = max_turns or None
        self.log_path = log_path
        self.session_id = session_id
        self.stream_model: str | None = None
        self.usage = Usage()
        self._narrative: list[str] = []
        self._seq = 0
        self._starts: dict[str, float] = {}
        if log_path is not None and log_path.exists():
            self._seq = sum(1 for _ in log_path.open(encoding="utf-8"))

    # -- the transcript, in the same shape the built-in agent writes, so /resume can replay it ----------
    def _log(self, msg: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**msg, "seq": self._seq, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, default=str) + "\n")
        self._seq += 1

    @property
    def label(self) -> str:
        return f"hermes · {self.model or self.stream_model or 'default'}"

    def argv(self) -> list[str]:
        """The invocation, without the prompt: it arrives on stdin (-), so quotes and backticks stay verbatim."""
        binary = hermes_binary()
        if binary is None:
            raise RuntimeError(missing_message())
        argv = [binary, "chat", "--query-file", "-", "--oneshot", "--format", "stream-json", "--in", str(self.scripts_dir), "--ignore-rules"]
        if self.max_turns:
            argv += ["--max-turns", str(self.max_turns)]
        if self.session_id:
            argv += ["--resume", self.session_id]
        if self.model:
            argv += ["-m", self.model]
        if self.provider:
            argv += ["--provider", self.provider]
        return argv

    # -- events -----------------------------------------------------------------------------------------
    def _flush_narrative(self) -> None:
        text = "".join(self._narrative).strip()
        self._narrative = []
        if text:
            self.on_event("thinking", {"text": text})
            self._log({"role": "assistant", "content": "", "reasoning": text})

    def _apply_usage(self, event: dict[str, Any]) -> None:
        tokens = event.get("tokens")
        if not isinstance(tokens, dict):
            return
        u = Usage(
            input_tokens=_as_int(tokens.get("input")),
            output_tokens=_as_int(tokens.get("output")),
            cache_read_tokens=_as_int(tokens.get("cache_read")),
            cache_creation_tokens=_as_int(tokens.get("cache_write")),
            requests=1,
        )
        model = self.model or self.stream_model or ""
        short = model.split("/")[-1].lower()
        if short and any(short.startswith(k) for k in PRICING):
            u.estimated_cost_usd = _price(model, u)
        elif u.input_tokens or u.output_tokens:
            self.on_event("progress", {"message": f"no price known for {model or 'this model'}; cost left at 0"})
        self.usage.add(u)  # a follow-up turn adds to the run's totals
        self.on_event("usage", self.usage.as_dict())

    def dispatch(self, event: dict[str, Any]) -> tuple[bool, str | None]:
        """Feed one event to the UI. Returns (the turn is over, the error when it failed)."""
        kind = event.get("type")
        if kind == "text":
            if isinstance(event.get("text"), str):
                self._narrative.append(event["text"])
            return False, None
        if kind == "system":
            if isinstance(event.get("model"), str) and event["model"]:
                self.stream_model = event["model"]
            if isinstance(event.get("session_id"), str) and event["session_id"]:
                self.session_id = event["session_id"]
            return False, None
        if kind == "tool_use":
            self._flush_narrative()
            name = str(event.get("name") or "tool")
            args = event.get("input") if isinstance(event.get("input"), dict) else {}
            call_id = str(event.get("id") or event.get("call_id") or f"h{self._seq}")
            self._starts[name] = time.time()
            self.on_event("tool_start", {"name": name, "input": args})
            self._log({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
            return False, None
        if kind == "tool_result":
            name = str(event.get("name") or "tool")
            out = event.get("output") if isinstance(event.get("output"), str) else ""
            is_error = bool(event.get("is_error"))
            ms = int((time.time() - self._starts.pop(name, time.time())) * 1000)
            self.on_event("tool_end", {"name": name, "is_error": is_error, "output": out, "duration_ms": ms, "input": {}})
            self._log({"role": "tool", "tool_call_id": str(event.get("id") or event.get("call_id") or ""), "name": name, "content": out, **({"is_error": True} if is_error else {})})
            return False, None
        if kind == "result":
            if isinstance(event.get("session_id"), str) and event["session_id"]:
                self.session_id = event["session_id"]
            final = event.get("text")
            if event.get("replay_gap") and isinstance(final, str):
                self._narrative = [final]  # the text events stopped partway; the result carries the whole reply
            self._apply_usage(event)
            self._flush_narrative()
            code = event.get("exit_code")
            if isinstance(code, int) and code != 0:
                # `error` carries the cause (a missing key, an unknown model); `text` is the final reply.
                reason = next((v for v in (event.get("error"), event.get("text")) if isinstance(v, str) and v.strip()), "")
                return True, reason.strip() or f"hermes exited with code {code}"
            return True, None
        return False, None

    # -- one turn ---------------------------------------------------------------------------------------
    def turn(self, prompt: str, shown_prompt: str | None = None) -> str | None:
        """Run one CLI turn and stream its events. Returns None on success, else the error."""
        self._narrative = []
        try:
            argv = self.argv()
        except RuntimeError as e:
            return str(e)
        self._log({"role": "user", "content": shown_prompt if shown_prompt is not None else prompt})
        self.on_event("thinking_start", {})
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(self.scripts_dir), start_new_session=True)
        lines: queue.Queue[bytes | None] = queue.Queue()
        stderr = bytearray()

        def pump_out() -> None:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                lines.put(raw)
            lines.put(None)

        def pump_err() -> None:
            assert proc.stderr is not None
            for chunk in iter(lambda: proc.stderr.read(65536), b""):  # type: ignore[union-attr]
                stderr.extend(chunk)

        threading.Thread(target=pump_out, daemon=True).start()
        err_thread = threading.Thread(target=pump_err, daemon=True)
        err_thread.start()
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass
        error: str | None = None
        saw_result = False
        last = time.time()
        try:
            while True:
                try:
                    raw = lines.get(timeout=0.25)  # short waits: a stop lands at once
                except queue.Empty:
                    if time.time() - last > STREAM_TIMEOUT_S:
                        error = f"hermes produced no output for {int(STREAM_TIMEOUT_S)}s"
                        break
                    continue
                if raw is None:
                    break
                last = time.time()
                event = parse_event(raw.decode("utf-8", errors="replace"))
                if event is None:
                    continue
                done, failure = self.dispatch(event)
                if done:
                    saw_result, error = True, failure
                    break
        except BaseException:
            self.stop(proc)
            raise
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.stop(proc)
        err_thread.join(timeout=10)
        extra = stderr.decode("utf-8", errors="replace").strip()
        if error:
            return f"{error}\n{extra}" if extra else error
        if not saw_result and "unrecognized arguments" in extra:
            return outdated_message()
        if proc.returncode not in (None, 0):
            return extra or f"hermes exited with code {proc.returncode}"
        if not saw_result:
            # Without the closing event the turn stopped early; a client from an earlier turn would make it look like a success.
            return extra or "hermes ended without a result event (the run did not finish)"
        return None

    @staticmethod
    def stop(proc: subprocess.Popen) -> None:
        """Stop the CLI for good: ask, wait, then force, so a stopped run leaves no process behind."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                return
            try:
                os.killpg(proc.pid, sig)
            except (OSError, AttributeError):
                (proc.terminate if sig == signal.SIGTERM else proc.kill)()
            try:
                proc.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                continue
