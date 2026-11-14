"""The tool-calling agent loop (provider-agnostic)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .llm import LLM, LLMError, Usage
from .tools import Context, Tool, run_tool

Event = Callable[[str, dict[str, Any]], None]

CONTEXT_CHARS = int(os.environ.get("REVAPI_CONTEXT_CHARS", str(360_000)))  # ~100k tokens
KEEP_RECENT_TOOL_RESULTS = 8


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        ctx: Context,
        system: str,
        log_path: Path | None = None,
        on_event: Event | None = None,
        max_turns: int = 120,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx
        self.log_path = log_path
        self.on_event = on_event or (lambda kind, data: None)
        self.max_turns = max_turns
        self.usage = Usage()
        self.messages: list[dict[str, Any]] = messages or [{"role": "system", "content": system}]
        if messages is None:
            self._log(self.messages[0])
        self.final_text = ""

    # -- persistence ----------------------------------------------------------
    def _log(self, msg: dict[str, Any]) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), **msg}, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def load_messages(path: Path) -> list[dict[str, Any]]:
        msgs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("role") in ("system", "user", "assistant", "tool"):
                m.pop("ts", None)
                m.pop("usage", None)
                msgs.append(m)
        # drop a dangling assistant tool_call without results at the tail
        while msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls"):
            msgs.pop()
        return msgs

    def _push(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)
        self._log(msg)

    # -- context management ---------------------------------------------------
    def _size(self) -> int:
        return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) for m in self.messages)

    def _compact(self) -> None:
        if self._size() <= CONTEXT_CHARS:
            return
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        for i in tool_idx[:-KEEP_RECENT_TOOL_RESULTS]:
            c = self.messages[i].get("content") or ""
            if len(c) > 600:
                self.messages[i] = {**self.messages[i], "content": c[:400] + f"\n...[older tool output compacted, {len(c)} chars; re-run the tool if needed]"}
            if self._size() <= CONTEXT_CHARS * 0.7:
                return
        # still too big: compact long assistant arguments (e.g. old write_file contents)
        for m in self.messages[:-6]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    a = tc["function"]["arguments"]
                    if len(a) > 1500:
                        tc["function"]["arguments"] = json.dumps({"_compacted": a[:300]})
            if self._size() <= CONTEXT_CHARS * 0.7:
                return

    # -- loop -----------------------------------------------------------------
    def send(self, user_text: str) -> dict[str, Any]:
        self._push({"role": "user", "content": user_text})
        return self.run()

    def run(self) -> dict[str, Any]:
        nudges = 0
        error: str | None = None
        error_kind: str | None = None
        schemas = [t.schema() for t in self.tools.values()]
        turn = 0
        try:
            while turn < self.max_turns:
                turn += 1
                self._compact()
                self.on_event("thinking_start", {"turn": turn})
                try:
                    reply = self.llm.chat(self.messages, schemas)
                except LLMError as e:
                    error, error_kind = str(e), e.kind
                    break
                self.usage.add(reply.usage)
                self.on_event("usage", self.usage.as_dict())
                if reply.reasoning:
                    self.on_event("thinking", {"text": reply.reasoning})
                if reply.content:
                    self.on_event("text", {"text": reply.content})
                asst: dict[str, Any] = {"role": "assistant", "content": reply.content or None}
                if reply.tool_calls:
                    asst["tool_calls"] = [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}} for tc in reply.tool_calls
                    ]
                self._push({**asst, "usage": reply.usage.as_dict()} if self.log_path else asst)
                if self.log_path:
                    self.messages[-1].pop("usage", None)
                if not reply.tool_calls:
                    self.final_text = reply.content or self.final_text
                    if "finish" not in self.tools:
                        break
                    if nudges >= 2:
                        break
                    nudges += 1
                    self._push({"role": "user", "content": "Continue working with the tools. When everything is done and tested, call `finish` with your summary."})
                    continue
                nudges = 0
                for tc in reply.tool_calls:
                    self.on_event("tool_start", {"name": tc["name"], "input": tc["arguments"]})
                    t0 = time.time()
                    out, is_err = run_tool(self.tools, tc["name"], tc["arguments"])
                    self.on_event("tool_end", {"name": tc["name"], "is_error": is_err, "output": out, "duration_ms": int((time.time() - t0) * 1000)})
                    self._push({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": out})
                if self.ctx.finished:
                    self.final_text = self.ctx.finished["summary"]
                    break
            else:
                error, error_kind = f"stopped after max_turns={self.max_turns}", "max_turns"
        except KeyboardInterrupt:
            error, error_kind = "interrupted by user", "interrupted"
        return {"summary": self.final_text, "error": error, "error_kind": error_kind, "usage": self.usage.as_dict(), "turns": turn, "finished": bool(self.ctx.finished)}
