"""The tool-calling agent loop (provider-agnostic), with streaming and lossless context compaction."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import compaction as cmp
from .llm import LLM, LLMError, Usage, wire
from .tools import Context, Tool, _obj, run_tool

Event = Callable[[str, dict[str, Any]], None]

COMPACT_AT = float(os.environ.get("REVAPI_COMPACT_AT", "0.8"))  # fraction of the window that triggers compaction
TAIL_SHARE = 0.25  # recent context kept verbatim, as a share of the window (capped at TAIL_MAX tokens)
TAIL_MAX = 40_000
CHARS_PER_TOKEN = 3.5
MAX_TOOL_SHARE = 0.2  # one tool output may fill at most this share of the window in live context


def _chars(m: dict[str, Any]) -> int:
    return len(json.dumps(wire(m), ensure_ascii=False))


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
        compact_at: float | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx
        self.log_path = log_path
        self.on_event = on_event or (lambda kind, data: None)
        self.max_turns = max_turns
        self.compact_at = compact_at or COMPACT_AT
        self.usage = Usage()
        self.final_text = ""
        self.compactions = 0
        self._compact_floor = 0
        self._forced = False
        self._measured: tuple[int, int] | None = None  # (messages sent, tokens the provider counted)
        self._seq = self._next_seq()
        if messages is None:
            self.messages: list[dict[str, Any]] = []
            self._push({"role": "system", "content": system})
        else:
            self.messages = messages
        if log_path is not None and "recall" not in self.tools:
            self.tools["recall"] = Tool(
                "recall",
                "Search or read the full, verbatim history of this session, including everything removed from your context by compaction. "
                "recall(id=N) returns message/tool output #N in full (page with offset); recall(query='regex') searches all of it; no arguments lists the index.",
                _obj({"id": {"type": "integer"}, "query": {"type": "string"}, "offset": {"type": "integer"}}),
                lambda id=None, query=None, offset=0: cmp.recall(self.log_path, id, query, offset or 0),  # noqa: A002
            )

    # -- persistence ----------------------------------------------------------
    def _next_seq(self) -> int:
        if not self.log_path or not self.log_path.exists():
            return 0
        recs = cmp.read_transcript(self.log_path)
        return (recs[-1]["seq"] + 1) if recs else 0

    def _log(self, rec: dict[str, Any]) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), **rec}, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def load_messages(path: Path) -> list[dict[str, Any]]:
        """Rebuild the live context: the latest compaction snapshot (if any) plus everything logged after it."""
        msgs: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("type") == "compaction" and isinstance(m.get("messages"), list):
                msgs = [{**{k: v for k, v in x.items() if k != "seq"}, **({"_seq": x["seq"]} if x.get("seq") is not None else {})} for x in m["messages"]]
                continue
            if m.get("role") in ("system", "user", "assistant", "tool"):
                m.pop("ts", None)
                m.pop("usage", None)
                if "seq" in m:
                    m["_seq"] = m.pop("seq")
                msgs.append(m)
        while msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls"):
            msgs.pop()  # a tool call whose results were never recorded
        return msgs

    def _push(self, msg: dict[str, Any]) -> None:
        msg = {**msg, "_seq": self._seq}
        self._seq += 1
        rec = {k: v for k, v in msg.items() if k != "_seq"}
        self._log({"seq": msg["_seq"], **rec})  # the transcript keeps the full text, always
        if msg.get("role") == "tool":
            cap = int(self._window() * MAX_TOOL_SHARE * CHARS_PER_TOKEN)
            if len(msg.get("content") or "") > cap:
                msg["content"] = cmp.shrink(msg["content"], msg["_seq"], keep=cap)
        self.messages.append(msg)

    # -- context accounting ---------------------------------------------------
    def _window(self) -> int:
        return int(getattr(self.llm, "context_window", 0) or 128_000)

    def _limit(self) -> int:
        w = self._window()
        reserve = min(int(getattr(self.llm, "max_tokens", 16000) or 16000), w // 4) + 2000
        return int(min(w * self.compact_at, w - reserve))

    def context_tokens(self, schemas: list[dict] | None = None) -> int:
        """Tokens the next request will carry: the provider's own count when it gave one, never less than
        a character-based estimate (some providers under-report, and overflowing is worse than compacting early)."""
        extra = len(json.dumps(schemas)) if schemas else 0
        estimate = int((sum(_chars(m) for m in self.messages) + extra) / CHARS_PER_TOKEN)
        if self._measured and self._measured[0] <= len(self.messages):
            n, tokens = self._measured
            measured = tokens + int(sum(_chars(m) for m in self.messages[n + 1 :]) / CHARS_PER_TOKEN)
            return max(measured, int(estimate * 0.85))
        return estimate

    def _original_request(self) -> str:
        for m in self.messages[1:]:
            if m.get("_summary"):
                return m.get("_original") or ""
            if m.get("role") == "user":
                return m.get("content") or ""
        return ""

    # -- compaction -----------------------------------------------------------
    def _estimate(self, schemas: list[dict] | None = None) -> int:
        extra = len(json.dumps(schemas)) if schemas else 0
        return int((sum(_chars(m) for m in self.messages) + extra) / CHARS_PER_TOKEN)

    def maybe_compact(self, schemas: list[dict] | None = None, force: bool = False) -> bool:
        est = self.context_tokens(schemas)
        if not force and est <= self._limit():
            return False
        if not force and self._compact_floor and est < self._compact_floor + self._window() * 0.08:
            return False  # the last attempt could not shrink this context; wait until it has really grown
        if len(self.messages) < 4:
            return False
        self.on_event("compaction_start", {"tokens": est, "window": self._window()})
        before = self._estimate(schemas)
        saved, saved_measured = list(self.messages), self._measured
        self._forced = force
        try:
            method = self._summarize(schemas) or self._mechanical(schemas)
        finally:
            self._forced = False
        after = self._estimate(schemas)
        if after >= before * 0.95:  # nothing gained: keep the conversation exactly as it was
            self.messages, self._measured = saved, saved_measured
            self._compact_floor = est
            self.on_event("progress", {"message": "context is mostly instructions and tools; nothing to compact yet"})
            return False
        self._measured = None
        self._compact_floor = 0
        self.compactions += 1
        self._log(
            {
                "type": "compaction",
                "method": method,
                "before": before,
                "after": after,
                "messages": [{**{k: v for k, v in m.items() if k != "_seq"}, "seq": m.get("_seq")} if m.get("_seq") is not None else m for m in self.messages],
            }
        )
        self.on_event("compaction", {"before": before, "after": after, "window": self._window(), "method": method, "count": self.compactions})
        return True

    def _split(self) -> int:
        """Index where the verbatim tail starts: a user or assistant message, so no tool result is orphaned.
        A manual /compact keeps only the latest exchange verbatim."""
        budget = 0 if self._forced else min(self._window() * TAIL_SHARE, TAIL_MAX) * CHARS_PER_TOKEN
        i, used = len(self.messages), 0
        while i > 2 and used + _chars(self.messages[i - 1]) <= budget:
            i -= 1
            used += _chars(self.messages[i])
        while i < len(self.messages) and self.messages[i]["role"] == "tool":
            i += 1
        if i >= len(self.messages):  # the last exchange alone is bigger than the budget: keep from its assistant turn
            i = max((j for j, m in enumerate(self.messages) if m["role"] in ("user", "assistant") and j >= 2), default=len(self.messages))
        return max(2, i)

    def _summarize(self, schemas: list[dict] | None) -> str | None:
        split = self._split()
        head = self.messages[1:split]
        if not head or all(m.get("_summary") for m in head):
            return None
        original = self._original_request()
        budget = int(self._window() * 0.55 * CHARS_PER_TOKEN)
        summary = ""
        for _attempt in range(3):
            try:
                summary = self._summarize_chunks(head, budget)
                break
            except LLMError as e:
                if e.kind == "context_length":
                    budget //= 2
                    continue
                self.on_event("progress", {"message": f"summary failed ({e}); compacting mechanically"})
                return None
        if not summary.strip():
            return None
        cont = cmp.CONTINUATION.format(original=original, summary=summary)
        tail = self.messages[split:]
        first: dict[str, Any] = {"role": "user", "content": cont, "_summary": True, "_original": original}
        if tail and tail[0]["role"] == "user":
            t0 = tail.pop(0)
            first = {**first, "content": cont + "\n\n" + (t0.get("content") or ""), "_seq": t0.get("_seq")}
        self.messages = [self.messages[0], first, *tail]
        return "summary"

    def _summarize_chunks(self, head: list[dict[str, Any]], budget: int) -> str:
        chunks: list[list[dict[str, Any]]] = [[]]
        size = 0
        for m in head:
            c = min(_chars(m), cmp.PER_MESSAGE_CHARS + 300) if not m.get("_summary") else _chars(m)
            if chunks[-1] and size + c > budget:
                chunks.append([])
                size = 0
            chunks[-1].append(m)
            size += c
        summary = ""
        for n, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                self.on_event("progress", {"message": f"summarizing part {n}/{len(chunks)}"})
            prev = f"<previous_summary>\n{summary}\n</previous_summary>\nMerge the previous summary with the new transcript below into one complete summary.\n\n" if summary else ""
            transcript = cmp.render_for_summary([m if not m.get("_summary") else {**m, "content": m["content"]} for m in chunk])
            reply = self.llm.chat(
                [{"role": "system", "content": cmp.SUMMARY_SYSTEM}, {"role": "user", "content": cmp.SUMMARY_PROMPT.format(previous=prev, transcript=transcript)}],
                None,
                on_retry=self._on_retry,
            )
            self.usage.add(reply.usage)
            summary = cmp.extract_summary(reply.content) or summary
        return summary

    def _mechanical(self, schemas: list[dict] | None) -> str:
        """No model summary available: shorten old outputs (pointers keep them recallable), then drop the oldest turns."""
        target = int(self._limit() * 0.6)
        keep_from = max(2, len(self.messages) - 8)
        for i in range(1, keep_from):
            m = self.messages[i]
            if m["role"] == "tool" and len(m.get("content") or "") > 800:
                self.messages[i] = {**m, "content": cmp.shrink(m["content"], m.get("_seq"), keep=500)}
            elif m["role"] == "assistant" and m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    a = tc["function"]["arguments"]
                    if len(a) > 1500:
                        a = json.dumps({"_archived": f"arguments archived ({len(a):,} chars) — recall(id={m.get('_seq')})"})
                    calls.append({**tc, "function": {**tc["function"], "arguments": a}})
                self.messages[i] = {**m, "tool_calls": calls}
        self._measured = None
        if self.context_tokens(schemas) <= target:
            return "trimmed"
        original = self._original_request()
        dropped: list[int] = []
        while self.context_tokens(schemas) > target and len(self.messages) > 4:
            j = 2 if self.messages[1].get("_summary") else 1
            dropped.append(self.messages[j].get("_seq") or 0)
            del self.messages[j]
            while len(self.messages) > j and self.messages[j]["role"] == "tool":
                dropped.append(self.messages[j].get("_seq") or 0)
                del self.messages[j]
        note = (
            f"Older turns (#{min(dropped)}–#{max(dropped)}) were removed from context to make room; they are archived verbatim — use recall(query=...) or recall(id=N) before redoing work."
            if dropped
            else "Older tool outputs were shortened; recall(id=N) returns any of them in full."
        )
        cont = cmp.CONTINUATION.format(original=original, summary=note)
        if self.messages[1].get("_summary"):
            self.messages[1] = {**self.messages[1], "content": self.messages[1]["content"] + "\n\n" + note}
        else:
            self.messages.insert(1, {"role": "user", "content": cont, "_summary": True, "_original": original})
            if len(self.messages) > 2 and self.messages[2]["role"] == "user":
                nxt = self.messages.pop(2)
                self.messages[1]["content"] += "\n\n" + (nxt.get("content") or "")
        return "dropped" if dropped else "trimmed"

    # -- loop -----------------------------------------------------------------
    def _on_retry(self, attempt: int, total: int, wait: float, reason: str) -> None:
        self.on_event("retry", {"attempt": attempt, "max": total, "wait": wait, "reason": reason})

    def send(self, user_text: str) -> dict[str, Any]:
        self._push({"role": "user", "content": user_text})
        return self.run()

    def run(self) -> dict[str, Any]:
        nudges = 0
        error: str | None = None
        error_kind: str | None = None
        schemas = [t.schema() for t in self.tools.values()]
        turn = 0
        partial: dict[str, str] = {"text": "", "thinking": ""}

        def on_delta(kind: str, payload: Any) -> None:
            if kind == "text":
                partial["text"] += payload
                self.on_event("text_delta", {"text": payload})
            elif kind == "thinking":
                partial["thinking"] += payload
                self.on_event("thinking_delta", {"text": payload})
            elif kind == "tool":
                self.on_event("tool_delta", payload)
            elif kind == "reset":
                partial["text"] = partial["thinking"] = ""
                self.on_event("stream_reset", {})

        try:
            while turn < self.max_turns:
                turn += 1
                self.maybe_compact(schemas)
                self.on_event("thinking_start", {"turn": turn})
                partial["text"] = partial["thinking"] = ""
                reply = None
                for overflow in range(4):
                    try:
                        reply = self.llm.chat(self.messages, schemas, on_delta=on_delta, on_retry=self._on_retry)
                        break
                    except LLMError as e:
                        if e.kind != "context_length" or overflow == 3:
                            error, error_kind = str(e), e.kind
                            break
                        # the provider says it does not fit: learn the real window, compact harder, ask again
                        learned = getattr(e, "window", None)
                        cur = self._window()
                        self.llm.context_window = learned if learned and learned < cur else int(cur * 0.7)
                        self.on_event("progress", {"message": f"context window is {self.llm.context_window:,} tokens; compacting"})
                        self.maybe_compact(schemas, force=True)
                if reply is None:
                    break
                self.usage.add(reply.usage)
                self._measured = (len(self.messages), reply.usage.input_tokens + reply.usage.output_tokens) if reply.usage.input_tokens else None
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
                if reply.reasoning:
                    asst["reasoning"] = reply.reasoning
                if reply.thinking_blocks:
                    asst["thinking_blocks"] = reply.thinking_blocks
                self._push(asst)
                self.on_event("context", {"tokens": self.context_tokens(schemas), "window": self._window(), "limit": self._limit()})
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
                    self.on_event("tool_end", {"name": tc["name"], "is_error": is_err, "output": out, "duration_ms": int((time.time() - t0) * 1000), "input": tc["arguments"]})
                    self._push({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": out})
                if self.ctx.finished:
                    self.final_text = self.ctx.finished["summary"]
                    break
            else:
                error, error_kind = f"stopped after max_turns={self.max_turns}", "max_turns"
        except KeyboardInterrupt:
            error, error_kind = "interrupted by user", "interrupted"
            # keep what was streamed so far, and close any tool call left without a result
            if self.messages and self.messages[-1]["role"] == "assistant" and self.messages[-1].get("tool_calls"):
                done = {m.get("tool_call_id") for m in self.messages if m["role"] == "tool"}
                for tc in self.messages[-1]["tool_calls"]:
                    if tc["id"] not in done:
                        self._push({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": "(interrupted by the user before this finished)"})
            elif partial["text"] or partial["thinking"]:
                self._push({"role": "assistant", "content": (partial["text"] + "\n[interrupted by the user]").strip(), **({"reasoning": partial["thinking"]} if partial["thinking"] else {})})
            self.on_event("interrupted", {})
        return {
            "summary": self.final_text,
            "error": error,
            "error_kind": error_kind,
            "usage": self.usage.as_dict(),
            "turns": turn,
            "finished": bool(self.ctx.finished),
            "compactions": self.compactions,
        }
