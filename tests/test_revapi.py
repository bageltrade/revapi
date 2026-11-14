"""Tests for revapi. Run: python -m pytest tests -q   (or: python tests/test_revapi.py)

Browser tests need a Chromium binary (skipped otherwise). LLM calls are replaced by a
scripted fake so the full agent/engineer/collector pipelines run deterministically offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ["REVAPI_HOME"] = tempfile.mkdtemp(prefix="revapi-test-home-")
os.environ["REVAPI_NO_SYNC"] = "1"

from fixture_site import serve  # noqa: E402

from revapi import har as harlib  # noqa: E402
from revapi import llm as llmmod  # noqa: E402
from revapi import modes, store  # noqa: E402
from revapi.browser import Browser, find_chromium, is_api_like  # noqa: E402
from revapi.cdp import _xor_mask  # noqa: E402
from revapi.llm import LLM, Reply, Usage  # noqa: E402
from revapi.prompts import load  # noqa: E402
from revapi import term as termmod  # noqa: E402
from revapi.agent import Agent  # noqa: E402
from revapi.tools import Context, Tool  # noqa: E402
from revapi.ui import Reporter  # noqa: E402

HAS_CHROMIUM = bool(find_chromium())
SAMPLE_HAR = Path("/home/user/.reverse-api/runs/har/59c7dc3b7609/recording.har")


class _Resp:
    def __init__(self, body: bytes, ctype: str = "application/json", die: bool = False) -> None:
        self.body, self.die = body, die
        self.headers = {"content-type": ctype}

    def read(self) -> bytes:
        return self.body

    def __iter__(self):  # noqa: ANN204
        for line in self.body.splitlines(keepends=True):
            yield line
        if self.die:
            raise ConnectionResetError("connection reset by peer")

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *a) -> None:  # noqa: ANN002
        return None


def _json(obj: dict) -> _Resp:
    return _Resp(json.dumps(obj).encode())


def _sse_resp(events: list[dict], die: bool = False) -> _Resp:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + ("" if die else "data: [DONE]\n\n")
    return _Resp(body.encode(), "text/event-stream", die)


def _http_error(code: int, text: str, headers: dict | None = None) -> Exception:
    import io
    import urllib.error

    return urllib.error.HTTPError("http://x", code, "err", headers or {}, io.BytesIO(text.encode()))


class FakeNet:
    """Stands in for urllib.request.urlopen: plays back responses/exceptions and records requests."""

    def __init__(self, plan: list) -> None:  # noqa: ANN001
        self.plan = list(plan)
        self.calls: list[dict] = []
        self._p = mock.patch("urllib.request.urlopen", self)

    def __call__(self, req, timeout=0):  # noqa: ANN001, ANN204
        self.calls.append({"url": req.full_url, "body": json.loads(req.data or b"{}"), "headers": dict(req.header_items())})
        step = self.plan.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    def __enter__(self):  # noqa: ANN204
        self._p.start()
        return self

    def __exit__(self, *a) -> None:  # noqa: ANN002
        self._p.stop()


class FakeLLM(LLM):
    """Plays back a script: each item is a list of (tool, args) or a final text."""

    def __init__(self, script, window: int = 128_000):  # noqa: ANN001
        self.provider, self.model, self.api_key, self.effort = "fake", "scripted", "x", None
        self.context_window, self.max_tokens = window, 1000
        self.script = list(script)
        self.seen: list[list[dict]] = []
        self.summaries = 0

    def chat(self, messages, tools=None, on_delta=None, on_retry=None, stream=None):  # noqa: ANN001
        if tools is None and "compacting the working memory" in (messages[0].get("content") or ""):
            self.summaries += 1
            return Reply(content=f"<analysis>x</analysis><summary>SUMMARY-{self.summaries}: endpoints GET /api/todos</summary>", usage=Usage(input_tokens=50, output_tokens=20, requests=1))
        self.seen.append(json.loads(json.dumps([llmmod.wire(m) for m in messages])))
        step = self.script.pop(0) if self.script else "done"
        r = Reply(usage=Usage(input_tokens=100, output_tokens=10, requests=1))
        if callable(step):
            step = step(messages)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            r.content = step
            if on_delta:
                for part in (step[: len(step) // 2], step[len(step) // 2 :]):
                    on_delta("text", part)
        else:
            r.tool_calls = [{"id": f"c{len(self.seen)}_{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(step)]
        return r


def _cfg() -> store.Config:
    c = store.Config(Path(tempfile.mkdtemp()) / "config.json")
    c.data["real_time_sync"] = False
    return c


# ============================================================================ unit


class Unit(unittest.TestCase):
    def test_xor_mask(self) -> None:
        data, mask = os.urandom(1000), os.urandom(4)
        self.assertEqual(_xor_mask(_xor_mask(data, mask), mask), data)
        self.assertEqual(_xor_mask(b"", mask), b"")
        slow = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.assertEqual(_xor_mask(data, mask), slow)

    def test_coerce_args(self) -> None:
        from revapi.tools import coerce_args

        schema = {"properties": {"all": {"type": "boolean"}, "last": {"type": "integer"}, "items": {"type": "array"}, "url": {"type": "string"}}}
        self.assertEqual(coerce_args(schema, {"all": "false", "last": "5", "items": '[{"a": 1}]', "url": 5, "junk": 1}), {"all": False, "last": 5, "items": [{"a": 1}], "url": "5"})

    def test_almost_json_repair(self) -> None:
        # real failure from gpt-oss-20b: closing quote of "content" escaped
        bad = '{"path":"api_client.py","content":"print(\\"hi\\")\\n\\"}'
        self.assertEqual(llmmod._parse_args(bad), {"path": "api_client.py", "content": 'print("hi")\n'})
        self.assertEqual(llmmod._parse_args('{"a": "x\ny"}'), {"a": "x\ny"})
        self.assertEqual(llmmod._parse_args('{"a": 1'), {"a": 1})
        self.assertIn("_raw", llmmod._parse_args("nope"))

    def test_clean_tool_names(self) -> None:
        self.assertEqual(llmmod.clean_name("browser_network_requests<|channel|>commentary"), "browser_network_requests")
        self.assertEqual(llmmod.clean_name("read file"), "read")
        self.assertEqual(llmmod.clean_name(""), "unknown_tool")

    def test_is_api_like(self) -> None:
        self.assertTrue(is_api_like("https://x.com/api/v1/items", "application/json", "XHR"))
        self.assertTrue(is_api_like("https://x.com/graphql", "", "Fetch"))
        self.assertFalse(is_api_like("https://x.com/app.js", "application/javascript", "Script"))
        self.assertFalse(is_api_like("https://www.google-analytics.com/collect", "", "XHR"))
        self.assertFalse(is_api_like("data:image/png;base64,xx", "", "Image"))
        self.assertTrue(is_api_like("wss://x.com/socket", "", "WebSocket"))

    def test_prompts_render(self) -> None:
        for lang in store.LANGUAGES:
            cg = modes.Codegen(lang, "client", Path("/tmp/s"))
            txt = cg.instructions()
            self.assertIn("/tmp/s", txt)
            self.assertNotIn("{scripts_dir}", txt)
            self.assertIn(cg.client_filename().split(".")[0], cg.output_files())
        d = modes.Codegen("python", "docs", Path("/tmp/s"))
        self.assertIn("openapi.json", d.instructions())
        self.assertEqual(d.client_filename(), "openapi.json")
        for name in ("agent_system", "agent_user", "engineer_system", "engineer_user", "collector_system", "collector_user"):
            self.assertTrue(load(name))
        self.assertIn('{"name"', load("collector_system"))

    def test_config_roundtrip(self) -> None:
        c = _cfg()
        c.set("output_language", "go")
        c.set("max_turns", "50")
        c.set("headless", "false")
        c2 = store.Config(c.path)
        self.assertEqual((c2.get("output_language"), c2.get("max_turns"), c2.get("headless")), ("go", 50, False))
        with self.assertRaises(ValueError):
            c.set("output_language", "cobol")

    def test_history(self) -> None:
        h = store.History()
        h.upsert(store.make_record("aaaa1111", "agent", "p1", "m"))
        h.upsert(store.make_record("bbbb2222", "engineer", "p2", "m"))
        self.assertEqual(h.get("latest")["run_id"], "bbbb2222")
        self.assertEqual(h.get("aaaa")["prompt"], "p1")
        h.update("aaaa1111", status="ok", paths={"script_path": "/x"})
        self.assertEqual(h.get("aaaa1111")["paths"]["script_path"], "/x")
        self.assertIn("har_dir", h.get("aaaa1111")["paths"])
        with self.assertRaises(ValueError):
            store.har_dir("../etc")

    def test_provider_detection_and_errors(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}, clear=False):
            self.assertEqual(llmmod.detect_provider(), "anthropic")
        with self.assertRaises(llmmod.LLMError):
            LLM(provider="openai", api_key=None) if not os.environ.get("OPENAI_API_KEY") else (_ for _ in ()).throw(llmmod.LLMError("x"))
        with self.assertRaises(llmmod.LLMError):
            LLM(provider="nope")

    def test_nvidia_provider(self) -> None:
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-x"}, clear=False):
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
                os.environ.pop(k, None)
            self.assertEqual(llmmod.detect_provider(), "nvidia")
            llm = LLM()
            self.assertEqual((llm.provider, llm.model, llm.base_url), ("nvidia", "stepfun-ai/step-5-preview", "https://integrate.api.nvidia.com/v1"))
        self.assertEqual(LLM(api_key="nvapi-abc").provider, "nvidia")  # key prefix picks the provider
        net = FakeNet(
            [
                _http_error(400, '{"detail":"max_tokens must be <= 8192"}'),
                _json({"choices": [{"message": {"content": "ok", "reasoning_content": "think"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            ]
        )
        with net:
            r = LLM(provider="nvidia", api_key="nvapi-k", stream=False).chat([{"role": "user", "content": "hi"}])
        self.assertEqual((r.content, r.reasoning), ("ok", "think"))
        self.assertEqual(net.calls[0]["url"], "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(net.calls[0]["headers"]["Authorization"], "Bearer nvapi-k")
        self.assertEqual(net.calls[1]["body"]["max_tokens"], 4096)  # retried with a smaller output cap

    def test_openai_and_anthropic_wire_format(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u", "_seq": 1},
            {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [{"id": "t0", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t0", "content": "res"},
        ]
        tools = [{"name": "x", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        net = FakeNet(
            [
                _json(
                    {
                        "choices": [{"message": {"content": "", "tool_calls": [{"id": "t1", "function": {"name": "x", "arguments": '{"a": 1}'}}]}, "finish_reason": "tool_calls"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                    }
                ),
                _json(
                    {
                        "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}}],
                        "usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 3},
                        "stop_reason": "tool_use",
                    }
                ),
            ]
        )
        with net:
            r = LLM(provider="openai", api_key="k", model="gpt-4.1", stream=False).chat(msgs, tools)
            self.assertEqual(r.tool_calls[0]["arguments"], {"a": 1})
            self.assertGreater(r.usage.estimated_cost_usd, 0)
            r2 = LLM(provider="anthropic", api_key="k", model="claude-sonnet-4-5", stream=False).chat(msgs, tools)
            self.assertEqual((r2.content, r2.tool_calls[0]["name"], r2.usage.input_tokens), ("hi", "x", 8))
        ob, ab = net.calls[0]["body"], net.calls[1]["body"]
        self.assertNotIn("_seq", ob["messages"][1])  # bookkeeping never reaches the provider
        self.assertNotIn("reasoning", ob["messages"][2])
        self.assertEqual(ab["system"][0]["text"], "sys")
        self.assertEqual(ab["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(ab["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(ob["tools"][0]["type"], "function")

    def test_openai_streaming_reasoning_and_tool_calls(self) -> None:
        events = [
            {"choices": [{"delta": {"role": "assistant", "reasoning_content": "Let me "}}]},
            {"choices": [{"delta": {"reasoning_content": "think."}}]},
            {"choices": [{"delta": {"content": "Writ"}}]},
            {"choices": [{"delta": {"content": "ing now"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "type": "function", "function": {"name": "write_file", "arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py", "content": "x=1"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 30}},
        ]
        net = FakeNet([_sse_resp(events)])
        seen: list[tuple[str, Any]] = []
        with net:
            r = LLM(provider="nvidia", api_key="nvapi-k").chat(
                [{"role": "user", "content": "go"}], [{"name": "write_file", "parameters": {"type": "object"}}], on_delta=lambda k, p: seen.append((k, p))
            )
        body = net.calls[0]["body"]
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(r.reasoning, "Let me think.")
        self.assertEqual(r.content, "Writing now")
        self.assertEqual(r.tool_calls, [{"id": "call_9", "name": "write_file", "arguments": {"path": "a.py", "content": "x=1"}}])
        self.assertEqual((r.usage.input_tokens, r.usage.output_tokens, r.finish_reason), (120, 30, "tool_calls"))
        kinds = [k for k, _ in seen]
        self.assertEqual(kinds[:4], ["thinking", "thinking", "text", "text"])
        self.assertIn("tool", kinds)

    def test_think_tags_split(self) -> None:
        got: list[tuple[str, str]] = []
        sp = llmmod.ThinkSplitter(lambda k, s: got.append((k, s)))
        for part in ["<thi", "nk>reason", "ing</th", "ink>\n\nAnswer: use <think> tags"]:
            sp.feed(part)
        sp.flush()
        self.assertEqual(sp.think, "reasoning")
        self.assertEqual(sp.text, "Answer: use <think> tags")  # a tag inside the answer stays text
        self.assertEqual(llmmod.split_think("<think>abc</think>\nHello", ""), ("Hello", "abc"))
        self.assertEqual(llmmod.split_think("abc</think>Hello", ""), ("Hello", "abc"))

    def test_anthropic_streaming_thinking_signature(self) -> None:
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 4}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan it"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Calling."}},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "tu1", "name": "shell", "input": {}}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"comm'}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": 'and": "ls"}'}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
            {"type": "message_stop"},
        ]
        net = FakeNet([_sse_resp(events)])
        llm = LLM(provider="anthropic", api_key="k", model="claude-sonnet-4-5", effort="low")
        with net:
            r = llm.chat([{"role": "user", "content": "x"}], [{"name": "shell", "parameters": {"type": "object"}}], on_delta=lambda k, p: None)
        self.assertEqual((r.reasoning, r.content, r.finish_reason), ("plan it", "Calling.", "tool_use"))
        self.assertEqual(r.thinking_blocks, [{"type": "thinking", "thinking": "plan it", "signature": "SIG"}])
        self.assertEqual(r.tool_calls[0]["arguments"], {"command": "ls"})
        self.assertEqual((r.usage.input_tokens, r.usage.output_tokens), (14, 7))
        self.assertEqual(net.calls[0]["body"]["thinking"]["type"], "enabled")
        # the signed thinking block travels back first on the next turn
        msg = {
            "role": "assistant",
            "content": "Calling.",
            "thinking_blocks": r.thinking_blocks,
            "tool_calls": [{"id": "tu1", "type": "function", "function": {"name": "shell", "arguments": '{"command": "ls"}'}}],
        }
        body = llm._anthropic_body([{"role": "user", "content": "x"}, msg, {"role": "tool", "tool_call_id": "tu1", "content": "a.py"}], None, True)
        self.assertEqual([b["type"] for b in body["messages"][1]["content"]], ["thinking", "text", "tool_use"])

    def test_retries_network_and_5xx_then_success(self) -> None:
        import urllib.error

        plan = [urllib.error.URLError("dns failure")] * 3 + [ConnectionResetError("reset")] * 2 + [_http_error(503, "busy"), _http_error(529, "overloaded")]
        plan.append(_json({"choices": [{"message": {"content": "back online"}}]}))
        net = FakeNet(plan)
        retries: list[tuple[int, int, str]] = []
        with net, mock.patch("time.sleep") as slept:
            r = LLM(provider="openai", api_key="k", stream=False).chat([{"role": "user", "content": "x"}], on_retry=lambda a, n, w, why: retries.append((a, n, why)))
        self.assertEqual(r.content, "back online")
        self.assertEqual([a for a, _, _ in retries], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(retries[0][1], 40)  # 40 retries allowed
        self.assertIn("network error", retries[0][2])
        self.assertIn("HTTP 503", retries[5][2])
        delays = [c.args[0] for c in slept.call_args_list]
        self.assertTrue(all(0 < d <= 69 for d in delays))
        self.assertGreater(delays[3], delays[0])  # exponential backoff

    def test_retries_give_up_after_40(self) -> None:
        import urllib.error

        net = FakeNet([urllib.error.URLError("offline")] * 60)
        with net, mock.patch("time.sleep"):
            with self.assertRaises(llmmod.LLMError) as cm:
                LLM(provider="openai", api_key="k", stream=False).chat([{"role": "user", "content": "x"}], on_retry=lambda *a: None)
        self.assertEqual(cm.exception.kind, "network")
        self.assertEqual(len(net.calls), 41)  # the first attempt + 40 retries
        self.assertIn("gave up after 40 retries", str(cm.exception))
        self.assertEqual(LLM(provider="openai", api_key="k", retries=5).retries, 5)

    def test_auth_errors_fail_fast_and_retry_after(self) -> None:
        net = FakeNet([_http_error(401, '{"error":"bad key"}')])
        with net, mock.patch("time.sleep"):
            with self.assertRaises(llmmod.LLMError) as cm:
                LLM(provider="openai", api_key="k", stream=False).chat([{"role": "user", "content": "x"}])
        self.assertEqual((cm.exception.kind, len(net.calls)), ("auth", 1))
        net = FakeNet([_http_error(429, "slow down", {"retry-after": "7"}), _json({"choices": [{"message": {"content": "ok"}}]})])
        with net, mock.patch("time.sleep") as slept:
            LLM(provider="openai", api_key="k", stream=False).chat([{"role": "user", "content": "x"}], on_retry=lambda *a: None)
        self.assertEqual(slept.call_args_list[0].args[0], 7.5)  # Retry-After honoured

    def test_stream_dropped_midway_is_retried(self) -> None:
        good = _sse_resp([{"choices": [{"delta": {"content": "complete answer"}, "finish_reason": "stop"}]}])
        broken = _sse_resp([{"choices": [{"delta": {"content": "half an ans"}}]}], die=True)
        cut = _Resp(b'data: {"choices": [{"delta": {"reasoning_content": "partial thou"}}]}\n\n', "text/event-stream")  # closed cleanly, but no [DONE]
        net = FakeNet([broken, cut, good])
        seen: list[str] = []
        with net, mock.patch("time.sleep"):
            r = LLM(provider="openai", api_key="k").chat([{"role": "user", "content": "x"}], on_delta=lambda k, p: seen.append(k), on_retry=lambda *a: None)
        self.assertEqual(r.content, "complete answer")
        self.assertEqual(seen.count("reset"), 2)
        self.assertEqual(len(net.calls), 3)

    def test_mid_stream_overload_error_is_retried(self) -> None:
        bad = _sse_resp([{"error": {"message": "Provider overloaded, try again", "code": 503}}])
        good = _sse_resp([{"choices": [{"delta": {"content": "fine"}, "finish_reason": "stop"}]}])
        net = FakeNet([bad, good])
        with net, mock.patch("time.sleep"):
            r = LLM(provider="openrouter", api_key="k").chat([{"role": "user", "content": "x"}], on_retry=lambda *a: None)
        self.assertEqual(r.content, "fine")

    def test_unsupported_optional_params_are_dropped(self) -> None:
        net = FakeNet(
            [
                _http_error(400, '{"error":{"message":"Unrecognized request argument supplied: stream_options"}}'),
                _http_error(400, '{"error":"chat_template_kwargs is not supported for this model"}'),
                _sse_resp([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}]),
            ]
        )
        llm = LLM(provider="nvidia", api_key="nvapi-k", effort="high")
        with net:
            self.assertEqual(llm.chat([{"role": "user", "content": "x"}]).content, "ok")
        self.assertIn("stream_options", net.calls[0]["body"])
        self.assertNotIn("stream_options", net.calls[1]["body"])
        self.assertIn("chat_template_kwargs", net.calls[1]["body"])
        self.assertNotIn("chat_template_kwargs", net.calls[2]["body"])
        self.assertEqual(llm._dropped, {"stream_options", "chat_template_kwargs"})

    def test_context_errors_and_windows(self) -> None:
        net = FakeNet([_http_error(400, '{"error":{"message":"This model\'s maximum context length is 131072 tokens. However, you requested 140000 tokens."}}')])
        with net:
            with self.assertRaises(llmmod.LLMError) as cm:
                LLM(provider="openai", api_key="k", stream=False).chat([{"role": "user", "content": "x"}])
        self.assertEqual((cm.exception.kind, cm.exception.window), ("context_length", 131072))
        self.assertEqual(llmmod.context_window("claude-sonnet-4-5"), 200_000)
        self.assertEqual(llmmod.context_window("moonshotai/kimi-k2.6"), 256_000)
        self.assertEqual(llmmod.context_window("unknown/model"), llmmod.DEFAULT_WINDOW)
        self.assertEqual(LLM(provider="openai", api_key="k", context=50_000).context_window, 50_000)

    def test_compaction_keeps_everything_recallable(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        big = lambda tag: f"BEGIN-{tag} " + ("payload " * 1200) + f" END-{tag}"  # noqa: E731
        tool = Tool("probe", "returns a big blob", {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}, lambda tag: big(tag))
        script = [[("probe", {"tag": f"T{i}"})] for i in range(8)] + ["all done"]
        llm = FakeLLM(script, window=9000)
        events: list[tuple[str, dict]] = []
        ag = Agent(llm, [tool], Context(workdir=tmp), "SYSTEM", tmp / "messages.jsonl", lambda k, d: events.append((k, d)))
        res = ag.send("ORIGINAL REQUEST: map the todo api")
        self.assertIsNone(res["error"])
        self.assertGreaterEqual(res["compactions"], 1)
        self.assertGreaterEqual(llm.summaries, 1)
        kinds = [k for k, _ in events]
        self.assertIn("compaction", kinds)
        comp = next(d for k, d in events if k == "compaction")
        self.assertLess(comp["after"], comp["before"])
        # the live context: system, then the continuation carrying the original request + summary
        self.assertEqual(ag.messages[0]["content"], "SYSTEM")
        self.assertIn("ORIGINAL REQUEST: map the todo api", ag.messages[1]["content"])
        self.assertIn("SUMMARY-", ag.messages[1]["content"])
        self.assertIn("recall", ag.messages[1]["content"])
        for m in ag.messages:  # no orphaned tool results
            if m["role"] == "tool":
                ids = {tc["id"] for x in ag.messages if x.get("tool_calls") for tc in x["tool_calls"]}
                self.assertIn(m["tool_call_id"], ids)
        # everything is still in the transcript, verbatim, and recall returns it
        recs = [json.loads(x) for x in (tmp / "messages.jsonl").read_text().splitlines()]
        t0 = next(r for r in recs if r.get("role") == "tool" and "BEGIN-T0" in r["content"])
        self.assertIn("END-T0", t0["content"])
        full = ag.tools["recall"].handler(id=t0["seq"])
        self.assertIn("BEGIN-T0", full)
        self.assertIn("END-T0", full)
        self.assertIn("END-T0", ag.tools["recall"].handler(id=t0["seq"], offset=9000))
        self.assertIn(f"#{t0['seq']}", ag.tools["recall"].handler(query="BEGIN-T0"))
        self.assertTrue(any(r.get("type") == "compaction" for r in recs))
        # resuming rebuilds the compacted context, not the whole history
        msgs = Agent.load_messages(tmp / "messages.jsonl")
        self.assertIn("SUMMARY-", msgs[1]["content"])
        self.assertLess(len(msgs), len([r for r in recs if r.get("role")]))
        self.assertTrue(all("_seq" in m for m in msgs[1:] if not m.get("_summary")))  # real messages keep their transcript ids
        self.assertTrue(msgs[1].get("_summary"))

    def test_context_overflow_error_forces_compaction(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        tool = Tool("probe", "x", {"type": "object", "properties": {}}, lambda: "data " * 400)
        err = llmmod.LLMError("context window exceeded", "context_length")
        err.window = 4000
        llm = FakeLLM([[("probe", {})], [("probe", {})], [("probe", {})], err, "done"], window=200_000)
        ag = Agent(llm, [tool], Context(workdir=tmp), "S", tmp / "m.jsonl")
        res = ag.send("REQ")
        self.assertIsNone(res["error"])
        self.assertEqual(llm.context_window, 4000)  # learned from the error
        self.assertGreaterEqual(res["compactions"], 1)

    def test_mechanical_compaction_when_summary_fails(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        tool = Tool("probe", "x", {"type": "object", "properties": {}}, lambda: "row " * 900)

        class NoSummary(FakeLLM):
            def chat(self, messages, tools=None, **kw):  # noqa: ANN001
                if tools is None:
                    raise llmmod.LLMError("summary model down", "network")
                return super().chat(messages, tools, **kw)

        llm = NoSummary([[("probe", {})] for _ in range(6)] + ["done"], window=6000)
        ag = Agent(llm, [tool], Context(workdir=tmp), "S", tmp / "m.jsonl")
        res = ag.send("REQ-MECH")
        self.assertIsNone(res["error"])
        self.assertGreaterEqual(res["compactions"], 1)
        self.assertIn("REQ-MECH", ag.messages[1]["content"])
        self.assertIn("recall", json.dumps(ag.messages))

    def test_markdown_and_wrapping(self) -> None:
        old = termmod.COLOR
        try:
            termmod.COLOR = 0
            out = termmod.render_markdown("# Title\n\nSome **bold** and `code` [docs](https://x.dev).\n\n- one\n- two\n\n```python\ndef f():\n    return 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |", 60)
            text = "\n".join(out)
            for want in ("Title", "Some bold and code docs (https://x.dev).", "• one", "def f():", "╭─ python", "│ a │ b │", "│ 1 │ 2 │"):
                self.assertIn(want, text)
            termmod.COLOR = 24
            styled = termmod.inline("a **bold** word and `x` " * 12)
            lines = termmod.wrap(styled, 40, "   ", "   ")
            self.assertGreater(len(lines), 3)
            self.assertTrue(all(termmod.vis_len(x) <= 40 for x in lines))
            self.assertEqual(termmod.vis_len(termmod.truncate(termmod.sty("x" * 100, "#ff0000"), 20)), 20)
            self.assertEqual(termmod.vis_len("日本"), 4)
            hl = termmod.highlight('def fetch(url: str) -> dict:  # get "x"', "python")
            self.assertIn("\x1b[", hl)
            self.assertEqual(termmod.strip_ansi(hl), 'def fetch(url: str) -> dict:  # get "x"')
            narrow = termmod.render_markdown("| endpoint | method | auth | notes |\n|---|---|---|---|\n| /api/todos/list/all | GET | cookie sid | paginated with cursor |", 40)
            self.assertTrue(any("endpoint:" in termmod.strip_ansi(x) for x in narrow))  # phone-width table fallback
        finally:
            termmod.COLOR = old

    def test_reporter_renders_a_session(self) -> None:
        import contextlib
        import io

        buf = io.StringIO()
        stream = io.StringIO()
        tmp = Path(tempfile.mkdtemp())
        client = tmp / "api_client.py"
        client.write_text("import requests\n\n\ndef list_todos():\n    return requests.get('https://x/api/todos').json()\n")
        with contextlib.redirect_stderr(buf):
            rep = Reporter(json_stream=stream)
            rep.run_header("agent", "abc123", "nvidia/stepfun-ai/step-5-preview", "python")
            rep("thinking_start", {"turn": 1})
            rep("thinking_delta", {"text": "I should open the page first."})
            rep("text_delta", {"text": "Opening the **site** now.\n"})
            rep("thinking", {"text": "I should open the page first."})
            rep("text", {"text": "Opening the **site** now."})
            rep("tool_start", {"name": "browser_navigate", "input": {"url": "https://example.com/app"}})
            rep("tool_end", {"name": "browser_navigate", "is_error": False, "duration_ms": 5, "output": "200 GET https://example.com/api/todos\nline2\nline3\nline4"})
            rep("retry", {"attempt": 2, "max": 39, "wait": 3.0, "reason": "HTTP 503 from provider"})
            rep("compaction", {"before": 150000, "after": 30000, "window": 200000, "method": "summary", "count": 1})
            rep("tool_start", {"name": "write_file", "input": {"path": "api_client.py", "content": client.read_text()}})
            rep("usage", {"input_tokens": 1200, "output_tokens": 300, "requests": 3, "estimated_cost_usd": 0.01})
            rep.card(client, "nvidia/stepfun-ai/step-5-preview")
            rep.footer()
        out = termmod.strip_ansi(buf.getvalue())
        for want in (
            "run abc123",
            "thought for",
            "I should open the page first.",
            "Opening the site now.",
            "01  navigate",
            "└ 200 GET",
            "+1 lines",
            "retry 2/39",
            "150.0k →",
            "150.0k → 30.0k",
            "02  write",
            "def list_todos",
            "api_client.py ready",
        ):
            self.assertIn(want, out)
        self.assertEqual(out.count("Opening the site now."), 1)  # streamed text is not printed twice
        ev = [json.loads(x)["event"] for x in stream.getvalue().splitlines()]
        for want in ("thinking", "text", "tool_start", "tool_end", "retry", "compaction"):
            self.assertIn(want, ev)

    @unittest.skipUnless(SAMPLE_HAR.exists(), "sample HAR not present")
    def test_har_tools_on_real_capture(self) -> None:
        ov = harlib.overview(SAMPLE_HAR)
        self.assertIn("endpoint shapes", ov)
        self.assertIn("json.excalidraw.com", ov)
        hit = harlib.search(SAMPLE_HAR, "api/v2", "url")
        self.assertIn("#", hit)
        idx = int(hit.split()[0][1:])
        self.assertIn("RESPONSE HEADERS", harlib.entry(SAMPLE_HAR, idx))
        out = Path(tempfile.mkdtemp()) / "api.har"
        n = harlib.filter_api(SAMPLE_HAR, out)
        self.assertEqual(len(harlib.load(out)), n)

    def test_run_script_auto_install_detection(self) -> None:
        d = store.scripts_dir("runtest01")
        (d / "api_client.py").write_text("import sys\nprint('hello', sys.argv[1:])\n")
        store.History().upsert(store.make_record("runtest01", "agent", "p", "m"))
        res = modes.run_script("runtest01", _cfg(), Reporter(quiet=True), args=["a"], capture=True, interactive=False)
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("hello ['a']", res["stdout"])
        (d / "api_client.py").write_text("import surely_not_a_real_module_xyz\n")
        res = modes.run_script("runtest01", _cfg(), Reporter(quiet=True), capture=True, interactive=False)
        self.assertEqual(res["missing_dependency"], "surely_not_a_real_module_xyz")
        self.assertEqual(res["status"], "error")

    def test_cli_basics(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        run = lambda *a: subprocess.run([sys.executable, "-m", "revapi", *a], capture_output=True, text=True, env=env, timeout=60)  # noqa: E731
        self.assertEqual(run("--json-schema-version").stdout.strip(), "1")
        self.assertEqual(json.loads(run("list", "--json").stdout).__class__, list)
        r = run("agent", "--json")
        self.assertEqual(json.loads(r.stdout)["error_kind"], "usage")
        r = run("show", "doesnotexist", "--json")
        self.assertEqual(json.loads(r.stdout)["error_kind"], "not_found")
        r = run("agent", "-p", "x", "-u", "https://example.com", "--dry-run", "--json")
        self.assertEqual(json.loads(r.stdout)["mode"], "dry-run")
        # `run` flags must be parsed by revapi, script args go after --
        d = store.scripts_dir("clirun01")
        (d / "api_client.py").write_text("import sys\nprint('ARGS', sys.argv[1:])\n")
        store.History().upsert(store.make_record("clirun01", "agent", "p", "m"))
        self.assertEqual(run("run", "clirun01", "--ls").stdout.strip(), "api_client.py")
        r = run("run", "clirun01", "--json", "--", "--limit", "5")
        out = json.loads(r.stdout)
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("ARGS ['--limit', '5']", out["stdout"])


# ============================================================================ browser + pipelines


@unittest.skipUnless(HAS_CHROMIUM, "no chromium")
class BrowserAndPipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.srv = serve()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.srv.shutdown()

    def test_browser_capture(self) -> None:
        b = Browser()
        try:
            b.navigate(self.base + "/old")
            snap = b.snapshot()
            self.assertIn("[e1]", snap)
            self.assertIn("New todo", snap)
            b.type("e1", "via cdp")
            b.click("e2")
            self.assertIn("via cdp", b.find_text("via cdp"))
            self.assertTrue(any(c["name"] == "sid" for c in b.cookies()))
            path = b.recorder.save(Path(tempfile.mkdtemp()) / "t.har")
        finally:
            b.close()
        entries = harlib.load(path)
        urls = [(e["request"]["method"], e["request"]["url"].replace(self.base, ""), e["response"]["status"]) for e in entries]
        self.assertIn(("GET", "/old", 302), urls)
        self.assertIn(("POST", "/api/todos", 201), urls)
        post = next(e for e in entries if e["request"]["method"] == "POST")
        self.assertEqual(json.loads(post["request"]["postData"]["text"]), {"title": "via cdp"})
        self.assertIn("via cdp", post["response"]["content"]["text"])
        self.assertTrue(any(c["name"] == "sid" for c in post["request"]["cookies"]))
        self.assertTrue(any(h["name"].lower() == "x-client" for h in post["request"]["headers"]))
        self.assertEqual(len([u for u in urls if u[1] == "/old"]), 1, "no duplicate entries")

    def test_agent_pipeline(self) -> None:
        """Full agent mode with a scripted model: browse, add todo, close, write client, run it, finish."""
        base = self.base
        client = (
            "import json, urllib.request\n"
            f"BASE = {base!r}\n"
            "H = {'Cookie': 'sid=abc123', 'X-Client': 'fixture', 'Content-Type': 'application/json'}\n"
            "def list_todos():\n    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + '/api/todos', headers=H)))\n"
            "def add_todo(t):\n    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + '/api/todos', data=json.dumps({'title': t}).encode(), headers=H)))\n"
            "if __name__ == '__main__':\n    print(add_todo('from client'))\n    print(list_todos())\n"
        )

        def check_requests(messages):  # noqa: ANN001
            last = messages[-1]["content"]
            assert "POST" in last and "/api/todos" in last, last
            return [("browser_network_request", {"index": next(int(l.split()[0][1:]) for l in last.splitlines() if l.startswith("#") and "POST" in l)})]

        script = [
            [("browser_navigate", {"url": base})],
            [("browser_type", {"ref": "e1", "text": "from agent"}), ("browser_click", {"ref": "e2"})],
            [("browser_network_requests", {})],
            check_requests,
            [("browser_close", {})],
            [("har_overview", {}), ("har_search", {"query": "sid=", "where": "request_headers"})],
            [("write_file", {"path": "api_client.py", "content": client}), ("write_file", {"path": "README.md", "content": "# Todo client\n"})],
            [("shell", {"command": "python3 api_client.py"})],
            [("finish", {"summary": "Todo API: GET/POST /api/todos with sid cookie. Client works.", "script_path": "api_client.py"})],
        ]
        fake = FakeLLM(script)
        with mock.patch.object(modes, "make_llm", lambda *a, **k: fake):
            res = modes.run_agent("todo api", base, _cfg(), Reporter(quiet=True))
        self.assertIsNone(res["error"], res)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["script_path"].endswith("api_client.py"))
        self.assertTrue(Path(res["har_path"]).exists())
        tool_outputs = [m["content"] for m in fake.seen[-1] if m["role"] == "tool"]
        self.assertTrue(any("[exit 0]" in o and "from client" in o for o in tool_outputs), tool_outputs[-3:])
        self.assertTrue(any('"from agent"' in o or "from agent" in o for o in tool_outputs))
        rec = store.History().get(res["run_id"])
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["usage"]["requests"], len(script))
        msgs = Path(rec["paths"]["messages_path"]).read_text().splitlines()
        self.assertGreater(len(msgs), 10)

        # follow-up via resume (re-opens browser; HAR must keep earlier traffic)
        n_before = len(harlib.load(res["har_path"]))
        fake2 = FakeLLM([[("browser_navigate", {"url": base + "/about"})], [("browser_close", {})], [("finish", {"summary": "added about"})]])
        with mock.patch.object(modes, "make_llm", lambda *a, **k: fake2):
            keep = modes.resume(res["run_id"], _cfg(), Reporter(quiet=True))
            self.assertGreater(len(keep["agent"].messages), 10)
            r2 = modes.follow_up(keep, "also capture the about page", Reporter(quiet=True))
        self.assertIsNone(r2["error"])
        self.assertGreater(len(harlib.load(res["har_path"])), n_before)

        # engineer mode on the same run: iterative edit sees existing client
        def check_prompt(messages):  # noqa: ANN001
            user = next(m["content"] for m in messages if m["role"] == "user")
            assert "existing_client" in user and "add a delete function" in user, user[-500:]
            return [("har_entry", {"index": 0, "part": "request"})]

        fake3 = FakeLLM(
            [
                check_prompt,
                [("edit_file", {"path": "api_client.py", "old_text": "def list_todos", "new_text": "def delete_todo(i):\n    raise NotImplementedError\n\ndef list_todos"})],
                [("finish", {"summary": "added delete"})],
            ]
        )
        with mock.patch.object(modes, "make_llm", lambda *a, **k: fake3):
            r3 = modes.run_engineer(res["run_id"], _cfg(), Reporter(quiet=True), prompt="add a delete function")
        self.assertIsNone(r3["error"], r3)
        self.assertIn("delete_todo", Path(r3["script_path"]).read_text())

        # run the generated client through `revapi run`
        rr = modes.run_script(res["run_id"], _cfg(), Reporter(quiet=True), capture=True, interactive=False)
        self.assertEqual(rr["exit_code"], 0, rr)
        self.assertIn("from client", rr["stdout"])

    def test_engineer_external_har_docs_mode(self) -> None:
        b = Browser()
        try:
            b.navigate(self.base)
            p = b.recorder.save(Path(tempfile.mkdtemp()) / "ext.har")
        finally:
            b.close()
        cfg = _cfg()
        cfg.data["output_mode"] = "docs"
        spec = {"openapi": "3.0.0", "info": {"title": "Todo", "version": "1"}, "paths": {"/api/todos": {"get": {}, "post": {}}}}

        def check_system(messages):  # noqa: ANN001
            assert "OpenAPI" in messages[0]["content"]
            return [("write_file", {"path": "openapi.json", "content": json.dumps(spec)}), ("write_file", {"path": "README.md", "content": "# docs"})]

        fake = FakeLLM([[("har_overview", {})], check_system, [("finish", {"summary": "spec written"})]])
        with mock.patch.object(modes, "make_llm", lambda *a, **k: fake):
            res = modes.run_engineer(None, cfg, Reporter(quiet=True), prompt="document it", har_file=p)
        self.assertIsNone(res["error"], res)
        self.assertTrue(res["script_path"].endswith("openapi.json"))
        self.assertIn("/api/todos", fake.seen[1][-1]["content"])

    def test_collector_pipeline(self) -> None:
        fake = FakeLLM(
            [
                [("web_fetch", {"url": self.base + "/about"})],
                [("save_items", {"items": [{"name": "A", "source_url": "u1"}, {"name": "B", "extra": [1, 2], "source_url": "u2"}]})],
                [("finish", {"summary": "2 items"})],
            ]
        )
        with mock.patch.object(modes, "make_llm", lambda *a, **k: fake):
            res = modes.run_collector("collect things", _cfg(), Reporter(quiet=True))
        self.assertIsNone(res["error"], res)
        self.assertEqual(res["items"], 2)
        out = Path(res["output_dir"])
        self.assertEqual(len(json.loads((out / "items.json").read_text())), 2)
        self.assertIn("name,source_url,extra", (out / "data.csv").read_text())
        self.assertIn("u2", (out / "README.md").read_text())
        self.assertIn("About page", fake.seen[1][-1]["content"])

    def test_manual_shell_capture(self) -> None:
        cmds = iter([f"go {self.base}", "type e1 manual one ⏎", "click e2", "reqs", "req 0", "cookies", "bogus", "done"])
        with mock.patch("builtins.input", lambda *a: next(cmds)), mock.patch("builtins.print"):
            res = modes.run_manual(_cfg(), Reporter(quiet=True), None, None, headed=False, engineer=False, interactive=False)
        entries = harlib.load(res["har_path"])
        self.assertTrue(any(e["request"]["method"] == "POST" and "manual one" in e["request"]["postData"]["text"] for e in entries))

    def test_cli_json_stream_with_fake(self) -> None:
        # json-stream: NDJSON events + final result line, via in-process main()
        import contextlib
        import io

        from revapi import cli

        fake = FakeLLM([[("browser_navigate", {"url": self.base})], [("browser_close", {})], [("write_file", {"path": "api_client.py", "content": "print(1)"})], [("finish", {"summary": "ok"})]])
        buf = io.StringIO()
        with mock.patch.object(modes, "make_llm", lambda *a, **k: fake), contextlib.redirect_stdout(buf):
            code = cli.main(["agent", "-p", "x", "-u", self.base, "--json-stream"])
        lines = [json.loads(ln) for ln in buf.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(lines[0]["event"], "run_started")
        self.assertIn("tool_start", {ln["event"] for ln in lines})
        self.assertEqual(lines[-1]["event"], "result")
        self.assertEqual(lines[-1]["status"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
