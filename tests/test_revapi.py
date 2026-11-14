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
from revapi.ui import Reporter  # noqa: E402

HAS_CHROMIUM = bool(find_chromium())
SAMPLE_HAR = Path("/home/user/.reverse-api/runs/har/59c7dc3b7609/recording.har")


class FakeLLM(LLM):
    """Plays back a script: each item is a list of (tool, args) or a final text."""

    def __init__(self, script):  # noqa: ANN001
        self.provider, self.model, self.api_key, self.effort = "fake", "scripted", "x", None
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):  # noqa: ANN001
        self.seen.append(json.loads(json.dumps(messages)))
        step = self.script.pop(0) if self.script else "done"
        r = Reply(usage=Usage(input_tokens=100, output_tokens=10, requests=1))
        if callable(step):
            step = step(messages)
        if isinstance(step, str):
            r.content = step
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
        calls = []

        def fake_post(url, body, headers, timeout):  # noqa: ANN001
            calls.append((url, body, headers))
            if len(calls) == 1:
                raise llmmod.LLMError('HTTP 400: {"detail":"max_tokens must be <= 8192"}')
            return {"choices": [{"message": {"content": "ok", "reasoning_content": "think"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        with mock.patch.object(llmmod, "_post", fake_post):
            r = LLM(provider="nvidia", api_key="nvapi-k").chat([{"role": "user", "content": "hi"}])
        self.assertEqual((r.content, r.reasoning), ("ok", "think"))
        self.assertEqual(calls[0][0], "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(calls[0][2]["Authorization"], "Bearer nvapi-k")
        self.assertEqual(calls[1][1]["max_tokens"], 4096)  # retried with a smaller output cap

    def test_openai_and_anthropic_wire_format(self) -> None:
        calls = []

        def fake_post(url, body, headers, timeout):  # noqa: ANN001
            calls.append((url, body, headers))
            if "anthropic" in url:
                return {
                    "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}}],
                    "usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 3},
                    "stop_reason": "tool_use",
                }
            return {
                "choices": [{"message": {"content": "", "tool_calls": [{"id": "t1", "function": {"name": "x", "arguments": '{"a": 1}'}}]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t0", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t0", "content": "res"},
        ]
        tools = [{"name": "x", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        with mock.patch.object(llmmod, "_post", fake_post):
            r = LLM(provider="openai", api_key="k", model="gpt-4.1").chat(msgs, tools)
            self.assertEqual(r.tool_calls[0]["arguments"], {"a": 1})
            self.assertGreater(r.usage.estimated_cost_usd, 0)
            r2 = LLM(provider="anthropic", api_key="k", model="claude-sonnet-4-5").chat(msgs, tools)
            self.assertEqual((r2.content, r2.tool_calls[0]["name"], r2.usage.input_tokens), ("hi", "x", 8))
        ab = calls[1][1]
        self.assertEqual(ab["system"][0]["text"], "sys")
        self.assertEqual(ab["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(ab["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(calls[0][1]["tools"][0]["type"], "function")

    def test_throttle_retry(self) -> None:
        import io
        import urllib.error

        n = {"i": 0}

        def fake_urlopen(req, timeout=0):  # noqa: ANN001
            n["i"] += 1
            if n["i"] == 1:
                raise urllib.error.HTTPError(req.full_url, 402, "x", {}, io.BytesIO(b'{"message":"Retry after 0s"}'))
            return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')

        with mock.patch("urllib.request.urlopen", fake_urlopen), mock.patch("time.sleep"):
            self.assertEqual(llmmod._post("http://x", {}, {}, 1)["choices"][0]["message"]["content"], "ok")

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
