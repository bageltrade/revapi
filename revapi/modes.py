"""The five modes: agent (auto), manual, engineer, collector, run."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, store
from . import har as harlib
from .agent import Agent
from .browser import Browser, find_chromium, is_api_like
from .llm import LLM, LLMError  # noqa: F401  (LLM re-exported for cli)
from .prompts import load
from .tools import Context, browser_tools, collector_tools, control_tools, file_tools, har_tools, web_tools
from .ui import Reporter, bold, cyan, dim, green, red, suspended, yellow

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------- codegen prompt bits


class Codegen:
    def __init__(self, language: str, output_mode: str, scripts: Path) -> None:
        self.language = language if language in store.LANGUAGES else "python"
        self.mode = output_mode
        self.scripts = scripts
        self.name = store.LANGUAGE_NAMES[self.language]

    @property
    def docs(self) -> bool:
        return self.mode == "docs"

    def client_filename(self) -> str:
        return "openapi.json" if self.docs else f"api_client{store.LANGUAGES[self.language]}"

    def run_command(self) -> str:
        d = self.scripts.resolve()
        q = lambda p: f"'{p}'"  # noqa: E731
        return {
            "java": f"mvn -q -f {q(d / 'pom.xml')} compile exec:exec",
            "csharp": f"dotnet run --project {q(d / 'ApiClient.csproj')}",
            "php": f"php {q(d / self.client_filename())}",
            "ruby": f"ruby {q(d / self.client_filename())}",
            "c": f"cc {q(d / 'api_client.c')} {q(d / 'cJSON.c')} -lcurl -o {q(d / 'api_client')} && {q(d / 'api_client')}",
            "powershell": f"pwsh -NoProfile -File {q(d / 'Example.ps1')}",
            "javascript": "node api_client.js",
            "typescript": "npx tsx api_client.ts",
            "go": "go run api_client.go",
        }.get(self.language, "python api_client.py")

    def instructions(self) -> str:
        if self.docs:
            return load("partials/_docs_instructions", scripts_dir=self.scripts)
        return load(f"partials/_language_{self.language}", scripts_dir=self.scripts, client_filename=self.client_filename(), run_command=self.run_command())

    def output_files(self) -> str:
        if self.docs:
            return f"1. `{self.scripts}/openapi.json` - OpenAPI 3.0 spec\n2. `{self.scripts}/README.md` - API documentation"
        base = f"1. `{self.scripts}/{self.client_filename()}` - Production {self.name} API client\n2. `{self.scripts}/README.md` - Documentation with usage examples"
        extra = {
            "javascript": "package.json - Only if external dependencies are needed",
            "typescript": "package.json - Dependencies and run scripts",
            "go": "go.mod and go.sum - Only if external dependencies are needed",
            "java": "pom.xml - Maven project file (Gson dependency, exec-maven-plugin)",
            "csharp": "ApiClient.csproj - .NET project file",
            "c": "cJSON.c and cJSON.h - Vendored JSON library",
            "powershell": "Example.ps1 - Imports the module and demonstrates usage",
        }.get(self.language)
        return base + (f"\n3. `{self.scripts}/{extra.split(' - ')[0]}` - {extra.split(' - ', 1)[1]}" if extra else "")


def ask_note(interactive: bool) -> str:
    if interactive:
        return "You can call `ask_user` to clarify requirements or choose between approaches when it genuinely matters; otherwise decide yourself."
    return "This is a non-interactive run: do not ask questions — make reasonable assumptions and document them in your summary."


# ----------------------------------------------------------------------------- helpers


HERMES_AGENT_NOTE = (
    "the hermes engine engineers saved captures only, so agent mode stops before opening a browser. "
    "Capture with the built-in engine (revapi config set engine revapi), then switch back "
    "(revapi config set engine hermes) and run: revapi engineer <run_id>"
)


def hermes_engine(cfg: store.Config) -> bool:
    return (cfg.get("engine") or "revapi") == "hermes"


def make_llm(cfg: store.Config, model: str | None = None, collector: bool = False) -> LLM:
    m = model or (cfg.get("collector_model") if collector else None) or cfg.get("model")
    return LLM(
        provider=cfg.get("provider", "auto"),
        model=m,
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        effort=cfg.get("effort"),
        stream=False if os.environ.get("REVAPI_NO_STREAM") else bool(cfg.get("stream", True)),
        retries=int(os.environ.get("REVAPI_RETRIES") or cfg.get("retries", 40)),
        context=int(os.environ.get("REVAPI_CONTEXT_WINDOW") or 0) or cfg.get("context_window"),
        max_tokens=cfg.get("max_tokens", 16000),
    )


def _setup_rep(rep: Reporter, cfg: store.Config, mode: str) -> None:
    rep.mode = mode
    if getattr(rep, "show_thinking", None) is not None and (cfg.get("show_thinking") is False or os.environ.get("REVAPI_HIDE_THINKING")):
        rep.show_thinking = False


def terminal_ask(question: str, options: list[str] | None) -> str:
    with suspended():
        print(f"\n {yellow('?')} {bold(question)}", file=sys.stderr)
        for i, o in enumerate(options or [], 1):
            print(f"     {dim(str(i) + '.')} {o}", file=sys.stderr)
        try:
            ans = input(f" {yellow('›')} ").strip()
        except EOFError:
            return ""
    if options and ans.isdigit() and 1 <= int(ans) <= len(options):
        return options[int(ans) - 1]
    return ans


def _main_script(scripts: Path, cg: Codegen, hint: str = "") -> str | None:
    if hint:
        p = Path(hint) if Path(hint).is_absolute() else scripts / hint
        if p.exists():
            return str(p)
    p = scripts / cg.client_filename()
    if p.exists():
        return str(p)
    found = store.discover_scripts(scripts)
    return str(found[0]) if found else None


def _start_sync(cfg: store.Config, scripts: Path, prompt: str, rep: Reporter) -> store.Syncer | None:
    if not cfg.get("real_time_sync") or os.environ.get("REVAPI_NO_SYNC"):
        return None
    dst = store.available_dir(Path.cwd() / "scripts", store.slugify(prompt))
    rep.log(dim(f"  syncing output → {dst}"))
    return store.Syncer(scripts, dst).start()


def _finish_record(hist: store.History, run_id: str, result: dict, extra: dict | None = None) -> None:
    hist.update(run_id, status="error" if result.get("error") else "ok", usage=result.get("usage", {}), error=result.get("error"), summary=(result.get("summary") or "")[:4000], **(extra or {}))


def payload(mode: str, run_id: str | None, prompt: str | None, result: dict, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error" if result.get("error") else "ok",
        "run_id": run_id,
        "prompt": prompt,
        "mode": mode,
        **extra,
        "script_path": result.get("script_path"),
        "summary": result.get("summary"),
        "usage": result.get("usage", {}),
        "error": result.get("error"),
        "error_kind": result.get("error_kind"),
    }


# ----------------------------------------------------------------------------- agent (auto) mode


def run_agent(
    prompt: str,
    url: str | None,
    cfg: store.Config,
    rep: Reporter,
    model: str | None = None,
    output_dir: str | None = None,
    headless: bool | None = None,
    interactive: bool = False,
    run_id: str | None = None,
    keep: dict | None = None,
) -> dict:
    output_dir = output_dir or cfg.get("output_dir")
    if hermes_engine(cfg):
        return payload("agent", None, prompt, {"error": HERMES_AGENT_NOTE, "error_kind": "config"}, url=url)
    llm = make_llm(cfg, model)
    run_id = run_id or store.new_run_id()
    scripts = store.scripts_dir(run_id, output_dir)
    har = store.har_path(run_id, output_dir)
    cg = Codegen(cfg.get("output_language"), cfg.get("output_mode"), scripts)
    hist = store.History()
    hist.upsert(store.make_record(run_id, "agent", prompt, llm.describe(), url, output_dir, output_mode=cg.mode, output_language=cg.language, provider=llm.provider))
    rep.emit("run_started", run_id=run_id, mode="agent", model=llm.describe(), har_path=str(har), scripts_dir=str(scripts))
    _setup_rep(rep, cfg, "agent")
    rep.run_header("agent", run_id, llm.describe(), cg.name if not cg.docs else "OpenAPI docs")
    ctx = Context(
        workdir=scripts,
        har_path=har,
        headless=cfg.get("headless", True) if headless is None else headless,
        interactive=interactive,
        shell_timeout=cfg.get("shell_timeout", 180),
        proxy=cfg.get("proxy"),
        user_data_dir=cfg.get("user_data_dir"),
        ask=terminal_ask,
    )
    system = load("agent_system", codegen_instructions=cg.instructions(), output_files=cg.output_files(), ask_user_note=ask_note(interactive))
    user = load("agent_user", prompt=prompt, start_url_block=f"\n<start_url>\n{url}\n</start_url>\n" if url else "", scripts_dir=scripts, har_path=har)
    tools = browser_tools(ctx) + har_tools(ctx) + file_tools(ctx) + web_tools(ctx) + control_tools(ctx)
    agent = Agent(llm, tools, ctx, system, store.messages_path(run_id, output_dir), rep, cfg.get("max_turns", 120), compact_at=float(os.environ.get("REVAPI_COMPACT_AT") or 0) or cfg.get("compact_at"))
    if keep is not None:
        keep.update(agent=agent, ctx=ctx, run_id=run_id, mode="agent", cg=cg, output_dir=output_dir)
    syncer = _start_sync(cfg, scripts, prompt, rep)
    try:
        result = agent.send(user)
    finally:
        ctx.close_browser()
        if syncer:
            syncer.stop()
    result["script_path"] = _main_script(scripts, cg, (ctx.finished or {}).get("script_path", ""))
    if not result.get("error") and not result["script_path"]:
        result["error"], result["error_kind"] = f"the agent finished without writing a client in {scripts}", "engine_failure"
    _finish_record(hist, run_id, result, {"paths": {"har_path": str(har) if har.exists() else None, "script_path": result["script_path"], "local_path": str(syncer.dst) if syncer else None}})
    if not result.get("error"):
        rep.card(result["script_path"], llm.describe())
    rep.footer()
    return payload("agent", run_id, prompt, result, url=url, har_path=str(har) if har.exists() else None, local_path=str(syncer.dst) if syncer else None, model=llm.describe())


# ----------------------------------------------------------------------------- engineer mode


def run_engineer(
    run_id: str,
    cfg: store.Config,
    rep: Reporter,
    prompt: str | None = None,
    fresh: bool = False,
    model: str | None = None,
    output_dir: str | None = None,
    interactive: bool = False,
    har_file: str | None = None,
    keep: dict | None = None,
) -> dict:
    hist = store.History()
    rec = hist.get(run_id) if run_id else None
    output_dir = output_dir or cfg.get("output_dir")
    if har_file:  # engineer an arbitrary HAR (e.g. exported from desktop DevTools)
        run_id = run_id or store.new_run_id()
        har = store.har_path(run_id, output_dir)
        shutil.copy(har_file, har)
    else:
        if rec:
            run_id = rec["run_id"]
            hp = (rec.get("paths") or {}).get("har_path")
            har = Path(hp) if hp else store.har_path(run_id, output_dir)
        else:
            har = store.har_path(run_id, output_dir)
        if not har.exists():
            return payload("engineer", run_id, prompt, {"error": f"no HAR for run {run_id} at {har}", "error_kind": "not_found"})
    use_hermes = hermes_engine(cfg)
    llm = None if use_hermes else make_llm(cfg, model)
    label = f"hermes · {cfg.get('hermes_model') or 'default'}" if llm is None else llm.describe()
    provider = "hermes" if llm is None else llm.provider
    scripts = store.scripts_dir(run_id, output_dir)
    cg = Codegen(cfg.get("output_language"), cfg.get("output_mode"), scripts)
    original = (rec or {}).get("prompt") or ""
    main_prompt = prompt if (fresh or not original) else original
    additional = prompt if (prompt and original and not fresh) else ""
    existing = None if (fresh or cg.docs) else _main_script(scripts, cg)
    if fresh and scripts.exists():
        for p in scripts.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    if not rec:
        hist.upsert(store.make_record(run_id, "engineer", main_prompt or "(engineer HAR)", label, None, output_dir, output_mode=cg.mode, output_language=cg.language, provider=provider))
    hist.update(run_id, status="running", model=label, last_engineered=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rep.emit("run_started", run_id=run_id, mode="engineer", model=label, har_path=str(har), scripts_dir=str(scripts))
    n_entries = len(harlib.load(har))
    _setup_rep(rep, cfg, "engineer")
    rep.run_header("engineer", run_id, label, f"{n_entries} HAR entries")
    ctx = Context(workdir=scripts, har_path=har, interactive=interactive, shell_timeout=cfg.get("shell_timeout", 180), ask=terminal_ask)
    attempt = (
        ""
        if cg.docs
        else (
            "If your first attempt doesn't work, analyze what went wrong and try again. Document each attempt and what you learned.\n\n"
            "<attempt_log>\nFor each attempt (up to 5), document:\n- Attempt number\n- What approach you tried\n- What error or issue occurred (if any)\n- What you changed for the next attempt\n</attempt_log>\n\n"
        )
    )
    system = load(
        "engineer_system",
        mode_description="generate an OpenAPI 3.0 specification documenting" if cg.docs else f"reverse engineer API calls and generate production-ready {cg.name} code that replicates",
        task_description="OpenAPI documentation" if cg.docs else f"{cg.name} API client",
        codegen_instructions=cg.instructions(),
        scratchpad_extra="" if cg.docs else "- Decide whether plain HTTP will be sufficient or if a browser context is needed",
        attempt_log_section=attempt,
        after_verb="documenting" if cg.docs else "testing",
        quality_check="The completeness and accuracy of the OpenAPI spec" if cg.docs else "Whether the implementation works",
        output_type="spec" if cg.docs else "code",
        ask_user_note=ask_note(interactive),
        tools_note=HERMES_TOOLS_NOTE
        if use_hermes
        else "Tools: har_overview / har_search / har_entry to explore traffic; read_file / write_file / edit_file / list_dir for files; shell to install dependencies and run tests; web_fetch for public docs. Call `finish` with your final response when done.",
    )
    guidance = ""
    if existing:
        guidance = (
            f"\nThere is already an existing {cg.name} client for this run:\n<existing_client>\n{existing}\n</existing_client>\n\n"
            f"**IMPORTANT: This is an iterative edit. Update that file in place and keep the implementation in {cg.name} unless the user explicitly asks for a fresh rewrite.**\n"
        )
    user = load(
        "engineer_user",
        har_path=har,
        prompt=main_prompt or "Reverse engineer every API in this traffic into a client.",
        scripts_dir=scripts,
        existing_client_guidance=guidance,
        additional_instructions=f"\n\nAdditional instructions:\n{additional}" if additional else "",
        tag_mode_label="Documentation" if cg.docs else "Re-engineer",
        run_id=run_id,
        har_parent=har.parent,
        existing_label="docs" if cg.docs else "scripts",
        messages_path=store.messages_path(run_id, output_dir).parent,
        is_fresh=str(fresh).lower(),
        existing_artifact="documentation" if cg.docs else "script",
    )
    if use_hermes:
        return _engineer_hermes(cfg, rep, run_id, prompt, fresh, har, scripts, cg, system, user, main_prompt, output_dir, hist, keep)
    assert llm is not None
    tools = har_tools(ctx) + file_tools(ctx) + [t for t in web_tools(ctx) if t.name == "web_fetch"] + control_tools(ctx)
    agent = Agent(llm, tools, ctx, system, store.messages_path(run_id, output_dir), rep, cfg.get("max_turns", 120), compact_at=float(os.environ.get("REVAPI_COMPACT_AT") or 0) or cfg.get("compact_at"))
    if keep is not None:
        keep.update(agent=agent, ctx=ctx, run_id=run_id, mode="engineer", cg=cg, output_dir=output_dir)
    syncer = _start_sync(cfg, scripts, main_prompt or run_id, rep)
    try:
        result = agent.send(user)
    finally:
        if syncer:
            syncer.stop()
    result["script_path"] = _main_script(scripts, cg, (ctx.finished or {}).get("script_path", ""))
    if not result.get("error") and not result["script_path"]:
        result["error"], result["error_kind"] = f"engineering finished without writing output in {scripts}", "engine_failure"
    _finish_record(hist, run_id, result, {"paths": {"har_path": str(har), "script_path": result["script_path"], "local_path": str(syncer.dst) if syncer else None}})
    if not result.get("error"):
        rep.card(result["script_path"], llm.describe())
    rep.footer()
    return payload("engineer", run_id, prompt, result, fresh=fresh, har_path=str(har), local_path=str(syncer.dst) if syncer else None, model=llm.describe())


HERMES_TOOLS_NOTE = (
    "Use your own tools: the HAR file is JSON, so read it with your file and terminal tools (python -c / jq to list "
    "entries, then print the ones that matter in full); write the files into the output directory; install "
    "dependencies and run the client against the live site until it works. Finish with a short summary."
)


def _engineer_hermes(
    cfg: store.Config,
    rep: Reporter,
    run_id: str,
    prompt: str | None,
    fresh: bool,
    har: Path,
    scripts: Path,
    cg: Codegen,
    system: str,
    user: str,
    main_prompt: str,
    output_dir: str | None,
    hist: store.History,
    keep: dict | None,
) -> dict:
    """Engineer through the local Hermes CLI instead of the built-in agent."""
    from .hermes import HermesRun, hermes_binary, missing_message

    if hermes_binary() is None:
        result = {"error": missing_message(), "error_kind": "config", "usage": {}}
        _finish_record(hist, run_id, result)
        return payload("engineer", run_id, prompt, result, fresh=fresh, har_path=str(har))
    hr = HermesRun(
        scripts,
        rep,
        model=cfg.get("hermes_model"),
        provider=cfg.get("hermes_provider"),
        max_turns=cfg.get("hermes_max_turns"),
        log_path=store.messages_path(run_id, output_dir),
        session_id=None if fresh else (hist.get(run_id) or {}).get("hermes_session"),
    )
    if keep is not None:
        keep.update(hermes=hr, agent=None, ctx=None, run_id=run_id, mode="engineer", cg=cg, output_dir=output_dir)
    syncer = _start_sync(cfg, scripts, main_prompt or run_id, rep)
    try:
        # A resumed Hermes session already holds the brief; it only gets the new instructions.
        turn = (prompt or "Continue.") if hr.session_id else f"{system}\n\n{user}"
        error = hr.turn(turn, shown_prompt=user if not hr.session_id else turn)
    finally:
        if syncer:
            syncer.stop()
    return _hermes_result(hr, rep, hist, run_id, prompt, cg, scripts, error, syncer, fresh=fresh, har=har)


def _hermes_result(hr: Any, rep: Reporter, hist: store.History, run_id: str, prompt: str | None, cg: Codegen | None, scripts: Path, error: str | None, syncer: Any = None, **extra: Any) -> dict:
    script = _main_script(scripts, cg) if cg else None
    if not error and not script:
        error = f"hermes finished without writing {cg.client_filename() if cg else 'the client'} in {scripts}"
    result: dict[str, Any] = {"usage": hr.usage.as_dict(), "script_path": script, "turns": hr.usage.requests}
    if error:
        result.update(error=error, error_kind="engine_failure")
    hist.update(run_id, hermes_session=hr.session_id)
    paths = {"script_path": script, "local_path": str(syncer.dst) if syncer else None}
    if extra.get("har"):
        paths["har_path"] = str(extra["har"])
    _finish_record(hist, run_id, result, {"paths": paths})
    if not error:
        rep.card(script, hr.label)
    rep.footer()
    har = extra.pop("har", None)
    return payload("engineer", run_id, prompt, result, har_path=str(har) if har else None, local_path=str(syncer.dst) if syncer else None, model=hr.label, **extra)


# ----------------------------------------------------------------------------- manual mode

MANUAL_HELP = """commands:
  go <url>            open a page              snap                 page snapshot with [refs]
  click <ref>         click element            type <ref> <text>    type into input (add ' ⏎' to submit)
  key <Key>           press key (Enter, Tab…)  select <ref> <value> choose option
  scroll [up|down]    scroll                   back                 history back
  js <expr>           evaluate JavaScript      wait <text|secs>     wait for text / seconds
  reqs [all] [filter] list captured requests   req <n>              request details
  tabs / tab <id>     list / switch tabs       cookies              list cookies
  shot                save screenshot          help                 this help
  done                stop capturing (save HAR)"""


def capture_manual(cfg: store.Config, rep: Reporter, url: str | None, run_id: str, output_dir: str | None, headed: bool) -> Path:
    """User drives the browser; everything is recorded. Headed if a display exists, else a command shell."""
    har = store.har_path(run_id, output_dir)
    b = Browser(headless=not headed, user_data_dir=cfg.get("user_data_dir"), proxy=cfg.get("proxy"))
    try:
        if url:
            rep.log(b.navigate(url))
        if headed:
            rep.log(f"{green('●')} Browse in the Chromium window. Everything is recorded. Press {bold('Enter')} here when done.")
            try:
                input()
            except EOFError:
                pass
            return Path(b.recorder.save(har))
        rep.log(MANUAL_HELP)
        shots = 0
        while True:
            try:
                line = input(cyan("capture> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            cmd, _, arg = line.partition(" ")
            try:
                if cmd in ("done", "exit", "quit", "q"):
                    break
                elif cmd == "help":
                    out = MANUAL_HELP
                elif cmd == "go":
                    out = b.navigate(arg) + "\n" + b.snapshot(4000)
                elif cmd == "snap":
                    out = b.snapshot(int(arg) if arg.isdigit() else 12000)
                elif cmd == "click":
                    out = b.click(arg.strip("[]"))
                elif cmd == "type":
                    ref, _, text = arg.partition(" ")
                    submit = text.endswith(" ⏎") or text.endswith(" \\n")
                    out = b.type(ref.strip("[]"), text[:-2].rstrip() if submit else text, submit=submit)
                elif cmd == "key":
                    out = b.press(arg or "Enter")
                elif cmd == "select":
                    ref, _, val = arg.partition(" ")
                    out = b.select(ref.strip("[]"), val)
                elif cmd == "scroll":
                    out = b.scroll(arg or "down")
                elif cmd == "back":
                    out = b.back()
                elif cmd == "js":
                    out = json.dumps(b.evaluate(arg), indent=1, ensure_ascii=False, default=str)
                elif cmd == "wait":
                    out = b.wait_for_text(arg) if not arg.replace(".", "").isdigit() else (time.sleep(float(arg)) or f"waited {arg}s")
                elif cmd == "reqs":
                    b.recorder.save(har)
                    entries = harlib.load(har)
                    words = arg.split()
                    idx = range(len(entries)) if "all" in words else harlib.api_indices(entries)
                    flt = next((w for w in words if w != "all"), "")
                    out = "\n".join(harlib.line(i, entries[i]) for i in idx if flt.lower() in entries[i]["request"]["url"].lower()) or "(none yet)"
                elif cmd == "req":
                    b.recorder.save(har)
                    out = harlib.entry(har, int(arg))
                elif cmd == "tabs":
                    out = json.dumps(b.tabs(), indent=1)
                elif cmd == "tab":
                    out = b.switch_tab(arg)
                elif cmd == "cookies":
                    out = "\n".join(f"{c['domain']}  {c['name']}={c['value'][:80]}" for c in b.cookies())
                elif cmd == "shot":
                    shots += 1
                    out = b.screenshot(har.parent / f"shot_{shots}.png")
                else:
                    out = f"unknown command {cmd!r} (help)"
            except Exception as e:  # noqa: BLE001
                out = red(f"{type(e).__name__}: {e}")
            for n in b.drain_notices():
                print(yellow(f"[notice] {n}"))
            print(out)
        return Path(b.recorder.save(har))
    finally:
        b.close()


def run_manual(
    cfg: store.Config,
    rep: Reporter,
    prompt: str | None,
    url: str | None,
    model: str | None = None,
    output_dir: str | None = None,
    headed: bool | None = None,
    engineer: bool = True,
    interactive: bool = True,
    keep: dict | None = None,
) -> dict:
    output_dir = output_dir or cfg.get("output_dir")
    run_id = store.new_run_id()
    if headed is None:
        headed = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not cfg.get("headless_manual", False)
    hist = store.History()
    hist.upsert(store.make_record(run_id, "manual", prompt or "", "-", url, output_dir, output_mode=cfg.get("output_mode"), output_language=cfg.get("output_language")))
    _setup_rep(rep, cfg, "manual")
    rep.run_header("manual", run_id, "headed window" if headed else "headless (command shell)")
    har = capture_manual(cfg, rep, url, run_id, output_dir, headed)
    entries = harlib.load(har)
    n_api = len(harlib.api_indices(entries))
    rep.log(f"{green('✓')} HAR saved: {har} ({len(entries)} requests, {n_api} API-like)")
    hist.update(run_id, status="captured", paths={"har_path": str(har)})
    if not engineer:
        return payload("manual", run_id, prompt, {"usage": {}}, har_path=str(har))
    if not prompt and interactive:
        try:
            prompt = input(f"{bold('What should the client do?')} (blank = all APIs) > ").strip()
        except EOFError:
            prompt = ""
    hist.update(run_id, prompt=prompt or "Reverse engineer every API in this traffic into a client.")
    res = run_engineer(run_id, cfg, rep, prompt=None, model=model, output_dir=output_dir, interactive=interactive, keep=keep)
    res["mode"] = "manual"
    return res


# ----------------------------------------------------------------------------- collector


def run_collector(prompt: str, cfg: store.Config, rep: Reporter, model: str | None = None, output_dir: str | None = None, interactive: bool = False, keep: dict | None = None) -> dict:
    output_dir = output_dir or cfg.get("output_dir")
    llm = make_llm(cfg, model, collector=True)
    run_id = store.new_run_id()
    out = store.collected_dir(run_id, output_dir)
    items = out / "items.jsonl"
    hist = store.History()
    hist.upsert(store.make_record(run_id, "collector", prompt, llm.describe(), None, output_dir, provider=llm.provider))
    rep.emit("run_started", run_id=run_id, mode="collector", model=llm.describe(), output_dir=str(out))
    _setup_rep(rep, cfg, "collector")
    rep.run_header("collector", run_id, llm.describe())
    ctx = Context(workdir=out, har_path=store.har_path(run_id, output_dir), items_path=items, interactive=interactive, ask=terminal_ask, headless=True)
    tools = web_tools(ctx) + collector_tools(ctx) + browser_tools(ctx) + [t for t in file_tools(ctx) if t.name in ("read_file", "shell")] + control_tools(ctx)
    agent = Agent(
        llm,
        tools,
        ctx,
        load("collector_system"),
        store.messages_path(run_id, output_dir),
        rep,
        cfg.get("max_turns", 120),
        compact_at=float(os.environ.get("REVAPI_COMPACT_AT") or 0) or cfg.get("compact_at"),
    )
    if keep is not None:
        keep.update(agent=agent, ctx=ctx, run_id=run_id, mode="collector", output_dir=output_dir)
    try:
        result = agent.send(load("collector_user", prompt=prompt, items_path=items))
    finally:
        ctx.close_browser()
    data = [json.loads(ln) for ln in items.read_text(encoding="utf-8").splitlines() if ln.strip()] if items.exists() else []
    export_collected(out, data, prompt)
    local = None
    if data and cfg.get("real_time_sync") and not os.environ.get("REVAPI_NO_SYNC"):
        local = store.available_dir(Path.cwd() / "collected", store.slugify(prompt))
        shutil.copytree(out, local)
    result["script_path"] = str(out / "items.json") if data else None
    if not result.get("error") and not data:
        result["error"], result["error_kind"] = "no items were collected", "empty"
    _finish_record(hist, run_id, result, {"paths": {"script_path": str(out), "local_path": str(local) if local else None}, "items": len(data)})
    rep.footer()
    return payload("collector", run_id, prompt, result, items=len(data), output_dir=str(out), local_path=str(local) if local else None, model=llm.describe())


def export_collected(out: Path, items: list[dict], prompt: str) -> None:
    (out / "items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    if items:
        cols: list[str] = []
        for it in items:
            for k in it:
                if k not in cols:
                    cols.append(k)
        with (out / "data.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for it in items:
                w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in it.items()})
    sources = sorted({str(it.get("source_url")) for it in items if it.get("source_url")})
    (out / "README.md").write_text(
        f"# Collected data\n\n**Mission:** {prompt}\n\n**Items:** {len(items)}\n\n## Files\n\n- `items.jsonl` - raw items as collected\n- `items.json` - All collected items in JSON format\n- `data.csv` - All collected items in CSV format\n\n## Sources\n\n"
        + "\n".join(f"- {s}" for s in sources[:200])
        + "\n"
    )


# ----------------------------------------------------------------------------- run generated scripts

_PY_MISSING = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")
_NODE_MISSING = re.compile(r"Cannot find (?:module|package) '([^'./][^']*)'")
PIP_NAMES = {
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "jwt": "pyjwt",
    "dotenv": "python-dotenv",
    "google": "google-api-python-client",
    "websocket": "websocket-client",
    "sklearn": "scikit-learn",
}


def run_script(
    run_id: str,
    cfg: store.Config,
    rep: Reporter,
    file_name: str | None = None,
    args: list[str] | None = None,
    list_only: bool = False,
    auto_install: bool = False,
    interactive: bool = True,
    capture: bool = False,
    output_dir: str | None = None,
) -> dict:
    hist = store.History()
    rec = hist.get(run_id)
    if rec:
        run_id = rec["run_id"]
    output_dir = output_dir or cfg.get("output_dir")
    d = Path((rec or {}).get("paths", {}).get("script_path") or "")
    d = d.parent if d.is_file() else d
    if not d.is_dir() or not str(d):
        d = store.runs_root(output_dir) / "scripts" / run_id
    scripts = store.discover_scripts(d)
    base = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "scripts_dir": str(d), "scripts": [str(s.relative_to(d)) for s in scripts]}
    if list_only:
        return {**base, "status": "ok", "error": None}
    if not scripts:
        return {**base, "status": "error", "error": f"no scripts in {d}", "error_kind": "not_found"}
    if file_name:
        target = next((s for s in scripts if s.name == file_name or str(s.relative_to(d)) == file_name), None)
        if not target:
            return {**base, "status": "error", "error": f"{file_name} not found", "error_kind": "not_found"}
    elif len(scripts) == 1 or not interactive:
        target = scripts[0]
    else:
        for i, s in enumerate(scripts, 1):
            rep.log(f"  {i}. {s.relative_to(d)}")
        try:
            ans = input("run which? [1] > ").strip() or "1"
        except EOFError:
            ans = "1"
        target = scripts[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(scripts) else scripts[0]
    cmd = store.run_command_for(target) + list(args or [])
    tried: set[str] = set()
    while True:
        rep.log(dim(f"$ {' '.join(cmd)}  (in {target.parent})"))
        if capture:
            r = subprocess.run(cmd, cwd=target.parent, capture_output=True, text=True)
            stdout, stderr = r.stdout, r.stderr
        else:
            r = subprocess.run(cmd, cwd=target.parent, stderr=subprocess.PIPE, text=True)
            stdout, stderr = None, r.stderr
            sys.stderr.write(stderr)
        missing = None
        m = _PY_MISSING.search(stderr or "") if target.suffix == ".py" else _NODE_MISSING.search(stderr or "")
        if r.returncode != 0 and m:
            missing = m.group(1).split(".")[0]
        if missing and missing not in tried:
            pkg = PIP_NAMES.get(missing, missing) if target.suffix == ".py" else missing
            if not re.fullmatch(r"[A-Za-z0-9_.@/-]+", pkg):
                break
            ok = auto_install
            if not ok and interactive:
                try:
                    ok = input(f"missing dependency {bold(pkg)} — install it? [Y/n] ").strip().lower() in ("", "y", "yes")
                except EOFError:
                    ok = False
            if ok:
                tried.add(missing)
                inst = [sys.executable, "-m", "pip", "install", pkg] if target.suffix == ".py" else ["npm", "install", pkg]
                rep.log(dim(f"$ {' '.join(inst)}"))
                subprocess.run(inst, cwd=target.parent)
                continue
        break
    return {
        **base,
        "status": "ok" if r.returncode == 0 else "error",
        "script": str(target),
        "exit_code": r.returncode,
        "stdout": stdout,
        "stderr": stderr if capture else None,
        "error": None if r.returncode == 0 else f"script exited with {r.returncode}",
        "missing_dependency": missing,
    }


# ----------------------------------------------------------------------------- follow-ups / resume


def tools_for(mode: str, ctx: Context) -> list:
    if mode == "collector":
        return web_tools(ctx) + collector_tools(ctx) + browser_tools(ctx) + [t for t in file_tools(ctx) if t.name in ("read_file", "shell")] + control_tools(ctx)
    if mode in ("engineer", "manual"):
        return har_tools(ctx) + file_tools(ctx) + [t for t in web_tools(ctx) if t.name == "web_fetch"] + control_tools(ctx)
    return browser_tools(ctx) + har_tools(ctx) + file_tools(ctx) + web_tools(ctx) + control_tools(ctx)


def resume(run_id: str, cfg: store.Config, rep: Reporter, model: str | None = None, interactive: bool = True) -> dict:
    """Rebuild a session from its transcript so the conversation can continue."""
    rec = store.History().get(run_id)
    if not rec:
        raise ValueError(f"unknown run {run_id}")
    run_id = rec["run_id"]
    output_dir = cfg.get("output_dir")
    mp = Path((rec.get("paths") or {}).get("messages_path") or store.messages_path(run_id, output_dir))
    if not mp.exists():
        raise ValueError(f"no transcript for {run_id}")
    mode = rec.get("mode", "agent")
    if rec.get("hermes_session") and hermes_engine(cfg):  # a Hermes conversation continues in Hermes
        from .hermes import HermesRun

        scripts = store.scripts_dir(run_id, output_dir)
        hr = HermesRun(scripts, rep, cfg.get("hermes_model"), cfg.get("hermes_provider"), cfg.get("hermes_max_turns"), mp, rec["hermes_session"])
        cg = Codegen(rec.get("output_language") or cfg.get("output_language"), rec.get("output_mode") or cfg.get("output_mode"), scripts)
        return {"hermes": hr, "agent": None, "ctx": None, "run_id": run_id, "mode": "engineer", "cg": cg, "output_dir": output_dir}
    msgs = Agent.load_messages(mp)
    scripts = store.scripts_dir(run_id, output_dir) if mode != "collector" else store.collected_dir(run_id, output_dir)
    ctx = Context(
        workdir=scripts,
        har_path=store.har_path(run_id, output_dir),
        interactive=interactive,
        ask=terminal_ask,
        shell_timeout=cfg.get("shell_timeout", 180),
        headless=cfg.get("headless", True),
        items_path=scripts / "items.jsonl" if mode == "collector" else None,
        proxy=cfg.get("proxy"),
        user_data_dir=cfg.get("user_data_dir"),
    )
    agent = Agent(
        make_llm(cfg, model, collector=mode == "collector"),
        tools_for(mode, ctx),
        ctx,
        "",
        mp,
        rep,
        cfg.get("max_turns", 120),
        messages=msgs,
        compact_at=float(os.environ.get("REVAPI_COMPACT_AT") or 0) or cfg.get("compact_at"),
    )
    cg = Codegen(rec.get("output_language") or cfg.get("output_language"), rec.get("output_mode") or cfg.get("output_mode"), scripts)
    return {"agent": agent, "ctx": ctx, "run_id": run_id, "mode": mode, "cg": cg, "output_dir": output_dir}


def follow_up(keep: dict, text: str, rep: Reporter) -> dict:
    """Continue the last (or resumed) session with a new instruction."""
    if keep.get("hermes") is not None:
        hr = keep["hermes"]
        hr.on_event = rep
        rep.mode, rep.model = "engineer", hr.label
        hist = store.History()
        hist.update(keep["run_id"], status="running")
        before = hr.usage.as_dict()
        error = hr.turn(text)
        res = _hermes_result(hr, rep, hist, keep["run_id"], text, keep.get("cg"), Path(hr.scripts_dir), error)
        res["usage_turn"] = {k: (v or 0) - (before.get(k) or 0) for k, v in hr.usage.as_dict().items() if isinstance(v, (int, float))}
        return res
    agent: Agent = keep["agent"]
    ctx: Context = keep["ctx"]
    ctx.finished = None
    rep.mode = keep["mode"] if keep["mode"] in ("agent", "engineer", "collector", "manual") else "agent"
    rep.model = agent.llm.describe()
    agent.on_event = rep
    hist = store.History()
    hist.update(keep["run_id"], status="running")
    try:
        result = agent.send(text)
    finally:
        ctx.close_browser()
    cg = keep.get("cg")
    if keep["mode"] == "collector":
        items = ctx.items_path
        data = [json.loads(ln) for ln in items.read_text(encoding="utf-8").splitlines() if ln.strip()] if items and items.exists() else []
        export_collected(ctx.workdir, data, text)
        result["script_path"] = str(ctx.workdir / "items.json")
    else:
        result["script_path"] = _main_script(ctx.workdir, cg, (ctx.finished or {}).get("script_path", "")) if cg else None
    prev = (hist.get(keep["run_id"]) or {}).get("usage") or {}
    total = {k: (prev.get(k) or 0) + (v or 0) for k, v in result["usage"].items() if isinstance(v, (int, float))}
    _finish_record(hist, keep["run_id"], {**result, "usage": total})
    if not result.get("error") and keep["mode"] != "collector":
        rep.card(result["script_path"], agent.llm.describe())
    rep.footer()
    return payload(keep["mode"], keep["run_id"], text, result)


# ----------------------------------------------------------------------------- updates

RELEASES_API = "https://api.github.com/repos/bageltrade/revapi/releases/latest"
UPDATE_HINT = "cd ~/revapi && git pull && pip install ."


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def update_notice(force: bool = False) -> str | None:
    """A note when a newer revapi release exists. Checked at most once a day (cached); never raises."""
    if os.environ.get("REVAPI_NO_UPDATE_CHECK") and not force:
        return None
    cache = store.app_dir() / "update_check.json"
    latest = None
    try:
        data = json.loads(cache.read_text()) if cache.exists() else {}
        if not force and time.time() - data.get("at", 0) < 86400:
            latest = data.get("latest")
        else:
            import urllib.request

            req = urllib.request.Request(RELEASES_API, headers={"User-Agent": "revapi", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=4) as r:
                latest = json.loads(r.read().decode()).get("tag_name")
            cache.write_text(json.dumps({"at": time.time(), "latest": latest}))
    except Exception:  # noqa: BLE001  (offline, rate limited: no notice)
        return None
    if latest and _vtuple(latest) > _vtuple(__version__):
        return f"revapi {latest.lstrip('v')} is out (you have {__version__}) · update: {UPDATE_HINT}"
    return None


# ----------------------------------------------------------------------------- doctor


def doctor(cfg: store.Config) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 10), sys.version.split()[0]))
    exe = find_chromium()
    checks.append(("chromium", bool(exe), exe or "not found — Termux: pkg install x11-repo && pkg install chromium"))
    if exe:
        try:
            b = Browser()
            v = b.version
            b.close()
            checks.append(("chromium headless launch", True, v))
        except Exception as e:  # noqa: BLE001
            checks.append(("chromium headless launch", False, str(e)[:300]))
    try:
        llm = make_llm(cfg)
        checks.append(("llm provider", True, llm.describe() + ("" if llm.api_key or llm.provider in ("pollinations", "ollama", "lmstudio") else " (no key)")))
    except LLMError as e:
        checks.append(("llm provider", False, str(e)))
    from .hermes import hermes_binary, missing_message

    hb = hermes_binary()
    if hermes_engine(cfg):
        checks.append(("engine: hermes", bool(hb), hb or missing_message()))
    elif hb:
        checks.append(("hermes cli (optional engine)", True, hb))
    checks.append(("update", True, update_notice(force=True) or f"revapi {__version__} is the latest"))
    for tool in ("bash", "node", "go", "php", "ruby", "java", "dotnet", "cc", "pwsh"):
        p = shutil.which(tool)
        checks.append((f"runtime: {tool}", bool(p), p or "not installed (only needed for that output language)"))
    checks.append(("data dir", True, str(store.app_dir())))
    return checks


__all__ = ["run_agent", "run_engineer", "run_manual", "run_collector", "run_script", "doctor", "is_api_like"]
