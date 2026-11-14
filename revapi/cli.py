"""revapi command line + interactive session."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__, modes, store
from . import har as harlib
from .browser import find_chromium
from .llm import PROVIDERS, LLMError, detect_provider
from .ui import Reporter, bold, cyan, dim, green, red, yellow

MODES = ("agent", "manual", "engineer", "collector")
URL_RE = re.compile(r"https?://[^\s'\"<>]+|\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|app|dev|co|xyz|me|tv|gg|so|sh|ng|uk|de|fr)(?:/[^\s]*)?\b", re.I)


# ----------------------------------------------------------------------------- output helpers


def _out(obj: Any, stream_mode: bool = False) -> None:
    if stream_mode:
        obj = {"event": "result", **obj}
    print(json.dumps(obj, ensure_ascii=False, default=str))


def _reporter(args: argparse.Namespace) -> Reporter:
    if getattr(args, "json_stream", False):
        return Reporter(quiet=True, json_stream=sys.stdout, verbose=False)
    if getattr(args, "json", False):
        return Reporter(quiet=True)
    return Reporter(verbose=getattr(args, "verbose", False))


def _interactive(args: argparse.Namespace) -> bool:
    return not (getattr(args, "no_interactive", False) or getattr(args, "json", False) or getattr(args, "json_stream", False)) and sys.stdin.isatty()


def _finish(res: dict, args: argparse.Namespace, rep: Reporter) -> int:
    if getattr(args, "json", False) or getattr(args, "json_stream", False):
        _out(res, getattr(args, "json_stream", False))
    else:
        if res.get("summary"):
            print("\n" + res["summary"].strip())
        if res.get("error"):
            print(red(f"\n✗ {res['error']}"), file=sys.stderr)
        else:
            for k in ("script_path", "local_path", "har_path"):
                if res.get(k):
                    print(f"{green('✓')} {k.replace('_', ' ')}: {res[k]}", file=sys.stderr)
            if res.get("run_id") and res.get("mode") != "collector":
                print(dim(f'  run it: revapi run {res["run_id"]}   ·   refine: revapi engineer {res["run_id"]} -p "..."'), file=sys.stderr)
    return 1 if res.get("error") else 0


def _llm_error(e: LLMError, args: argparse.Namespace, mode: str) -> int:
    res = {"schema_version": modes.SCHEMA_VERSION, "status": "error", "mode": mode, "run_id": None, "error": str(e), "error_kind": e.kind}
    return _finish(res, args, Reporter())


# ----------------------------------------------------------------------------- commands


def cmd_agent(args: argparse.Namespace, cfg: store.Config) -> int:
    if not args.prompt:
        if not _interactive(args):
            return _finish({"status": "error", "error": "--prompt is required (non-interactive)", "error_kind": "usage", "mode": "agent"}, args, Reporter())
        args.prompt = input(bold("What should the agent reverse engineer? ") + "> ").strip()
    url = args.url
    if not url:
        m = URL_RE.search(args.prompt)
        url = m.group(0) if m else None
    headless = False if args.headed else (True if args.headless else None)
    if args.dry_run:
        return _finish(dry_run(cfg, args, url, headless), args, Reporter())
    rep = _reporter(args)
    try:
        res = modes.run_agent(args.prompt, url, cfg, rep, model=args.model, output_dir=args.output_dir, headless=headless, interactive=_interactive(args))
    except LLMError as e:
        return _llm_error(e, args, "agent")
    return _finish(res, args, rep)


def dry_run(cfg: store.Config, args: argparse.Namespace, url: str | None, headless: bool | None) -> dict:
    checks = []
    exe = find_chromium()
    checks.append({"name": "chromium", "ok": bool(exe), "detail": exe or "not found"})
    try:
        llm = modes.make_llm(cfg, args.model)
        checks.append({"name": "llm", "ok": True, "detail": llm.describe()})
        model = llm.describe()
    except LLMError as e:
        checks.append({"name": "llm", "ok": False, "detail": str(e)})
        model = None
    bad = [c for c in checks if not c["ok"]]
    return {
        "schema_version": modes.SCHEMA_VERSION,
        "status": "error" if bad else "ok",
        "run_id": None,
        "prompt": args.prompt,
        "url": url,
        "mode": "dry-run",
        "har_path": None,
        "script_path": None,
        "usage": {},
        "error": "; ".join(f"{c['name']}: {c['detail']}" for c in bad) or None,
        "error_kind": "preflight" if bad else None,
        "would_run": {
            "model": model,
            "output_dir": str(store.runs_root(args.output_dir or cfg.get("output_dir"))),
            "headless": cfg.get("headless", True) if headless is None else headless,
            "output_language": cfg.get("output_language"),
            "output_mode": cfg.get("output_mode"),
        },
        "checks": checks,
    }


def cmd_manual(args: argparse.Namespace, cfg: store.Config) -> int:
    rep = _reporter(args)
    headed = True if args.headed else (False if args.headless else None)
    try:
        res = modes.run_manual(cfg, rep, args.prompt, args.url, model=args.model, output_dir=args.output_dir, headed=headed, engineer=not args.no_engineer, interactive=sys.stdin.isatty())
    except LLMError as e:
        return _llm_error(e, args, "manual")
    return _finish(res, args, rep)


def cmd_engineer(args: argparse.Namespace, cfg: store.Config) -> int:
    rid = args.run_id
    if not rid and not args.har:
        latest = store.History().get("latest")
        if not latest:
            return _finish({"status": "error", "error": "run_id is required (no previous runs)", "error_kind": "usage", "mode": "engineer"}, args, Reporter())
        rid = latest["run_id"]
    rep = _reporter(args)
    try:
        res = modes.run_engineer(rid, cfg, rep, prompt=args.prompt, fresh=args.fresh, model=args.model, output_dir=args.output_dir, interactive=_interactive(args), har_file=args.har)
    except LLMError as e:
        return _llm_error(e, args, "engineer")
    return _finish(res, args, rep)


def cmd_collector(args: argparse.Namespace, cfg: store.Config) -> int:
    if not args.prompt:
        if not _interactive(args):
            return _finish({"status": "error", "error": "--prompt is required", "error_kind": "usage", "mode": "collector"}, args, Reporter())
        args.prompt = input(bold("What should be collected? ") + "> ").strip()
    rep = _reporter(args)
    try:
        res = modes.run_collector(args.prompt, cfg, rep, model=args.model, output_dir=args.output_dir, interactive=_interactive(args))
    except LLMError as e:
        return _llm_error(e, args, "collector")
    return _finish(res, args, rep)


def _filter_runs(args: argparse.Namespace) -> list[dict]:
    runs = store.History().all()
    if args.mode:
        runs = [r for r in runs if r.get("mode") == args.mode or (args.mode == "auto" and r.get("mode") == "agent")]
    if args.model:
        runs = [r for r in runs if args.model.lower() in str(r.get("model", "")).lower()]
    if args.search:
        runs = [r for r in runs if args.search.lower() in str(r.get("prompt", "")).lower()]
    return runs[: args.limit] if args.limit else runs


def cmd_list(args: argparse.Namespace, cfg: store.Config) -> int:
    runs = _filter_runs(args)
    if args.json:
        _out(
            [
                {k: r.get(k) for k in ("run_id", "timestamp", "mode", "model", "status", "prompt", "url")}
                | {"script_path": (r.get("paths") or {}).get("script_path"), "cost": (r.get("usage") or {}).get("estimated_cost_usd")}
                for r in runs
            ]
        )  # type: ignore[arg-type]
        return 0
    if not runs:
        print('no runs yet — try: revapi agent -p "..." -u https://example.com')
        return 0
    for r in runs:
        st = {"ok": green("✓"), "error": red("✗"), "running": yellow("…"), "captured": cyan("◆")}.get(r.get("status", ""), " ")
        line = f"{st} {bold(r['run_id'])}  {dim(str(r.get('timestamp', ''))[:16])}  {r.get('mode', ''):<9} {(r.get('prompt') or '')[:60].replace(chr(10), ' ')}"
        if args.full:
            u = r.get("usage") or {}
            line += f"\n    model={r.get('model')} url={r.get('url') or '-'} lang={r.get('output_language', '-')} cost=${(u.get('estimated_cost_usd') or 0):.4f}\n    {(r.get('paths') or {}).get('script_path') or ''}"
        print(line)
    return 0


def cmd_show(args: argparse.Namespace, cfg: store.Config) -> int:
    rec = store.History().get(args.run_id or "latest")
    if not rec:
        if args.json:
            _out({"schema_version": modes.SCHEMA_VERSION, "status": "error", "run_id": args.run_id, "error": "run not found", "error_kind": "not_found"})
        else:
            print(red(f"run {args.run_id} not found"))
        return 1
    d = Path(store.runs_root(cfg.get("output_dir")) / "scripts" / rec["run_id"])
    rec = {**rec, "schema_version": modes.SCHEMA_VERSION, "scripts": [str(p.relative_to(d)) for p in store.discover_scripts(d)] if d.exists() else []}
    if args.json:
        _out(rec)
        return 0
    for k in ("run_id", "status", "mode", "model", "timestamp", "url", "output_language", "output_mode"):
        print(f"{bold(k):<20} {rec.get(k)}")
    print(f"{bold('prompt'):<20} {rec.get('prompt')}")
    for k, v in (rec.get("paths") or {}).items():
        if v:
            print(f"{bold(k):<20} {v}")
    u = rec.get("usage") or {}
    if u:
        print(f"{bold('usage'):<20} in={u.get('input_tokens')} out={u.get('output_tokens')} cost=${(u.get('estimated_cost_usd') or 0):.4f}")
    if rec["scripts"]:
        print(f"{bold('scripts'):<20} {', '.join(rec['scripts'])}")
    if rec.get("error"):
        print(f"{bold('error'):<20} {red(rec['error'])}")
    if rec.get("summary"):
        print("\n" + rec["summary"])
    return 0


def cmd_run(args: argparse.Namespace, cfg: store.Config) -> int:
    interactive = _interactive(args) and not args.auto_install
    rep = Reporter(quiet=bool(args.json or args.json_stream), json_stream=sys.stdout if args.json_stream else None)
    res = modes.run_script(
        args.run_id or "latest", cfg, rep, file_name=args.file, args=args.args, list_only=args.ls, auto_install=args.auto_install, interactive=interactive, capture=bool(args.json or args.json_stream)
    )
    if args.json or args.json_stream:
        _out(res, args.json_stream)
    elif args.ls:
        print("\n".join(res.get("scripts") or []) or "(no scripts)")
    elif res.get("error") and res.get("error_kind") == "not_found":
        print(red(res["error"]), file=sys.stderr)
    return 0 if res.get("status") == "ok" else 1


def cmd_config(args: argparse.Namespace, cfg: store.Config) -> int:
    if args.action in (None, "list"):
        for k, v in cfg.data.items():
            shown = ("***" + str(v)[-4:]) if k == "api_key" and v else v
            print(f"{k:<18} {shown}")
        print(dim(f"\n{cfg.path}"))
    elif args.action == "get":
        print(cfg.get(args.key))
    elif args.action == "set":
        try:
            cfg.set(args.key, args.value)
        except ValueError as e:
            print(red(str(e)))
            return 1
        print(f"{args.key} = {cfg.get(args.key)}")
    elif args.action == "unset":
        cfg.set(args.key, store.DEFAULTS.get(args.key))
        print(f"{args.key} reset")
    elif args.action == "path":
        print(cfg.path)
    return 0


def cmd_doctor(args: argparse.Namespace, cfg: store.Config) -> int:
    ok_all = True
    for name, ok, detail in modes.doctor(cfg):
        critical = not name.startswith("runtime")
        ok_all &= ok or not critical
        mark = green("✓") if ok else (red("✗") if critical else dim("·"))
        print(f"{mark} {name:<26} {detail}")
    print(dim(f"\nproviders: {', '.join(PROVIDERS)} (auto → {detect_provider()})"))
    return 0 if ok_all else 1


def cmd_har(args: argparse.Namespace, cfg: store.Config) -> int:
    p = Path(args.target)
    if not p.exists():
        rec = store.History().get(args.target)
        p = Path((rec or {}).get("paths", {}).get("har_path") or store.runs_root(cfg.get("output_dir")) / "har" / (rec or {"run_id": args.target})["run_id"] / "recording.har")
    if not p.exists():
        print(red(f"no HAR found for {args.target}"))
        return 1
    if args.entry is not None:
        print(harlib.entry(p, args.entry, max_body=args.max_body))
    elif args.search:
        print(harlib.search(p, args.search))
    elif args.api_only:
        n = harlib.filter_api(p, args.api_only)
        print(f"wrote {n} API entries to {args.api_only}")
    else:
        print(harlib.overview(p, include_all=args.all))
    return 0


# ----------------------------------------------------------------------------- interactive session

REPL_COMMANDS = {
    "/new": "start a new capture",
    "/model": "choose the model (/model <name>, /model provider:<name>)",
    "/effort": "how hard the agent thinks (low|medium|high|none)",
    "/mode": "choose the mode (agent|manual|engineer|collector)",
    "/language": "output language (python, javascript, typescript, go, java, csharp, php, ruby, c, powershell)",
    "/docs": "toggle OpenAPI docs output instead of a client",
    "/settings": "all settings (/settings <key> <value> to change)",
    "/history": "past runs: their details, resume or re-engineer",
    "/resume": "a past run's conversation, to continue it (/resume <run_id>)",
    "/engineer": "re-engineer a past run (/engineer <run_id> [instructions])",
    "/run": "run the latest client (/run [run_id] [args])",
    "/output": "everything from this session, unfolded",
    "/open": "open the latest client",
    "/clear": "clear the conversation and start over",
    "/help": "what revapi does, its modes, commands and keys",
    "/exit": "quit revapi",
}

HELP = (
    f"""{bold("revapi")} turns websites into APIs. Headless Chromium captures every request; an AI agent
reads the traffic and writes a working client.

{bold("modes")}
  agent      the agent browses the site itself, captures traffic, writes + tests a client
  manual     you drive the browser (window if a display exists, else a command shell), then the agent engineers
  engineer   (re)generate a client from a past run's HAR; follow-ups refine it
  collector  the agent searches/fetches the web and collects structured data (JSONL/JSON/CSV)

Type a request to start (include a URL to set the start page). Follow-up messages continue the same run.
{bold("commands")}
"""
    + "\n".join(f"  {k:<11} {v}" for k, v in REPL_COMMANDS.items())
    + "\n\nCtrl-C interrupts the agent; Ctrl-D quits."
)


def repl(cfg: store.Config) -> int:
    try:
        import readline  # noqa: F401  (history + line editing where available)

        hist_file = store.app_dir() / "repl_history"
        with contextlib.suppress(OSError):
            readline.read_history_file(hist_file)
        import atexit

        atexit.register(lambda: readline.write_history_file(hist_file))
        readline.set_completer(lambda text, state: ([c for c in REPL_COMMANDS if c.startswith(text)] + [None])[state])
        readline.parse_and_bind("tab: complete")
    except Exception:  # noqa: BLE001
        pass
    mode = cfg.get("repl_mode", "agent")
    keep: dict[str, Any] = {}
    session: list[dict] = []
    try:
        desc = modes.make_llm(cfg).describe()
    except LLMError as e:
        desc = red(f"no model ({e})")
    print(f"{bold('revapi')} {__version__} · {desc} · mode {cyan(mode)} · {cfg.get('output_language')}{' (docs)' if cfg.get('output_mode') == 'docs' else ''}")
    print(dim("type a request, or /help"))
    while True:
        try:
            line = input(f"\n{cyan(mode)} {bold('›')} ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/exit", "/quit", "/q"):
                return 0
            if cmd == "/help":
                print(HELP)
            elif cmd in ("/new", "/clear"):
                keep = {}
                if cmd == "/clear":
                    session = []
                    print("\033[2J\033[H" if sys.stdout.isatty() else "", end="")
                print(green("new session"))
            elif cmd == "/mode":
                if arg not in MODES:
                    arg = _pick("mode", list(MODES), mode)
                mode = arg
                cfg.set("repl_mode", mode)
                keep = {}
                print(f"mode → {cyan(mode)}")
            elif cmd == "/model":
                if not arg:
                    print(
                        f"provider={cfg.get('provider')} (auto → {detect_provider()}) model={cfg.get('model') or 'default'}\nproviders: {', '.join(PROVIDERS)}\nusage: /model <model>  or  /model <provider>:<model>"
                    )
                else:
                    if ":" in arg and arg.split(":", 1)[0] in PROVIDERS:
                        prov, _, m = arg.partition(":")
                        cfg.set("provider", prov)
                        cfg.set("model", m or None)
                    else:
                        cfg.set("model", arg)
                    keep = {}
                    print(f"model → {cfg.get('provider')}/{cfg.get('model') or 'default'}")
            elif cmd == "/effort":
                try:
                    cfg.set("effort", arg or _pick("effort", ["low", "medium", "high", "none"], cfg.get("effort") or "none"))
                    print(f"effort → {cfg.get('effort')}")
                except ValueError as e:
                    print(red(str(e)))
            elif cmd == "/language":
                try:
                    cfg.set("output_language", arg or _pick("language", list(store.LANGUAGES), cfg.get("output_language")))
                    print(f"language → {cfg.get('output_language')}")
                except ValueError as e:
                    print(red(str(e)))
            elif cmd == "/docs":
                cfg.set("output_mode", "client" if cfg.get("output_mode") == "docs" else "docs")
                print(f"output → {cfg.get('output_mode')}")
            elif cmd == "/settings":
                if arg:
                    k, _, v = arg.partition(" ")
                    try:
                        cfg.set(k, v)
                        print(f"{k} = {cfg.get(k)}")
                    except ValueError as e:
                        print(red(str(e)))
                else:
                    cmd_config(argparse.Namespace(action="list"), cfg)
            elif cmd == "/history":
                runs = store.History().all()[:20]
                for r in runs:
                    print(f"  {bold(r['run_id'])} {r.get('mode', ''):<9} {r.get('status', ''):<8} {(r.get('prompt') or '')[:60]}")
                if runs:
                    print(dim("  /resume <id> to continue · /engineer <id> [instructions] · /run <id>"))
            elif cmd == "/resume":
                rid = arg or (store.History().get("latest") or {}).get("run_id")
                try:
                    keep = modes.resume(rid, cfg, Reporter())
                    mode = keep["mode"] if keep["mode"] in MODES else "agent"
                    print(green(f"resumed {keep['run_id']} ({len(keep['agent'].messages)} messages) — type a follow-up"))
                except (ValueError, LLMError) as e:
                    print(red(str(e)))
            elif cmd == "/engineer":
                rid, _, extra = arg.partition(" ")
                rid = rid or (store.History().get("latest") or {}).get("run_id")
                keep = {}
                session.append(_safe(lambda: modes.run_engineer(rid, cfg, Reporter(), prompt=extra or None, interactive=True, keep=keep)))
                _show(session[-1])
            elif cmd == "/run":
                parts = arg.split()
                rid = parts[0] if parts else (keep.get("run_id") or "latest")
                modes.run_script(rid, cfg, Reporter(), args=parts[1:], interactive=True)
            elif cmd == "/output":
                for r in session:
                    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
                if not session:
                    print(dim("nothing yet"))
            elif cmd == "/open":
                last = next((r for r in reversed(session) if r.get("script_path")), None) or {}
                path = last.get("script_path") or ((store.History().get("latest") or {}).get("paths") or {}).get("script_path")
                if path and Path(path).is_file():
                    opener = shutil.which("termux-open") or shutil.which("xdg-open") or shutil.which("open")
                    if opener and not os.environ.get("REVAPI_NO_OPEN"):
                        subprocess.run([opener, path], check=False)
                    else:
                        print(Path(path).read_text(encoding="utf-8")[:20000])
                    print(dim(path))
                else:
                    print(dim("no client yet"))
            else:
                print(red(f"unknown command {cmd}") + dim("  (/help)"))
            continue

        # ---- a request
        rep = Reporter()
        if keep.get("agent"):
            res = _safe(lambda: modes.follow_up(keep, line, rep))
        elif mode == "agent":
            m = URL_RE.search(line)
            res = _safe(lambda: modes.run_agent(line, m.group(0) if m else None, cfg, rep, interactive=True, keep=keep))
        elif mode == "manual":
            m = URL_RE.search(line)
            res = _safe(lambda: modes.run_manual(cfg, rep, line if not (m and m.group(0) == line) else None, m.group(0) if m else None, interactive=True, keep=keep))
        elif mode == "engineer":
            rid = line if store.History().get(line) and " " not in line else (store.History().get("latest") or {}).get("run_id")
            extra = None if rid == line else line
            res = _safe(lambda: modes.run_engineer(rid, cfg, rep, prompt=extra, interactive=True, keep=keep)) if rid else {"error": "no runs to engineer yet"}
        else:
            res = _safe(lambda: modes.run_collector(line, cfg, rep, interactive=True, keep=keep))
        session.append(res)
        _show(res)


def _safe(fn) -> dict:  # noqa: ANN001
    try:
        return fn()
    except LLMError as e:
        return {"status": "error", "error": str(e), "error_kind": e.kind}
    except KeyboardInterrupt:
        return {"status": "error", "error": "interrupted", "error_kind": "interrupted"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "error_kind": "unknown"}


def _show(res: dict) -> None:
    if res.get("summary"):
        print("\n" + res["summary"].strip())
    if res.get("error"):
        print(red(f"✗ {res['error']}"))
    for k in ("script_path", "local_path", "har_path"):
        if res.get(k):
            print(f"{green('✓')} {k.replace('_', ' ')}: {res[k]}")


def _pick(label: str, options: list[str], current: str | None) -> str:
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}{dim(' (current)') if o == current else ''}")
    try:
        a = input(f"{label} > ").strip()
    except EOFError:
        return current or options[0]
    if a.isdigit() and 1 <= int(a) <= len(options):
        return options[int(a) - 1]
    return a or current or options[0]


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="revapi", description="Reverse engineer website APIs with headless Chromium + an AI agent. Run without a command for the interactive session.")
    p.add_argument("--version", action="version", version=f"revapi {__version__}")
    p.add_argument("--json-schema-version", action="store_true", help="print the JSON schema_version emitted and exit")
    sub = p.add_subparsers(dest="cmd")

    def scripted(sp: argparse.ArgumentParser, stream: bool = True) -> None:
        sp.add_argument("--no-interactive", action="store_true", help="fail fast instead of prompting (for scripts/agents)")
        sp.add_argument("--json", action="store_true", help="emit a single JSON result on stdout (logs go to stderr)")
        if stream:
            sp.add_argument("--json-stream", action="store_true", help='emit NDJSON progress events, then a final {"event":"result",...}')
        sp.add_argument("-v", "--verbose", action="store_true", help="show reasoning and tool outputs")

    a = sub.add_parser("agent", help="autonomous: browse, capture, and generate a client")
    a.add_argument("-p", "--prompt")
    a.add_argument("-u", "--url", help="starting URL")
    a.add_argument("-m", "--model")
    a.add_argument("-o", "--output-dir")
    a.add_argument("--headless", action="store_true", help="force headless (default)")
    a.add_argument("--headed", action="store_true", help="show the browser window (needs a display)")
    a.add_argument("--dry-run", action="store_true", help="check prerequisites and print what would run")
    scripted(a)

    m = sub.add_parser("manual", help="you browse (window or command shell); then the agent engineers the HAR")
    m.add_argument("-p", "--prompt")
    m.add_argument("-u", "--url")
    m.add_argument("-m", "--model")
    m.add_argument("-o", "--output-dir")
    m.add_argument("--headed", action="store_true")
    m.add_argument("--headless", action="store_true", help="command-shell capture even if a display exists")
    m.add_argument("--no-engineer", action="store_true", help="only capture the HAR")
    m.add_argument("--json", action="store_true")
    m.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("engineer", help="generate/refine a client from a past run's HAR")
    e.add_argument("run_id", nargs="?")
    e.add_argument("-p", "--prompt")
    e.add_argument("--fresh", action="store_true", help="ignore previous scripts and re-engineer from scratch")
    e.add_argument("--har", help="engineer an external .har file (e.g. exported from DevTools)")
    e.add_argument("-m", "--model")
    e.add_argument("-o", "--output-dir")
    scripted(e)

    c = sub.add_parser("collector", help="collect structured data from the web")
    c.add_argument("-p", "--prompt")
    c.add_argument("-m", "--model")
    c.add_argument("-o", "--output-dir")
    scripted(c, stream=True)

    ls = sub.add_parser("list", help="list past runs")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--full", action="store_true")
    ls.add_argument("-n", "--limit", type=int)
    ls.add_argument("-m", "--mode")
    ls.add_argument("--model")
    ls.add_argument("-s", "--search")

    sh = sub.add_parser("show", help="show one run")
    sh.add_argument("run_id", nargs="?")
    sh.add_argument("--json", action="store_true")

    r = sub.add_parser("run", help="run a generated script")
    r.add_argument("run_id", nargs="?")
    r.add_argument("args", nargs="*", help="arguments passed to the script (put flags for the script after --, e.g. revapi run ID -- --limit 5)")
    r.add_argument("-f", "--file")
    r.add_argument("--ls", action="store_true")
    r.add_argument("--auto-install", action="store_true", help="install missing dependencies without prompting")
    scripted(r)

    cf = sub.add_parser("config", help="view or change settings")
    cf.add_argument("action", nargs="?", choices=["list", "get", "set", "unset", "path"])
    cf.add_argument("key", nargs="?")
    cf.add_argument("value", nargs="?")

    sub.add_parser("doctor", help="check Chromium, model provider and runtimes")

    h = sub.add_parser("har", help="inspect a HAR (run id or file)")
    h.add_argument("target")
    h.add_argument("--entry", type=int)
    h.add_argument("--search")
    h.add_argument("--all", action="store_true")
    h.add_argument("--api-only", metavar="OUT", help="write a HAR with only API traffic")
    h.add_argument("--max-body", type=int, default=6000)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json_schema_version:
        print(modes.SCHEMA_VERSION)
        return 0
    cfg = store.Config()
    if getattr(args, "json", False) or getattr(args, "json_stream", False):
        # keep stdout clean for machine output
        pass
    handlers = {
        "agent": cmd_agent,
        "manual": cmd_manual,
        "engineer": cmd_engineer,
        "collector": cmd_collector,
        "list": cmd_list,
        "show": cmd_show,
        "run": cmd_run,
        "config": cmd_config,
        "doctor": cmd_doctor,
        "har": cmd_har,
    }
    if not args.cmd:
        if not sys.stdin.isatty():
            parser.print_help()
            return 0
        return repl(cfg)
    try:
        return handlers[args.cmd](args, cfg)
    except KeyboardInterrupt:
        print(red("\ninterrupted"), file=sys.stderr)
        return 130


def entry() -> None:
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())


if __name__ == "__main__":
    entry()

_ = io  # silence unused import on some linters
