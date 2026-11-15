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
from . import term as T
from .ui import LAST_THINKING, Reporter, bold, cyan, dim, green, red, yellow

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
    return Reporter(verbose=getattr(args, "verbose", False), show_thinking=store.Config().get("show_thinking", True) is not False)


def _interactive(args: argparse.Namespace) -> bool:
    return not (getattr(args, "no_interactive", False) or getattr(args, "json", False) or getattr(args, "json_stream", False)) and sys.stdin.isatty()


def _finish(res: dict, args: argparse.Namespace, rep: Reporter) -> int:
    if getattr(args, "json", False) or getattr(args, "json_stream", False):
        _out(res, getattr(args, "json_stream", False))
    else:
        if res.get("summary") and not (sys.stdout.isatty() and getattr(rep, "summary_shown", False)):
            print("\n" + res["summary"].strip())  # piped output gets the plain summary
        if res.get("error"):
            print(f"\n {T.sty('!', T.PINK, bold=True)}  {red(str(res['error']))}", file=sys.stderr)
            if res.get("error_kind") in ("network", "rate_limit", "engine_failure", "interrupted", "max_turns") and res.get("run_id"):
                print(dim(f"    nothing is lost — continue with: revapi   then /resume {res['run_id']}"), file=sys.stderr)
        else:
            for k in ("script_path", "local_path", "har_path"):
                if res.get(k):
                    label = {"script_path": "server" if res.get("mcp") else "client", "local_path": "copy", "har_path": "capture"}[k]
                    print(f"   {T.sty('└', T.HAIRLINE)} {T.sty(f'{label:<7}', T.DIM)} {T.sty(T.link(str(res[k]), Path(str(res[k])).resolve().as_uri()), T.INK_SOFT)}", file=sys.stderr)
            if res.get("mcp"):
                n = res["mcp"]["name"]
                print(dim(f'\n   test it: revapi mcp test {n}  ·  add tools: revapi mcp from {res["run_id"]} "..."'), file=sys.stderr)
            elif res.get("run_id") and res.get("mode") != "collector":
                print(dim(f'\n   run it: revapi run {res["run_id"]}  ·  refine: revapi engineer {res["run_id"]} -p "..."'), file=sys.stderr)
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
    if modes.hermes_engine(cfg):  # like RAE: agent mode cannot run on the hermes engine
        checks.append({"name": "engine", "ok": False, "detail": modes.HERMES_AGENT_NOTE})
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


SKIP_MODELS = ("embed", "guard", "rerank", "safety", "reward", "retriev", "ocr", "whisper", "tts", "dall-e", "image", "audio", "moderation")


def cmd_models(args: argparse.Namespace, cfg: store.Config) -> int:
    try:
        llm = modes.make_llm(cfg) if not args.provider else modes.LLM(provider=args.provider, api_key=cfg.get("api_key") if cfg.get("provider") == args.provider else None, require_key=False)
        ids = llm.list_models()
    except (LLMError, OSError, ValueError) as e:
        print(red(f"could not list models: {e}"))
        return 1
    if not args.all:
        ids = [i for i in ids if not any(k in i.lower() for k in SKIP_MODELS)]
    if args.filter:
        ids = [i for i in ids if args.filter.lower() in i.lower()]
    if args.json:
        _out({"provider": llm.provider, "current": llm.model, "models": ids})
        return 0
    for i in ids:
        print(f"{green('●') if i == llm.model else ' '} {i}")
    print(dim(f"\n{len(ids)} models on {llm.provider} · current: {llm.model} · change: revapi config set model <id>"))
    return 0


def _handoff_say(message: str, data: dict) -> None:
    stage = data.get("stage")
    if stage == "grace":
        print(dim(f"◇ {message}"), file=sys.stderr)
    elif stage == "waiting":
        t = int(data.get("timeout", 600))
        mins = f"{t // 60}m" if t >= 60 else f"{t}s"
        print(f"\n{bold('your turn')} · {data.get('label')} on {data.get('url')}", file=sys.stderr)
        print(f"  open {cyan(data.get('live_url', ''))}", file=sys.stderr)
        print(dim(f"  it mirrors revapi's browser · tap the check yourself · carries on once it clears · waits {mins} · ctrl+c stops\n"), file=sys.stderr)


def cmd_refresh(args: argparse.Namespace, cfg: store.Config) -> int:
    """Open the site in revapi's browser (saved profile); hand any human check to you; save the session."""
    from .browser import Browser
    from .cdp import CDPError
    from .handoff import hand_off, host_of, save_session
    from .tools import shared_profile

    opts = modes._handoff_opts(cfg)
    prof = cfg.get("user_data_dir") or shared_profile()
    try:
        b = Browser(headless=True if not args.headed else False, user_data_dir=prof, proxy=cfg.get("proxy"))
    except CDPError as e:
        print(red(str(e)), file=sys.stderr)
        return 1
    try:
        print(dim(b.navigate(args.url)), file=sys.stderr)
        res = hand_off(
            b, _handoff_say, timeout=args.timeout or opts["handoff_timeout"], port=opts["handoff_port"], host=opts["handoff_host"], open_url=opts["handoff_open"] and not args.no_open, live=True
        )
        if res is None:
            path = str(save_session(b, args.url, source="no check showing"))
            res = {"passed": True, "how": "no check showing", "session": path, "url": args.url, "label": "", "seconds": 0}
    except KeyboardInterrupt:
        print(yellow("stopped"), file=sys.stderr)
        return 130
    finally:
        b.close()
    if args.json:
        _out({k: res.get(k) for k in ("passed", "how", "session", "url", "label", "seconds")})
    elif res["passed"]:
        print(f"{green('✓')} {host_of(args.url)} · {res['how']} · session saved to {res['session']}")
    else:
        print(f"{red('✗')} {host_of(args.url)} · {res['how']}")
    return 0 if res["passed"] else 1


def cmd_cookies(args: argparse.Namespace, cfg: store.Config) -> int:
    from .handoff import host_of, load_session, parse_cookies, session_path, sessions_dir, write_session

    if args.action == "list":
        rows = []
        for f in sorted(sessions_dir().glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            names = {c.get("name") for c in d.get("cookies", [])}
            rows.append({"host": d.get("host", f.stem), "cookies": len(names), "cf_clearance": "cf_clearance" in names, "saved_at": d.get("saved_at"), "source": d.get("source"), "path": str(f)})
        if args.json:
            _out({"sessions": rows})
        elif not rows:
            print(dim("no saved sessions · revapi refresh <url> or revapi cookies import <host>"))
        for r in rows if not args.json else []:
            cf = green(" cf_clearance") if r["cf_clearance"] else ""
            print(f"{bold(r['host']):<40} {r['cookies']:>3} cookies{cf}  {dim(str(r['saved_at']) + ' · ' + str(r['source']))}")
        return 0
    if not args.host:
        print(red(f"revapi cookies {args.action} needs a host"), file=sys.stderr)
        return 2
    host = host_of(args.host)
    if args.action == "show":
        d = load_session(host)
        if not d:
            print(red(f"no saved session for {host}"), file=sys.stderr)
            return 1
        print(json.dumps(d, indent=1, ensure_ascii=False))
        return 0
    if args.action == "clear":
        p = session_path(host)
        if p.exists():
            p.unlink()
            print(f"cleared the saved session for {host}")
        else:
            print(dim(f"no saved session for {host}"))
        return 0
    # import
    if args.file:
        text = Path(args.file).expanduser().read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            print(dim("Paste the cookies (a Cookie header, a JSON export, or cookies.txt), then press ctrl+d:"), file=sys.stderr)
        text = sys.stdin.read()
    try:
        cookies = parse_cookies(text, host)
    except (ValueError, TypeError) as e:
        print(red(f"could not read those cookies: {e}"), file=sys.stderr)
        return 1
    if not cookies:
        print(red("no cookies found in that input"), file=sys.stderr)
        return 1
    ua = args.ua or ""
    if not ua and "cf_clearance" in {c["name"] for c in cookies}:
        print(yellow("note: cf_clearance only works with the exact user agent of the browser that passed the check · add --ua '<that browser's user agent>'"), file=sys.stderr)
    path = write_session(host, cookies, ua, "imported")
    print(f"{green('✓')} {len(cookies)} cookies for {host} saved to {path}")
    return 0


def _use_mcp(cfg: store.Config) -> None:
    """MCP output for this command only (the saved output_mode is left alone)."""
    cfg.data["output_mode"] = "mcp"


def cmd_mcp(args: argparse.Namespace, cfg: store.Config) -> int:
    from . import mcpgen

    act = args.mcp_cmd
    if act in (None, "list"):
        rows = mcpgen.installed()
        if getattr(args, "json", False):
            _out({"servers": rows, "root": str(mcpgen.root())})
            return 0
        if not rows:
            print(dim('no MCP servers yet · make one: revapi mcp new "tools to search and read posts" -u https://site.com'))
            return 0
        for r in rows:
            tools = r.get("tools") or []
            print(f"{bold(r['name'])}  {dim(str(len(tools)) + ' tools')}  {dim(r.get('url') or '')}")
            if tools:
                print("   " + dim(", ".join(tools)))
        print(dim(f"\n{mcpgen.root()} · revapi mcp config <name> · revapi mcp add <name> <client>"))
        return 0
    if act == "new":
        prompt = " ".join(args.words or []) or args.prompt
        if not prompt:
            if not sys.stdin.isatty():
                print(red('describe the tools: revapi mcp new "search products, product details, reviews" -u https://site.com'), file=sys.stderr)
                return 2
            prompt = input(bold("Which tools should the MCP server have? ") + "> ").strip()
        _use_mcp(cfg)
        ns = argparse.Namespace(
            prompt=prompt,
            url=args.url,
            model=args.model,
            output_dir=args.output_dir,
            headed=args.headed,
            headless=False,
            dry_run=False,
            json=args.json,
            json_stream=False,
            verbose=args.verbose,
            no_interactive=args.no_interactive,
        )
        return cmd_agent(ns, cfg)
    if act == "from":
        _use_mcp(cfg)
        ns = argparse.Namespace(
            run_id=args.run_id,
            har=args.har,
            prompt=" ".join(args.words or []) or None,
            fresh=False,
            model=args.model,
            output_dir=args.output_dir,
            json=args.json,
            json_stream=False,
            verbose=args.verbose,
            no_interactive=args.no_interactive,
        )
        return cmd_engineer(ns, cfg)
    try:
        server = mcpgen.resolve(args.target)
    except FileNotFoundError as e:
        print(red(str(e)), file=sys.stderr)
        return 1
    meta_path = server.parent / "revapi-mcp.json"
    name = json.loads(meta_path.read_text(encoding="utf-8")).get("name") if meta_path.exists() else mcpgen.slug(server.parent.name)
    if act == "install":
        rec = store.History().get(args.target) or {}
        meta = mcpgen.install(server.parent, args.name or mcpgen.name_for_run(rec.get("run_id", ""), rec.get("url"), rec.get("prompt")), rec.get("run_id", ""), rec.get("url"), rec.get("prompt") or "")
        print(f"{green('✓')} installed {bold(meta['name'])} · {len(meta['tools'])} tools · {meta['path']}")
        if meta.get("error"):
            print(yellow(meta["error"][:500]))
        return 0 if not meta.get("error") else 1
    if act == "test":
        try:
            info = mcpgen.handshake(server)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"{red('✗')} {name}: {e}")
            return 1
        print(f"{green('✓')} {info['server'].get('name', name)} · MCP {info['protocol']} · {len(info['tools'])} tools")
        for t in info["tools"]:
            req = t.get("inputSchema", {}).get("required", [])
            params = ", ".join(f"{k}{'' if k in req else '?'}" for k in t.get("inputSchema", {}).get("properties", {}))
            print(f"  {bold(t['name'])}({params})  {dim((t.get('description') or '').splitlines()[0][:90] if t.get('description') else '')}")
        if args.call:
            r = subprocess.run([mcpgen.python_cmd(), str(server), "--call", args.call, args.arguments or "{}"], cwd=str(server.parent))
            return r.returncode
        return 0
    if act == "config":
        snippets = mcpgen.config(server, name)
        if args.json:
            _out(snippets)
            return 0
        pick = [args.client] if args.client else ["claude", "codex", "gemini", "opencode", "hermes", "http"]
        titles = {
            "claude": "Claude Code",
            "codex": "Codex CLI",
            "gemini": "Gemini CLI",
            "opencode": "opencode",
            "hermes": "Hermes",
            "http": "Streamable HTTP (any client that takes a URL)",
            "json": "mcpServers JSON",
        }
        for k in pick:
            print(bold(titles.get(k, k)))
            print(snippets[k] + "\n")
        print(dim(f"or let revapi add it: revapi mcp add {name} claude|codex|gemini|opencode"))
        return 0
    if act == "add":
        try:
            print(f"{green('✓')} {mcpgen.add_to(args.client, server, name)}")
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            print(red(str(e)), file=sys.stderr)
            return 1
        return 0
    if act == "serve":
        cmd = [mcpgen.python_cmd(), str(server)] + (["--http", str(args.http), "--host", args.host] if args.http else [])
        try:
            return subprocess.run(cmd, cwd=str(server.parent)).returncode
        except KeyboardInterrupt:
            return 0
    if act == "remove":
        if server.parent.parent != mcpgen.root():
            print(red(f"{server} is not an installed server (only ~/.revapi/mcp/<name> can be removed)"), file=sys.stderr)
            return 1
        shutil.rmtree(server.parent)
        print(f"removed {name} · also remove it from your MCP client's config")
        return 0
    return 2


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
    "/models": "list models available on the current provider (/models <filter>)",
    "/effort": "how hard the agent thinks (low|medium|high|xhigh|max|none)",
    "/mode": "choose the mode (agent|manual|engineer|collector)",
    "/language": "output language (python, javascript, typescript, go, java, csharp, php, ruby, c, powershell)",
    "/docs": "toggle OpenAPI docs output instead of a client",
    "/mcp": "toggle MCP server output instead of a client",
    "/settings": "all settings (/settings <key> <value> to change)",
    "/history": "past runs: their details, resume or re-engineer",
    "/resume": "a past run's conversation, to continue it (/resume <run_id>)",
    "/engineer": "re-engineer a past run (/engineer <run_id> [instructions])",
    "/run": "run the latest client (/run [run_id] [args])",
    "/thinking": "the model's full reasoning from the last reply",
    "/context": "how full the context window is",
    "/compact": "summarize the conversation now (full history stays recallable)",
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


def _banner(cfg: store.Config, mode: str, desc: str) -> None:
    a = T.MODE_COLORS.get(mode, T.PINK)
    lang = "MCP server" if cfg.get("output_mode") == "mcp" else cfg.get("output_language") + (" (docs)" if cfg.get("output_mode") == "docs" else "")
    effort = cfg.get("effort")
    print(file=sys.stderr)
    print(
        f" {T.sty('*', T.PINK, bold=True)} {T.sty('revapi', T.INK, bold=True)}  {T.sty('v' + __version__, T.DIM)}   {T.sty('turn websites into', T.INK_SOFT)} {T.sty('APIs', T.PINK, italic=True)}",
        file=sys.stderr,
    )
    print(f"   {T.sty(' · '.join(x for x in (desc, 'effort ' + effort if effort else '', lang) if x), T.DIM)}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"   {T.sty('describe the API you want · / for commands · mode', T.DIM)} {T.sty(mode, a, italic=True)}", file=sys.stderr)


def _prompt(mode: str, model: str) -> str:
    """The seam above the input (mode left, model right) and the › marker; escapes wrapped for readline."""
    a = T.MODE_COLORS.get(mode, T.PINK)
    w = T.cols()
    left = f" ┄┄ {mode} "
    right = f" {model} ┄┄"
    fill = max(1, w - len(left) - len(right) - 1)
    seam = T.sty(" ┄┄ ", T.HAIRLINE) + T.sty(mode, a, italic=True) + " " + T.sty("┄" * fill, T.HAIRLINE) + " " + T.sty(model, T.INK_SOFT) + T.sty(" ┄┄", T.HAIRLINE)
    marker = T.sty(" › ", a, bold=True)
    if T.COLOR:
        marker = re.sub(r"(\x1b\[[0-9;]*m)", "\x01\\1\x02", marker)
    print("\n" + seam, file=sys.stderr)
    return marker


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
        desc = f"no model ({e})"
    _banner(cfg, mode, desc)
    note = modes.update_notice()
    if note:
        print(dim(f"   {note}"), file=sys.stderr)
    while True:
        try:
            short = (modes.make_llm(cfg).model if not desc.startswith("no model") else "no model").split("/")[-1]
        except LLMError:
            short = "no model"
        try:
            line = input(_prompt(mode, short)).strip()
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
            elif cmd == "/models":
                cmd_models(argparse.Namespace(provider=None, filter=arg or None, all=False, json=False), cfg)
            elif cmd == "/effort":
                try:
                    cfg.set("effort", arg or _pick("effort", [*store.EFFORTS, "none"], cfg.get("effort") or "none"))
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
            elif cmd == "/mcp":
                cfg.set("output_mode", "client" if cfg.get("output_mode") == "mcp" else "mcp")
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
            elif cmd == "/thinking":
                text = LAST_THINKING[0]
                if text:
                    for ln in T.render_markdown(text, T.cols() - 1, "      " + T.sty("┊", T.HAIRLINE) + " "):
                        print(T.sty(T.strip_ansi(ln), T.DIM, italic=True) if T.COLOR else ln, file=sys.stderr)
                else:
                    print(dim("   no reasoning yet — models that think show it here (try /effort high)"))
            elif cmd in ("/context", "/compact"):
                ag = keep.get("agent")
                if not ag:
                    print(dim("   no conversation yet"))
                else:
                    if cmd == "/compact":
                        rep = Reporter(mode=mode)
                        ag.on_event = rep
                        _safe(lambda: ag.maybe_compact([t.schema() for t in ag.tools.values()], force=True))
                        rep.stop()
                    used, win = ag.context_tokens([t.schema() for t in ag.tools.values()]), ag._window()
                    pct = round(100 * used / win) if win else 0
                    bar_w = 24
                    fill = min(bar_w, round(bar_w * used / win)) if win else 0
                    color = T.MINT if pct < 60 else T.AMBER if pct < 80 else T.PINK
                    print(f"   {T.sty('▰' * fill, color)}{T.sty('▱' * (bar_w - fill), T.HAIRLINE)}  {T.sty(f'{T.k(used)} / {T.k(win)}', T.INK)} {T.sty(f'tokens · {pct}%', T.DIM)}", file=sys.stderr)
                    print(
                        T.truncate(T.sty(f"   compacts at {int(ag.compact_at * 100)}% · {ag.compactions} compacted · {len(ag.messages)} messages live · full history recallable", T.DIM), T.cols() - 1),
                        file=sys.stderr,
                    )
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
        rep = Reporter(mode=mode, show_thinking=cfg.get("show_thinking", True) is not False)
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
        rep.stop()
        session.append(res)
        _show(res, shown=rep.summary_shown)


def _safe(fn) -> dict:  # noqa: ANN001
    try:
        return fn()
    except LLMError as e:
        return {"status": "error", "error": str(e), "error_kind": e.kind}
    except KeyboardInterrupt:
        return {"status": "error", "error": "interrupted", "error_kind": "interrupted"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "error_kind": "unknown"}


def _show(res: dict, shown: bool = False) -> None:
    """A run's outcome in the conversation: the summary as Markdown (unless already shown), then where its files are."""
    if res.get("summary") and not shown:
        lines = T.render_markdown(res["summary"].strip(), T.cols() - 1, "   ", f" {T.sty('¶', T.PINK)} ")
        print("\n".join(lines), file=sys.stderr)
    if res.get("error"):
        print(f"\n {T.sty('!', T.PINK, bold=True)}  {T.sty(str(res['error'])[:2000], T.PINK)}", file=sys.stderr)
        if res.get("error_kind") in ("network", "rate_limit", "engine_failure", "interrupted") and res.get("run_id"):
            print(f"    {dim('nothing is lost — type a follow-up or /resume ' + str(res['run_id']) + ' to continue')}", file=sys.stderr)
    for k in ("script_path", "local_path", "har_path"):
        if res.get(k):
            label = {"script_path": "server" if res.get("mcp") else "client", "local_path": "copy", "har_path": "capture"}[k]
            print(f"   {T.sty('└', T.HAIRLINE)} {T.sty(f'{label:<7}', T.DIM)} {T.sty(T.link(str(res[k]), Path(str(res[k])).resolve().as_uri()), T.INK_SOFT)}", file=sys.stderr)


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
    a.add_argument("--mcp", action="store_true", help="generate an MCP server instead of a client")
    scripted(a)

    m = sub.add_parser("manual", help="you browse (window or command shell); then the agent engineers the HAR")
    m.add_argument("-p", "--prompt")
    m.add_argument("-u", "--url")
    m.add_argument("-m", "--model")
    m.add_argument("-o", "--output-dir")
    m.add_argument("--headed", action="store_true")
    m.add_argument("--headless", action="store_true", help="command-shell capture even if a display exists")
    m.add_argument("--no-engineer", action="store_true", help="only capture the HAR")
    m.add_argument("--mcp", action="store_true", help="generate an MCP server instead of a client")
    m.add_argument("--json", action="store_true")
    m.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("engineer", help="generate/refine a client from a past run's HAR")
    e.add_argument("run_id", nargs="?")
    e.add_argument("-p", "--prompt")
    e.add_argument("--fresh", action="store_true", help="ignore previous scripts and re-engineer from scratch")
    e.add_argument("--har", help="engineer an external .har file (e.g. exported from DevTools)")
    e.add_argument("--mcp", action="store_true", help="generate an MCP server instead of a client")
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

    mo = sub.add_parser("models", help="list models available from the current (or given) provider")
    mo.add_argument("--provider", choices=list(PROVIDERS))
    mo.add_argument("-f", "--filter")
    mo.add_argument("--all", action="store_true", help="include embedding/guard/etc. models")
    mo.add_argument("--json", action="store_true")

    mc = sub.add_parser("mcp", help="MCP servers for any website: new, from, list, test, config, add, serve, install, remove")
    msub = mc.add_subparsers(dest="mcp_cmd")
    mn = msub.add_parser("new", help="browse a site and generate an MCP server with the tools you describe")
    mn.add_argument("words", nargs="*", help="the tools you want, e.g. search products, product details, reviews")
    mn.add_argument("-p", "--prompt")
    mn.add_argument("-u", "--url", help="the website")
    mn.add_argument("-m", "--model")
    mn.add_argument("-o", "--output-dir")
    mn.add_argument("--headed", action="store_true")
    scripted(mn, stream=False)
    mf = msub.add_parser("from", help="turn a past run's capture into an MCP server")
    mf.add_argument("run_id", nargs="?")
    mf.add_argument("words", nargs="*", help="the tools you want")
    mf.add_argument("--har", help="an external .har file instead of a run")
    mf.add_argument("-m", "--model")
    mf.add_argument("-o", "--output-dir")
    scripted(mf, stream=False)
    ml = msub.add_parser("list", help="installed MCP servers")
    ml.add_argument("--json", action="store_true")
    for nm, hp in (
        ("test", "start a server and check it speaks MCP; list its tools"),
        ("config", "print the config for MCP clients"),
        ("serve", "run a server (stdio, or --http PORT)"),
        ("install", "install a run's server under ~/.revapi/mcp"),
        ("remove", "delete an installed server"),
    ):
        sp = msub.add_parser(nm, help=hp)
        sp.add_argument("target", help="installed name, run id, folder or server.py")
        if nm == "test":
            sp.add_argument("--call", help="also call this tool")
            sp.add_argument("arguments", nargs="?", help="JSON arguments for --call")
        if nm == "config":
            sp.add_argument("client", nargs="?", choices=["claude", "codex", "gemini", "opencode", "hermes", "http", "json"])
            sp.add_argument("--json", action="store_true")
        if nm == "serve":
            sp.add_argument("--http", type=int, metavar="PORT", help="Streamable HTTP instead of stdio")
            sp.add_argument("--host", default="127.0.0.1")
        if nm == "install":
            sp.add_argument("--name")
    ma = msub.add_parser("add", help="register a server with an MCP client on this device")
    ma.add_argument("target")
    ma.add_argument("client", choices=["claude", "codex", "gemini", "opencode", "hermes"])

    rf = sub.add_parser("refresh", help="open a site, pass any Cloudflare/CAPTCHA check yourself in the live view, save the session")
    rf.add_argument("url")
    rf.add_argument("--timeout", type=int, help="seconds to wait for you (default: handoff_timeout)")
    rf.add_argument("--no-open", action="store_true", help="only print the live view link")
    rf.add_argument("--headed", action="store_true", help="use a visible Chromium window (needs a display)")
    rf.add_argument("--json", action="store_true")

    ck = sub.add_parser("cookies", help="saved site sessions: list, show, clear, or import cookies from your own browser")
    ck.add_argument("action", choices=["list", "show", "clear", "import"])
    ck.add_argument("host", nargs="?")
    ck.add_argument("-f", "--file", help="read cookies from a file instead of stdin")
    ck.add_argument("--ua", help="user agent of the browser the cookies came from")
    ck.add_argument("--json", action="store_true")

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
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if argv[:1] == ["run"] and "--" in argv:  # everything after -- goes to the script (argparse differs across Python versions)
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1 :]
    args = parser.parse_args(argv)
    if passthrough:
        args.args = list(getattr(args, "args", None) or []) + passthrough
    if args.json_schema_version:
        print(modes.SCHEMA_VERSION)
        return 0
    cfg = store.Config()
    if getattr(args, "mcp", False):
        _use_mcp(cfg)
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
        "models": cmd_models,
        "refresh": cmd_refresh,
        "mcp": cmd_mcp,
        "cookies": cmd_cookies,
    }
    if not args.cmd:
        if not sys.stdin.isatty():
            parser.print_help()
            return 0
        from . import session

        if session.available():
            return session.run(cfg)
        return repl(cfg)
    try:
        return handlers[args.cmd](args, cfg)
    except KeyboardInterrupt:
        print(red("\ninterrupted"), file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001  (a short error; REVAPI_DEBUG=1 shows the traceback)
        if os.environ.get("REVAPI_DEBUG"):
            raise
        print(red(f"error: {type(e).__name__}: {e}"), file=sys.stderr)
        print(dim("   set REVAPI_DEBUG=1 for the traceback"), file=sys.stderr)
        return 1


def entry() -> None:
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())


if __name__ == "__main__":
    entry()

_ = io  # silence unused import on some linters
