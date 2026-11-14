# revapi

Turn any website into a working API client from your terminal (including **Termux on Android**).

**Headless Chromium** records every request the site makes: headers, cookies, bodies, redirects and websocket frames. An **AI agent** then reads that traffic and writes a client, runs it, and fixes it until it works.

- **Pure Python standard library.** No pip dependencies, no Playwright, no Node.
- **Talks to Chromium directly** over the DevTools Protocol, using its own small WebSocket client.
- **Works with almost any LLM.** Anthropic, OpenAI, OpenRouter, Gemini, **NVIDIA NIM**, Groq, DeepSeek, Mistral, xAI, Together, Cerebras and OpenCode are supported. So are local models through Ollama or LM Studio, any OpenAI-compatible URL, and a free keyless fallback.

revapi is a from-scratch rebuild of [reverse-api-engineer](https://github.com/kalil0321/reverse-api-engineer) (MIT) with the same modes, commands and outputs. Its code-generation prompt partials are adapted from that project.

---

## Install

### Termux (Android)

```bash
pkg update && pkg install -y git
git clone https://github.com/bageltrade/revapi.git   # private repo: username = bageltrade, password = a GitHub token
cd revapi && bash install-termux.sh                   # installs python + chromium (x11-repo) + revapi, runs doctor
```

Or do it by hand:

```bash
pkg install python git x11-repo
pkg install chromium          # from x11-repo; provides chromium-browser
pip install .                 # or run without installing: python -m revapi
revapi doctor
```

Or install a release wheel directly (download it from the repo's Releases page first):

```bash
pip install revapi-0.1.1-py3-none-any.whl
```

### Linux / macOS / WSL

```bash
sudo apt install chromium     # or have Google Chrome installed
pip install .
revapi doctor
```

If your browser binary isn't found automatically, set `REVAPI_CHROMIUM=/path/to/chrome`. Any Chromium-family browser works: Chromium, Chrome, Edge, Brave or chrome-headless-shell.

### Pick a model

Export one key and revapi picks the matching provider automatically (`provider: auto`):

```bash
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY,
                                 # GROQ_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, XAI_API_KEY ...
```

#### NVIDIA (build.nvidia.com)

```bash
export NVIDIA_API_KEY=nvapi-...          # get one at https://build.nvidia.com (keys start with nvapi-)
revapi doctor                            # → llm provider  nvidia/moonshotai/kimi-k2.6
revapi models                            # browse NVIDIA's catalog
revapi config set model deepseek-ai/deepseek-v4.1-flash   # optional: pick another model
```

The default is `moonshotai/kimi-k2.6` (strong at agentic coding and tool use). Other good picks are `deepseek-ai/deepseek-v4.1-flash`, `z-ai/glm-5.3-flash`, `minimaxai/minimax-m3.1-flash`, `nvidia/nemotron-3-super-120b-a12b` and `mistralai/mistral-large-4-1t-a52b`. You can also save the key in config instead of the environment: `revapi config set api_key nvapi-...` (an `nvapi-` key auto-selects NVIDIA).

You can also choose explicitly:

```bash
revapi config set provider openrouter
revapi config set model anthropic/claude-sonnet-4.5
revapi config set provider ollama && revapi config set model qwen2.5-coder:14b     # local
revapi config set provider custom && revapi config set base_url https://my-host/v1  # any OpenAI-compatible API
```

With no key and no local Ollama, revapi falls back to **Pollinations**. It is free and needs no key, but it is slow: about one reply every 45 seconds, which revapi waits out for you. A real model gives far better clients.

---

## Modes

| Mode | What happens |
|---|---|
| `agent` | The agent drives headless Chromium itself, captures the traffic, writes a client in your chosen language, runs it and iterates until it works. |
| `manual` | You drive the browser and everything is recorded; then the agent engineers the HAR. You get a real window if a display exists. On Termux you get a command shell instead (`go`, `snap`, `click e3`, `type e5 hello ⏎`, `reqs`, `req 4`, …). |
| `engineer` | Generate or refine a client from a past run's HAR, or from any `.har` file (`--har`). Follow-up instructions edit the existing client in place; `--fresh` rewrites it from scratch. |
| `collector` | The agent searches and fetches the web (falling back to the browser for JS-heavy pages) and saves structured items as `items.jsonl`, `items.json`, `data.csv` and `README.md`. |
| `run` | Run a generated client. Missing Python or Node dependencies are detected and installed with a prompt, or automatically with `--auto-install`. |

### Examples

```bash
revapi                                    # interactive session (all modes, slash commands)

revapi agent -p "get product search and product details" -u https://shop.example.com
revapi agent -p "..." -u https://site.com --json          # one JSON result on stdout
revapi agent -p "..." -u https://site.com --json-stream   # NDJSON progress events + final result
revapi agent -p "..." -u https://site.com --dry-run       # check Chromium + model, run nothing

revapi manual -u https://site.com         # you browse; then it engineers
revapi manual -u https://site.com --no-engineer           # capture only

revapi engineer <run_id> -p "add pagination and a search() helper"
revapi engineer <run_id> --fresh -p "rewrite in TypeScript"
revapi engineer --har ~/Downloads/devtools-export.har -p "client for the orders API"

revapi collector -p "list 30 Lagos fintech startups with website and founding year"

revapi list [--json] [--full] [-n 10] [-m agent] [--model claude] [-s search]
revapi show <run_id|latest> [--json]
revapi run <run_id|latest> [-f file] [--ls] [--auto-install] [--json] [-- script args...]
revapi har <run_id|file.har> [--search token] [--entry 12] [--all] [--api-only out.har]
revapi models [-f filter] [--provider nvidia] [--json]   # list models on the provider
revapi config [list | get k | set k v | unset k | path]
revapi doctor
```

### Output languages and docs mode

```bash
revapi config set output_language typescript   # python javascript typescript go java csharp php ruby c powershell
revapi config set output_mode docs             # OpenAPI 3.0 spec + README instead of a client
```

### Interactive session

Run `revapi` with no arguments. Type a request; if it contains a URL, that becomes the start page. Follow-up messages keep refining the same run.

| Command | Does |
|---|---|
| `/mode` | Switch mode: agent, manual, engineer or collector |
| `/model` | Change model; `/model nvidia:moonshotai/kimi-k2.6` switches provider too |
| `/models` | List the models available on the current provider (`/models flash` filters) |
| `/effort` | Reasoning effort: low, medium, high or none |
| `/language` | Output language |
| `/docs` | Toggle OpenAPI docs output |
| `/settings` | Show settings; `/settings <key> <value>` changes one |
| `/history` | List past runs |
| `/resume` | `/resume <id>` reopens a past conversation so you can continue it |
| `/engineer` | `/engineer <id> [instructions]` re-engineers a past run |
| `/run` | Run the latest client |
| `/open` | Open the latest client |
| `/output` | Show everything from this session |
| `/new` | Start a new run |
| `/clear` | Clear the session and start over |
| `/help` | Help |
| `/exit` | Quit |

Ctrl-C interrupts the agent; Ctrl-D quits.

---

## What the agent can do (tools)

- **Browser** (headless Chromium over CDP):
  - Page actions: `browser_navigate`, `browser_snapshot` (text page model with `[e12]` element refs), `browser_click`, `browser_type`, `browser_press_key`, `browser_select`, `browser_scroll`, `browser_back`, `browser_wait_for`, `browser_find_text`.
  - Scripting and screenshots: `browser_evaluate` (JavaScript), `browser_take_screenshot`.
  - Tabs and state: `browser_tabs` / `browser_switch_tab` (popups are tracked), `browser_cookies` (including HttpOnly), `browser_storage` (localStorage and sessionStorage).
  - Network: `browser_fetch` (probe an endpoint from inside the page), `browser_network_requests`, `browser_network_request`, `browser_close`.
- **HAR analysis:**
  - `har_overview`: hosts, auth headers, endpoint shapes and a numbered request list.
  - `har_search`: regex search over URLs, headers, bodies and websocket frames, e.g. to find where a token comes from.
  - `har_entry`: full detail of one request, with paging through long bodies.
- **Files and shell:** `read_file`, `write_file`, `edit_file`, `list_dir`, and `shell` (install dependencies, run and test the client).
- **Web:** `web_search` (DuckDuckGo), `web_fetch`.
- **Control:**
  - `ask_user`: interactive runs only; in non-interactive runs the agent makes and documents assumptions.
  - `save_items`: collector mode.
  - `finish`: ends the run.

### What gets recorded

- **Format:** HAR 1.2 with full request headers (including `Cookie`, taken from Chromium's extra-info events), request cookies, POST bodies (fetched separately when large), response headers including `Set-Cookie`, decoded response bodies (binary bodies kept as base64), redirect chains with `redirectURL`, timings and server IP.
- **Websockets:** frames are stored in `_webSocketMessages`.
- **Coverage:** every tab, iframe and worker the page opens.
- **Limits:** the browser cache is disabled so every call hits the network. Bodies over `REVAPI_MAX_BODY_BYTES` (3 MB by default) are omitted. Images, fonts and media skip body capture unless `REVAPI_RECORD_MEDIA=1` is set.

---

## Files

```
~/.revapi/
  config.json
  history.json                      every run: prompt, mode, model, status, usage/cost, paths
  runs/har/<run_id>/recording.har
  runs/scripts/<run_id>/            generated client + README (+ package.json / go.mod / ...)
  runs/messages/<run_id>.jsonl      full agent transcript (used by /resume)
  runs/collected/<run_id>/          collector output
  runs/screenshots/<run_id>/
./scripts/<prompt_slug>/            live mirror of the generated files (real_time_sync)
```

`--output-dir` (or `config set output_dir`) moves `runs/` elsewhere.

## Config keys

| Key | Default | Meaning |
|---|---|---|
| `provider` | `auto` | anthropic, openai, openrouter, gemini, nvidia, groq, deepseek, mistral, xai, together, cerebras, opencode, ollama, lmstudio, pollinations or custom |
| `model` | provider default | Model id |
| `api_key` / `base_url` | – | Optional overrides; environment variables are preferred for keys |
| `effort` | – | low, medium or high (reasoning effort / thinking budget) |
| `collector_model` | – | Separate model for collector mode |
| `output_language` | `python` | Language of the generated client |
| `output_mode` | `client` | `client` or `docs` |
| `output_dir` | `~/.revapi/runs` | Where runs are stored |
| `real_time_sync` | `true` | Mirror generated files into `./scripts/` |
| `headless` | `true` | Run the agent's browser headless |
| `user_data_dir` | – | Persistent Chromium profile, which keeps you logged in between runs |
| `proxy` | – | e.g. `http://127.0.0.1:8080` |
| `max_turns` | `120` | Agent step limit |
| `shell_timeout` | `180` | Seconds allowed per shell command |

### Environment variables

| Variable | Effect |
|---|---|
| `REVAPI_CHROMIUM` | Path to the browser binary |
| `REVAPI_HOME` | Data directory |
| `REVAPI_NO_SANDBOX=1` | Always set automatically on Termux and when running as root |
| `REVAPI_USER_AGENT` | Override the browser's User-Agent |
| `REVAPI_CONTEXT_CHARS` | Context budget before old tool outputs are compacted |
| `REVAPI_VERBOSE=1` | Show reasoning and tool output |
| `NO_COLOR=1` | Disable colored output |

## JSON output (for scripts and other agents)

`agent`, `engineer`, `collector` and `run` accept `--json`, which prints one object on stdout; logs go to stderr. They also accept `--json-stream`, which emits NDJSON events (`run_started`, `text`, `thinking`, `tool_start`, `tool_end`, …) and then a final line of the form `{"event":"result", ...}`.

Every result has the fields `schema_version`, `status`, `run_id`, `prompt`, `mode`, `script_path`, `summary`, `usage` (tokens and `estimated_cost_usd`), `error` and `error_kind`.

`revapi --json-schema-version` prints the schema version.

## Tests

```bash
python tests/test_revapi.py        # or: python -m pytest tests -q
```

The test suite:
- launches the real Chromium against a local fixture app;
- runs the agent, engineer (including docs mode and external HAR input), collector, manual-shell and `--json-stream` pipelines end to end with a scripted model;
- checks the OpenAI and Anthropic wire formats, throttle handling, config, history and the script runner.

## Architecture

| Module | Contents |
|---|---|
| `cdp.py` | RFC 6455 WebSocket client and CDP connection (flattened sessions, events, waiters) |
| `browser.py` | Chromium discovery and launch, page actions, HAR recorder |
| `har.py` | HAR overview, search, entry detail and API filter |
| `llm.py` | OpenAI-compatible and Anthropic clients, retries, usage and cost |
| `agent.py` | Tool-calling loop, context compaction, transcript log, resume |
| `tools.py` | All agent tools |
| `modes.py` | agent, engineer, manual, collector, run and doctor |
| `store.py` | Config, run paths, history, script discovery, live sync |
| `cli.py` | argparse CLI and interactive session |
| `prompts/` | Prompt templates and per-language code-generation partials |
