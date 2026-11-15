**Generate an MCP server** (Model Context Protocol) that exposes the site's functionality as tools, so any MCP client (Claude Code, Codex, Gemini CLI, opencode, Hermes, Cursor, …) can use it.

**The tools are what the user asked for.** Turn the user's request into a focused set of tools. Each tool is one useful action, such as `search_products`, `get_product`, `add_to_cart` or `list_orders`, not one tool per raw endpoint. If the request is broad ("everything"), cover the main read actions first, then the write actions you could verify.

**Runtime (already in place; do not edit or rewrite it):** `{scripts_dir}/revapi_mcp.py` is a dependency-free MCP framework. It uses only the Python standard library, so it runs on Termux with nothing to compile or install. Never use the `mcp`/`fastmcp` packages, pydantic, `requests` or `httpx`.

Write `{scripts_dir}/server.py` like this:

```python
from __future__ import annotations

from typing import Literal

from revapi_mcp import MCP, Session, ToolError

mcp = MCP("<short-site-name>", instructions="<one paragraph: what these tools do and when to use them>")
http = Session("https://api.example.com", headers={{"X-Client": "web"}})  # exact headers the browser sent


@mcp.tool(read_only=True)
def search_products(query: str, limit: int = 20, sort: Literal["relevance", "price"] = "relevance") -> dict:
    """Search the catalogue by keyword.

    Args:
        query: words to search for
        limit: number of results, 1-100
        sort: result order
    """
    data = http.get("/v2/search", params={{"q": query, "limit": limit, "sort": sort}}).json()
    return {{"total": data["total"], "items": [{{"id": p["id"], "name": p["title"], "price": p["price"]["amount"]}} for p in data["results"]]}}


if __name__ == "__main__":
    mcp.run()
```

Rules:
- **Schemas come from the type hints and docstrings.** Use `str`, `int`, `float`, `bool`, `list[str]`, `dict`, `Literal[...]` and `X | None = None`. The first docstring paragraph is the tool description, and `Args:` lines describe the parameters. Write them for a model that has never seen the site.
- **`Session` is a small requests-like client on urllib.** It has `get/post/put/patch/delete(path, params=, json=, data=, headers=)`, returning `.status`, `.headers`, `.text` and `.json()`. It also gives you:
  - a cookie jar, and `set_cookie(name, value)`;
  - retries on network errors, 429 and 5xx;
  - an `HTTPError` for 4xx/5xx (pass `allow_errors=True` to inspect the response yourself).
- **Saved sessions load automatically.** `Session` loads revapi's saved session for the host (`~/.revapi/sessions/<host>.json`: cookies + user agent) and raises `SessionExpired` with the renew command when the site shows a Cloudflare/CAPTCHA check again. Don't duplicate that logic.
- **Auth:** replay what the browser did. Hardcode the tokens and cookies you observed as defaults, and allow overriding them with environment variables (e.g. `os.environ.get("SITE_TOKEN", "<observed>")`). If you saw a login or token refresh flow, implement it inside the server, so tools re-authenticate on their own.
- **Return compact JSON.** Pick the useful fields instead of dumping huge raw payloads. Paginate with `page`/`cursor` parameters when the API does.
- **Errors:** raise `ToolError("clear message")` for bad input or empty results the model should know about. Never print to stdout: stdout carries the protocol, so log to stderr if needed.
- **Annotations:** mark tools that only read with `read_only=True`, and tools that delete or pay with `destructive=True`.

**Test before finishing.** Use the shell tool in `{scripts_dir}`:
- `python server.py --list`: every tool and its JSON schema.
- `python server.py --call <tool> '<json arguments>'`: run each tool against the live site and check the output is right. Fix it until every tool works.
- Check the stdio protocol with: `printf '%s\n' '{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-11-25","capabilities":{{}},"clientInfo":{{"name":"t","version":"1"}}}}}}' '{{"jsonrpc":"2.0","id":2,"method":"tools/list"}}' | python server.py`

**README.md:** what the server does, the tool list with one example each, and the auth/env variables. Include how to add it to a client, which revapi prints with `revapi mcp config <name>`. The command is `python {scripts_dir}/server.py`.
