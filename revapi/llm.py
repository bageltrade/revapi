"""LLM backends over plain urllib: any OpenAI-compatible endpoint + native Anthropic.

Internally messages use the OpenAI chat format:
  {"role": "system"|"user"|"assistant"|"tool", "content": str, "tool_calls": [...], "tool_call_id": str}
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# name -> (base_url, env var holding the key, default model)
PROVIDERS: dict[str, tuple[str, str | None, str]] = {
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-sonnet-4-5"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "anthropic/claude-sonnet-4.5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-large-latest"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY", "grok-4"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "gpt-oss-120b"),
    "opencode": ("https://opencode.ai/zen/v1", "OPENCODE_API_KEY", "big-pickle"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "moonshotai/kimi-k2.6"),  # NVIDIA NIM (build.nvidia.com), keys start with nvapi-
    "ollama": ("http://127.0.0.1:11434/v1", None, "qwen2.5-coder:7b"),
    "lmstudio": ("http://127.0.0.1:1234/v1", None, "local-model"),
    "pollinations": ("https://text.pollinations.ai/openai", "POLLINATIONS_TOKEN", "openai"),  # free, no key needed
    "custom": ("", "REVAPI_API_KEY", ""),
}

# USD per 1M tokens (input, output) — rough public list prices, used for estimates only.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15, 75),
    "claude-opus-4-1": (15, 75),
    "claude-opus-4-5": (5, 25),
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4": (3, 15),
    "claude-sonnet-4-5": (3, 15),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-5": (1.25, 10),
    "gpt-5-mini": (0.25, 2),
    "o4-mini": (1.1, 4.4),
    "gemini-2.5-pro": (1.25, 10),
    "gemini-2.5-flash": (0.3, 2.5),
    "deepseek-chat": (0.27, 1.1),
    "deepseek-reasoner": (0.55, 2.19),
    "grok-4": (3, 15),
    "mistral-large-latest": (2, 6),
}


class LLMError(RuntimeError):
    def __init__(self, msg: str, kind: str = "engine_failure") -> None:
        super().__init__(msg)
        self.kind = kind


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    requests: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["estimated_cost_usd"] = round(d["estimated_cost_usd"], 6)
        d["cost"] = d["estimated_cost_usd"]
        return d


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"id","name","arguments"(dict)}]
    reasoning: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


def _price(model: str, u: Usage) -> float:
    m = model.split("/")[-1].lower()
    best = max((k for k in PRICING if m.startswith(k)), key=len, default=None)
    if not best:
        return 0.0
    i, o = PRICING[best]
    return ((u.input_tokens - u.cache_read_tokens) * i + u.cache_read_tokens * i * 0.1 + u.output_tokens * o) / 1e6


def _note(msg: str) -> None:
    if os.environ.get("REVAPI_QUIET_RETRIES"):
        return
    import sys

    print(f"  \033[2m… {msg}\033[0m" if sys.stderr.isatty() else f"  … {msg}", file=sys.stderr, flush=True)


def _post(url: str, body: dict, headers: dict[str, str], timeout: float) -> dict:
    data = json.dumps(body).encode()
    last: Exception | None = None
    attempt = 0
    throttled = 0
    while attempt < 6:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "revapi", **headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")[:1500]
            m = re.search(r"[Rr]etry after (\d+(?:\.\d+)?)\s*s", text)
            if e.code in (402, 429) and m and throttled < 200:
                # free tiers (e.g. Pollinations: 1 reply / 45s) tell us exactly how long to wait
                throttled += 1
                wait = float(m.group(1)) + 1
                _note(f"provider throttled (HTTP {e.code}); waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if e.code == 400 and any(k in text for k in ("unexpected tokens", "UpstreamError", "failed to parse", "Failed to parse")) and attempt < 4:
                # the provider choked on the model's own (malformed) generation — resample
                attempt += 1
                _note("provider failed to parse the model output; resampling")
                time.sleep(2)
                last = LLMError(f"HTTP 400: {text}")
                continue
            if e.code in (401, 403):
                raise LLMError(f"auth failed ({e.code}): {text}", "auth") from None
            if e.code in (429, 500, 502, 503, 504, 520, 522, 524, 529) and attempt < 5:
                retry = e.headers.get("retry-after")
                wait = min(60, float(retry)) if retry and retry.replace(".", "").isdigit() else 2 * 2**attempt
                _note(f"HTTP {e.code} from provider; retrying in {wait:.0f}s")
                time.sleep(wait)
                attempt += 1
                last = LLMError(f"HTTP {e.code}: {text}", "rate_limit" if e.code == 429 else "engine_failure")
                continue
            raise LLMError(f"HTTP {e.code}: {text}", "rate_limit" if e.code == 429 else "engine_failure") from None
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
            last = LLMError(f"network error: {e}", "network")
            _note(f"network error ({e}); retrying")
            attempt += 1
            if attempt < 6:
                time.sleep(2 * 2**attempt)
                continue
            break
        except json.JSONDecodeError as e:
            last = LLMError(f"bad JSON from provider: {e}")
            attempt += 1
            if attempt < 3:
                time.sleep(2)
                continue
            break
    raise last or LLMError("request failed")


def clean_name(name: str) -> str:
    """Strip leaked chat-template tokens (e.g. 'tool<|channel|>commentary') and invalid chars."""
    name = re.split(r"<\||\s", name or "", maxsplit=1)[0]
    return re.sub(r"[^A-Za-z0-9_-]", "", name)[:64] or "unknown_tool"


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    s = str(raw).strip()
    # weaker models emit almost-JSON: literal newlines, invalid escapes, an escaped closing quote, missing brace...
    candidates = [s]
    if s.endswith('\\"}'):
        candidates.append(s[:-3] + '"}')
    candidates += [s + '"}', s + "}", re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", s)]
    end = s.rfind("}")
    if end > 0:
        candidates.append(s[: end + 1])
    if s.startswith("```"):
        candidates.append(s.strip("`").removeprefix("json").strip())
    for cand in candidates:
        for strict in (True, False):
            try:
                v = json.loads(cand, strict=strict)
            except json.JSONDecodeError:
                continue
            return v if isinstance(v, dict) else {"value": v}
    return {"_raw": s}


class LLM:
    def __init__(
        self,
        provider: str = "auto",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        effort: str | None = None,
        timeout: float = 300,
        max_tokens: int = 16000,
        require_key: bool = True,
    ) -> None:
        if provider == "auto":
            provider = "nvidia" if (api_key or "").startswith("nvapi-") else detect_provider()
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}", "config")
        default_base, env, default_model = PROVIDERS[provider]
        self.provider = provider
        self.base_url = (base_url or os.environ.get("REVAPI_BASE_URL") if provider == "custom" else base_url) or default_base
        if provider == "ollama":
            self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/v1"
        self.api_key = (
            api_key or (os.environ.get(env) if env else None) or (os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else None) or (os.environ.get("NVAPI_KEY") if provider == "nvidia" else None)
        )
        self.model = model or default_model
        self.effort = effort
        self.timeout = timeout
        self.max_tokens = max_tokens
        if not self.base_url:
            raise LLMError("custom provider needs base_url (config `base_url` or REVAPI_BASE_URL)", "config")
        if require_key and env and not self.api_key and provider not in ("pollinations", "custom"):
            raise LLMError(f"{provider} needs an API key: export {env}=... (or `revapi config set api_key ...`)", "auth")

    def describe(self) -> str:
        return f"{self.provider}/{self.model}"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        if self.provider == "anthropic":
            return self._anthropic(messages, tools)
        try:
            return self._openai(messages, tools)
        except LLMError as e:
            # some hosted models cap output tokens lower than our default -> retry smaller once
            if self.max_tokens > 4096 and re.search(r"max_(completion_)?tokens|max_new_tokens|maximum.*tokens|tokens.*exceed", str(e), re.I):
                self.max_tokens = 4096
                return self._openai(messages, tools)
            raise

    def list_models(self) -> list[str]:
        """Model ids from the provider's /models endpoint (OpenAI-compatible providers and Anthropic)."""
        if self.provider == "pollinations":
            url, headers = "https://text.pollinations.ai/models", {}
        elif self.provider == "anthropic":
            url, headers = self.base_url.rstrip("/") + "/models", {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}
        else:
            url, headers = self.base_url.rstrip("/") + "/models", ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        req = urllib.request.Request(url, headers={"User-Agent": "revapi", **headers})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        items = data if isinstance(data, list) else data.get("data") or data.get("models") or []
        return sorted(str(m.get("id") or m.get("name")) if isinstance(m, dict) else str(m) for m in items)

    # -- OpenAI-compatible ----------------------------------------------------
    def _openai(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        msgs = []
        for m in messages:
            if m.get("role") == "tool" and "name" in m:
                m = {**m, "name": clean_name(m["name"])}
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [{**tc, "function": {**tc["function"], "name": clean_name(tc["function"]["name"])}} for tc in m["tool_calls"]]}
            msgs.append(m)
        body: dict[str, Any] = {"model": self.model, "messages": msgs}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        if self.provider not in ("pollinations",):
            key = "max_completion_tokens" if self.provider == "openai" else "max_tokens"
            body[key] = self.max_tokens
        if self.effort and self.provider in ("openai", "openrouter", "groq", "xai", "gemini", "cerebras"):
            body["reasoning_effort"] = self.effort
        if self.provider == "openrouter":
            body["usage"] = {"include": True}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = self.base_url if self.provider == "pollinations" else self.base_url.rstrip("/") + "/chat/completions"
        data = _post(url, body, headers, self.timeout)
        if "error" in data and not data.get("choices"):
            raise LLMError(f"provider error: {json.dumps(data['error'])[:800]}")
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        reply = Reply(content=msg.get("content") or "", reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "", finish_reason=ch.get("finish_reason") or "")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            reply.tool_calls.append({"id": tc.get("id") or f"call_{len(reply.tool_calls)}", "name": clean_name(fn.get("name", "")), "arguments": _parse_args(fn.get("arguments"))})
        u = data.get("usage") or {}
        reply.usage = Usage(
            input_tokens=u.get("prompt_tokens", 0) or 0,
            output_tokens=u.get("completion_tokens", 0) or 0,
            reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            cache_read_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
            requests=1,
        )
        reply.usage.estimated_cost_usd = float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else _price(self.model, reply.usage)
        return reply

    # -- Anthropic ------------------------------------------------------------
    def _anthropic(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conv: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or "(empty)"}
                if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                    conv[-1]["content"].append(block)
                else:
                    conv.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": clean_name(tc["function"]["name"]), "input": _parse_args(tc["function"]["arguments"])})
                conv.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
            else:
                if conv and conv[-1]["role"] == "user":
                    c = conv[-1]["content"]
                    conv[-1]["content"] = (c if isinstance(c, list) else [{"type": "text", "text": c}]) + [{"type": "text", "text": m["content"]}]
                else:
                    conv.append({"role": "user", "content": m["content"]})
        # prompt caching on the system prompt and last message
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": conv}
        if system:
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            body["tools"] = [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]} for t in tools]
            body["tools"][-1]["cache_control"] = {"type": "ephemeral"}
        if conv and isinstance(conv[-1]["content"], list) and conv[-1]["content"]:
            conv[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        if self.effort in ("low", "medium", "high"):
            budget = {"low": 2048, "medium": 8000, "high": 24000}[self.effort]
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(self.max_tokens, budget + 4000)
        data = _post(self.base_url.rstrip("/") + "/messages", body, {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}, self.timeout)
        reply = Reply(finish_reason=data.get("stop_reason") or "")
        for b in data.get("content", []):
            if b["type"] == "text":
                reply.content += b["text"]
            elif b["type"] == "thinking":
                reply.reasoning += b.get("thinking", "")
            elif b["type"] == "tool_use":
                reply.tool_calls.append({"id": b["id"], "name": clean_name(b["name"]), "arguments": b.get("input") or {}})
        u = data.get("usage") or {}
        cr, cw = u.get("cache_read_input_tokens", 0) or 0, u.get("cache_creation_input_tokens", 0) or 0
        reply.usage = Usage(input_tokens=(u.get("input_tokens", 0) or 0) + cr + cw, output_tokens=u.get("output_tokens", 0) or 0, cache_read_tokens=cr, cache_creation_tokens=cw, requests=1)
        reply.usage.estimated_cost_usd = _price(self.model, reply.usage) + cw * PRICING.get(self.model, (0, 0))[0] * 0.25 / 1e6
        return reply


def detect_provider() -> str:
    """First provider with a key in the environment; else a local Ollama; else free Pollinations."""
    for name in ("anthropic", "openai", "openrouter", "gemini", "nvidia", "groq", "deepseek", "mistral", "xai", "together", "cerebras", "opencode"):
        env = PROVIDERS[name][1]
        if env and os.environ.get(env):
            return name
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("NVAPI_KEY"):
        return "nvidia"
    if os.environ.get("REVAPI_BASE_URL"):
        return "custom"
    try:
        with urllib.request.urlopen(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/tags", timeout=1):
            return "ollama"
    except Exception:  # noqa: BLE001
        pass
    return "pollinations"
