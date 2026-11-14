"""LLM backends over plain urllib: any OpenAI-compatible endpoint + native Anthropic.

Internally messages use the OpenAI chat format:
  {"role": "system"|"user"|"assistant"|"tool", "content": str, "tool_calls": [...], "tool_call_id": str}
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
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
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "stepfun-ai/step-5-preview"),  # NVIDIA NIM (build.nvidia.com), keys start with nvapi-
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
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)  # Anthropic signed thinking, echoed back on tool use


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


# ----------------------------------------------------------------------------- transport + retries
#
# Every request goes through _send(). It retries anything that is not the caller's fault — DNS
# failures, dropped Wi-Fi, resets mid-stream, timeouts, 408/409/425/429, every 5xx, Cloudflare 52x,
# "overloaded" errors inside a 200 stream, empty or garbled bodies — up to `retries` times (default
# 40) with exponential backoff (1.5s → 60s cap, ±15% jitter) and the server's Retry-After when it
# sends one. Free-tier "retry after Ns" throttles are waited out without spending an attempt.
# Auth errors, unknown models and bad requests fail at once: retrying cannot fix them.

DEFAULT_RETRIES = 40
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 507, 508, 509, 520, 521, 522, 523, 524, 525, 526, 527, 529, 530, 598, 599}
RESAMPLE_HINTS = ("unexpected tokens", "UpstreamError", "failed to parse", "Failed to parse", "invalid tool call", "Invalid tool call")
CONTEXT_HINTS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
    "input is too long",
    "tokens exceed",
    "exceeds the maximum",
    "model_context_window_exceeded",
    "string too long",
)
OVERLOAD_HINTS = ("overloaded", "rate limit", "rate_limit", "capacity", "temporarily", "try again", "timeout", "unavailable", "internal error", "server_error")
NETWORK_ERRORS: tuple[type[BaseException], ...] = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException, ssl.SSLError, OSError)

RetryHook = Callable[[int, int, float, str], None]  # (retry number, max retries, wait seconds, reason)
DeltaHook = Callable[[str, Any], None]  # ("text"|"thinking"|"tool"|"reset", payload)


class StreamDropped(Exception):
    """The connection died after the response started; the request is repeated from scratch."""


def _backoff(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None and retry_after >= 0:
        return min(300.0, retry_after + 0.5)
    return min(60.0, 1.5 * 2 ** min(attempt - 1, 6)) * random.uniform(0.85, 1.15)


def _retry_after(headers: Any, text: str) -> float | None:
    v = (headers.get("retry-after") if headers else None) or ""
    if re.fullmatch(r"\d+(\.\d+)?", v.strip()):
        return float(v)
    m = re.search(r"[Rr]etry (?:after|in) (\d+(?:\.\d+)?)\s*s", text)
    return float(m.group(1)) if m else None


def is_context_error(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in CONTEXT_HINTS) and not ("rate" in low and "limit" in low and "per minute" in low)


def window_from_error(text: str) -> int | None:
    """'maximum context length is 131072 tokens' -> 131072, so the next compaction aims right."""
    for m in re.finditer(r"(?:context(?: length| window)?|maximum|max|limit)[^0-9]{0,60}?(\d{1,3}(?:[,_]\d{3})+|\d{4,7})\s*(?:tokens)?", text, re.I):
        n = int(re.sub(r"[,_]", "", m.group(1)))
        if 2048 <= n <= 20_000_000:
            return n
    return None


def _send(
    url: str,
    body: dict,
    headers: dict[str, str],
    timeout: float,
    *,
    consume: Callable[[Any], Any],
    retries: int = DEFAULT_RETRIES,
    on_retry: RetryHook | None = None,
    on_reset: Callable[[], None] | None = None,
) -> Any:
    data = json.dumps(body).encode()
    attempt = 0  # failed attempts so far
    throttled = 0
    resampled = 0

    def again(reason: str, kind: str, wait: float | None = None) -> None:
        nonlocal attempt
        attempt += 1
        if attempt > retries:
            raise LLMError(f"{reason} (gave up after {retries} retries)", kind)
        delay = _backoff(attempt, wait)
        (on_retry or (lambda a, n, w, r: _note(f"{r}; retry {a}/{n} in {w:.0f}s")))(attempt, retries, delay, reason)
        _sleep(delay)

    while True:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "revapi", **headers}, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            try:
                text = e.read().decode(errors="replace")[:4000]
            except Exception:  # noqa: BLE001
                text = ""
            wait = _retry_after(e.headers, text)
            if e.code in (402, 429) and wait is not None and throttled < 400 and re.search(r"[Rr]etry (?:after|in) \d", text):
                throttled += 1  # free tiers say exactly how long to wait; not a failure
                (on_retry or (lambda a, n, w, r: _note(f"{r}; waiting {w:.0f}s")))(0, retries, wait + 1, f"provider throttled (HTTP {e.code})")
                _sleep(wait + 1)
                continue
            if e.code in (400, 413, 422) and is_context_error(text):
                err = LLMError(f"context window exceeded: {text[:600]}", "context_length")
                err.window = window_from_error(text)  # type: ignore[attr-defined]
                raise err from None
            if e.code in (400, 422) and any(k in text for k in RESAMPLE_HINTS) and resampled < 6:
                resampled += 1  # the provider choked on the model's own malformed generation — sample again
                again(f"provider could not parse the model output (HTTP {e.code})", "engine_failure", 1.5)
                continue
            if e.code in (400, 404, 422):
                err = LLMError(f"HTTP {e.code}: {text}", "bad_request" if e.code != 404 else "config")
                err.status, err.body = e.code, text  # type: ignore[attr-defined]
                raise err from None
            if e.code in (401, 403) and not any(h in text.lower() for h in ("cloudflare", "temporar", "overload")):
                raise LLMError(f"auth failed ({e.code}): {text[:600]}", "auth") from None
            if e.code in RETRYABLE_HTTP or e.code >= 500 or e.code in (401, 403):
                again(f"HTTP {e.code} from provider", "rate_limit" if e.code == 429 else "engine_failure", wait)
                continue
            raise LLMError(f"HTTP {e.code}: {text}", "engine_failure") from None
        except NETWORK_ERRORS as e:
            again(f"network error ({type(e).__name__}: {str(e)[:120] or 'no detail'})", "network")
            continue
        try:
            with resp:
                return consume(resp)
        except StreamDropped as e:
            if on_reset:
                on_reset()
            again(f"stream interrupted ({e})", "network")
        except LLMError as e:
            if getattr(e, "retryable", False):
                if on_reset:
                    on_reset()
                again(str(e), e.kind)
                continue
            raise
        except json.JSONDecodeError as e:
            again(f"garbled response from provider ({e})", "engine_failure")
        except NETWORK_ERRORS as e:
            if on_reset:
                on_reset()
            again(f"connection lost mid-response ({type(e).__name__}: {str(e)[:120]})", "network")


def _read_json(resp: Any) -> dict:
    raw = resp.read()
    if not raw.strip():
        err = LLMError("empty response from provider")
        err.retryable = True  # type: ignore[attr-defined]
        raise err
    return json.loads(raw.decode())


def _sleep(seconds: float) -> None:
    """Wait in short slices, so stopping a run (esc in the session) never waits out a long backoff."""
    end = time.monotonic() + max(0.0, seconds)
    while (left := end - time.monotonic()) > 0:
        time.sleep(min(0.2, left))


def _post(url: str, body: dict, headers: dict[str, str], timeout: float, retries: int = DEFAULT_RETRIES, on_retry: RetryHook | None = None) -> dict:
    """Plain JSON POST with the full retry policy."""
    return _send(url, body, headers, timeout, consume=_read_json, retries=retries, on_retry=on_retry)


def _sse(resp: Any, state: dict[str, Any] | None = None) -> Any:
    """Yield the JSON payload of each server-sent event; plain-JSON answers are yielded whole.
    ``state["done"]`` becomes True when the server ends the stream with [DONE]."""
    state = state if state is not None else {}
    ctype = (resp.headers.get("content-type") or "").lower() if getattr(resp, "headers", None) else ""
    if ctype and "event-stream" not in ctype and "ndjson" not in ctype:
        yield {"__whole__": _read_json(resp)}
        return
    buf: list[str] = []
    got_any = False
    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if buf:
                    payload = "\n".join(buf)
                    buf = []
                    if payload.strip() == "[DONE]":
                        state["done"] = True
                        return
                    got_any = True
                    yield json.loads(payload)
                continue
            if line.startswith(":"):
                continue  # keep-alive comment
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
            elif not line.startswith(("event:", "id:", "retry:")) and line.lstrip().startswith("{"):
                got_any = True
                yield json.loads(line)  # some servers stream bare NDJSON
        if buf and buf[0].strip() == "[DONE]":
            state["done"] = True
        elif buf:
            got_any = True
            yield json.loads("\n".join(buf))
    except (http.client.IncompleteRead, ConnectionError, socket.timeout, TimeoutError, ssl.SSLError) as e:
        raise StreamDropped(f"{type(e).__name__}") from None
    if not got_any:
        raise StreamDropped("stream ended before any data")


class ThinkSplitter:
    """Routes <think>…</think> spans in streamed content to the thinking channel (DeepSeek-R1, Qwen, Kimi…)."""

    OPEN = ("<think>", "<thinking>")
    CLOSE = ("</think>", "</thinking>")

    def __init__(self, emit: Callable[[str, str], None]) -> None:
        self.emit = emit
        self.inside = False
        self.buf = ""
        self.text = ""
        self.think = ""

    def feed(self, s: str) -> None:
        self.buf += s
        while self.buf:
            # an opening tag only counts before the answer starts, so code that mentions <think> stays text
            tags = self.CLOSE if self.inside else (self.OPEN if not self.text.strip() else ())
            hits = [(self.buf.find(t), t) for t in tags if t in self.buf]
            if not self.inside:  # an opening tag counts only when nothing but whitespace precedes it
                hits = [(p, t) for p, t in hits if not (self.text + self.buf[:p]).strip()]
            if hits:
                pos, tag = min(hits)
                self._out(self.buf[:pos])
                self.buf = self.buf[pos + len(tag) :]
                self.inside = not self.inside
                continue
            keep = max((k for t in tags for k in range(1, len(t)) if self.buf.endswith(t[:k])), default=0)
            self._out(self.buf[: len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep :]
            break

    def _out(self, s: str) -> None:
        if not s:
            return
        if self.inside:
            self.think += s
            self.emit("thinking", s)
        else:
            if not self.text:
                s = s.lstrip()  # whitespace between </think> and the answer
                if not s:
                    return
            self.text += s
            self.emit("text", s)

    def flush(self) -> None:
        if self.buf:
            b, self.buf = self.buf, ""
            self._out(b)


def split_think(content: str, reasoning: str) -> tuple[str, str]:
    """Final clean-up for non-streamed replies: inline <think> blocks, or a lone </think>, become reasoning."""
    m = re.match(r"\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*(.*)\Z", content, re.S)
    if m:
        return m.group(2), (reasoning + "\n" + m.group(1)).strip()
    if "</think>" in content and "<think>" not in content:
        a, _, b = content.partition("</think>")
        return b.lstrip(), (reasoning + "\n" + a).strip()
    return content, reasoning


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


# Context windows (tokens) by model-id prefix; longest prefix wins. Unknown models get DEFAULT_WINDOW,
# and a provider's "maximum context length is N" error teaches the real number at runtime.
CONTEXT_WINDOWS: dict[str, int] = {
    "claude": 200_000,
    "claude-sonnet-4": 200_000,
    "gpt-4.1": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-5": 400_000,
    "o3": 200_000,
    "o4": 200_000,
    "gpt-oss": 131_072,
    "gemini": 1_000_000,
    "deepseek": 128_000,
    "deepseek-v4": 1_000_000,
    "kimi": 256_000,
    "glm": 200_000,
    "minimax": 1_000_000,
    "qwen3": 256_000,
    "qwen": 128_000,
    "grok-4": 256_000,
    "grok": 128_000,
    "mistral-large": 128_000,
    "llama-4": 1_000_000,
    "llama": 128_000,
    "nemotron": 128_000,
    "nemotron-3": 1_000_000,
    "step": 256_000,
    "big-pickle": 200_000,
    "openai": 128_000,
}
DEFAULT_WINDOW = 128_000
# RAE's five efforts. OpenAI-style APIs take low|medium|high, so xhigh and max ask for "high" there; Anthropic gets
# a thinking budget per level.
THINKING_BUDGET = {"low": 2048, "medium": 8000, "high": 24000, "xhigh": 32000, "max": 48000}


def wire_effort(effort: str) -> str:
    return "high" if effort in ("xhigh", "max") else effort


OPTIONAL_KEYS = ("stream_options", "chat_template_kwargs", "reasoning", "include_reasoning", "reasoning_effort", "tool_choice", "parallel_tool_calls", "usage")
INTERNAL_KEYS = ("reasoning", "thinking_blocks", "_seq", "seq", "ts", "usage")


def context_window(model: str) -> int:
    m = model.split("/")[-1].lower()
    best = max((k for k in CONTEXT_WINDOWS if m.startswith(k) or f"/{k}" in model.lower()), key=len, default=None)
    return CONTEXT_WINDOWS[best] if best else DEFAULT_WINDOW


def wire(m: dict) -> dict:
    """A message as providers see it: our bookkeeping keys removed."""
    return {k: v for k, v in m.items() if k not in INTERNAL_KEYS and not k.startswith("_")}


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
        stream: bool | None = None,
        retries: int | None = None,
        context: int | None = None,
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
        self.stream = (os.environ.get("REVAPI_NO_STREAM") is None) if stream is None else stream
        self.retries = max(1, int(retries or os.environ.get("REVAPI_RETRIES") or DEFAULT_RETRIES))
        self.context_window = int(context or os.environ.get("REVAPI_CONTEXT_WINDOW") or context_window(self.model))
        self._dropped: set[str] = set()  # optional request fields this provider rejected
        if not self.base_url:
            raise LLMError("custom provider needs base_url (config `base_url` or REVAPI_BASE_URL)", "config")
        if require_key and env and not self.api_key and provider not in ("pollinations", "custom"):
            raise LLMError(f"{provider} needs an API key: export {env}=... (or `revapi config set api_key ...`)", "auth")

    def describe(self) -> str:
        return f"{self.provider}/{self.model}"

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: DeltaHook | None = None,
        on_retry: RetryHook | None = None,
        stream: bool | None = None,
    ) -> Reply:
        """One model turn. Streams text/thinking/tool-call progress to ``on_delta`` when given."""
        use_stream = self.stream if stream is None else stream
        for _ in range(len(OPTIONAL_KEYS) + 3):
            try:
                if self.provider == "anthropic":
                    return self._anthropic(messages, tools, on_delta if use_stream else None, on_retry, use_stream)
                return self._openai(messages, tools, on_delta if use_stream else None, on_retry, use_stream)
            except LLMError as e:
                text = getattr(e, "body", "") or str(e)
                if e.kind != "bad_request":
                    raise
                # a provider that rejects one of our optional fields: drop it and ask again
                low = text.lower()
                bad = [k for k in OPTIONAL_KEYS if k in low and k not in self._dropped]
                if "stream_options" in low:
                    bad = ["stream_options"]
                if bad:
                    self._dropped.update(bad)
                    continue
                if use_stream and re.search(r"\bstream(ing)?\b.{0,40}(not supported|unsupported|invalid)|(not support|unsupported).{0,40}\bstream", low):
                    use_stream = self.stream = False
                    continue
                if any(h in low for h in ("extra inputs", "extra fields", "unrecognized", "unknown field", "unknown parameter", "additional properties", "not permitted")):
                    before = len(self._dropped)
                    self._dropped.update(OPTIONAL_KEYS)
                    if len(self._dropped) > before:
                        continue
                if self.max_tokens > 4096 and re.search(r"max_(completion_)?tokens|max_new_tokens|maximum.*tokens|tokens.*exceed", text, re.I):
                    self.max_tokens = 4096  # some hosted models cap output lower than our default
                    continue
                raise
        raise LLMError("provider kept rejecting the request", "bad_request")

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
    def _openai_body(self, messages: list[dict], tools: list[dict] | None, stream: bool) -> dict[str, Any]:
        msgs = []
        for m in messages:
            m = wire(m)
            if m.get("role") == "tool" and "name" in m:
                m = {**m, "name": clean_name(m["name"])}
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [{**tc, "function": {**tc["function"], "name": clean_name(tc["function"]["name"])}} for tc in m["tool_calls"]]}
            msgs.append(m)
        body: dict[str, Any] = {"model": self.model, "messages": msgs}
        opt: dict[str, Any] = {}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            opt["tool_choice"] = "auto"
        if self.provider not in ("pollinations",):
            body["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.max_tokens
        if self.effort:
            level = wire_effort(self.effort)
            if self.provider == "openrouter":
                opt["reasoning"] = {"effort": level}
            elif self.provider in ("openai", "groq", "xai", "gemini", "cerebras", "together", "custom"):
                opt["reasoning_effort"] = level
            elif self.provider in ("nvidia", "ollama", "lmstudio"):
                opt["chat_template_kwargs"] = {"thinking": True, "enable_thinking": True}
        if self.provider == "openrouter":
            opt["usage"] = {"include": True}
            opt["include_reasoning"] = True
        if stream:
            body["stream"] = True
            opt["stream_options"] = {"include_usage": True}
        body.update({k: v for k, v in opt.items() if k not in self._dropped})
        return body

    def _openai(self, messages: list[dict], tools: list[dict] | None, on_delta: DeltaHook | None, on_retry: RetryHook | None, stream: bool) -> Reply:
        body = self._openai_body(messages, tools, stream)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if stream:
            headers["Accept"] = "text/event-stream"
        url = self.base_url if self.provider == "pollinations" else self.base_url.rstrip("/") + "/chat/completions"
        emit = on_delta or (lambda kind, payload: None)
        if not stream:
            data = _send(url, body, headers, self.timeout, consume=_read_json, retries=self.retries, on_retry=on_retry)
            return self._openai_reply(data)

        def consume(resp: Any) -> Reply:
            split = ThinkSplitter(lambda kind, s: emit(kind, s))
            reasoning = ""
            calls: dict[int, dict[str, str]] = {}
            usage: dict[str, Any] = {}
            finish = ""
            state: dict[str, Any] = {}
            for ev in _sse(resp, state):
                if "__whole__" in ev:  # the server ignored stream=true
                    reply = self._openai_reply(ev["__whole__"])
                    if reply.reasoning:
                        emit("thinking", reply.reasoning)
                    if reply.content:
                        emit("text", reply.content)
                    return reply
                if ev.get("error"):
                    err_text = json.dumps(ev["error"])[:800]
                    err = LLMError(f"provider error mid-stream: {err_text}")
                    err.retryable = any(h in err_text.lower() for h in OVERLOAD_HINTS) or bool(re.search(r'"code":\s*"?(429|5\d\d)', err_text))  # type: ignore[attr-defined]
                    if is_context_error(err_text):
                        err.kind = "context_length"
                        err.window = window_from_error(err_text)  # type: ignore[attr-defined]
                        err.retryable = False  # type: ignore[attr-defined]
                    raise err
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or ch.get("message") or {}
                    r = d.get("reasoning_content") or d.get("reasoning") or ""
                    if not r and isinstance(d.get("reasoning_details"), list):
                        r = "".join(x.get("text") or x.get("summary") or "" for x in d["reasoning_details"] if isinstance(x, dict))
                    if r:
                        reasoning += r
                        emit("thinking", r)
                    if d.get("content"):
                        split.feed(d["content"])
                    for tc in d.get("tool_calls") or []:
                        idx = tc.get("index")
                        if idx is None:
                            idx = next((i for i, c in calls.items() if tc.get("id") and c["id"] == tc.get("id")), len(calls))
                        slot = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        name = fn.get("name") or ""
                        if name and not slot["name"].endswith(name):
                            slot["name"] = name if name.startswith(slot["name"]) else slot["name"] + name
                        a = fn.get("arguments")
                        if isinstance(a, dict):
                            slot["args"] = json.dumps(a)
                        elif a:
                            slot["args"] += a
                        emit("tool", {"index": idx, "name": clean_name(slot["name"]), "chars": len(slot["args"]), "args": slot["args"]})
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
            if not finish and not state.get("done"):
                raise StreamDropped("stream ended before the reply was complete")  # cut mid-reply: ask again
            split.flush()
            reply = Reply(content=split.text, reasoning=(reasoning + split.think).strip(), finish_reason=finish)
            for _, c in sorted(calls.items()):
                if c["name"] or c["args"]:
                    reply.tool_calls.append({"id": c["id"] or f"call_{len(reply.tool_calls)}", "name": clean_name(c["name"]), "arguments": _parse_args(c["args"])})
            if not reply.content and not reply.tool_calls and not reply.reasoning and not finish:
                raise StreamDropped("stream closed without a reply")
            reply.usage = self._usage(usage, messages, reply)
            return reply

        return _send(url, body, headers, self.timeout, consume=consume, retries=self.retries, on_retry=on_retry, on_reset=lambda: emit("reset", None))

    def _openai_reply(self, data: dict) -> Reply:
        if "error" in data and not data.get("choices"):
            err_text = json.dumps(data["error"])[:800]
            err = LLMError(f"provider error: {err_text}")
            if is_context_error(err_text):
                err.kind = "context_length"
                err.window = window_from_error(err_text)  # type: ignore[attr-defined]
            else:
                err.retryable = any(h in err_text.lower() for h in OVERLOAD_HINTS)  # type: ignore[attr-defined]
            raise err
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content, reasoning = split_think(msg.get("content") or "", msg.get("reasoning") or msg.get("reasoning_content") or "")
        reply = Reply(content=content, reasoning=reasoning, finish_reason=ch.get("finish_reason") or "")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            reply.tool_calls.append({"id": tc.get("id") or f"call_{len(reply.tool_calls)}", "name": clean_name(fn.get("name", "")), "arguments": _parse_args(fn.get("arguments"))})
        reply.usage = self._usage(data.get("usage") or {}, None, reply)
        return reply

    def _usage(self, u: dict, messages: list[dict] | None, reply: Reply) -> Usage:
        usage = Usage(
            input_tokens=u.get("prompt_tokens", 0) or 0,
            output_tokens=u.get("completion_tokens", 0) or 0,
            reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            cache_read_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
            requests=1,
        )
        if not usage.input_tokens and messages is not None:  # provider sent no usage: estimate (~3.5 chars/token)
            usage.input_tokens = int(sum(len(json.dumps(wire(m), ensure_ascii=False)) for m in messages) / 3.5)
            usage.output_tokens = int((len(reply.content) + len(reply.reasoning) + len(json.dumps(reply.tool_calls))) / 3.5)
        usage.estimated_cost_usd = float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else _price(self.model, usage)
        return usage

    # -- Anthropic ------------------------------------------------------------
    def _anthropic_body(self, messages: list[dict], tools: list[dict] | None, stream: bool) -> dict[str, Any]:
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
                # extended thinking must travel back with tool use, signature intact
                for tb in m.get("thinking_blocks") or []:
                    if tb.get("type") == "redacted_thinking":
                        blocks.append({"type": "redacted_thinking", "data": tb.get("data", "")})
                    elif tb.get("signature"):
                        blocks.append({"type": "thinking", "thinking": tb.get("thinking", ""), "signature": tb["signature"]})
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
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": conv}
        if system:
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            body["tools"] = [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]} for t in tools]
            body["tools"][-1]["cache_control"] = {"type": "ephemeral"}
        if conv and isinstance(conv[-1]["content"], list) and conv[-1]["content"]:
            conv[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        if self.effort in THINKING_BUDGET:
            budget = THINKING_BUDGET[self.effort]
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(self.max_tokens, budget + 4000)
        if stream:
            body["stream"] = True
        return body

    def _anthropic(self, messages: list[dict], tools: list[dict] | None, on_delta: DeltaHook | None, on_retry: RetryHook | None, stream: bool) -> Reply:
        body = self._anthropic_body(messages, tools, stream)
        url = self.base_url.rstrip("/") + "/messages"
        headers = {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}
        emit = on_delta or (lambda kind, payload: None)
        if not stream:
            return self._anthropic_reply(_send(url, body, headers, self.timeout, consume=_read_json, retries=self.retries, on_retry=on_retry))

        def consume(resp: Any) -> Reply:
            blocks: dict[int, dict[str, Any]] = {}
            u: dict[str, Any] = {}
            stop = ""
            ended = False
            for ev in _sse(resp):
                if "__whole__" in ev:
                    return self._anthropic_reply(ev["__whole__"])
                t = ev.get("type")
                if t == "error":
                    err_text = json.dumps(ev.get("error"))[:800]
                    err = LLMError(f"provider error mid-stream: {err_text}")
                    err.retryable = any(h in err_text.lower() for h in OVERLOAD_HINTS + ("api_error",))  # type: ignore[attr-defined]
                    raise err
                if t == "message_start":
                    u.update((ev.get("message") or {}).get("usage") or {})
                elif t == "content_block_start":
                    b = dict(ev.get("content_block") or {})
                    b.setdefault("text", "")
                    b.setdefault("thinking", "")
                    b["json"] = ""
                    blocks[ev.get("index", len(blocks))] = b
                elif t == "content_block_delta":
                    b = blocks.setdefault(ev.get("index", 0), {"type": "text", "text": "", "thinking": "", "json": ""})
                    d = ev.get("delta") or {}
                    dt = d.get("type")
                    if dt == "text_delta":
                        b["text"] += d.get("text", "")
                        emit("text", d.get("text", ""))
                    elif dt == "thinking_delta":
                        b["thinking"] += d.get("thinking", "")
                        emit("thinking", d.get("thinking", ""))
                    elif dt == "signature_delta":
                        b["signature"] = b.get("signature", "") + d.get("signature", "")
                    elif dt == "input_json_delta":
                        b["json"] += d.get("partial_json", "")
                        emit("tool", {"index": ev.get("index", 0), "name": clean_name(b.get("name", "")), "chars": len(b["json"]), "args": b["json"]})
                elif t == "message_delta":
                    stop = (ev.get("delta") or {}).get("stop_reason") or stop
                    u.update(ev.get("usage") or {})
                elif t == "message_stop":
                    ended = True
            if not stop and not ended:
                raise StreamDropped("stream ended before the reply was complete")
            content = []
            for _, b in sorted(blocks.items()):
                if b.get("type") == "tool_use":
                    content.append({"type": "tool_use", "id": b.get("id"), "name": b.get("name"), "input": _parse_args(b["json"]) if b["json"] else (b.get("input") or {})})
                elif b.get("type") == "thinking":
                    content.append({"type": "thinking", "thinking": b["thinking"], "signature": b.get("signature", "")})
                elif b.get("type") == "redacted_thinking":
                    content.append({"type": "redacted_thinking", "data": b.get("data", "")})
                else:
                    content.append({"type": "text", "text": b["text"]})
            if not content and not stop:
                raise StreamDropped("stream closed without a reply")
            return self._anthropic_reply({"content": content, "stop_reason": stop, "usage": u})

        return _send(url, body, headers, self.timeout, consume=consume, retries=self.retries, on_retry=on_retry, on_reset=lambda: emit("reset", None))

    def _anthropic_reply(self, data: dict) -> Reply:
        if data.get("type") == "error" or ("error" in data and not data.get("content")):
            raise LLMError(f"provider error: {json.dumps(data.get('error'))[:800]}")
        reply = Reply(finish_reason=data.get("stop_reason") or "")
        for b in data.get("content", []):
            if b["type"] == "text":
                reply.content += b["text"]
            elif b["type"] == "thinking":
                reply.reasoning += b.get("thinking", "")
                reply.thinking_blocks.append({"type": "thinking", "thinking": b.get("thinking", ""), "signature": b.get("signature", "")})
            elif b["type"] == "redacted_thinking":
                reply.thinking_blocks.append({"type": "redacted_thinking", "data": b.get("data", "")})
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
