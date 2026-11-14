"""Prompt templates (.md). Partials adapted from reverse-api-engineer (MIT, (c) 2025 kalil0321)."""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


class _Keep(dict):
    def __missing__(self, key: str) -> str:  # leave unknown {placeholders} untouched
        return "{" + key + "}"


def load(name: str, **kwargs: object) -> str:
    text = (_DIR / f"{name}.md").read_text(encoding="utf-8")
    while "{include:" in text:
        a = text.index("{include:")
        b = text.index("}", a)
        text = text[:a] + (_DIR / "partials" / f"{text[a + 9 : b]}.md").read_text(encoding="utf-8") + text[b + 1 :]
    return text.format_map(_Keep({k: str(v) for k, v in kwargs.items()}))
