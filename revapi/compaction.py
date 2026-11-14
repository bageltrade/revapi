"""Context compaction that loses nothing.

When a conversation nears the model's context window, the older part is summarised by the model
itself, in the structured form Claude Code uses for /compact, adapted for API reverse engineering.
The verbatim history is never deleted: every message stays in the run's transcript (messages.jsonl)
under a sequence number, summaries cite those numbers, and the agent gets a ``recall`` tool that
searches the transcript and returns any earlier message or tool output in full.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SUMMARY_SYSTEM = """You are compacting the working memory of an autonomous API reverse-engineering agent.
The conversation below is being removed from its context window and replaced by your summary, so the
summary must let the agent continue the task exactly where it left off, without redoing any work.
Be precise and complete; prefer concrete facts (URLs, header names, field names, file paths, error
messages, code) over prose. Never invent anything that is not in the transcript."""

SUMMARY_PROMPT = """Write a detailed summary of the conversation so far, paying close attention to the
user's explicit requests and the agent's previous actions. First think through the transcript in
chronological order inside <analysis> tags, then write the summary inside <summary> tags with these
sections:

1. Primary Request and Intent — every explicit request of the user, in detail.
2. Target and Environment — sites, start URLs, accounts or logins used, output language and directory.
3. Discovered API Surface — every endpoint found so far: method, full URL or pattern, query/body
   parameters, required headers, auth mechanism (cookies, bearer tokens, CSRF, signatures) and where
   each token comes from, request/response shapes, pagination, rate limits. Cite transcript ids like
   [#57] for the tool outputs that show them.
4. Files and Code — every file written or edited, with its path, purpose and current state; include
   the important function signatures and any code snippets the agent will need again.
5. Errors and Fixes — every error met and how it was fixed, plus anything that did not work and
   should not be retried. Include user feedback verbatim.
6. Problem Solving — what has been verified to work (scripts run, tests passed) and open questions.
7. All User Messages — every message the user sent (not tool results), verbatim or nearly.
8. Pending Tasks — what is still left to do.
9. Current Work — precisely what was being done immediately before this summary, with file names
   and code where relevant.
10. Next Step — the single next step that continues the current work, in line with the user's most
   recent request. Quote the latest instruction if there is one.

{previous}<transcript>
{transcript}
</transcript>"""

CONTINUATION = """This session is being continued from a previous conversation that ran out of context. \
The earlier messages were replaced by the summary below. Nothing was lost: the full transcript is \
archived, and the `recall` tool returns any earlier message or tool output verbatim — `recall(id=57)` \
for an id cited as [#57], or `recall(query="authorization")` to search it. Use it before redoing work.

<original_request>
{original}
</original_request>

<summary>
{summary}
</summary>

Continue the task from where it left off without asking the user anything new; the next step is \
described above. Recent messages follow verbatim."""

PER_MESSAGE_CHARS = 2400  # how much of each old message the summariser sees (full text stays recallable)


def render_for_summary(messages: list[dict[str, Any]], per_message: int = PER_MESSAGE_CHARS) -> str:
    out = []
    for m in messages:
        seq = m.get("_seq")
        tag = f"#{seq} " if seq is not None else ""
        role = m.get("role")
        if role == "tool":
            body = m.get("content") or ""
            head = f"[{tag}tool result: {m.get('name', '')}]"
        elif role == "assistant":
            parts = [m.get("content") or ""]
            for tc in m.get("tool_calls") or []:
                parts.append(f"→ call {tc['function']['name']}({tc['function']['arguments']})")
            body = "\n".join(p for p in parts if p)
            head = f"[{tag}assistant]"
        else:
            body = m.get("content") or ""
            head = f"[{tag}{role}]"
        if len(body) > per_message:
            keep = per_message - 200
            body = body[: int(keep * 0.75)] + f"\n…[{len(body) - keep} chars omitted here; recall(id={seq}) has them]…\n" + body[-int(keep * 0.25) :]
        out.append(f"{head}\n{body}")
    return "\n\n".join(out)


def extract_summary(text: str) -> str:
    m = re.search(r"<summary>(.*?)(?:</summary>|\Z)", text, re.S)
    s = (m.group(1) if m else re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.S)).strip()
    return s


def shrink(content: str, seq: int | None, keep: int = 2000) -> str:
    """Head and tail of a long text, with a pointer to the archived original."""
    if len(content) <= keep + 200:
        return content
    pointer = f"recall(id={seq})" if seq is not None else "the transcript"
    return content[: int(keep * 0.75)] + f"\n…[{len(content) - keep:,} chars archived — {pointer} returns the full text]…\n" + content[-int(keep * 0.25) :]


# ----------------------------------------------------------------------------- recall


def read_transcript(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("seq") is not None:
            out.append(rec)
    return out


def _record_text(rec: dict[str, Any]) -> str:
    parts = [rec.get("content") or ""]
    for tc in rec.get("tool_calls") or []:
        parts.append(f"→ call {tc['function']['name']}({tc['function']['arguments']})")
    if rec.get("reasoning"):
        parts.append(f"(thinking) {rec['reasoning']}")
    return "\n".join(p for p in parts if p)


def _label(rec: dict[str, Any]) -> str:
    role = rec.get("role", "?")
    return f"tool result: {rec.get('name', '')}" if role == "tool" else role


def recall(path: Path, id: int | None = None, query: str | None = None, offset: int = 0, max_chars: int = 20000) -> str:  # noqa: A002
    recs = read_transcript(path)
    if not recs:
        return "the transcript is empty"
    if id is not None:
        rec = next((r for r in recs if r.get("seq") == id), None)
        if rec is None:
            return f"no message #{id}; ids run from {recs[0]['seq']} to {recs[-1]['seq']}"
        text = _record_text(rec)
        chunk = text[offset : offset + max_chars]
        more = len(text) - offset - len(chunk)
        tail = f"\n…[{more:,} more chars — recall(id={id}, offset={offset + len(chunk)})]" if more > 0 else ""
        return f"#{id} {_label(rec)} ({len(text):,} chars)\n{chunk}{tail}"
    if query:
        try:
            rx = re.compile(query, re.I)
        except re.error:
            rx = re.compile(re.escape(query), re.I)
        hits = []
        for r in recs:
            text = _record_text(r)
            m = rx.search(text)
            if m:
                a = max(0, m.start() - 160)
                snippet = text[a : m.end() + 240].replace("\n", " ")
                hits.append(f"#{r['seq']} {_label(r)}: …{snippet}…")
        if not hits:
            return f"nothing in the transcript matches {query!r}"
        shown = hits[-25:]
        return f"{len(hits)} match(es) for {query!r}" + (" (latest 25)" if len(hits) > 25 else "") + ":\n" + "\n".join(shown) + "\n\nrecall(id=N) returns one in full."
    lines = [f"#{r['seq']:<4} {_label(r):<32} {len(_record_text(r)):>8,} chars" for r in recs[-60:]]
    return "transcript index (latest 60):\n" + "\n".join(lines) + "\n\nrecall(id=N) for one message, recall(query=...) to search."
