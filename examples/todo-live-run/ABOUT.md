# Live run example

Produced by `revapi agent` against `tests/fixture_site.py` using the free Pollinations model (gpt-oss-20b). The run was resumed once with `/resume` after the model emitted malformed JSON (since fixed by the JSON repair in `llm.py`).

- `recording.har`: what headless Chromium captured
- `transcript.jsonl`: the full agent conversation
- `api_client.py` / `README.md`: what the agent wrote, ran (exit 0) and shipped
