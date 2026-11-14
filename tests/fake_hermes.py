#!/usr/bin/env python3
"""A stand-in for the Hermes CLI: speaks `hermes chat --format stream-json` (NDJSON on stdout).

Writes api_client.py into the workspace (--in), resumes when --resume is given, and fails when the prompt says so.
"""

import json
import sys
from pathlib import Path


def emit(**event):
    print(json.dumps(event), flush=True)


args = sys.argv[1:]
workspace = Path(args[args.index("--in") + 1])
resumed = args[args.index("--resume") + 1] if "--resume" in args else None
prompt = sys.stdin.read()
Path(workspace / "argv.json").write_text(json.dumps({"argv": args, "prompt": prompt}))
print("hermes banner, not json")
emit(type="system", model="claude-sonnet-4-5", session_id=resumed or "sess-1")
if "FAIL" in prompt:
    emit(type="result", exit_code=2, error="no API key configured", tokens={})
    sys.exit(2)
emit(type="text", text="Reading the capture. ")
emit(type="text", text="Writing the client.")
code = "import requests\n\n\ndef list_todos():\n    return requests.get('http://x/api/todos').json()\n"
emit(type="tool_use", id="t1", name="write_file", input={"path": "api_client.py", "content": code})
(workspace / "api_client.py").write_text(code)
emit(type="tool_result", id="t1", name="write_file", output="wrote api_client.py")
emit(type="text", text="Done: api_client.py lists the todos.")
emit(type="result", exit_code=0, session_id=resumed or "sess-1", text="Done", tokens={"input": 1200, "output": 300, "cache_read": 100})
