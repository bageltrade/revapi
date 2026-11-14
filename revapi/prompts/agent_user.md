<mission>
{prompt}
</mission>
{start_url_block}
<output_directory>
{scripts_dir}
</output_directory>

## Workflow

Use this workflow as a guide — adapt as needed:

### Phase 1: BROWSE
Use the browser tools to accomplish the mission goal. Start with browser_navigate, then browser_snapshot to see the page and its element refs.

### Phase 2: MONITOR
Call `browser_network_requests()` regularly to watch API traffic (it lists API-like requests by default; pass all=true for everything). Use `browser_network_request(index)` to see full headers and bodies of one request. Look for auth patterns (cookies, tokens, headers), endpoints, and response structures.

### Phase 3: CAPTURE
revapi records every request and response to `{har_path}`. Call `browser_close()` once you have browsed what the mission needs; this writes the HAR. After closing, use har_overview / har_search / har_entry to dig into the full recording.

### Phase 4: REVERSE ENGINEER
Extract the API calls, authentication patterns, and data structures, generate the code into the output directory with write_file, run it with the shell tool, and iterate until it works. Then call `finish`.
