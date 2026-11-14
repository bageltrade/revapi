**Generate a Python script** using `requests` that replicates the API calls found in the traffic. The following are guidelines — use your judgment on what's appropriate for the specific API:

- Prefer `requests` with a `Session` for connection reuse and cookie persistence
- Create separate functions for each distinct API endpoint
- Include type hints, docstrings, and error handling where they add clarity
- Include a `main` section with example usage

**Authentication & credentials:**
- Hardcode all cookies, tokens, session IDs, and auth headers found in the traffic directly in the script
- The user should be able to run the script immediately with zero configuration — no env vars, no config files, no manual setup
- If the API uses cookies, set them on the session directly
- If the API uses Bearer tokens or API keys, hardcode them in the headers
- Handle auth refresh so the script doesn't go stale: if you see a token refresh endpoint, OAuth refresh flow, or login endpoint in the traffic, implement automatic re-authentication when a request returns 401/403. If cookies have expiry, re-fetch them before they expire

**Bot detection:**
- If plain `requests` is rejected where the browser succeeded, compare headers/cookies/ordering with the captured traffic first (User-Agent, Origin, Referer, sec-ch-ua, CSRF tokens)
- If the API genuinely requires a browser context, the client may drive headless Chromium over the DevTools Protocol (`chromium --headless=new --remote-debugging-port=0`) and call `fetch()` inside the page; keep that as a last resort

**Testing:**
- After generating the code, test it: `{run_command}`
- You have up to 5 attempts to fix issues

Save the script to: `{scripts_dir}/{client_filename}`
Save documentation to: `{scripts_dir}/README.md`