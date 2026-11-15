## Human checks (Cloudflare, Turnstile, CAPTCHA)

revapi never solves or works around these checks. When a page shows one, the browser tools hand it to the user in a live view and the tool result starts with `[human check]`.
- Passed: continue normally. The session (cookies and the exact user agent) is saved to `~/.revapi/sessions/<host>.json`.
- Still showing: stop browsing that site. Don't treat the check page or the `/cdn-cgi/` and `challenges.cloudflare.com` requests as the site's API. Tell the user to run `revapi refresh <url>` or `revapi cookies import <host>`, then finish with what you have.

**Session file:** if `~/.revapi/sessions/<host>.json` exists for the target host (check with the shell tool), the generated client should do three things:
- At startup, load `cookie_header` (or `cookies`) and `user_agent` from it. Keep the values you hardcoded as the fallback when the file is missing.
- Send exactly that `User-Agent` with those cookies.
- Treat a 403/503 with a `cf-mitigated: challenge` header, or a body containing "Just a moment", as an expired session. Raise a clear error: `session expired: run  revapi refresh <url>  then try again`. Don't retry in a loop.
