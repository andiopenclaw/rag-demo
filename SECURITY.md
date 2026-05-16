# Security Notes

This project is a **local demo** tool. The notes below explain what's been
hardened and what to address before any broader deployment.

## Fixed

| Issue | File | Fix |
|---|---|---|
| Path traversal via `space_key` | `fetch_confluence.py` | `space_key` now sanitized; path checked against `OUTPUT_DIR` |
| Path traversal via article title | `fetch_articles.py` | Title sanitized; path checked against `OUTPUT_DIR` |
| Unbounded response size | `fetch_confluence.py` | `resp.read()` capped at 10 MB |
| Server error body leaked in exceptions | `fetch_confluence.py` | HTTP error body no longer included in raised message |
| `KeyError` on missing OpenAI key | `app.py` | `get_openai_key()` raises clearly; no bare dict access |
| Prompt stuffing via long input | `app.py` | Queries capped at 1,000 characters |

## Known gaps (acceptable for local demo, fix before production)

### 1. No authentication on the Chainlit app
Anyone who can reach port 8000 can query the knowledge base.

**For production:** Put Chainlit behind a reverse proxy (nginx/Caddy) with
HTTP basic auth, OAuth, or restrict to `127.0.0.1` only:
```bash
chainlit run app.py --host 127.0.0.1 --port 8000
```

### 2. No rate limiting
A user can send unlimited queries, driving up OpenAI API costs.

**For production:** Add per-IP rate limiting at the proxy layer, or use
Chainlit's session hooks to track and limit requests per session.

### 3. Prompt injection
User queries are passed into the LLM prompt without semantic filtering.
A crafted query could attempt to override system instructions.

**For production:** This is hard to fully prevent at the input level. The
stronger mitigations are at the output level: system prompt pinning,
output filtering, and monitoring for unusual responses.

### 4. Confluence credentials in environment
Credentials are read from env vars — don't put them in `.env` files that
get committed. The `.gitignore` already excludes `.env`.

### 5. Article content stored as plaintext JSON
If your wiki contains sensitive data, the `articles/` folder will contain
it in plaintext. The `.gitignore` excludes `articles/` from commits, but
protect the directory on disk if needed.
