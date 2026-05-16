"""
Confluence fetcher — replaces fetch_articles.py for internal wikis.
Tested against Confluence Server / Data Center 7.x–9.x (REST API v1).

Usage:
    python fetch_confluence.py

Configuration (edit the CONFIG block below, or use environment variables):
    CONFLUENCE_URL      Base URL, e.g. https://wiki.yourcompany.com
    CONFLUENCE_USER     Your username (for basic auth)
    CONFLUENCE_PASSWORD Your password or Personal Access Token
    CONFLUENCE_TOKEN    Personal Access Token (alternative to user+password)
    CONFLUENCE_SPACES   Comma-separated space keys, e.g. ENG,HR,PRODUCT

Personal Access Tokens (recommended over passwords):
    Confluence → Profile → Personal Access Tokens → Create token
    Then set CONFLUENCE_TOKEN=your-token (leave USER/PASSWORD blank)

Authentication modes:
    - PAT (recommended):  set CONFLUENCE_TOKEN
    - Basic auth:         set CONFLUENCE_USER + CONFLUENCE_PASSWORD
"""

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these or set as environment variables
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "url":      os.environ.get("CONFLUENCE_URL",      "https://wiki.yourcompany.com"),
    "user":     os.environ.get("CONFLUENCE_USER",     ""),
    "password": os.environ.get("CONFLUENCE_PASSWORD", ""),
    "token":    os.environ.get("CONFLUENCE_TOKEN",    ""),
    "spaces":   os.environ.get("CONFLUENCE_SPACES",   ""),   # e.g. "ENG,PRODUCT"
    "max_pages": int(os.environ.get("CONFLUENCE_MAX_PAGES", "50")),  # per space
}

OUTPUT_DIR = "articles"


# ─────────────────────────────────────────────────────────────
# HTML → plain text stripper
# ─────────────────────────────────────────────────────────────
class HTMLStripper(HTMLParser):
    """Strip HTML/Confluence storage format tags, keep meaningful text."""
    SKIP_TAGS = {"script", "style", "ac:parameter", "ac:plain-text-body"}

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in ("p", "h1", "h2", "h3", "h4", "li", "tr", "br"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.text_parts.append(data)

    def get_text(self):
        raw = "".join(self.text_parts)
        # Collapse whitespace / blank lines
        lines = [l.strip() for l in raw.splitlines()]
        lines = [l for l in lines if l]
        return "\n".join(lines)


def strip_html(html: str) -> str:
    parser = HTMLStripper()
    parser.feed(html)
    return parser.get_text()


# ─────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────
def make_headers() -> dict:
    headers = {
        "Accept": "application/json",
        "User-Agent": "RAG-Demo/1.0",
    }
    if CONFIG["token"]:
        headers["Authorization"] = f"Bearer {CONFIG['token']}"
    elif CONFIG["user"] and CONFIG["password"]:
        import base64
        creds = base64.b64encode(
            f"{CONFIG['user']}:{CONFIG['password']}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {creds}"
    return headers


def get_json(path: str, params: dict = None) -> dict:
    base = CONFIG["url"].rstrip("/")
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=make_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason} — {body[:200]}") from e


# ─────────────────────────────────────────────────────────────
# Confluence API helpers
# ─────────────────────────────────────────────────────────────
def list_spaces() -> list[dict]:
    """Return all spaces this user can access."""
    spaces = []
    start = 0
    limit = 50
    while True:
        data = get_json("/rest/api/space", {"start": start, "limit": limit, "type": "global"})
        spaces.extend(data["results"])
        if data["start"] + len(data["results"]) >= data["size"]:
            break
        start += limit
    return spaces


def list_pages(space_key: str, max_pages: int) -> list[dict]:
    """Return up to max_pages pages from a space."""
    pages = []
    start = 0
    limit = 25
    while len(pages) < max_pages:
        batch_limit = min(limit, max_pages - len(pages))
        data = get_json("/rest/api/content", {
            "spaceKey": space_key,
            "type": "page",
            "status": "current",
            "start": start,
            "limit": batch_limit,
        })
        pages.extend(data["results"])
        if data["start"] + len(data["results"]) >= data["size"]:
            break
        start += batch_limit
        time.sleep(0.1)
    return pages


def fetch_page(page_id: str) -> dict:
    """Fetch full page content (storage format)."""
    return get_json(
        f"/rest/api/content/{page_id}",
        {"expand": "body.storage,ancestors,space,version"}
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate config
    if "yourcompany" in CONFIG["url"]:
        print("⚠️  You haven't set CONFLUENCE_URL yet.")
        print("    Edit CONFIG in this file or set environment variables:\n")
        print("    export CONFLUENCE_URL=https://wiki.yourcompany.com")
        print("    export CONFLUENCE_TOKEN=your-personal-access-token")
        print("    export CONFLUENCE_SPACES=ENG,PRODUCT  # optional: specific spaces")
        print("    python fetch_confluence.py")
        return

    print(f"Connecting to: {CONFIG['url']}")

    # Determine which spaces to index
    if CONFIG["spaces"]:
        space_keys = [s.strip() for s in CONFIG["spaces"].split(",") if s.strip()]
        print(f"Spaces: {space_keys} (from config)\n")
    else:
        print("Discovering spaces...", end=" ", flush=True)
        spaces = list_spaces()
        space_keys = [s["key"] for s in spaces]
        names = [f"{s['key']} ({s['name']})" for s in spaces[:10]]
        print(f"found {len(space_keys)}: {', '.join(names)}" + (" ..." if len(space_keys) > 10 else "") + "\n")

    total_saved = 0

    for space_key in space_keys:
        print(f"── Space: {space_key} ──────────────────────")
        try:
            pages = list_pages(space_key, CONFIG["max_pages"])
        except RuntimeError as e:
            print(f"  ✗ Could not list pages: {e}")
            continue

        print(f"  Found {len(pages)} pages (max={CONFIG['max_pages']})")

        for page in pages:
            title = page["title"]
            page_id = page["id"]
            print(f"  Fetching: {title} ...", end=" ", flush=True)

            try:
                full = fetch_page(page_id)
                storage_html = full["body"]["storage"]["value"]
                plain_text = strip_html(storage_html)
                page_url = (
                    CONFIG["url"].rstrip("/")
                    + full.get("_links", {}).get("webui", f"/pages/viewpage.action?pageId={page_id}")
                )

                article = {
                    "title": title,
                    "url": page_url,
                    "space": space_key,
                    "page_id": page_id,
                    "version": full.get("version", {}).get("number", 1),
                    "summary": plain_text[:500],
                    "text": plain_text,
                    "word_count": len(plain_text.split()),
                }

                safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
                filename = f"{space_key}_{safe_title[:60].replace(' ', '_').lower()}.json"
                path = os.path.join(OUTPUT_DIR, filename)

                with open(path, "w") as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)

                print(f"✓  ({article['word_count']:,} words)")
                total_saved += 1

            except Exception as e:
                print(f"✗  {e}")

            time.sleep(0.15)  # polite rate limiting

        print()

    print(f"✅ Done. {total_saved} pages saved to ./{OUTPUT_DIR}/")
    print()
    print("Next steps:")
    print("  python3 build_index.py   # build the vector index")
    print("  python3 demo.py          # run queries")


if __name__ == "__main__":
    main()
