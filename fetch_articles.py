"""
Step 1: Fetch Wikipedia articles and save them locally.
Topic: Space Exploration — great for demos because it's factual,
specific dates/numbers trip up LLMs without grounding, and everyone
finds it interesting.
"""

import json
import os
import time
import urllib.request
import urllib.parse

ARTICLES = [
    "Apollo program",
    "James Webb Space Telescope",
    "SpaceX",
    "International Space Station",
    "Exploration of Mars",
]

OUTPUT_DIR = "articles"


def fetch_wikipedia(title: str) -> dict:
    """Fetch article text via Wikipedia's REST API (no API key needed)."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "RAG-Demo/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        summary = json.loads(resp.read())

    # Also get full extract via action API
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    })
    full_url = f"https://en.wikipedia.org/w/api.php?{params}"
    req2 = urllib.request.Request(full_url, headers={"User-Agent": "RAG-Demo/1.0"})
    with urllib.request.urlopen(req2, timeout=15) as resp2:
        data = json.loads(resp2.read())

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    full_text = page.get("extract", summary.get("extract", ""))

    return {
        "title": title,
        "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "summary": summary.get("extract", ""),
        "text": full_text,
        "word_count": len(full_text.split()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for title in ARTICLES:
        print(f"Fetching: {title} ...", end=" ", flush=True)
        try:
            article = fetch_wikipedia(title)
            safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
            filename = safe_title.replace(" ", "_").lower()[:80] + ".json"
            path = os.path.join(OUTPUT_DIR, filename)
            # Guard against path traversal
            if not os.path.abspath(path).startswith(os.path.abspath(OUTPUT_DIR)):
                print(f"✗  Skipping unsafe filename: {filename}")
                continue
            with open(path, "w") as f:
                json.dump(article, f, indent=2)
            print(f"✓  ({article['word_count']:,} words)")
            results.append({"title": title, "file": filename, "words": article["word_count"]})
        except OSError as e:
            print(f"✗  File error for '{title}': {e}")
        except Exception as e:
            print(f"✗  Fetch error for '{title}': {e}")
        time.sleep(0.5)  # polite rate limiting

    print(f"\nFetched {len(results)}/{len(ARTICLES)} articles.")
    print("Saved to ./articles/")
    return results


if __name__ == "__main__":
    main()
