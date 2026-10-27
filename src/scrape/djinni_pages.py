"""
Phase 1: collect vacancy URLs from the public Djinni listing.

Only /jobs/ is requested (allowed by robots.txt). Sequential, with a pause
between requests; stops on 403/429 so we never hammer the site.
"""
import argparse
import random
import re
import time
from pathlib import Path

import httpx

BASE = "https://djinni.co"
LISTING = BASE + "/jobs/"
USER_AGENT = "Vacancy-Role-Classifier/0.1 (educational project; +https://github.com/LV-Engineer/Vacancy-Role-Classifier)"
JOB_LINK = re.compile(r'href="(/jobs/\d+-[^"/]+/)"')
DELAY = (2.0, 3.5)
RETRIES = 3

class Blocked(Exception):
    """Raised on 403/429: stop the whole run."""

def fetch(client: httpx.Client, page: int) -> str | None:
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.get(LISTING, params={"page": page}, timeout=20)
        except httpx.HTTPError:
            resp = None
        if resp is not None:
            if resp.status_code in (403, 429):
                raise Blocked(f"HTTP {resp.status_code} on page {page}")
            if resp.status_code == 200 and JOB_LINK.search(resp.text):
                return resp.text
            print(f"page {page}: attempt {attempt} got status {resp.status_code}, no vacancy links")
        else:
            print(f"page {page}: attempt {attempt} network error")
            
        time.sleep(5 * attempt)
    return None

def extract_urls(html: str) -> set[str]:
    return {BASE + path for path in JOB_LINK.findall(html)}

def load_seen(path: Path) -> set[str]:
    return set(path.read_text().split()) if path.exists() else set()

def append_urls(path: Path, urls: set[str]) -> None:
    with path.open("a") as f:
        f.writelines(url + "\n" for url in sorted(urls))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=60, help="last listing page")
    parser.add_argument("--out", type=Path, default=Path("data/raw/djinni_urls.txt"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.out)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for page in range(args.start, args.end + 1):
            html = fetch(client, page)
            if html is None:
                print(f"page {page}: failed after {RETRIES} retries, skipping")
                continue
            new = extract_urls(html) - seen
            seen |= new
            append_urls(args.out, new)
            print(f"page {page}: +{len(new)} (total {len(seen)})")
            time.sleep(random.uniform(*DELAY))


if __name__ == "__main__":
    main()