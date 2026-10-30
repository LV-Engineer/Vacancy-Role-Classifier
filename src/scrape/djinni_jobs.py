"""
Phase 2: download each vacancy from djinni_urls.txt and save clean text.

Output: one JSON object per line (id, title, description, experience_years,
language, source). Resumable: already saved ids are skipped.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
from selectolax.lexbor import LexborHTMLParser

from scrape.djinni_pages import DELAY, RETRIES, USER_AGENT, Blocked

TITLE_SELECTOR = "h1"
DESCRIPTION_SELECTOR = "div.job-post__description"
MIN_DESCRIPTION_CHARS = 200

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
HANDLE = re.compile(r"(?<!\w)@\w{3,}|t\.me/\S+")
EXPERIENCE_SELECTOR = "aside ul.list-unstyled strong"
EXPERIENCE = re.compile(r"(\d+(?:[.,]\d+)?)\s*\+?\s*(?:років|року|рік|роки|years?)", re.IGNORECASE)
NO_EXPERIENCE = re.compile(r"без досвіду|no experience", re.IGNORECASE)
CYRILLIC = re.compile(r"[а-яіїєґ]", re.IGNORECASE)
JOB_ID = re.compile(r"/jobs/(\d+)-")

def clean_text(text: str) -> str:
    for pattern in (EMAIL, PHONE, HANDLE):
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    share = len(CYRILLIC.findall(text)) / len(letters)
    return "uk" if share > 0.3 else "en"

def parse_experience(text: str) -> float | None:
    if NO_EXPERIENCE.search(text):
        return 0.0
    found_match = EXPERIENCE.search(text)
    return float(found_match.group(1).replace(",", ".")) if found_match else None

def parse_experience_from_panel(tree: LexborHTMLParser) -> float | None:
    for node in tree.css(EXPERIENCE_SELECTOR):
        years = parse_experience(node.text())
        if years is not None:
            return years
    return None

def job_id_from_url(url: str) -> str:
    match = JOB_ID.search(url)
    if match is None:
        raise ValueError(f"not a vacancy url: {url}")
    return match.group(1)

def parse_job(url: str, html: str) -> dict[str, Any] | None:
    tree = LexborHTMLParser(html)
    title_node = tree.css_first(TITLE_SELECTOR)
    desc_node = tree.css_first(DESCRIPTION_SELECTOR)
    if title_node is None or desc_node is None:
        return None
    description = clean_text(desc_node.text(separator=" "))
    if len(description) < MIN_DESCRIPTION_CHARS:
        return None
    job_id = job_id_from_url(url)
    return {
        "id": f"djinni_{job_id}",
        "title": clean_text(title_node.text()),
        "description": description,
        "experience_years": parse_experience_from_panel(tree),
        "language": detect_language(description),
        "source": "djinni",
    }

def fetch_job(client: httpx.Client, url: str) -> str | None:
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.get(url, timeout=20)
        except httpx.HTTPError:
            resp = None
        if resp is not None:
            if resp.status_code in (403, 429):
                raise Blocked(f"HTTP {resp.status_code} on {url}")
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
        if attempt < RETRIES:
            time.sleep(5 * attempt)
    return None

def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def append_job(path: Path, job: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=Path, default=Path("data/raw/djinni_urls.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/djinni_jobs.jsonl"))
    parser.add_argument("--limit", type=int, default=20, help="max vacancies this run")
    args = parser.parse_args()

    urls = args.urls.read_text().split()
    done = load_done_ids(args.out)
    saved = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for url in urls:
            if saved >= args.limit:
                break
            if f"djinni_{job_id_from_url(url)}" in done:
                continue
            html = fetch_job(client, url)
            job = parse_job(url, html) if html else None
            if job is None:
                print(f"skip {url}")
            else:
                append_job(args.out, job)
                saved += 1
                print(f"saved {job['id']} ({job['language']}, {job['experience_years']} y)")
            time.sleep(random.uniform(*DELAY))


if __name__ == "__main__":
    main()