"""
fetch_papers.py
----------------
Fetches this week's papers from Hugging Face's "Daily Papers" feed
(curated by AK and the research community) and returns a normalized list.

Primary source:  https://huggingface.co/papers/week/{YYYY-Www}
Fallback source:  https://huggingface.co/papers/date/{YYYY-MM-DD}  (unioned Mon-Sun)

The fallback exists because the weekly archive page is sometimes thin/unavailable
for the current, still-in-progress ISO week -- in that case we reconstruct the
week ourselves from the daily pages, which are always present.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; paper-digest-agent/1.0; +https://github.com/)"
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    upvotes: int = 0
    url: str = field(init=False)

    def __post_init__(self):
        self.url = f"https://huggingface.co/papers/{self.arxiv_id}"


def current_iso_week(today: dt.date | None = None) -> tuple[int, int]:
    """Return (iso_year, iso_week) for the given date (defaults to today)."""
    today = today or dt.date.today()
    iso_year, iso_week, _ = today.isocalendar()
    return iso_year, iso_week


def _parse_paper_cards(html: str) -> list[Paper]:
    """Parse a Hugging Face papers listing page (daily/weekly/monthly all share
    the same card markup) into a list of Paper objects."""
    soup = BeautifulSoup(html, "html.parser")
    papers: dict[str, Paper] = {}

    # Every paper card links to /papers/<arxiv_id> via its <h3><a> title link.
    for h3 in soup.select("h3 a[href^='/papers/']"):
        href = h3.get("href", "")
        m = re.match(r"^/papers/([0-9]{4}\.[0-9]{4,5})", href)
        if not m:
            continue
        arxiv_id = m.group(1)
        title = h3.get_text(strip=True)

        # Upvote count is rendered as a small number near the submitter's
        # avatar, inside the same card container. We walk up to the card
        # and grab the first standalone integer we find.
        upvotes = 0
        card = h3.find_parent("article") or h3.find_parent("div")
        if card:
            text_nums = re.findall(r"\b(\d{1,4})\b", card.get_text(" ", strip=True))
            if text_nums:
                # Heuristic: the vote count is usually the first 1-3 digit
                # number in the card; skip 4-digit numbers (likely years/ids).
                for n in text_nums:
                    if len(n) <= 3:
                        upvotes = int(n)
                        break

        if arxiv_id not in papers or upvotes > papers[arxiv_id].upvotes:
            papers[arxiv_id] = Paper(arxiv_id=arxiv_id, title=title, upvotes=upvotes)

    return list(papers.values())


def fetch_week(iso_year: int, iso_week: int) -> list[Paper]:
    """Try the weekly archive page first; fall back to unioning daily pages."""
    week_str = f"{iso_year}-W{iso_week:02d}"
    url = f"https://huggingface.co/papers/week/{week_str}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.ok:
            papers = _parse_paper_cards(resp.text)
            if papers:
                return papers
    except requests.RequestException:
        pass

    # Fallback: reconstruct the week from daily pages (Mon-Sun of that ISO week).
    monday = dt.date.fromisocalendar(iso_year, iso_week, 1)
    all_papers: dict[str, Paper] = {}
    for offset in range(7):
        day = monday + dt.timedelta(days=offset)
        if day > dt.date.today():
            break
        day_url = f"https://huggingface.co/papers/date/{day.isoformat()}"
        try:
            resp = requests.get(day_url, headers=HEADERS, timeout=20)
            if not resp.ok:
                continue
            for p in _parse_paper_cards(resp.text):
                if p.arxiv_id not in all_papers or p.upvotes > all_papers[p.arxiv_id].upvotes:
                    all_papers[p.arxiv_id] = p
        except requests.RequestException:
            continue

    return list(all_papers.values())


def fetch_paper_abstract(arxiv_id: str) -> dict:
    """Fetch the individual HF paper page and return title/abstract/community notes."""
    url = f"https://huggingface.co/papers/{arxiv_id}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else arxiv_id

    # The long-form abstract lives in the section following the "Abstract" heading.
    abstract = ""
    abstract_heading = soup.find(lambda tag: tag.name in ("h2", "h3") and "Abstract" in tag.get_text())
    if abstract_heading:
        parts = []
        for sib in abstract_heading.find_next_siblings():
            if sib.name in ("h2", "h3"):
                break
            text = sib.get_text(" ", strip=True)
            if text:
                parts.append(text)
        abstract = "\n\n".join(parts)

    # First author-submitted community comment, if any (often a short human summary).
    community_note = ""
    comment_el = soup.select_one(".community, [class*='comment']")
    if comment_el:
        community_note = comment_el.get_text(" ", strip=True)[:1000]

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "community_note": community_note,
        "url": url,
    }


if __name__ == "__main__":
    year, week = current_iso_week()
    found = fetch_week(year, week)
    print(f"Week {year}-W{week:02d}: {len(found)} papers found")
    for p in sorted(found, key=lambda x: -x.upvotes)[:10]:
        print(f"  [{p.upvotes:>3}] {p.title}  ({p.arxiv_id})")
