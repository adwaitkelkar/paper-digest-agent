"""
main.py
--------
Orchestrator for the weekly paper-digest agent. Runs the full pipeline:

  1. Fetch this week's Hugging Face Daily Papers.
  2. Shortlist by upvotes.
  3. Select 2 papers (CV-preference + random rule).
  4. Fetch full abstracts for the selected papers.
  5. Summarize each into a structured, plain-English write-up (Claude).
  6. Publish each as a standalone Notion page.
  7. Print a run summary (also useful as CI log output).

Run manually:
    ANTHROPIC_API_KEY=... NOTION_TOKEN=... NOTION_PARENT_PAGE_ID=... python src/main.py

Run on a schedule: see .github/workflows/weekly_digest.yml (defaults to every
Sunday). The GitHub Actions cron is UTC-based -- adjust to your timezone.
"""

from __future__ import annotations

import os
import sys

import anthropic
from notion_client import Client as NotionClient

from fetch_papers import current_iso_week, fetch_week
from select_papers import build_shortlist, select_two, enrich_with_abstract
from summarize import summarize_paper, build_page_content


def run(dry_run: bool = False) -> None:
    year, week = current_iso_week()
    week_str = f"{year}-W{week:02d}"
    print(f"=== Weekly Paper Digest: {week_str} ===")

    papers = fetch_week(year, week)
    print(f"Fetched {len(papers)} papers for the week.")
    if not papers:
        print("No papers found -- nothing to do.")
        return

    shortlist = build_shortlist(papers)
    print(f"Shortlist: {len(shortlist)} papers")
    for p in shortlist:
        print(f"  [{p.upvotes:>3}] {p.title}")

    anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    selections = select_two(shortlist, anthropic_client, seed=None)

    print("\nSelected papers:")
    for s in selections:
        print(f"  ({s.method}) {s.paper['title']}")

    # Enrich with full abstracts, then summarize each.
    enriched = [enrich_with_abstract(s) for s in selections]

    pages_to_publish = []
    for s in enriched:
        print(f"\nSummarizing: {s.paper['title']} ...")
        writeup = summarize_paper(anthropic_client, s.paper)
        s.paper["writeup"] = writeup
        content = build_page_content(s.paper, s.method, week_str)
        pages_to_publish.append(
            {
                "title": s.paper["title"],
                "content": content,
                "method": s.method,
            }
        )

    if dry_run:
        print("\n[DRY RUN] Skipping Notion publish. Generated content:\n")
        for p in pages_to_publish:
            print("=" * 80)
            print(p["title"], f"({p['method']})")
            print(p["content"][:1000], "...\n")
        return

    notion_client = NotionClient(auth=os.environ["NOTION_TOKEN"])
    from notion_publish import publish_digest

    results = publish_digest(notion_client, pages_to_publish)

    print("\n=== Done ===")
    for r in results:
        print(f"  ({r['method']}) {r['title']} -> {r['notion_url']}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run(dry_run=dry)
