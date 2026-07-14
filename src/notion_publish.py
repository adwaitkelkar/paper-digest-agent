"""
notion_publish.py
-------------------
Publishes one Notion page per selected paper under a configured parent
page, using the official `notion-client` SDK. Converts the Markdown
write-up into Notion blocks (headings + paragraphs), since the Notion
API doesn't accept raw Markdown for page content.
"""

from __future__ import annotations

import os
import re

from notion_client import Client

# A small rotation of paper-themed emoji so pages don't all look identical.
ICONS = ["📄", "🧠", "🔬", "🤖", "📊", "🛰️", "🧩"]


def _parent_page_id() -> str:
    """Read lazily (not at import time) so this module can be imported/tested
    without NOTION_* env vars being set -- only publish calls need them."""
    return os.environ["NOTION_PARENT_PAGE_ID"]


def _markdown_to_blocks(markdown: str) -> list[dict]:
    """Very small Markdown -> Notion block converter.

    Supports: ## headings, plain paragraphs, and a leading '---' divider.
    This is intentionally minimal -- the write-ups we generate only use
    H2 headers and paragraphs by design (see summarize.py's prompt).
    """
    blocks: list[dict] = []
    for raw_line in markdown.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line == "---":
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            continue
        if line.startswith("## "):
            blocks.append(_heading_block(line[3:].strip(), level=2))
            continue
        if line.startswith("**") and line.endswith("  "):
            # Bold metadata lines from the header block (Link/arXiv ID/Week/etc.)
            blocks.append(_paragraph_block(line.strip()))
            continue
        blocks.append(_paragraph_block(line))
    return blocks


def _rich_text(text: str) -> list[dict]:
    """Parse a very small subset of inline Markdown (**bold**, [text](url))
    into Notion rich_text objects. Falls back to plain text otherwise."""
    tokens: list[dict] = []
    pattern = re.compile(r"(\*\*(.+?)\*\*)|(\[(.+?)\]\((.+?)\))")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            tokens.append({"type": "text", "text": {"content": text[pos:m.start()]}})
        if m.group(1):  # bold
            tokens.append(
                {
                    "type": "text",
                    "text": {"content": m.group(2)},
                    "annotations": {"bold": True},
                }
            )
        elif m.group(3):  # link
            tokens.append(
                {
                    "type": "text",
                    "text": {"content": m.group(4), "link": {"url": m.group(5)}},
                }
            )
        pos = m.end()
    if pos < len(text):
        tokens.append({"type": "text", "text": {"content": text[pos:]}})
    return tokens or [{"type": "text", "text": {"content": text}}]


def _heading_block(text: str, level: int = 2) -> dict:
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rich_text(text)}}


def _paragraph_block(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich_text(text)}}


def publish_paper_page(client: Client, title: str, content_markdown: str, icon: str | None = None) -> str:
    """Create a standalone Notion page for one paper. Returns the page URL."""
    blocks = _markdown_to_blocks(content_markdown)
    # Notion's API caps children at 100 blocks per request.
    page = client.pages.create(
        parent={"type": "page_id", "page_id": _parent_page_id()},
        icon={"type": "emoji", "emoji": icon or "📄"},
        properties={"title": {"title": [{"type": "text", "text": {"content": title}}]}},
        children=blocks[:100],
    )
    return page.get("url", "")


def publish_digest(client: Client, papers_with_content: list[dict]) -> list[dict]:
    """papers_with_content: list of {"title": str, "content": str, "icon": str}
    Returns the same list with a "notion_url" key added to each item."""
    results = []
    for i, item in enumerate(papers_with_content):
        icon = item.get("icon") or ICONS[i % len(ICONS)]
        url = publish_paper_page(client, item["title"], item["content"], icon=icon)
        results.append({**item, "notion_url": url})
    return results
