"""
summarize.py
-------------
Turns a paper's abstract (+ any community notes) into a plain-English,
structured write-up using Claude. The prompt enforces a fixed section
order and forbids inventing facts not present in the source text --
Claude is explicitly told to reason carefully from the abstract rather
than fabricate numbers or mechanisms it can't support.
"""

from __future__ import annotations

import os

import anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

WRITEUP_PROMPT = """You are writing a weekly AI research digest for a general
technical audience (not necessarily ML researchers). Explain the paper below in
simple, plain English. Define any technical term you must use. Do not use jargon
without explaining it.

Paper title: {title}
arXiv ID: {arxiv_id}
Paper URL: {url}

Abstract:
{abstract}

Community note (if any, may be empty):
{community_note}

Write the explanation with exactly these five sections, in this order, using
Markdown headers (##):

## The Problem
What exact problem is the paper trying to solve? Be specific about the setting/domain.

## Why It Needed Solving
The motivation -- why this problem matters, what breaks or falls short without a solution.

## The Core Idea / Method
This should be the most detailed and thorough section. Explain the core technique(s)
step by step: how the method actually works, what makes it different from prior
approaches, and any key mechanisms, architectures, or training procedures involved.
Do not just summarize in one paragraph -- unpack the mechanics so a curious
non-expert could follow the logic.

## Outcomes
What results did they get? Include concrete numbers/benchmarks where available in
the abstract. If no numbers are given, say so rather than inventing any.

## Drawbacks
Limitations, open questions, or reasons to be skeptical (e.g. self-reported
benchmarks, cost tradeoffs, narrow scope, generalization concerns). If the abstract
doesn't state limitations explicitly, reason carefully about plausible ones implied
by the method, and say these are inferred, not stated.

Rules:
- Do NOT fabricate results, numbers, or mechanisms not supported by the abstract/notes.
- If a detail isn't available from the source text, say so explicitly rather than
  guessing at specifics.
- Keep the tone direct and simple. No filler, no marketing language.
"""


def summarize_paper(client: anthropic.Anthropic, paper: dict) -> str:
    """Return the Markdown body (5 sections) for one paper."""
    prompt = WRITEUP_PROMPT.format(
        title=paper["title"],
        arxiv_id=paper["arxiv_id"],
        url=paper["url"],
        abstract=paper.get("abstract") or "(no abstract text was retrievable)",
        community_note=paper.get("community_note") or "(none)",
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def build_page_content(paper: dict, method: str, week_str: str) -> str:
    """Assemble the full Notion-markdown page body: link header + write-up."""
    header = (
        f"**Link:** [{paper['url']}]({paper['url']})  \n"
        f"**arXiv ID:** {paper['arxiv_id']}  \n"
        f"**Week:** {week_str}  \n"
        f"**Selected via:** {'CV-preference rule (highest-upvoted vision/vision-language paper)' if method == 'cv_preference' else 'random draw from the shortlist'}\n\n"
        "---\n\n"
    )
    return header + paper["writeup"]
