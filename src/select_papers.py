"""
select_papers.py
------------------
Turns the raw weekly paper list into a shortlist, then applies the
selection rule that picks exactly 2 papers for the digest:

  1. Build a shortlist of substantive papers (drop pure technical
     reports / thin submissions when better options exist).
  2. Ask Claude to flag which shortlisted papers are computer-vision /
     vision-language papers (broadly defined: any paper whose input or
     output involves images or video -- detection, segmentation, 3D,
     image/video generation, VLMs, embodied/video world models, etc.).
  3. If at least one CV paper exists, the highest-upvoted one is
     GUARANTEED a slot.
  4. The remaining slot(s) are filled by true random sampling from the
     rest of the shortlist (both slots are random if no CV paper
     qualifies).

This mirrors a common real-world editorial pattern: a hard business
rule (always cover computer vision, our team's focus area) combined
with randomness to keep the "AI pick" from always being the single
most-upvoted item every week.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import anthropic

from fetch_papers import Paper, fetch_paper_abstract

SHORTLIST_SIZE = int(os.environ.get("SHORTLIST_SIZE", "15"))
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


@dataclass
class Selection:
    paper: dict          # full paper detail (title, abstract, url, arxiv_id)
    method: str           # "cv_preference" or "random"


def build_shortlist(papers: list[Paper], size: int = SHORTLIST_SIZE) -> list[Paper]:
    """Rank by upvotes and take the top N as the shortlist.

    Upvotes are a reasonable proxy for "substantive contribution" on HF
    Papers, since low-effort submissions rarely accumulate votes. This
    keeps the shortlist small enough for a single Claude call to classify.
    """
    ranked = sorted(papers, key=lambda p: -p.upvotes)
    return ranked[:size]


def classify_cv_papers(client: anthropic.Anthropic, shortlist: list[Paper]) -> set[str]:
    """Ask Claude which shortlisted papers are CV / vision-language papers.

    Returns a set of arxiv_ids judged to be CV-related. We only pass
    titles here (abstracts are fetched later, only for the papers that
    end up selected, to save API calls) -- titles are almost always
    enough to tell whether a paper is visual in nature ("video", "image",
    "3D", "VLM", "diffusion", "world model", "segmentation", etc.), and
    Claude is asked to say when it's unsure rather than guess.
    """
    listing = "\n".join(f"- {p.arxiv_id}: {p.title}" for p in shortlist)
    prompt = f"""Below is a shortlist of AI research paper titles with their arXiv IDs.

Definition of "computer vision / vision-language paper" (broad): any paper whose
input and/or output involves images or video. This includes classic CV (detection,
segmentation, 3D reconstruction), image/video generation, video understanding,
vision-language / multimodal models, embodied or robotic models that consume visual
observations (e.g. RGB-D, video-based world models), and OCR/document vision.
It EXCLUDES papers that are purely about text-only LLMs, audio-only models, agentic
tool-use frameworks with no visual component, and general ML systems/infra papers.

Papers:
{listing}

Return ONLY a JSON array of arXiv IDs (strings) that qualify as CV/vision-language
papers under the definition above. If none qualify, return an empty array. Do not
include any other text."""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Be defensive about the model wrapping the JSON in prose or code fences.
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return set()
    try:
        ids = json.loads(text[start : end + 1])
        return {str(i) for i in ids}
    except json.JSONDecodeError:
        return set()


def select_two(shortlist: list[Paper], client: anthropic.Anthropic, seed: int | None = None) -> list[Selection]:
    """Apply the CV-preference + random selection rule to pick exactly 2 papers."""
    if seed is not None:
        random.seed(seed)

    cv_ids = classify_cv_papers(client, shortlist)
    cv_candidates = [p for p in shortlist if p.arxiv_id in cv_ids]

    selections: list[Selection] = []
    remaining = list(shortlist)

    if cv_candidates:
        best_cv = max(cv_candidates, key=lambda p: p.upvotes)
        selections.append(Selection(paper=_to_dict(best_cv), method="cv_preference"))
        remaining = [p for p in remaining if p.arxiv_id != best_cv.arxiv_id]

    slots_left = 2 - len(selections)
    if slots_left > 0 and remaining:
        picks = random.sample(remaining, k=min(slots_left, len(remaining)))
        for p in picks:
            selections.append(Selection(paper=_to_dict(p), method="random"))

    return selections


def _to_dict(p: Paper) -> dict:
    return {"arxiv_id": p.arxiv_id, "title": p.title, "upvotes": p.upvotes, "url": p.url}


def enrich_with_abstract(selection: Selection) -> Selection:
    """Fetch the full abstract/community note for a selected paper."""
    details = fetch_paper_abstract(selection.paper["arxiv_id"])
    selection.paper.update(details)
    return selection


if __name__ == "__main__":
    from fetch_papers import current_iso_week, fetch_week

    year, week = current_iso_week()
    papers = fetch_week(year, week)
    shortlist = build_shortlist(papers)
    print(f"Shortlist ({len(shortlist)} papers):")
    for p in shortlist:
        print(f"  [{p.upvotes:>3}] {p.title}")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    picks = select_two(shortlist, client)
    print("\nSelected:")
    for s in picks:
        print(f"  ({s.method}) {s.paper['title']}")
