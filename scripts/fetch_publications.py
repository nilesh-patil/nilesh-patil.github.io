#!/usr/bin/env python3
"""Fetch publications from Google Scholar + arxiv and emit _publications/*.md.

Two-source pipeline per docs/specs.md §8:

  Source 1 — Google Scholar (primary) via `scholarly`, profile IIabY1sAAAAJ.
  Source 2 — arxiv API (supplementary) for Dream11 preprints not on Scholar.

Idempotent: skips files that already exist.

  python scripts/fetch_publications.py             # only adds new entries
  python scripts/fetch_publications.py --refresh   # also updates managed
                                                   # fields on existing files

Managed fields (overwritten on --refresh): excerpt, authors, venue, paper_url,
date, citation_count.

NEVER overwritten on --refresh: citation, code_url, tags, body content, and
any other manual edits to the YAML frontmatter or markdown body.

Behavior on failure:
  - Scholar throttles (HTTP 429 / blank profile): log + exit 0 with no writes.
  - arxiv API error: log + continue (Scholar results are still emitted).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import frontmatter  # python-frontmatter

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLICATIONS_DIR = REPO_ROOT / "_publications"
AUTHORS_FILE = REPO_ROOT / "scripts" / "dream11_authors.txt"

SCHOLAR_USER_ID = "IIabY1sAAAAJ"
BASE_AUTHOR = "Nilesh Patil"

# Fields the script manages. Anything else in an existing file is preserved.
MANAGED_FIELDS = {
    "excerpt",
    "authors",
    "venue",
    "paper_url",
    "paperurl",
    "date",
    "citation_count",
}

DUPE_SIMILARITY_THRESHOLD = 0.85

logger = logging.getLogger("fetch_publications")


# ---------------------------------------------------------------------------
# Slug + dedupe helpers
# ---------------------------------------------------------------------------


def slugify(title: str, max_len: int = 60) -> str:
    """Lowercased, dash-separated slug truncated to ~max_len chars."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0]
    return s or "untitled"


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def titles_match(a: str, b: str) -> bool:
    """Jaccard-like fuzzy match on normalized titles."""
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio() >= DUPE_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Source 1 — Google Scholar
# ---------------------------------------------------------------------------


def fetch_scholar(user_id: str) -> list[dict[str, Any]]:
    """Return a list of publication dicts from Google Scholar."""
    try:
        from scholarly import scholarly  # type: ignore
    except ImportError:
        logger.error("scholarly not installed. Run: pip install -r scripts/requirements.txt")
        return []

    try:
        author = scholarly.search_author_id(user_id)
        scholarly.fill(author, sections=["publications"])
    except Exception as exc:
        # Scholar throttling, DOM changes, or network errors all land here.
        # Per spec: log + exit cleanly without partial writes.
        logger.warning("Google Scholar fetch failed (%s); skipping Scholar source.", exc)
        return []

    out: list[dict[str, Any]] = []
    for pub in author.get("publications", []):
        try:
            scholarly.fill(pub)
        except Exception as exc:
            logger.warning("Failed to fill publication; skipping. (%s)", exc)
            continue
        bib = pub.get("bib", {}) or {}
        title = bib.get("title")
        if not title:
            continue
        venue = bib.get("journal") or bib.get("conference") or bib.get("book") or bib.get("publisher", "")
        out.append({
            "title": title,
            "authors": bib.get("author", BASE_AUTHOR),
            "venue": venue,
            "year": bib.get("pub_year") or bib.get("year"),
            "abstract": bib.get("abstract", ""),
            "paper_url": pub.get("pub_url") or pub.get("eprint_url") or "",
            "citation_count": pub.get("num_citations"),
            "source": "scholar",
        })
    logger.info("Scholar: %d publications fetched.", len(out))
    return out


# ---------------------------------------------------------------------------
# Source 2 — arxiv
# ---------------------------------------------------------------------------


def fetch_arxiv(authors: Iterable[str]) -> list[dict[str, Any]]:
    """Return arxiv hits across the given author list."""
    try:
        import arxiv  # type: ignore
    except ImportError:
        logger.error("arxiv not installed. Run: pip install -r scripts/requirements.txt")
        return []

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for name in authors:
        query = f'au:"{name}"'
        try:
            search = arxiv.Search(query=query, max_results=30)
            client = arxiv.Client()
            for result in client.results(search):
                if result.entry_id in seen_ids:
                    continue
                seen_ids.add(result.entry_id)
                out.append({
                    "title": result.title,
                    "authors": ", ".join(a.name for a in result.authors),
                    "venue": "arXiv preprint",
                    "year": result.published.year if result.published else None,
                    "abstract": result.summary,
                    "paper_url": result.entry_id,
                    "source": "arxiv",
                })
        except Exception as exc:
            logger.warning("arxiv query for %r failed (%s); continuing.", name, exc)
    logger.info("arxiv: %d unique publications fetched across %d authors.", len(out), sum(1 for _ in authors))
    return out


def load_dream11_authors() -> list[str]:
    """Read scripts/dream11_authors.txt; return list with BASE_AUTHOR included."""
    authors = [BASE_AUTHOR]
    if AUTHORS_FILE.exists():
        for line in AUTHORS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                authors.append(line)
    return authors


# ---------------------------------------------------------------------------
# Dedupe + write
# ---------------------------------------------------------------------------


def dedupe_against_existing(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_titles: list[tuple[str, Path]] = []
    for path in PUBLICATIONS_DIR.glob("*.md"):
        try:
            post = frontmatter.load(path)
            t = post.metadata.get("title")
            if t:
                existing_titles.append((t, path))
        except Exception:
            continue

    kept = []
    for item in items:
        if any(titles_match(item["title"], t) for t, _ in existing_titles):
            logger.info("Skipping duplicate of existing: %s", item["title"])
            continue
        kept.append(item)
    return kept


def dedupe_pair(scholar: list[dict[str, Any]], arxiv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop arxiv hits that match a scholar hit by fuzzy title."""
    merged = list(scholar)
    for item in arxiv:
        if any(titles_match(item["title"], s["title"]) for s in scholar):
            continue
        merged.append(item)
    return merged


def bold_author(text: str) -> str:
    """Wrap the author name in <strong> for case-insensitive variants.

    Liquid does not escape HTML in `{{ ... }}`, so <strong> survives to render.
    Variants handled: 'Nilesh Patil', 'N. Patil', 'Patil, N.', 'Patil N'.
    """
    if not text:
        return text
    patterns = [
        r"(?<!\w)Nilesh Patil(?!\w)",
        r"(?<!\w)N\. Patil(?!\w)",
        r"(?<!\w)Patil, N\.?(?!\w)",
        r"(?<!\w)Patil N(?!\w)",
    ]
    for pat in patterns:
        text = re.sub(pat, lambda m: f"<strong>{m.group(0)}</strong>", text, flags=re.IGNORECASE)
    # Collapse any double-wraps from overlapping patterns.
    text = re.sub(r"<strong>(<strong>.*?</strong>)</strong>", r"\1", text)
    return text


def build_frontmatter(item: dict[str, Any]) -> dict[str, Any]:
    year = item.get("year")
    date = f"{year}-01-01" if year else ""
    slug = slugify(item["title"])
    excerpt = item.get("abstract", "") or ""
    if excerpt:
        excerpt = excerpt.split("\n\n")[0][:300]
    url = item.get("paper_url", "")
    authors = bold_author(item.get("authors", BASE_AUTHOR))
    fm: dict[str, Any] = {
        "title": item["title"],
        "collection": "publications",
        "permalink": f"/publications/{slug}/",
        "date": date,
        "venue": item.get("venue", ""),
        "paper_url": url,
        # academicpages templates read `paperurl` (one word); spec uses
        # `paper_url`. We emit both so manual edits + templates agree.
        "paperurl": url,
        "authors": authors,
        "excerpt": excerpt,
    }
    if item.get("citation_count") is not None:
        fm["citation_count"] = item["citation_count"]
    return fm


def write_new(item: dict[str, Any]) -> Path | None:
    slug = slugify(item["title"])
    year = item.get("year") or "undated"
    out_path = PUBLICATIONS_DIR / f"{year}-{slug}.md"
    if out_path.exists():
        logger.info("Already exists, skipping: %s", out_path.name)
        return None
    post = frontmatter.Post(content="", **build_frontmatter(item))
    out_path.write_bytes(frontmatter.dumps(post).encode("utf-8") + b"\n")
    logger.info("Wrote new: %s", out_path.name)
    return out_path


def refresh_existing(item: dict[str, Any]) -> Path | None:
    """Merge managed fields into an existing file matching this title."""
    target: Path | None = None
    for path in PUBLICATIONS_DIR.glob("*.md"):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        if titles_match(post.metadata.get("title", ""), item["title"]):
            target = path
            break
    if not target:
        return None
    post = frontmatter.load(target)
    fm = build_frontmatter(item)
    for key, value in fm.items():
        if key in MANAGED_FIELDS and value not in (None, ""):
            post.metadata[key] = value
    target.write_bytes(frontmatter.dumps(post).encode("utf-8") + b"\n")
    logger.info("Refreshed: %s", target.name)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Update managed fields on existing entries (does not overwrite "
        "citation, code_url, tags, or body).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Log INFO + DEBUG messages."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    PUBLICATIONS_DIR.mkdir(exist_ok=True)

    scholar_items = fetch_scholar(SCHOLAR_USER_ID)
    arxiv_items = fetch_arxiv(load_dream11_authors())
    merged = dedupe_pair(scholar_items, arxiv_items)

    if not merged:
        logger.warning("No publications fetched from either source. Nothing to do.")
        return 0

    if args.refresh:
        for item in merged:
            if refresh_existing(item) is None:
                write_new(item)
    else:
        for item in dedupe_against_existing(merged):
            write_new(item)

    return 0


if __name__ == "__main__":
    sys.exit(main())
