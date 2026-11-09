"""
Evidence selection for the LLM context in BioR&D Radar.

A run can retrieve up to 50 papers. All of them are still stored,
charted, keyword-analyzed and listed in the Papers tab, but sending
every full abstract to a local 8B model produces very long prompts
that can exceed the model's context window (silently truncating the
evidence) and slow generation down.

Selection logic (deterministic for a given input):

1. Each paper gets a relevance score. When per-paper semantic
   similarity to the topic is available (cosine similarity of
   Sentence-Transformer embeddings), that score is used. Otherwise
   PubMed's own relevance order is used (earlier = more relevant).
2. Papers are ranked by score; ties break by original PubMed order.
3. The top `max_sources` papers are kept.
4. Kept papers are presented in their original order and keep their
   original [Source N] numbers, so citations in the LLM output match
   the numbering in the Papers tab.
5. Abstracts longer than `max_abstract_chars` are cut at a word
   boundary and marked as truncated. Title, authors, journal, year,
   PMID and URL are always kept in full.
"""

from src.retriever import build_context


DEFAULT_MAX_SOURCES = 15
DEFAULT_MAX_ABSTRACT_CHARS = 1500


def truncate_abstract(text, max_chars=DEFAULT_MAX_ABSTRACT_CHARS):
    """Shorten an abstract at a word boundary, marking the cut."""
    text = str(text or "").strip()

    if max_chars is None or len(text) <= max_chars:
        return text

    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut} ... [abstract truncated]"


def select_evidence(articles, scores=None, max_sources=DEFAULT_MAX_SOURCES):
    """
    Choose which papers go into the LLM context.

    Args:
        articles: Retrieved papers, in PubMed relevance order.
        scores: Optional per-paper relevance scores (same length as
            articles, higher = more relevant).
        max_sources: Maximum number of papers to keep.

    Returns:
        List of (source_number, article) tuples in original order,
        where source_number is the paper's 1-based position in
        `articles` (its [Source N] number in the UI).
    """
    if scores is not None and len(scores) != len(articles):
        raise ValueError("scores must have one value per article")

    positions = list(range(len(articles)))

    if scores is None:
        ranked = positions
    else:
        ranked = sorted(positions, key=lambda i: (-float(scores[i]), i))

    keep = sorted(ranked[:max_sources])

    return [(i + 1, articles[i]) for i in keep]


def build_evidence_context(
    articles,
    scores=None,
    max_sources=DEFAULT_MAX_SOURCES,
    max_abstract_chars=DEFAULT_MAX_ABSTRACT_CHARS,
):
    """
    Build the LLM context from a bounded, ranked subset of papers.

    Returns (context_text, source_numbers_used).
    """
    selected = select_evidence(articles, scores, max_sources)

    results = [
        {
            "title": a.get("title", ""),
            "authors": a.get("authors", ""),
            "journal": a.get("journal", ""),
            "year": a.get("year", ""),
            "pmid": a.get("pmid", ""),
            "url": a.get("url", ""),
            "abstract": truncate_abstract(
                a.get("abstract", ""), max_abstract_chars
            ),
        }
        for _, a in selected
    ]
    source_numbers = [number for number, _ in selected]

    context = build_context(results, source_numbers=source_numbers)

    return context, source_numbers
