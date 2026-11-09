"""
BioR&D Radar end-to-end pipeline.

SEARCH -> STRUCTURE -> CONNECT -> ANALYZE -> (returned for)
VISUALIZE / OPPORTUNITY DISCOVERY in the Streamlit app layer.

This module orchestrates existing, reused components
(topic_search, database, embedder/connections, generator) rather
than reimplementing any of them. If a step fails (e.g. no network
to PubMed, or no local Ollama server running), the pipeline
degrades gracefully and reports exactly what succeeded, instead of
silently fabricating output.
"""

from src.topic_search import fetch_topic_literature
from src.database import (
    init_db,
    insert_papers,
    log_analysis_run,
    link_run_papers,
    papers_by_year,
    journal_frequency,
)
from src.analysis import (
    extract_keywords_and_phrases,
    recent_vs_older_keywords,
)
from src.connections import (
    build_run_embeddings,
    most_similar_pairs,
    topic_similarity_scores,
)
from src.evidence import build_evidence_context, DEFAULT_MAX_SOURCES
from src.generator import generate_structured_analysis, generate_opportunities


def run_radar_analysis(
    topic,
    start_year,
    end_year,
    max_results=30,
    max_context_sources=DEFAULT_MAX_SOURCES,
):
    """
    Run the full BioR&D Radar pipeline for one topic.

    Returns a result dict with every stage's output and a
    `warnings` list describing any stage that could not complete
    (so the UI can show a clear partial result instead of a
    silent failure or fabricated content).
    """
    warnings = []
    result = {
        "topic": topic,
        "start_year": start_year,
        "end_year": end_year,
        "articles": [],
        "num_papers": 0,
        "keywords": [],
        "phrases": [],
        "emerging_keywords": [],
        "activity_by_year": [],
        "journal_counts": [],
        "connections": [],
        "run_id": None,
        "context_sources": [],
        "context_selection": None,
        "structured_analysis": None,
        "opportunities": None,
        "warnings": warnings,
    }

    init_db()

    # 1. SEARCH
    try:
        articles = fetch_topic_literature(
            topic, start_year, end_year, max_results=max_results
        )
    except Exception as exc:
        warnings.append(f"PubMed retrieval failed: {exc}")
        articles = []

    if not articles:
        warnings.append(
            "No papers with usable abstracts were retrieved for this "
            "topic/date range. Try a broader topic or wider date range."
        )
        return result

    result["articles"] = articles
    result["num_papers"] = len(articles)

    # 2. STRUCTURE (store + log)
    # Papers are deduplicated per topic across runs; run_papers links
    # this run to exactly the papers it retrieved, so the activity and
    # journal aggregations below reflect only the current topic/date
    # range, not papers kept from earlier runs.
    run_id = None
    try:
        insert_papers(articles, query_topic=topic)
        run_id = log_analysis_run(
            topic, len(articles), start_year=start_year, end_year=end_year
        )
        link_run_papers(run_id, articles, query_topic=topic)
        result["run_id"] = run_id
    except Exception as exc:
        warnings.append(f"Database storage failed: {exc}")

    if run_id is not None:
        try:
            result["activity_by_year"] = papers_by_year(topic, run_id=run_id)
            result["journal_counts"] = journal_frequency(topic, run_id=run_id)
        except Exception as exc:
            warnings.append(f"SQL analysis queries failed: {exc}")

    try:
        kw_result = extract_keywords_and_phrases(articles)
        result["keywords"] = kw_result["keywords"]
        result["phrases"] = kw_result["phrases"]
        result["emerging_keywords"] = recent_vs_older_keywords(articles)
    except Exception as exc:
        warnings.append(f"Keyword/theme extraction failed: {exc}")

    # 3. CONNECT (semantic similarity on this run's papers)
    relevance_scores = None
    embeddings = None
    try:
        embeddings, _index = build_run_embeddings(articles)
        result["connections"] = most_similar_pairs(articles, embeddings)
    except Exception as exc:
        embeddings = None
        warnings.append(
            f"Semantic connection analysis unavailable "
            f"(embeddings/FAISS step failed): {exc}"
        )

    if embeddings is not None:
        try:
            relevance_scores = topic_similarity_scores(topic, embeddings)
        except Exception as exc:
            warnings.append(
                f"Topic-similarity ranking failed; using PubMed relevance "
                f"order to select LLM evidence: {exc}"
            )

    # 4. ANALYZE (grounded context + structured LLM analysis)
    # Every retrieved paper stays in the result and the Papers tab.
    # Only the LLM context is bounded: the papers most semantically
    # similar to the topic (or PubMed's relevance order if embeddings
    # are unavailable) are kept, with their original [Source N]
    # numbers. See src/evidence.py.
    context, result["context_sources"] = build_evidence_context(
        articles,
        scores=relevance_scores,
        max_sources=max_context_sources,
    )
    result["context_selection"] = (
        "semantic similarity to the topic"
        if relevance_scores is not None
        else "PubMed relevance order"
    )

    try:
        result["structured_analysis"] = generate_structured_analysis(
            topic, context
        )
    except Exception as exc:
        warnings.append(
            f"AI structured analysis unavailable (is Ollama running "
            f"with the qwen3:8b model pulled?): {exc}"
        )

    try:
        result["opportunities"] = generate_opportunities(topic, context)
    except Exception as exc:
        warnings.append(
            f"R&D opportunity generation unavailable (is Ollama "
            f"running with the qwen3:8b model pulled?): {exc}"
        )

    return result


if __name__ == "__main__":
    out = run_radar_analysis(
        topic="non-invasive glucose monitoring",
        start_year=2022,
        end_year=2026,
        max_results=20,
    )

    print(f"Papers retrieved: {out['num_papers']}")
    print(f"Warnings: {out['warnings']}")
    print("Top keywords:", out["keywords"][:5])
