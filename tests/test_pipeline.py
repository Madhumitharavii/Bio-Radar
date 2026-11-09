"""
Automated checks for BioR&D Radar.

Data-processing checks run against the real 1,088-paper dataset in
data/papers.csv. Pipeline checks replace PubMed, the embedding model
and Ollama with deterministic stand-ins, so they run offline; live
retrieval and LLM generation still need to be verified by running
the app (see README).

Run with: python tests/test_pipeline.py   (pytest also collects it)
"""

import sys
import os
import re
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.database import (
    init_db, insert_papers, papers_by_year, journal_frequency, parse_year,
    log_analysis_run, link_run_papers,
)
from src.analysis import extract_keywords_and_phrases, recent_vs_older_keywords
from src.evidence import select_evidence, build_evidence_context
from src.relevance import screen_paper
import src.radar_pipeline as radar_pipeline

PAPERS_CSV = ROOT / "data" / "papers.csv"


TEST_DB = os.path.join(
    tempfile.mkdtemp(prefix="biorad_test_"),
    "test.db",
)


def test_year_parsing():
    cases = {
        "2024": 2024,
        "2024 Nov-Dec 01": 2024,
        "2025 Sep-Oct 01": 2025,
        "": None,
        None: None,
    }
    for raw, expected in cases.items():
        assert parse_year(raw) == expected, f"parse_year({raw!r}) failed"
    print("[PASS] year parsing handles messy PubMed date formats")


def test_sqlite_storage_and_dedup():
    df = pd.read_csv(PAPERS_CSV).head(30)
    articles = df.to_dict("records")

    init_db(db_path=TEST_DB)
    first = insert_papers(articles, query_topic="unit-test-topic", db_path=TEST_DB)
    second = insert_papers(articles, query_topic="unit-test-topic", db_path=TEST_DB)

    assert first == len(articles), "expected all rows inserted on first pass"
    assert second == 0, "expected zero new rows on duplicate re-insert"
    print("[PASS] SQLite storage inserts correctly and dedupes on re-run")


def test_sql_aggregations():
    activity = papers_by_year("unit-test-topic", db_path=TEST_DB)
    journals = journal_frequency("unit-test-topic", db_path=TEST_DB)

    assert len(activity) > 0, "expected at least one year bucket"
    assert all("year" in row and "paper_count" in row for row in activity)
    assert len(journals) > 0, "expected at least one journal"
    print("[PASS] SQL GROUP BY aggregations (year, journal) return sane shapes")


def test_keyword_extraction_no_hardcoding():
    df = pd.read_csv(PAPERS_CSV)

    raman_subset = df[
        df["title"].str.lower().str.contains("raman", na=False)
        | df["abstract"].str.lower().str.contains("raman", na=False)
    ].to_dict("records")

    glucose_subset = df[
        df["title"].str.lower().str.contains("glucose", na=False)
    ].to_dict("records")

    for subset, expected_term in [
        (raman_subset, "raman"),
        (glucose_subset, "glucose"),
    ]:
        if len(subset) < 2:
            print(f"[SKIP] not enough rows for '{expected_term}' subset test")
            continue

        result = extract_keywords_and_phrases(subset, top_n=15)
        keywords = [k for k, _ in result["keywords"]]

        assert expected_term in keywords, (
            f"expected '{expected_term}' in top keywords for its own "
            f"topic subset, got {keywords}"
        )

    print(
        "[PASS] keyword extraction is topic-agnostic: correctly surfaces "
        "each subset's own defining term without any hard-coded topic logic"
    )


def test_emerging_keywords_runs():
    df = pd.read_csv(PAPERS_CSV)
    subset = df[
        df["title"].str.lower().str.contains("raman", na=False)
    ].to_dict("records")

    # Should not raise, regardless of date spread in the subset.
    _ = recent_vs_older_keywords(subset)
    print("[PASS] emerging-keyword comparison runs without error")


# ---------------------------------------------------------------------
# Run-scoped literature activity (date-range correctness)
# ---------------------------------------------------------------------

def _paper(pmid, year, journal="Journal A", title=None, abstract=None):
    return {
        "pmid": str(pmid),
        "title": title or f"Paper {pmid}",
        "abstract": abstract or f"Abstract text for paper {pmid}.",
        "authors": "Doe J",
        "journal": journal,
        "year": str(year),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _new_db():
    path = os.path.join(tempfile.mkdtemp(prefix="biorad_run_"), "runs.db")
    init_db(db_path=path)
    return path


def test_activity_is_scoped_to_current_run():
    db = _new_db()
    topic = "raman cancer"

    # Older run: 2015-2018, published mostly in "Old Journal".
    old_run_papers = [
        _paper(1, 2015, "Old Journal"),
        _paper(2, 2016, "Old Journal"),
        _paper(3, 2017, "Old Journal"),
        _paper(4, 2018, "Shared Journal"),
    ]
    insert_papers(old_run_papers, topic, db_path=db)
    old_run = log_analysis_run(topic, 4, db_path=db, start_year=2015, end_year=2018)
    link_run_papers(old_run, old_run_papers, topic, db_path=db)

    # Current run: 2023-2025. Papers stored by the older run (e.g.
    # PMID 4 in "Shared Journal") must not leak into its counts.
    current_papers = [
        _paper(10, 2023, "New Journal"),
        _paper(11, 2024, "New Journal"),
        _paper(12, 2025, "Shared Journal"),
    ]
    insert_papers(current_papers, topic, db_path=db)
    current_run = log_analysis_run(topic, 3, db_path=db, start_year=2023, end_year=2025)
    linked = link_run_papers(current_run, current_papers, topic, db_path=db)
    assert linked == 3

    activity = papers_by_year(topic, db_path=db, run_id=current_run)
    assert activity == [
        {"year": 2023, "paper_count": 1},
        {"year": 2024, "paper_count": 1},
        {"year": 2025, "paper_count": 1},
    ], activity

    journals = {
        row["journal"]: row["paper_count"]
        for row in journal_frequency(topic, db_path=db, run_id=current_run)
    }
    assert journals == {"New Journal": 2, "Shared Journal": 1}, journals

    # Without a run_id the historical, topic-wide view is still available.
    all_years = [r["year"] for r in papers_by_year(topic, db_path=db)]
    assert all_years == [2015, 2016, 2017, 2018, 2023, 2024, 2025]
    print("[PASS] activity/journal SQL is scoped to the current run only")


def test_rerun_overlapping_papers_dedupes_and_links():
    db = _new_db()
    topic = "glucose sensing"

    first = [_paper(1, 2020), _paper(2, 2021), _paper(3, 2022)]
    insert_papers(first, topic, db_path=db)
    run_a = log_analysis_run(topic, 3, db_path=db, start_year=2020, end_year=2022)
    link_run_papers(run_a, first, topic, db_path=db)

    # Narrower re-run that retrieves a subset of the same papers.
    second = [_paper(2, 2021), _paper(3, 2022)]
    inserted = insert_papers(second, topic, db_path=db)
    run_b = log_analysis_run(topic, 2, db_path=db, start_year=2021, end_year=2022)
    linked = link_run_papers(run_b, second, topic, db_path=db)

    assert inserted == 0, "already-stored papers must not be duplicated"
    assert linked == 2, "existing rows must still be linked to the new run"

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3
    conn.close()

    years = [r["year"] for r in papers_by_year(topic, db_path=db, run_id=run_b)]
    assert years == [2021, 2022], years
    print("[PASS] re-runs dedupe stored papers but link them to the new run")


def test_init_db_migrates_legacy_analysis_runs_table():
    path = os.path.join(tempfile.mkdtemp(prefix="biorad_legacy_"), "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT, timestamp TEXT, number_of_papers INTEGER
        );
        INSERT INTO analysis_runs (topic, timestamp, number_of_papers)
        VALUES ('old topic', '2026-01-01T00:00:00', 5);
        """
    )
    conn.commit()
    conn.close()

    init_db(db_path=path)
    init_db(db_path=path)  # idempotent

    run_id = log_analysis_run("new topic", 2, db_path=path, start_year=2020, end_year=2021)
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT topic, start_year, end_year FROM analysis_runs ORDER BY id"
    ).fetchall()
    conn.close()
    assert rows == [("old topic", None, None), ("new topic", 2020, 2021)]
    assert run_id == 2
    print("[PASS] init_db migrates a pre-existing radar.db in place")


def test_pipeline_activity_matches_selected_date_range():
    """
    Two pipeline runs on the same topic with different date ranges,
    using the real SQLite code and offline stand-ins for PubMed, the
    embedding model and Ollama. The second run's Literature Activity
    and journal counts must only reflect the second run's papers.
    """
    batches = {
        (2010, 2014): [_paper(100 + y, y, "Old Journal") for y in range(2010, 2015)],
        (2022, 2024): [_paper(200 + y, y, "New Journal") for y in range(2022, 2025)],
    }
    prompts = []

    patches = {
        "fetch_topic_literature": lambda topic, s, e, max_results=30: [
            dict(a) for a in batches[(s, e)]
        ],
        "build_run_embeddings": lambda articles: (
            np.eye(len(articles), dtype="float32"), None
        ),
        "topic_similarity_scores": lambda topic, emb: np.arange(
            emb.shape[0], dtype="float32"
        ),
        "generate_structured_analysis": lambda topic, context: (
            prompts.append(context) or "analysis"
        ),
        "generate_opportunities": lambda topic, context: "opportunities",
    }
    originals = {name: getattr(radar_pipeline, name) for name in patches}
    cwd = os.getcwd()
    workdir = tempfile.mkdtemp(prefix="biorad_pipeline_")
    os.makedirs(os.path.join(workdir, "data"))

    try:
        for name, fn in patches.items():
            setattr(radar_pipeline, name, fn)
        os.chdir(workdir)  # DB_PATH is relative: data/radar.db

        first = radar_pipeline.run_radar_analysis("topic x", 2010, 2014)
        second = radar_pipeline.run_radar_analysis(
            "topic x", 2022, 2024, max_context_sources=2
        )
    finally:
        os.chdir(cwd)
        for name, fn in originals.items():
            setattr(radar_pipeline, name, fn)

    assert first["warnings"] == [], first["warnings"]
    assert second["warnings"] == [], second["warnings"]
    assert second["run_id"] == first["run_id"] + 1

    years = [row["year"] for row in second["activity_by_year"]]
    assert years == [2022, 2023, 2024], years
    journals = [row["journal"] for row in second["journal_counts"]]
    assert journals == ["New Journal"], journals

    # All papers stay available to the UI; only the LLM context is bounded.
    assert second["num_papers"] == 3
    assert second["context_sources"] == [2, 3]
    assert "SOURCE 2" in prompts[-1] and "SOURCE 3" in prompts[-1]
    assert "SOURCE 1\n" not in prompts[-1]
    print("[PASS] pipeline Literature Activity matches the selected date range")


# ---------------------------------------------------------------------
# Emerging keywords
# ---------------------------------------------------------------------

def test_emerging_keywords_detects_recent_term():
    older = [
        _paper(i, 2018, abstract="Raman spectra classified with support "
               "vector machine and principal component analysis.")
        for i in range(1, 5)
    ]
    recent = [
        _paper(i, 2024, abstract="Raman spectra classified with a "
               "transformer network for tissue diagnosis.")
        for i in range(5, 9)
    ]
    emerging = {term: (r, o) for term, r, o in recent_vs_older_keywords(older + recent)}

    assert "transformer" in emerging, emerging
    assert emerging["transformer"] == (1.0, 0.0)
    assert "raman" not in emerging, "terms common to both periods are not emerging"
    assert "vector" not in emerging, "older-only terms are not emerging"

    # Too few dated papers in one period -> no signal rather than noise.
    assert recent_vs_older_keywords(recent) == []
    print("[PASS] emerging keywords flag recent-only terms, not shared ones")


# ---------------------------------------------------------------------
# LLM evidence selection
# ---------------------------------------------------------------------

def test_evidence_selection_ranks_and_keeps_numbering():
    articles = [_paper(i, 2020 + i % 5) for i in range(1, 9)]
    scores = [0.1, 0.9, 0.3, 0.9, 0.2, 0.8, 0.05, 0.4]

    selected = select_evidence(articles, scores, max_sources=3)
    # Highest scores are sources 2, 4 (tie -> original order) and 6;
    # returned in original order with original 1-based numbers.
    assert [n for n, _ in selected] == [2, 4, 6]
    assert [a["pmid"] for _, a in selected] == ["2", "4", "6"]
    assert select_evidence(articles, scores, 3) == selected  # deterministic

    # No scores -> PubMed relevance order.
    assert [n for n, _ in select_evidence(articles, None, 3)] == [1, 2, 3]
    # Small runs keep everything.
    assert len(select_evidence(articles[:2], None, 15)) == 2

    long_abstract = "word " * 1000
    articles[1]["abstract"] = long_abstract
    context, used = build_evidence_context(
        articles, scores, max_sources=3, max_abstract_chars=200
    )
    assert used == [2, 4, 6]
    assert re.findall(r"^SOURCE (\d+)$", context, flags=re.M) == ["2", "4", "6"]
    for pmid in ("2", "4", "6"):
        assert f"PMID: {pmid}\n" in context
        assert f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/" in context
    assert "[abstract truncated]" in context
    assert len(context) < len(long_abstract)
    print("[PASS] evidence selection is ranked, bounded, and keeps [Source N] numbering")


# ---------------------------------------------------------------------
# Experimental hyperspectral Raman screening module
# ---------------------------------------------------------------------

def test_hyperspectral_raman_screening_labels():
    strong = screen_paper({
        "title": "Hyperspectral Raman imaging of thyroid tumour biopsies",
        "abstract": "Deep learning classification for cancer diagnosis.",
    })
    assert strong["relevance_label"] == "strong_candidate"
    assert strong["uses_hyperspectral_raman"] == "yes"
    assert strong["ai_ml"] == "yes"

    food = screen_paper({
        "title": "Hyperspectral Raman imaging for milk adulteration",
        "abstract": "Detection of adulterants in milk powder.",
    })
    assert food["relevance_label"] == "adjacent_method"

    unrelated = screen_paper({
        "title": "Gut microbiome and diet",
        "abstract": "A cohort study of dietary fibre intake.",
    })
    assert unrelated["relevance_label"] == "out_of_scope"
    assert strong["verification_status"] == "requires_full_text_methods_review"
    print("[PASS] experimental Raman screening assigns expected preliminary labels")


# ---------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------

def test_no_hard_coded_email_in_source():
    pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    files = [ROOT / "app.py", ROOT / "fetch_batch.py",
             ROOT / "screen_candidates.py", *(ROOT / "src").glob("*.py")]
    offenders = [
        f"{path.name}:{n}"
        for path in files
        for n, line in enumerate(path.read_text().splitlines(), start=1)
        if pattern.search(line)
    ]
    assert not offenders, offenders
    print("[PASS] no hard-coded email addresses in source code")


def main():
    print("Running BioR&D Radar manual verification...\n")
    print("NOTE: PubMed live retrieval and Ollama LLM generation are")
    print("NOT exercised here -- they require network access and a")
    print("running local Ollama server respectively. Verify those by")
    print("running `streamlit run app.py` with Ollama running and")
    print("entering a real topic.\n")

    test_year_parsing()
    test_sqlite_storage_and_dedup()
    test_sql_aggregations()
    test_keyword_extraction_no_hardcoding()
    test_emerging_keywords_runs()
    test_activity_is_scoped_to_current_run()
    test_rerun_overlapping_papers_dedupes_and_links()
    test_init_db_migrates_legacy_analysis_runs_table()
    test_pipeline_activity_matches_selected_date_range()
    test_emerging_keywords_detects_recent_term()
    test_evidence_selection_ranks_and_keeps_numbering()
    test_hyperspectral_raman_screening_labels()
    test_no_hard_coded_email_in_source()

    print("\nAll automated checks passed.")


if __name__ == "__main__":
    main()
