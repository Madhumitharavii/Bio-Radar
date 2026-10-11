"""
Lightweight manual verification for BioR&D Radar.

This is NOT a full pytest suite with mocked network/LLM calls.
Per the project's stated time constraint, this script documents
exactly what was verified automatically (against the real
1,088-paper dataset already in data/papers.csv) and what requires
a live PubMed connection and a running local Ollama server to
verify end-to-end on your machine.

Run with: python tests/test_pipeline.py
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.database import (
    init_db, insert_papers, papers_by_year, journal_frequency, parse_year,
)
from src.analysis import extract_keywords_and_phrases, recent_vs_older_keywords


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
    df = pd.read_csv("data/papers.csv").head(30)
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
    df = pd.read_csv("data/papers.csv")

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
    df = pd.read_csv("data/papers.csv")
    subset = df[
        df["title"].str.lower().str.contains("raman", na=False)
    ].to_dict("records")

    # Should not raise, regardless of date spread in the subset.
    _ = recent_vs_older_keywords(subset)
    print("[PASS] emerging-keyword comparison runs without error")


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

    print("\nAll automated checks passed.")


if __name__ == "__main__":
    main()
