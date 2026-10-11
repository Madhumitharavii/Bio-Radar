
"""
Dynamic, user-driven PubMed retrieval for BioR&D Radar.
"""

import re

from src.data_loader import (
    search_pubmed,
    fetch_pubmed_records,
    extract_articles,
)


def build_dated_query(topic, start_year, end_year):
    """Wrap a topic in a PubMed publication-date filter."""
    return (
        f"({topic}) AND "
        f'("{start_year}"[Date - Publication] : '
        f'"{end_year}"[Date - Publication])'
    )


def _publication_year(article):
    """Return the electronic year, then issue year, then legacy year."""
    for field in ("pub_year_electronic", "pub_year_issue", "year"):
        value = str(article.get(field, "") or "").strip()
        match = re.search(r"\b(19|20)\d{2}\b", value)
        if match:
            return int(match.group(0))
    return None


def fetch_topic_literature(topic, start_year, end_year, max_results=30):
    """
    Retrieve papers and enforce the requested year range after retrieval.

    Papers without a usable publication year are excluded because their
    compliance with the requested date range cannot be verified.
    """
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("start_year and end_year must be integers")
    if start_year > end_year:
        raise ValueError("start_year cannot be later than end_year")

    query = build_dated_query(topic, start_year, end_year)
    pmids = search_pubmed(query, max_results=max_results)

    if not pmids:
        print("PubMed returned no papers.")
        return []

    records = fetch_pubmed_records(pmids)
    articles = extract_articles(records)

    with_abstract = [
        article for article in articles
        if article.get("abstract", "").strip()
    ]

    in_range = []
    dropped_date = 0

    for article in with_abstract:
        year = _publication_year(article)

        if year is None or not start_year <= year <= end_year:
            dropped_date += 1
            continue

        # Standardize year for downstream charts and filters.
        article["year"] = str(year)
        in_range.append(article)

    print(f"Retrieved records: {len(articles)}")
    print(f"Removed for missing abstract: {len(articles) - len(with_abstract)}")
    print(f"Removed for missing/out-of-range year: {dropped_date}")
    print(f"Papers retained: {len(in_range)}")

    return in_range
