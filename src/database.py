"""
SQLite storage layer for BioR&D Radar.

Keeps retrieved PubMed literature in a structured, queryable
form (papers table) and logs each analysis run (analysis_runs
table), so publication-activity and journal-frequency questions
can be answered with plain SQL rather than re-parsing CSVs.
"""

import sqlite3
from datetime import datetime, timezone


DB_PATH = "data/radar.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT NOT NULL,
    title TEXT,
    abstract TEXT,
    authors TEXT,
    journal TEXT,
    publication_date TEXT,
    year INTEGER,
    pubmed_url TEXT,
    query_topic TEXT,
    UNIQUE(pmid, query_topic)
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    timestamp TEXT,
    number_of_papers INTEGER
);
"""


def get_connection(db_path=DB_PATH):
    """
    Open a SQLite connection with row access by column name.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """
    Create the papers/analysis_runs tables if they do not exist.
    Safe to call on every app startup.
    """
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def parse_year(raw_year):
    """
    PubMed year fields are messy ("2024", "2024 Nov-Dec 01",
    "2024-2025"). Pull out the first 4-digit number we find.
    Returns None if nothing parseable is present.
    """
    if raw_year is None:
        return None

    text = str(raw_year)

    digits = ""
    for i in range(len(text) - 3):
        chunk = text[i:i + 4]
        if chunk.isdigit():
            digits = chunk
            break

    return int(digits) if digits else None


def insert_papers(articles, query_topic, db_path=DB_PATH):
    """
    Insert a list of article dicts (as produced by
    data_loader.extract_articles) for a given topic.
    Duplicate (pmid, query_topic) pairs are ignored, so
    re-running the same topic does not create duplicate rows.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    inserted = 0

    for article in articles:
        year = parse_year(article.get("year", ""))

        cur.execute(
            """
            INSERT OR IGNORE INTO papers
            (pmid, title, abstract, authors, journal,
             publication_date, year, pubmed_url, query_topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article.get("pmid", ""),
                article.get("title", ""),
                article.get("abstract", ""),
                article.get("authors", ""),
                article.get("journal", ""),
                article.get("year", ""),
                year,
                article.get("url", ""),
                query_topic,
            ),
        )

        if cur.rowcount:
            inserted += 1

    conn.commit()
    conn.close()

    return inserted


def log_analysis_run(topic, number_of_papers, db_path=DB_PATH):
    """
    Record that an analysis run happened, for auditability.
    """
    conn = get_connection(db_path)
    conn.execute(
        """
        INSERT INTO analysis_runs (topic, timestamp, number_of_papers)
        VALUES (?, ?, ?)
        """,
        (
            topic,
            datetime.now(timezone.utc).isoformat(),
            number_of_papers,
        ),
    )
    conn.commit()
    conn.close()


def papers_for_topic(topic, db_path=DB_PATH):
    """
    All stored papers for a given topic, most recent year first.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT * FROM papers
        WHERE query_topic = ?
        ORDER BY year DESC
        """,
        (topic,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def papers_by_year(topic, db_path=DB_PATH):
    """
    GROUP BY year, COUNT(*) for a topic's publication activity.
    Rows with no parseable year are excluded.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT year, COUNT(*) AS paper_count
        FROM papers
        WHERE query_topic = ? AND year IS NOT NULL
        GROUP BY year
        ORDER BY year ASC
        """,
        (topic,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def journal_frequency(topic, db_path=DB_PATH, limit=10):
    """
    Which journals appear most often for this topic's retrieved set.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT journal, COUNT(*) AS paper_count
        FROM papers
        WHERE query_topic = ? AND journal IS NOT NULL AND journal != ''
        GROUP BY journal
        ORDER BY paper_count DESC, journal ASC
        LIMIT ?
        """,
        (topic, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def recent_papers(topic, db_path=DB_PATH, limit=10):
    """
    Most recent papers by parsed year for a topic.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT * FROM papers
        WHERE query_topic = ?
        ORDER BY year DESC
        LIMIT ?
        """,
        (topic, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def distinct_topics(db_path=DB_PATH):
    """
    Every topic that has ever been analyzed and stored.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT DISTINCT query_topic FROM papers ORDER BY query_topic ASC"
    ).fetchall()
    conn.close()
    return [row["query_topic"] for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"Initialized SQLite database at {DB_PATH}")
