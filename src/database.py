"""
SQLite storage layer for BioR&D Radar.

Keeps retrieved PubMed literature in a structured, queryable
form (papers table) and logs each analysis run (analysis_runs
table), so publication-activity and journal-frequency questions
can be answered with plain SQL rather than re-parsing CSVs.

Papers are stored once per (pmid, query_topic), so the database
accumulates literature across runs. Each analysis run records which
stored papers it actually retrieved in the run_papers link table.
Run-level aggregations (papers_by_year, journal_frequency) filter on
that link when given a run_id, so papers kept from earlier runs or
other date ranges for the same topic do not leak into the current
analysis.
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
    number_of_papers INTEGER,
    start_year INTEGER,
    end_year INTEGER
);

CREATE TABLE IF NOT EXISTS run_papers (
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id),
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    PRIMARY KEY (run_id, paper_id)
);
"""

# Columns added to analysis_runs after the first release; older
# databases are migrated in place by init_db().
_ANALYSIS_RUN_MIGRATIONS = {
    "start_year": "ALTER TABLE analysis_runs ADD COLUMN start_year INTEGER",
    "end_year": "ALTER TABLE analysis_runs ADD COLUMN end_year INTEGER",
}


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

    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(analysis_runs)")
    }
    for column, statement in _ANALYSIS_RUN_MIGRATIONS.items():
        if column not in existing:
            conn.execute(statement)

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


def log_analysis_run(
    topic,
    number_of_papers,
    db_path=DB_PATH,
    start_year=None,
    end_year=None,
):
    """
    Record that an analysis run happened, for auditability.
    Returns the new run's id.
    """
    conn = get_connection(db_path)
    cur = conn.execute(
        """
        INSERT INTO analysis_runs
        (topic, timestamp, number_of_papers, start_year, end_year)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            topic,
            datetime.now(timezone.utc).isoformat(),
            number_of_papers,
            start_year,
            end_year,
        ),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def link_run_papers(run_id, articles, query_topic, db_path=DB_PATH):
    """
    Record which stored papers belong to a run.

    Must be called after insert_papers(), so every article already has
    a (pmid, query_topic) row -- either newly inserted or kept from an
    earlier run. Returns the number of papers linked.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    for article in articles:
        cur.execute(
            """
            INSERT OR IGNORE INTO run_papers (run_id, paper_id)
            SELECT ?, id FROM papers
            WHERE pmid = ? AND query_topic = ?
            """,
            (run_id, article.get("pmid", ""), query_topic),
        )

    linked = conn.execute(
        "SELECT COUNT(*) FROM run_papers WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]

    conn.commit()
    conn.close()
    return linked


def _run_filter(run_id):
    """
    SQL fragment + params restricting `papers` to one run's papers.
    With run_id=None, no restriction is applied (all stored papers
    for the topic, accumulated across runs).
    """
    if run_id is None:
        return "", ()
    return (
        " AND id IN (SELECT paper_id FROM run_papers WHERE run_id = ?)",
        (run_id,),
    )


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


def papers_by_year(topic, db_path=DB_PATH, run_id=None):
    """
    GROUP BY year, COUNT(*) for a topic's publication activity.
    Rows with no parseable year are excluded.

    Pass run_id to count only the papers retrieved in that run
    (what the dashboard shows). Without it, counts cover every
    paper ever stored for the topic.
    """
    run_sql, run_params = _run_filter(run_id)
    conn = get_connection(db_path)
    rows = conn.execute(
        f"""
        SELECT year, COUNT(*) AS paper_count
        FROM papers
        WHERE query_topic = ? AND year IS NOT NULL{run_sql}
        GROUP BY year
        ORDER BY year ASC
        """,
        (topic, *run_params),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def journal_frequency(topic, db_path=DB_PATH, limit=10, run_id=None):
    """
    Which journals appear most often for this topic's retrieved set.

    Pass run_id to restrict to the papers retrieved in that run.
    """
    run_sql, run_params = _run_filter(run_id)
    conn = get_connection(db_path)
    rows = conn.execute(
        f"""
        SELECT journal, COUNT(*) AS paper_count
        FROM papers
        WHERE query_topic = ? AND journal IS NOT NULL AND journal != ''{run_sql}
        GROUP BY journal
        ORDER BY paper_count DESC, journal ASC
        LIMIT ?
        """,
        (topic, *run_params, limit),
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
