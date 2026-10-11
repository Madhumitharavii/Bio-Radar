
"""
Data-driven analysis of retrieved biomedical literature.

Extracts keywords and phrases from paper titles and abstracts using
TF-IDF. Compares recent and older publications to identify terms
that are relatively more prominent in the recent subset.

These signals describe only the retrieved literature sample. They
do not establish that a technology is commercially promising or
that a research area is growing across the entire literature.
"""

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from src.database import papers_by_year, parse_year


SIMPLE_STOPWORDS = {
    "study",
    "studies",
    "result",
    "results",
    "method",
    "methods",
    "using",
    "based",
    "used",
    "use",
    "paper",
    "papers",
    "approach",
    "research",
    "analysis",
    "data",
    "patient",
    "patients",
    "clinical",
    "group",
    "groups",
    "effect",
    "effects",
    "resulting",
    "provide",
    "provides",
    "provided",
    "potential",
    "associated",
    "respectively",
    "however",
    "therefore",
    "also",
    "may",
    "might",
    "could",
    "would",
    "et",
    "al",
}


def clean_text(text):
    """Normalize text and remove markup from PubMed abstracts."""
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _prepare_texts(articles):
    """Combine and clean the title and abstract of each article."""
    return [
        clean_text(
            f"{article.get('title', '')} "
            f"{article.get('abstract', '')}"
        )
        for article in articles
    ]


def _valid_term(term):
    """Exclude generic terms and terms that contain no letters."""
    if term in SIMPLE_STOPWORDS:
        return False

    if not re.search(r"[a-z]", term):
        return False

    return True


def _rank_terms(texts, ngram_range=(1, 1), top_n=15):
    """
    Rank terms by their summed TF-IDF weight across the document set.

    Returns a list of (term, score) tuples.
    """
    texts = [text for text in texts if text.strip()]

    if len(texts) < 2:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=ngram_range,
        min_df=1,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Handles empty vocabularies and abstracts with no usable terms.
        return []

    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for term, score in ranked:
        if ngram_range == (1, 1) and not _valid_term(term):
            continue

        results.append((term, round(float(score), 3)))

        if len(results) >= top_n:
            break

    return results


def extract_keywords_and_phrases(articles, top_n=15):
    """
    Extract single-word keywords and two-word phrases.

    The vocabulary is learned from the supplied articles, so the
    function can be used for different biomedical research topics.

    Args:
        articles: List of dictionaries containing title and abstract.
        top_n: Maximum number of keywords and phrases to return.

    Returns:
        Dictionary with 'keywords' and 'phrases' lists. Each list
        contains (term, score) tuples.
    """
    texts = _prepare_texts(articles)
    texts = [text for text in texts if text.strip()]

    if len(texts) < 2:
        return {
            "keywords": [],
            "phrases": [],
        }

    keywords = _rank_terms(
        texts,
        ngram_range=(1, 1),
        top_n=top_n,
    )

    phrases = _rank_terms(
        texts,
        ngram_range=(2, 2),
        top_n=top_n,
    )

    return {
        "keywords": keywords,
        "phrases": phrases,
    }


def publication_activity(topic):
    """
    Return publication counts grouped by year for a topic.

    Delegates the SQL aggregation to database.py.
    """
    return papers_by_year(topic)


def _term_document_frequencies(texts, top_n=100):
    """
    Calculate the proportion of documents containing each keyword.

    Each document contributes at most once per term. This makes the
    resulting scores easier to compare between groups of different
    sizes than raw summed TF-IDF values.
    """
    texts = [text for text in texts if text.strip()]

    if len(texts) < 2:
        return {}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 1),
        min_df=1,
        binary=True,
        use_idf=False,
        norm=None,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return {}

    document_counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, document_counts),
        key=lambda item: item[1],
        reverse=True,
    )

    frequencies = {}

    for term, count in ranked:
        if not _valid_term(term):
            continue

        frequencies[term] = float(count) / len(texts)

        if len(frequencies) >= top_n:
            break

    return frequencies


def recent_vs_older_keywords(articles, recent_years=2):
    """
    Identify keywords relatively more prevalent in recent papers.

    The recent period is determined from the latest publication year
    in the supplied sample, not the current calendar year.

    A keyword is flagged when:
      - At least two articles are available in each period.
      - It occurs in at least two recent articles.
      - Its recent document frequency is more than 1.5 times its
        older document frequency, or it is absent from the older set.

    Returns:
        A list of (term, recent_score, older_score) tuples.

    Important:
        This is a descriptive signal within the retrieved sample.
        Retrieval limits, topic wording, publication dates, and sample
        size can affect the results. It is not proof of a field-wide
        trend, novelty, or commercial opportunity.
    """
    if recent_years < 1:
        raise ValueError("recent_years must be at least 1.")

    dated_articles = []

    for article in articles:
        year = parse_year(article.get("year", ""))

        if year is not None:
            dated_articles.append((article, year))

    if not dated_articles:
        return []

    latest_year = max(year for _, year in dated_articles)
    cutoff_year = latest_year - recent_years + 1

    recent_articles = [
        article
        for article, year in dated_articles
        if year >= cutoff_year
    ]

    older_articles = [
        article
        for article, year in dated_articles
        if year < cutoff_year
    ]

    if len(recent_articles) < 2 or len(older_articles) < 2:
        return []

    recent_texts = _prepare_texts(recent_articles)
    older_texts = _prepare_texts(older_articles)

    recent_frequencies = _term_document_frequencies(
        recent_texts,
        top_n=100,
    )

    older_frequencies = _term_document_frequencies(
        older_texts,
        top_n=100,
    )

    emerging = []

    for term, recent_score in recent_frequencies.items():
        older_score = older_frequencies.get(term, 0.0)

        # Require at least two recent documents containing the term.
        recent_count = round(recent_score * len(recent_texts))

        if recent_count < 2:
            continue

        if older_score == 0.0:
            emerging.append((term, recent_score, older_score))
        elif recent_score / older_score > 1.5:
            emerging.append((term, recent_score, older_score))

    emerging.sort(
        key=lambda item: (
            item[1] - item[2],
            item[1],
        ),
        reverse=True,
    )

    return [
        (
            term,
            round(recent_score, 3),
            round(older_score, 3),
        )
        for term, recent_score, older_score in emerging[:10]
    ]