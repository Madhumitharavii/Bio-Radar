
import os

from Bio import Entrez
import pandas as pd


# Configure a contact email through the environment.
# Set NCBI_EMAIL before running the application.
Entrez.email = os.getenv("NCBI_EMAIL", "").strip() or None


def search_pubmed(query, max_results=15):
    """Search PubMed and return PubMed IDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    try:
        record = Entrez.read(handle)
        return record["IdList"]
    finally:
        handle.close()


def fetch_pubmed_records(pmids):
    """Fetch detailed PubMed records for a list of PMIDs."""
    handle = Entrez.efetch(
        db="pubmed",
        id=pmids,
        retmode="xml",
    )
    try:
        return Entrez.read(handle)
    finally:
        handle.close()


def _extract_year(pub_date):
    """Extract a four-digit year from a PubMed date dictionary."""
    if not pub_date:
        return ""

    year = str(pub_date.get("Year", "")).strip()
    if year.isdigit() and len(year) == 4:
        return year

    medline_date = str(pub_date.get("MedlineDate", "")).strip()
    match = __import__("re").search(r"\b(19|20)\d{2}\b", medline_date)
    return match.group(0) if match else ""


def extract_articles(records):
    """Convert PubMed records into dictionaries with publication dates."""
    articles = []

    for article_record in records["PubmedArticle"]:
        medline = article_record["MedlineCitation"]
        article = medline["Article"]

        pmid = str(medline["PMID"])
        title = str(article.get("ArticleTitle", ""))

        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(part) for part in abstract_parts)

        journal = str(article.get("Journal", {}).get("Title", ""))

        journal_issue = article.get("Journal", {}).get("JournalIssue", {})
        issue_date = journal_issue.get("PubDate", {})
        pub_year_issue = _extract_year(issue_date)

        # ArticleDate contains electronic/other article-specific dates.
        article_dates = article.get("ArticleDate", [])
        pub_year_electronic = ""

        for article_date in article_dates:
            date_parts = [
                str(article_date.get(key, "")).strip()
                for key in ("Year", "Month", "Day")
            ]
            candidate_year = date_parts[0]
            if candidate_year.isdigit() and len(candidate_year) == 4:
                pub_year_electronic = candidate_year
                break

        # Keep the original year field for existing callers.
        year = pub_year_electronic or pub_year_issue

        authors = []
        for author in article.get("AuthorList", []):
            last_name = author.get("LastName", "")
            initials = author.get("Initials", "")
            if last_name:
                authors.append(f"{last_name} {initials}".strip())

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": "; ".join(authors),
            "journal": journal,
            "year": year,
            "pub_year_issue": pub_year_issue,
            "pub_year_electronic": pub_year_electronic,
            "publication_date": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return articles


if __name__ == "__main__":
    queries = [
        "artificial intelligence biomedical research",
        "machine learning biomedical research",
        "deep learning biomedical research",
        "artificial intelligence medical imaging",
        "deep learning medical image analysis",
        "machine learning biomedical signal processing",
        "machine learning genomics",
        "machine learning proteomics",
        "machine learning drug discovery",
        "deep learning digital pathology",
        "machine learning Raman spectroscopy",
        "generative artificial intelligence biomedical",
    ]

    all_pmids = []

    print("Searching PubMed...\n")
    for query in queries:
        print(f"Query: {query}")
        pmids = search_pubmed(query, max_results=100)
        print(f"  Found {len(pmids)} papers")
        all_pmids.extend(pmids)

    unique_pmids = list(dict.fromkeys(all_pmids))

    print(f"\nTotal papers retrieved: {len(all_pmids)}")
    print(f"Unique papers after deduplication: {len(unique_pmids)}")

    print("\nFetching PubMed records...")
    records = fetch_pubmed_records(unique_pmids)
    articles = extract_articles(records)

    df = pd.DataFrame(articles)
    df = df[df["abstract"].str.strip() != ""]
    df = df.drop_duplicates(subset="pmid")

    output_path = "data/papers.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} papers to {output_path}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nDataset preview:")
    print(df[["pmid", "title", "journal", "year"]].head(10).to_string(index=False))
