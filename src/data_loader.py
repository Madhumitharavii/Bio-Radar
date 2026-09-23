from Bio import Entrez
import pandas as pd


Entrez.email = "madhumitharavi@gmail.com"


def search_pubmed(query, max_results=15):
    """
    Search PubMed and return PubMed IDs.
    """

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )

    record = Entrez.read(handle)
    handle.close()

    return record["IdList"]


def fetch_pubmed_records(pmids):
    """
    Fetch detailed PubMed records for a list of PMIDs.
    """

    handle = Entrez.efetch(
        db="pubmed",
        id=pmids,
        retmode="xml"
    )

    records = Entrez.read(handle)
    handle.close()

    return records


def extract_articles(records):
    """
    Convert PubMed records into simple dictionaries.
    """

    articles = []

    for article_record in records["PubmedArticle"]:

        medline = article_record["MedlineCitation"]
        article = medline["Article"]

        pmid = str(medline["PMID"])

        title = str(article.get("ArticleTitle", ""))

        abstract_parts = article.get(
            "Abstract",
            {}
        ).get(
            "AbstractText",
            []
        )

        abstract = " ".join(
            str(part) for part in abstract_parts
        )

        journal = str(
            article["Journal"].get("Title", "")
        )

        year = ""

        journal_issue = article["Journal"].get(
            "JournalIssue",
            {}
        )

        pub_date = journal_issue.get(
            "PubDate",
            {}
        )

        if "Year" in pub_date:
            year = str(pub_date["Year"])

        elif "MedlineDate" in pub_date:
            year = str(pub_date["MedlineDate"])

        authors = []

        author_list = article.get(
            "AuthorList",
            []
        )

        for author in author_list:

            last_name = author.get(
                "LastName",
                ""
            )

            initials = author.get(
                "Initials",
                ""
            )

            if last_name:

                authors.append(
                    f"{last_name} {initials}".strip()
                )

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": "; ".join(authors),
                "journal": journal,
                "year": year,
                "url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/"
                    f"{pmid}/"
                ),
            }
        )

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

        pmids = search_pubmed(
            query,
            max_results=100
        )

        print(f"  Found {len(pmids)} papers")

        all_pmids.extend(pmids)

    unique_pmids = list(dict.fromkeys(all_pmids))

    print(
        f"\nTotal papers retrieved: {len(all_pmids)}"
    )

    print(
        f"Unique papers after deduplication: "
        f"{len(unique_pmids)}"
    )

    print("\nFetching PubMed records...")

    records = fetch_pubmed_records(unique_pmids)

    articles = extract_articles(records)

    df = pd.DataFrame(articles)

    df = df[
        df["abstract"].str.strip() != ""
    ]

    df = df.drop_duplicates(
        subset="pmid"
    )

    output_path = "data/papers.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved {len(df)} papers to "
        f"{output_path}"
    )

    print("\nColumns:")
    print(list(df.columns))

    print("\nDataset preview:")

    print(
        df[
            ["pmid", "title", "journal", "year"]
        ]
        .head(10)
        .to_string(index=False)
    )