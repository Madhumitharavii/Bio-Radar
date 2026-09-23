import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

DATA_PATH = "data/papers.csv"
INDEX_PATH = "data/papers.index"


def load_resources():
    """
    Load the literature dataset, FAISS index,
    and embedding model.
    """

    df = pd.read_csv(DATA_PATH)

    index = faiss.read_index(INDEX_PATH)

    model = SentenceTransformer(MODEL_NAME)

    return df, index, model


def search(query, df, index, model, top_k=5):
    """
    Search the literature corpus using semantic similarity.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        paper = df.iloc[index_position]

        results.append(
            {
                "score": float(score),
                "pmid": paper["pmid"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": paper["authors"],
                "journal": paper["journal"],
                "year": paper["year"],
                "url": paper["url"],
            }
        )

    return results


def build_context(results):
    """
    Convert retrieved papers into context
    that can be provided to the language model.
    """

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""SOURCE {rank}
Title: {result['title']}
Authors: {result['authors']}
Journal: {result['journal']}
Year: {result['year']}
PMID: {result['pmid']}
URL: {result['url']}

Abstract:
{result['abstract']}
"""
        )

    return "\n\n".join(context_parts)


def print_results(results):
    """
    Print retrieved papers in a readable format.
    """

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"[Score: {result['score']:.4f}] "
            f"{result['title']}"
        )

        print(
            f"   PMID: {result['pmid']}"
        )

        print(
            f"   Journal: {result['journal']} "
            f"({result['year']})"
        )

        print(
            f"   URL: {result['url']}"
        )

        print()


if __name__ == "__main__":

    print("Loading BioLit-RAG resources...")

    df, index, model = load_resources()

    print(f"Loaded {len(df)} papers.")

    print(
        f"FAISS index contains "
        f"{index.ntotal} documents."
    )

    query = (
        "How can Raman spectroscopy be used "
        "for cancer diagnosis?"
    )

    print("\nQuery:")
    print(query)

    results = search(
        query,
        df,
        index,
        model,
        top_k=5
    )

    print("\nTop retrieved papers:\n")

    print_results(results)

    context = build_context(results)

    print("RAG CONTEXT PREVIEW")
    print("=" * 80)

    print(context[:3000])