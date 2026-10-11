"""
Semantic connections between retrieved papers.

Reuses the existing Sentence-Transformer + FAISS infrastructure
from embedder.py (same model, same cosine-similarity-via-inner-
product convention) rather than building a separate NLP pipeline
or a knowledge graph. Scope is intentionally limited to:
nearest-neighbor paper pairs and the top terms that characterize
each side of a connection -- not a graph database, not clustering
beyond simple nearest-neighbor grouping.
"""

import numpy as np

from src.embedder import create_embedding_text, create_embeddings, create_faiss_index


def build_run_embeddings(articles):
    """
    Build embeddings + a FAISS index for exactly the papers
    retrieved in this run (not the static historical corpus).
    Small per-run sets (tens of papers) keep this fast.
    """
    df_like = {
        "title": [a.get("title", "") for a in articles],
        "abstract": [a.get("abstract", "") for a in articles],
    }

    import pandas as pd
    df = pd.DataFrame(df_like)

    texts = create_embedding_text(df)
    embeddings = create_embeddings(texts)
    index = create_faiss_index(embeddings)

    return embeddings, index


def most_similar_pairs(articles, embeddings, top_k_neighbors=1, max_pairs=6):
    """
    For each paper, find its nearest neighbor(s) by cosine
    similarity (embeddings are already normalized, so inner
    product == cosine similarity). Returns the strongest,
    de-duplicated cross-paper connections found in the set.

    This characterizes connections at the PAPER level; the
    caller can present the two papers' titles as a human-
    readable "these research directions are related" signal,
    explained by the literature itself rather than an opaque
    graph.
    """
    n = embeddings.shape[0]

    if n < 2:
        return []

    similarity_matrix = embeddings @ embeddings.T
    np.fill_diagonal(similarity_matrix, -1.0)

    pairs = []
    seen = set()

    for i in range(n):
        neighbor_indices = np.argsort(similarity_matrix[i])[::-1][:top_k_neighbors]

        for j in neighbor_indices:
            j = int(j)
            score = float(similarity_matrix[i, j])

            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)

            pairs.append(
                {
                    "paper_a": articles[i]["title"],
                    "paper_b": articles[j]["title"],
                    "similarity": round(score, 3),
                    "pmid_a": articles[i].get("pmid", ""),
                    "pmid_b": articles[j].get("pmid", ""),
                }
            )

    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    return pairs[:max_pairs]
