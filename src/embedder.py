import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

DATA_PATH = "data/papers.csv"
INDEX_PATH = "data/papers.index"
EMBEDDINGS_PATH = "data/embeddings.npy"


def load_papers(path=DATA_PATH):
    """
    Load the PubMed literature dataset.
    """

    df = pd.read_csv(path)

    return df


def create_embedding_text(df):
    """
    Combine paper title and abstract into the text
    that will be converted into an embedding.
    """

    return (
        df["title"].fillna("")
        + ". "
        + df["abstract"].fillna("")
    ).tolist()


def create_embeddings(texts):
    """
    Convert literature text into numerical embeddings.
    """

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


def create_faiss_index(embeddings):
    """
    Create a FAISS index using cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


if __name__ == "__main__":

    print("Loading PubMed papers...")

    df = load_papers()

    print(f"Loaded {len(df)} papers.")

    texts = create_embedding_text(df)

    print("Creating embeddings...")

    embeddings = create_embeddings(texts)

    print("\nCreating FAISS index...")

    index = create_faiss_index(embeddings)

    os.makedirs("data", exist_ok=True)

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    print("\nIndex creation complete!")

    print("Number of documents:", index.ntotal)
    print("Embedding dimensions:", embeddings.shape[1])
    print("Saved embeddings:", EMBEDDINGS_PATH)
    print("Saved FAISS index:", INDEX_PATH)