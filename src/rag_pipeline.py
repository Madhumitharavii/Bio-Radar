from src.retriever import (
    load_resources,
    search,
    build_context
)

from src.generator import generate_answer


def answer_question(
    query,
    df,
    index,
    model,
    top_k=5
):
    """
    Run the complete BioLit-RAG pipeline.

    1. Retrieve relevant biomedical literature.
    2. Build context from the retrieved papers.
    3. Generate a grounded answer using the local LLM.
    """

    results = search(
        query,
        df,
        index,
        model,
        top_k=top_k
    )

    context = build_context(
        results
    )

    answer = generate_answer(
        query,
        context
    )

    return answer, results


if __name__ == "__main__":

    print("Loading BioLit-RAG...")

    df, index, model = load_resources()

    print(
        f"Loaded {len(df)} papers."
    )

    print(
        f"FAISS index contains "
        f"{index.ntotal} documents."
    )

    query = (
        "How can Raman spectroscopy "
        "be used for cancer diagnosis?"
    )

    print("\nUser question:")
    print(query)

    print(
        "\nRetrieving relevant literature..."
    )

    answer, results = answer_question(
        query,
        df,
        index,
        model,
        top_k=5
    )

    print("\nRetrieved papers:")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"[{result['score']:.4f}] "
            f"{result['title']}"
        )

    print("\nGenerated answer:")
    print("=" * 80)
    print(answer)