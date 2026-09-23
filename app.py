import streamlit as st

from src.rag_pipeline import (
    answer_question,
    load_resources
)


st.set_page_config(
    page_title="BioLit-RAG",
    page_icon="🧬",
    layout="wide"
)


@st.cache_resource
def load_rag_resources():
    return load_resources()


def main():
    st.title("🧬 BioLit-RAG")

    st.subheader(
        "Biomedical AI/ML Literature Retrieval & Question Answering"
    )

    st.write(
        "Ask a question about artificial intelligence, "
        "machine learning, and computational methods in "
        "biomedical research."
    )

    st.divider()

    query = st.text_input(
        "Ask a biomedical AI/ML question:",
        placeholder=(
            "Example: How is deep learning used "
            "for medical image segmentation?"
        )
    )

    ask_button = st.button(
        "Ask BioLit-RAG",
        type="primary"
    )

    if ask_button and query.strip():

        with st.spinner(
            "Retrieving literature and generating answer..."
        ):

            df, index, model = load_rag_resources()

            answer, results = answer_question(
                query,
                df,
                index,
                model,
                top_k=5
            )

        st.subheader("Answer")

        st.write(answer)

        st.divider()

        st.subheader(
            "Retrieved Literature"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            st.markdown(
                f"### {rank}. {result['title']}"
            )

            st.write(
                f"**Similarity score:** "
                f"{result['score']:.4f}"
            )

            st.write(
                f"**Journal:** "
                f"{result['journal']} "
                f"({result['year']})"
            )

            st.write(
                f"**PMID:** {result['pmid']}"
            )

            st.markdown(
                f"[View on PubMed]({result['url']})"
            )

            with st.expander(
                "Show abstract"
            ):
                st.write(
                    result["abstract"]
                )


if __name__ == "__main__":
    main()