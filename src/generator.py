import ollama


MODEL_NAME = "qwen3:8b"


def generate_answer(query, context):
    """
    Generate a grounded answer using only
    the retrieved biomedical literature context.
    """

    system_prompt = """
You are BioLit-RAG, a biomedical literature question-answering assistant.

Answer the user's question using ONLY the biomedical literature
provided in the context.

Do not use information that is not supported by the provided sources.

If the retrieved literature does not contain enough information
to answer the question, clearly say that the available literature
does not provide enough evidence.

Do not invent facts, citations, PMID numbers, or study findings.

When making a claim based on a source, cite it using:
[Source 1], [Source 2], etc.

Keep the answer scientifically accurate and appropriately cautious.

This system is for literature exploration and is not a diagnostic
or medical advice tool.
"""

    user_prompt = f"""
USER QUESTION:
{query}

RETRIEVED BIOMEDICAL LITERATURE:
{context}

Using only the retrieved literature above, answer the user's question.
Include source citations such as [Source 1] and [Source 2].
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    print("Testing local Qwen3 model...")

    test_context = """
SOURCE 1
Title: Raman Spectroscopy and Imaging for Cancer Diagnosis
PMID: 29977484

Abstract:
Raman spectroscopy provides molecular information based on
vibrational signatures and has been investigated for cancer
diagnosis and biomedical imaging.
"""

    test_query = (
        "How can Raman spectroscopy help in cancer diagnosis?"
    )

    answer = generate_answer(
        test_query,
        test_context
    )

    print("\nGenerated answer:\n")
    print(answer)