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


def generate_structured_analysis(topic, context):
    """
    Produce the BioR&D Radar structured analysis: Research
    Landscape, Emerging Directions, Common Challenges, and
    Technology Connections -- grounded strictly in the supplied
    literature context, with explicit anti-hallucination rules.
    """

    system_prompt = """
You are an AI research-intelligence assistant analyzing biomedical
literature for an R&D technology-scouting workflow.

You must analyze ONLY the literature supplied to you in the context.

Strict rules:
- Do not invent papers, PMIDs, authors, journals, or findings.
- Do not cite a source number that was not provided in the context.
- Clearly distinguish EVIDENCE (what the papers directly report)
  from INTERPRETATION (your synthesis across papers).
- If the supplied literature is insufficient to support a section,
  say so explicitly rather than filling the gap with assumptions.
- This is a research-exploration and R&D-scouting tool. It is not
  a medical advice system, a clinical decision system, or a
  commercial/market forecasting system. Do not make market-size,
  investment, or regulatory-approval claims.

Cite supporting sources using [Source N] matching the numbering
in the provided context.
"""

    user_prompt = f"""
RESEARCH TOPIC:
{topic}

RETRIEVED BIOMEDICAL LITERATURE:
{context}

Using ONLY the literature above, produce a structured analysis with
exactly these four sections, using these exact headings:

### Research Landscape
What are the major themes visible across the retrieved literature?

### Emerging Directions
What research directions appear increasingly visible in the more
recent papers in this set?

### Common Challenges
What limitations, gaps, or recurring problems are mentioned across
multiple papers?

### Technology Connections
What methods, technologies, or related research areas appear
connected across the retrieved papers?

For every claim, cite the supporting source(s) as [Source N]. If a
section cannot be well supported by the retrieved literature, state
that explicitly instead of speculating.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"]


def generate_opportunities(topic, context, max_opportunities=3):
    """
    Generate a small number of literature-supported R&D
    opportunity hypotheses. These are explicitly framed as
    AI-generated hypotheses, never as predictions or validated
    commercial opportunities.
    """

    system_prompt = """
You are an AI research-intelligence assistant helping an R&D team
generate POTENTIAL research directions from biomedical literature.

You must base every opportunity on patterns actually present in the
supplied literature context. Do not invent papers, PMIDs, or results.

CRITICAL FRAMING RULES:
- Never claim something "will be" a big market or technology.
- Never claim something is "commercially validated."
- Use only hedged, literature-grounded language such as:
  "potential R&D opportunity", "literature-supported hypothesis",
  "potential research direction", "AI-generated hypothesis".
- Each opportunity must clearly separate:
  (a) evidence from the papers,
  (b) the observed pattern/trend across papers,
  (c) your AI-generated hypothesis built on top of (a) and (b).
- If the literature does not support a confident opportunity,
  generate fewer opportunities rather than inventing weak ones.

This tool produces exploratory hypotheses for human R&D
researchers to evaluate further. It does not produce validated
business or clinical recommendations.
"""

    user_prompt = f"""
RESEARCH TOPIC:
{topic}

RETRIEVED BIOMEDICAL LITERATURE:
{context}

Generate up to {max_opportunities} potential R&D opportunity
hypotheses based on recurring patterns, limitations, or gaps you
observe across the retrieved literature.

For EACH opportunity, use exactly this structure:

#### Opportunity: <short title>
- **Observed evidence/trend:** <what the papers show, with [Source N] citations>
- **Repeated limitation/problem:** <the recurring issue this addresses, with [Source N] citations>
- **Potential R&D direction:** <the hypothesis -- framed as "potential", never as certain>
- **Supporting papers:** <list the [Source N] references used>
- **Evidence level:** <Low / Moderate / Higher, based on how many independent sources support it>
- **Why this could be worth investigating:** <1-2 sentences>

If fewer than {max_opportunities} well-supported opportunities exist
in this literature set, generate only as many as are genuinely
supportable, and say so.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
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