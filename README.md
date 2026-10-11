# 🧭 BioR&D Radar

### Biomedical R&D Intelligence & Opportunity Discovery

BioR&D Radar explores a biomedical research/technology area, structures the
literature, identifies research activity and emerging themes, surfaces
semantically connected research directions, and generates evidence-linked
**R&D opportunity hypotheses** — all grounded in literature actually
retrieved for the query, with sources traceable back to PMIDs.

## Why BioR&D Radar Exists

Biomedical literature is large, fast-changing, and hard to turn into
structured, trustworthy knowledge quickly. Most RAG demos stop at
"ask a question, get an answer." BioR&D Radar demonstrates a different
workflow — one closer to real technology-scouting / R&D-intelligence work:

```
SEARCH → STRUCTURE → CONNECT → ANALYZE → VISUALIZE → OPPORTUNITY DISCOVERY
```

## How It Evolved From BioLit-RAG

BioR&D Radar is **not a rewrite** — it's an extension of an existing,
working RAG project, [BioLit-RAG](.), built by reusing its PubMed
retrieval, embedding, and LLM-generation components rather than
replacing them.

| | BioLit-RAG | BioR&D Radar |
|---|---|---|
| Input | A question | A research topic + date range |
| Retrieval | Fixed, pre-built 1,088-paper corpus | Dynamic, live PubMed search per topic |
| Storage | CSV only | CSV + SQLite (structured, queryable) |
| Output | One grounded answer | Structured landscape, publication activity, semantic connections, AI analysis, and R&D opportunity hypotheses |
| Philosophy | Chatbot over literature | R&D intelligence tool over literature |

Reused unchanged: `data_loader.py`'s PubMed functions, `embedder.py`'s
embedding/FAISS functions, the anti-hallucination grounding philosophy in
`generator.py`. Extended: a dated-query wrapper, a SQLite layer, TF-IDF-based
structuring, a semantic-connections module, two new structured LLM prompts,
and a 5-tab Streamlit dashboard.

## The DNP / R&D Problem Being Addressed

This project was built to demonstrate the workflow described in DNP's
R&D technology-scouting role: searching and selecting information,
checking source reliability, turning large and fast-changing information
into **structured, searchable knowledge with clear sources and evidence**,
connecting signals across technologies, and communicating findings clearly.
Biomedical literature is used here as one deliberately scoped first
vertical — the same workflow (search → structure → connect → analyze →
opportunity discovery) generalizes to other information sources (patents,
startups, policy) as future work.

## Architecture

```
USER ENTERS A BIOMEDICAL TOPIC + DATE RANGE
                 ↓
            PUBMED API (reused from BioLit-RAG)
                 ↓
       RECENT LITERATURE (with abstracts)
                 ↓
         CLEAN + STRUCTURE → SQLITE
                 ↓
     SENTENCE EMBEDDINGS (reused) → FAISS (per-run)
                 ↓
   ┌─────────────┼─────────────┐
   ↓             ↓             ↓
TRENDS        THEMES       CONNECTIONS
 (SQL)        (TF-IDF)    (cosine similarity)
   └─────────────┼─────────────┘
                 ↓
         RAG CONTEXT → QWEN3 8B (Ollama, reused)
                 ↓
   ┌─────────────┼─────────────┐
   ↓             ↓             ↓
LANDSCAPE    CHALLENGES    DIRECTIONS
                 ↓
      R&D OPPORTUNITY HYPOTHESES
                 ↓
           STREAMLIT UI (5 tabs)
```

## Features

1. **Technology/Research Explorer** — enter a topic, date range, and
   paper count; retrieves recent PubMed literature with title, authors,
   year, journal, abstract, PMID, and PubMed link.
2. **Structured Research Landscape** — TF-IDF-derived keywords, phrases,
   and emerging terms, computed fresh from whatever was retrieved (no
   hard-coded topic logic, so it works for arbitrary biomedical areas).
3. **Literature Activity** — publications-by-year (SQL `GROUP BY`/`COUNT`)
   and journal frequency. Explicitly labeled "literature activity," not
   "market growth" — publication frequency alone doesn't establish
   commercial importance.
4. **AI-Assisted Research Analysis** — a structured, four-section LLM
   analysis (Research Landscape / Emerging Directions / Common Challenges
   / Technology Connections), grounded only in retrieved papers, with
   [Source N] citations and explicit instructions against inventing
   papers, PMIDs, or findings.
5. **R&D Opportunity Discovery** — a small number of opportunity
   hypotheses, each separating observed evidence, repeated limitations,
   the AI-generated hypothesis itself, supporting papers, and an evidence
   level — framed throughout as "potential," never as prediction or
   validated opportunity.

## Technology Stack

- **PubMed / BioPython Entrez** — literature retrieval (reused from BioLit-RAG)
- **SQLite** — structured storage for papers and analysis runs; demonstrates
  `GROUP BY`/`COUNT`/`ORDER BY`/`WHERE` for literature-activity queries
- **Sentence-Transformers (`all-MiniLM-L6-v2`) + FAISS** — semantic
  embeddings and nearest-neighbor similarity (reused from BioLit-RAG)
- **scikit-learn (TF-IDF)** — data-driven keyword/theme/phrase extraction
- **Qwen3 8B via Ollama** — local, grounded structured analysis and
  opportunity generation (reused from BioLit-RAG)
- **Streamlit + Plotly** — interactive dashboard

## Evaluation

Given the project's scope, evaluation is intentionally lightweight and
honestly split into two parts:

**Automated** (`tests/test_pipeline.py`, run against the real 1,088-paper
dataset already bundled in `data/papers.csv`):
- Year parsing on messy real PubMed date strings
- SQLite insert + de-duplication on repeated topic runs
- SQL `GROUP BY` aggregations (publications-by-year, journal frequency)
- Keyword extraction is topic-agnostic: correctly surfaces each subset's
  own defining term with zero hard-coded topic logic
- Emerging-keyword comparison runs without error

**Manual, requires your machine** (PubMed network access + a running
local Ollama server with `qwen3:8b` pulled — neither was available in the
sandbox this was built in):
- Did retrieval return relevant papers for a real topic?
- Are AI analysis sections grounded in retrieved papers, with correct
  [Source N] references?
- Are PMIDs/source links present and correct?
- Are opportunity hypotheses traceable to the cited supporting papers?

No numerical accuracy/precision metrics are reported — this is explicitly
a manual, qualitative evaluation given the project's scope, and that is
stated honestly rather than fabricated.

## Responsible AI

BioR&D Radar is a **biomedical research intelligence and exploratory R&D
analysis tool**. It is explicitly **not**: a medical advice system, a
clinical decision system, a commercial prediction engine, or a market
forecasting system. AI-generated opportunities are always labeled as
hypotheses. The system is instructed never to fabricate papers, PMIDs,
or results, and to separate evidence from interpretation throughout.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <your-github-repository-url>
cd BioR-D-Radar
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and pull the Qwen3 8B model:

```bash
ollama pull qwen3:8b
```

## Running

```bash
streamlit run app.py
```

## Example Query

Topic: `Raman spectroscopy for cancer diagnostics`
Date range: `2020–2026`
Papers: `30`

Produces: recent PubMed papers on Raman-based cancer diagnostics,
structured keywords (e.g. "surface enhanced", "machine learning",
"spectral classification"), a publication-activity chart, semantic
connections between related papers (e.g. SERS ↔ machine-learning
classification papers), a grounded four-section AI analysis, and a
small set of literature-supported R&D opportunity hypotheses.

## Limitations

- Per-run semantic connections are computed over a small set (10-50
  papers), not the full literature — treat connections as suggestive,
  not comprehensive.
- TF-IDF keyword/theme extraction is a statistical, not a biological,
  summary — the LLM analysis layer provides the interpretive reading,
  and should be checked against the cited sources.
- Publication counts reflect PubMed retrieval volume for the given
  query and date range, not the full research landscape or any
  commercial signal.
- All AI-generated content depends on the retrieved abstract text only
  (not full papers), and inherits whatever gaps exist in PubMed's
  metadata (e.g. inconsistent year formatting, which is parsed
  defensively but not perfectly for every edge case).
- Opportunity hypotheses are exploratory starting points for human
  researchers to evaluate further, not validated conclusions.

## Future Work (Post-Submission)

- Technology comparison across multiple topics side-by-side
- Markdown/PDF export of a full analysis run
- More advanced clustering of themes (beyond TF-IDF nearest-neighbor)
- More advanced trend detection (e.g. proper time-series modeling of
  keyword emergence, rather than a simple recent-vs-older split)
- Additional information sources (startups, investment, policy, patents)
  as additional "verticals" alongside the biomedical-literature one
