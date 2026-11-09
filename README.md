# 🧭 BioR&D Radar

### Biomedical Research Discovery & R&D Intelligence

BioR&D Radar is a biomedical research discovery platform that helps explore scientific literature, identify research themes, examine publication activity, and discover potential directions for future investigation. It combines date-filtered PubMed retrieval, SQL-based literature activity analysis, TF-IDF theme extraction, semantic similarity between papers, and source-linked AI assistance in one workflow.

## Why BioR&D Radar Exists

Biomedical literature is growing rapidly, making it difficult to identify emerging research directions, connect findings across studies, and recognize gaps that could motivate further investigation.

BioR&D Radar aims to make this process more systematic by turning retrieved literature into a structured, searchable research landscape.

Its workflow follows:

**SEARCH → STRUCTURE → CONNECT → ANALYZE → VISUALIZE → DISCOVER**

Unlike a basic literature question-answering system, the project brings literature retrieval, publication analysis, semantic connections, and exploratory research-opportunity generation into a single workflow.

## The Problem

Researchers often need to review large numbers of publications before they can understand the state of a research area. Relevant findings may be distributed across different studies, while recurring limitations and possible directions for further investigation can be difficult to identify systematically.

BioR&D Radar addresses this challenge by organizing retrieved biomedical literature, extracting research themes, summarizing publication activity, connecting related papers, and generating evidence-linked hypotheses for further investigation.

The platform is intended to support research exploration, not replace critical reading, scientific judgment, or expert validation.

## How It Evolved From BioLit-RAG

BioR&D Radar extends the existing [BioLit-RAG](https://github.com/Madhumitharavii/BioLit-RAG) project rather than rebuilding it from scratch. It reuses BioLit-RAG's PubMed loader (`src/data_loader.py`), Sentence-Transformer embedding code (`src/embedder.py`), context builder (`src/retriever.py`) and Ollama generator (`src/generator.py`), extending each where Radar needs more.

The main difference in retrieval: BioLit-RAG performs **semantic retrieval** (a question is embedded and FAISS returns the most similar papers from a fixed corpus). Radar retrieves papers with a **PubMed keyword query** for the topic and date range, then uses embeddings for **semantic similarity** analysis *within* that retrieved set: finding closely related paper pairs and ranking which papers go into the LLM context.

| Aspect | BioLit-RAG | BioR&D Radar |
|---|---|---|
| Primary purpose | Biomedical literature question answering | Biomedical research discovery and R&D intelligence |
| Input | A biomedical question | A research topic and date range |
| Retrieval | Existing 1,088-paper corpus | Topic-specific PubMed retrieval |
| Storage | CSV-based corpus | CSV and SQLite |
| Analysis | Retrieval-grounded answers | Research themes, publication activity, semantic paper connections, and structured AI analysis |
| Output | An answer supported by literature | A research landscape and exploratory opportunity hypotheses |

BioR&D Radar adds date-filtered retrieval, structured database storage, TF-IDF-based theme extraction, semantic connections between papers, structured LLM prompts, and a multi-tab Streamlit dashboard.

## Architecture

```text
USER ENTERS A BIOMEDICAL TOPIC + DATE RANGE
                    |
                    v
             PUBMED RETRIEVAL
                    |
                    v
       RETRIEVED PAPERS AND ABSTRACTS
                    |
                    v
        CLEANING AND STRUCTURING
                    |
                    v
              SQLITE STORAGE
                    |
                    v
   SENTENCE EMBEDDINGS (SIMILARITY)
                    |
          +---------+---------+
          |         |         |
          v         v         v
        TRENDS     THEMES  CONNECTIONS
        (SQL)     (TF-IDF)  (SIMILARITY)
          |         |         |
          +---------+---------+
                    |
                    v
   SELECTED EVIDENCE CONTEXT (TOP 15)
                    |
                    v
           QWEN3 8B VIA OLLAMA
                    |
                    v
       STRUCTURED RESEARCH ANALYSIS
                    |
                    v
       EXPLORATORY OPPORTUNITY
               HYPOTHESES
                    |
                    v
           STREAMLIT DASHBOARD
```

## Features

### 1. Research Topic Explorer

Enter a biomedical research topic, date range, and desired number of papers. The application retrieves matching PubMed records containing available metadata such as titles, authors, publication years, journals, abstracts, PMIDs, and PubMed links.

### 2. Structured Research Landscape

Extract keywords, phrases, and potentially emerging terms from the retrieved literature using TF-IDF. The analysis is based on the retrieved corpus rather than a fixed list of predefined research topics.

### 3. Literature Activity Analysis

Summarize publication activity by year and journal using SQL aggregations over **the papers retrieved in the current run**.

Retrieved papers are stored in a local SQLite database (`data/radar.db`, not committed) and deduplicated per topic, so the database accumulates literature across runs. Each run is logged in `analysis_runs` with its date range, and a `run_papers` table records exactly which stored papers that run retrieved. The Literature Activity chart and journal counts are filtered by that run, so papers kept from earlier runs or other date ranges for the same topic do not appear in the current analysis.

Publication counts are presented as **literature activity**, not as a measure of market growth, commercial potential, or scientific importance.

### 4. Semantic Paper Connections

Embed each retrieved paper's title and abstract with `all-MiniLM-L6-v2` and use cosine similarity to find the most closely related pairs of papers within the retrieved set (literature connection analysis, not retrieval). These connections can help researchers explore relationships between studies that may not use identical terminology.

Similarity indicates textual or semantic relatedness; it does not establish scientific agreement, causation, or novelty.

### 5. AI-Assisted Research Analysis

Generate a structured analysis covering:

- Research landscape
- Emerging directions
- Common challenges
- Technology connections

The LLM is instructed to use retrieved papers as evidence and provide source references. Generated statements and citations still require verification against the original records.

**Evidence selection.** All retrieved papers are stored, charted, keyword-analyzed and listed in the Papers tab. To keep prompts within the model's context window, the LLM receives a bounded evidence set (`src/evidence.py`):

- Papers are ranked by semantic similarity between the topic and each paper's embedding (falling back to PubMed's relevance order if embeddings are unavailable); ties keep PubMed order, so selection is deterministic.
- The top 15 papers are kept. Runs with 15 or fewer papers send everything.
- Abstracts longer than 1,500 characters are truncated at a word boundary and marked `[abstract truncated]`; title, authors, journal, year, PMID and URL are always included.
- Selected papers keep their original `[Source N]` numbers, so citations match the Papers tab. The AI Analysis tab lists which sources were used.
- Ollama is asked for a 16,384-token context window (`ANALYSIS_NUM_CTX` in `src/generator.py`) so the evidence is not silently truncated by a smaller default.

### 6. R&D Opportunity Discovery

Generate a small number of exploratory research-opportunity hypotheses. Each hypothesis is intended to distinguish between:

- Observations supported by retrieved literature
- Recurring limitations or unresolved challenges
- The proposed research direction
- Supporting papers and their source identifiers
- An indicative evidence level

These outputs are hypotheses for human investigation, not validated discoveries, predictions, or guaranteed research opportunities.

### Evidence vs. AI-Generated Hypotheses

| Output | Source | Status |
|---|---|---|
| Papers, years, journals, PMIDs, abstracts | PubMed | Retrieved records |
| Literature activity, journal counts | SQL over the current run's papers | Deterministic counts |
| Keywords, phrases, emerging keywords | TF-IDF / document frequency over the current run | Deterministic, sample-specific signals |
| Semantic paper connections | Embedding cosine similarity | Deterministic similarity scores |
| AI Analysis | Qwen3 8B, prompted with the selected evidence | LLM interpretation; verify citations |
| R&D Opportunities | Qwen3 8B, prompted with the selected evidence | AI-generated hypotheses, not evidence |

## Technology Stack

- **PubMed / BioPython Entrez:** Biomedical literature retrieval
- **SQLite:** Structured storage and SQL-based literature analysis
- **Sentence Transformers (`all-MiniLM-L6-v2`):** Text embeddings for semantic similarity
- **FAISS:** Inner-product index reused from BioLit-RAG (Radar builds one per run; pairwise similarities are computed directly from the normalized embeddings)
- **scikit-learn (TF-IDF):** Keyword, phrase, and theme extraction
- **Qwen3 8B via Ollama:** Local LLM-based analysis and opportunity generation
- **Streamlit:** Interactive application interface
- **Plotly:** Data visualization
- **Python, NumPy and Pandas:** Data processing and analysis

## Evaluation and Testing

The project includes automated tests in `tests/test_pipeline.py`. Data-processing tests use the existing 1,088-paper dataset in `data/papers.csv`; pipeline tests replace PubMed, the embedding model and Ollama with deterministic stand-ins, so the suite runs offline:

```bash
python tests/test_pipeline.py
```

The tests cover:

- Parsing publication years from PubMed date formats
- SQLite insertion and deduplication, including re-runs that overlap earlier runs
- SQL aggregations for publication years and journal frequency
- Run-scoped Literature Activity: a second run with a different date range shows only its own papers
- Migration of an existing `radar.db` created before run tracking was added
- Topic-agnostic keyword extraction and emerging-keyword detection
- Evidence selection for the LLM context (ranking, limit, truncation, `[Source N]` numbering)
- Preliminary labels from the experimental hyperspectral Raman screening module
- No hard-coded email addresses in source code

One keyword check is skipped automatically when the dataset has too few matching papers (the `glucose` subset).

**Important:** Passing these tests establishes that the tested data-processing components behave as expected. It does not establish that the complete application works correctly under all conditions or that its generated research insights are scientifically accurate.

### End-to-End Validation Status

The updated application still requires end-to-end validation covering:

- Live PubMed retrieval for different biomedical topics and date ranges
- Relevance and date accuracy of retrieved records
- Semantic similarity and paper-connection results
- Successful generation through a running Ollama server with `qwen3:8b`
- Correct source references in AI-generated analysis
- Evidence traceability of opportunity hypotheses
- Streamlit interface behavior, including loading states and error handling

Until these checks are completed, the project should be considered a development-stage MVP rather than a fully validated research-intelligence system.

## Responsible AI

BioR&D Radar is designed for biomedical literature exploration and research support. It is not a medical advice system, a clinical decision-support system, a commercial prediction engine, or a market forecasting tool.

The system aims to distinguish retrieved evidence from AI-generated interpretation and to retain links to original PubMed records. Source references and generated claims must still be checked against the original publications.

The platform does not independently establish the scientific validity, feasibility, novelty, or commercial value of a proposed research direction.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- Internet access for PubMed retrieval
- Ollama for local LLM-based analysis

Clone the repository:

```bash
git clone https://github.com/Madhumitharavii/Bio-Radar.git
cd Bio-Radar
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and download the model:

```bash
ollama pull qwen3:8b
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NCBI_EMAIL` | Recommended | Contact email NCBI asks Entrez/PubMed users to provide. Read in `src/data_loader.py`; if unset, requests are sent without an email. |

```bash
export NCBI_EMAIL="you@example.com"
```

The app reads this from the shell environment. Do not commit API credentials, a `.env` file, or the local `data/radar.db` database; all are listed in `.gitignore`.

The first run downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face; it is cached afterwards.

## Running the Application

Start Ollama and ensure the required model is available. Then, from the project directory with the virtual environment activated, run:

```bash
streamlit run app.py
```

Streamlit will display a local application URL, usually `http://localhost:8501`. Open that address in your browser.

This command runs the application locally. Other users cannot automatically access it just because the repository is on GitHub; public access requires a separate deployment.

## Example Query

**Topic:** `Raman spectroscopy for cancer diagnostics`

**Date range:** `2020–2026`

**Requested papers:** `30`

Depending on the records returned by PubMed, the application can present retrieved papers, extracted keywords, publication activity, semantic connections, structured AI-assisted analysis, and exploratory research-opportunity hypotheses.

The example illustrates a possible use case, not a claim that the query has already passed end-to-end validation.

## Experimental Module: Hyperspectral Raman Screening

`src/relevance.py` and `screen_candidates.py` are an **experimental, domain-specific module** and are **not part of the Streamlit app or the Radar pipeline**. They support a separate literature-screening task: triaging papers on hyperspectral Raman imaging for biomedical diagnosis.

- `src/relevance.py` assigns rule-based preliminary labels (`strong_candidate`, `potential_candidate`, `adjacent_method`, `out_of_scope`) using regular-expression matches for method wording, biomedical/diagnostic terms and AI/ML terms, and records the matched evidence.
- `screen_candidates.py` applies it to `data/hyperspectral_candidates.csv` and writes `data/hyperspectral_screening.csv` (38 papers in the committed version).
- The candidate CSV is committed as-is; the step that produced it from a PubMed batch retrieval (see `fetch_batch.py`) is not scripted in this repository.

Labels are keyword-based screening priorities only. Every row is marked `requires_full_text_methods_review`; none is a verified scientific conclusion.

```bash
python screen_candidates.py
```

## Limitations


- **Retrieval:** Depends on PubMed coverage, query, and date range.
- **Semantic analysis:** Limited to 10–50 papers per run; similarity is computed only within the retrieved set.
- **Emerging keywords:** Compares the latest two publication years in the retrieved sample against earlier years. Results depend on PubMed's relevance ranking, the requested paper count, topic wording and date range; they need at least two dated papers in each period, and are not evidence of a field-wide trend, novelty or commercial potential.
- **LLM evidence limit:** The AI analysis sees at most 15 papers (abstracts truncated to 1,500 characters), chosen by similarity to the topic; papers outside that set can still appear in the charts and Papers tab.
- **Theme extraction:** TF-IDF identifies terms, not biological significance.
- **AI analysis:** Uses abstracts; claims and citations require verification.
- **Publication trends:** Counts do not establish research or commercial value.
- **Opportunity discovery:** Hypotheses require expert validation.
- **Generalization:** Performance across biomedical topics remains to be evaluated.

## Future Work

- Validate retrieval relevance and source attribution using a manually reviewed benchmark
- Evaluate generated analyses and opportunity hypotheses against expert judgments
- Compare multiple research topics and technology areas side by side
- Export research landscapes and analysis reports to Markdown or PDF
- Explore more advanced theme clustering and time-series trend analysis
- Extend the platform to additional information sources, such as patents, startups, funding activity, and policy documents
- Improve deployment, usability, and error handling

---

**Project status:** Development-stage MVP. Automated tests cover selected pipeline components; complete live retrieval, generation, and interface validation remains to be demonstrated.
