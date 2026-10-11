# 🧭 BioR&D Radar

### Biomedical Research Discovery & R&D Intelligence

BioR&D Radar is a biomedical research discovery platform that helps explore scientific literature, identify research themes, examine publication activity, and discover potential directions for future investigation. It combines PubMed literature retrieval, semantic search, structured analysis, and source-linked AI assistance in one workflow.

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

BioR&D Radar extends the existing BioLit-RAG project rather than rebuilding it from scratch. It reuses and extends components for PubMed retrieval, embeddings, semantic search, and LLM-based answer generation.

| Aspect | BioLit-RAG | BioR&D Radar |
|---|---|---|
| Primary purpose | Biomedical literature question answering | Biomedical research discovery and R&D intelligence |
| Input | A biomedical question | A research topic and date range |
| Retrieval | Existing 1,088-paper corpus | Topic-specific PubMed retrieval |
| Storage | CSV-based corpus | CSV and SQLite |
| Analysis | Retrieval-grounded answers | Research themes, publication activity, semantic connections, and structured AI analysis |
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
       SENTENCE EMBEDDINGS + FAISS
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
        RETRIEVED LITERATURE CONTEXT
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

Summarize publication activity by year and journal using SQL aggregations.

Publication counts are presented as **literature activity**, not as a measure of market growth, commercial potential, or scientific importance.

### 4. Semantic Connections

Use sentence embeddings and cosine similarity to identify papers with related textual content. These connections can help researchers explore relationships between studies that may not use identical terminology.

Similarity indicates textual or semantic relatedness; it does not establish scientific agreement, causation, or novelty.

### 5. AI-Assisted Research Analysis

Generate a structured analysis covering:

- Research landscape
- Emerging directions
- Common challenges
- Technology connections

The LLM is instructed to use retrieved papers as evidence and provide source references. Generated statements and citations still require verification against the original records.

### 6. R&D Opportunity Discovery

Generate a small number of exploratory research-opportunity hypotheses. Each hypothesis is intended to distinguish between:

- Observations supported by retrieved literature
- Recurring limitations or unresolved challenges
- The proposed research direction
- Supporting papers and their source identifiers
- An indicative evidence level

These outputs are hypotheses for human investigation, not validated discoveries, predictions, or guaranteed research opportunities.

## Technology Stack

- **PubMed / BioPython Entrez:** Biomedical literature retrieval
- **SQLite:** Structured storage and SQL-based literature analysis
- **Sentence Transformers (`all-MiniLM-L6-v2`):** Text embeddings
- **FAISS:** Vector similarity search
- **scikit-learn (TF-IDF):** Keyword, phrase, and theme extraction
- **Qwen3 8B via Ollama:** Local LLM-based analysis and opportunity generation
- **Streamlit:** Interactive application interface
- **Plotly:** Data visualization
- **Python, NumPy and Pandas:** Data processing and analysis

## Evaluation and Testing

The project includes automated tests in `tests/test_pipeline.py`, using the existing 1,088-paper dataset in `data/papers.csv`.

The tests cover:

- Parsing publication years from PubMed date formats
- SQLite insertion and deduplication
- SQL aggregations for publication years and journal frequency
- Topic-agnostic keyword extraction
- Emerging-keyword comparison

These tests have passed in the development environment.

**Important:** Passing these tests establishes that the tested data-processing components behave as expected. It does not establish that the complete application works correctly under all conditions or that its generated research insights are scientifically accurate.

### End-to-End Validation Status

The updated application still requires end-to-end validation covering:

- Live PubMed retrieval for different biomedical topics and date ranges
- Relevance and date accuracy of retrieved records
- Semantic search and similarity results
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

If PubMed retrieval requires an email address, configure it using the environment variable supported by the application. Do not commit API credentials, private configuration, or local database files to GitHub.

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

## Limitations


- **Retrieval:** Depends on PubMed coverage, query, and date range.
- **Semantic analysis:** Limited to 10–50 papers per run.
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
