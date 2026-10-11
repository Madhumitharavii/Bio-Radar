import streamlit as st
import pandas as pd
import plotly.express as px

from src.radar_pipeline import run_radar_analysis


st.set_page_config(
    page_title="BioR&D Radar",
    page_icon="🧭",
    layout="wide",
)


def main():
    with st.sidebar:
        st.title("🧭 BioR&D Radar")
        st.caption(
            "Biomedical R&D Intelligence & Opportunity Discovery "
            "— built as an evolution of BioLit-RAG."
        )

        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. Raman spectroscopy for cancer diagnostics",
        )

        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year", min_value=2000, max_value=2026, value=2020
            )
        with col2:
            end_year = st.number_input(
                "End Year", min_value=2000, max_value=2026, value=2026
            )

        max_results = st.slider(
            "Number of Papers", min_value=10, max_value=50, value=30, step=10
        )

        run_button = st.button("Run Analysis", type="primary")

        st.divider()
        st.caption(
            "Outputs are literature-derived summaries and AI-generated "
            "research hypotheses -- not medical advice, not validated "
            "market predictions, and not a clinical decision system."
        )

    if "radar_result" not in st.session_state:
        st.session_state.radar_result = None

    if run_button:
        if not topic.strip():
            st.warning("Enter a research topic first.")
        elif start_year > end_year:
            st.warning("Start year must be before end year.")
        else:
            with st.spinner(
                f"Searching PubMed, structuring literature, and analyzing "
                f"'{topic}'..."
            ):
                st.session_state.radar_result = run_radar_analysis(
                    topic, int(start_year), int(end_year), max_results
                )

    result = st.session_state.radar_result

    if result is None:
        st.title("🧭 BioR&D Radar")
        st.write(
            "Enter a biomedical research/technology topic in the sidebar "
            "and click **Run Analysis** to explore the literature, "
            "identify research patterns and technology connections, "
            "and surface AI-generated R&D opportunity hypotheses."
        )
        return

    for warning in result["warnings"]:
        st.warning(warning)

    if result["num_papers"] == 0:
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Research Landscape",
            "AI Analysis",
            "R&D Opportunities",
            "Papers",
        ]
    )

    # ---------------- TAB 1: OVERVIEW ----------------
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Topic", result["topic"])
        c2.metric("Papers Analyzed", result["num_papers"])
        c3.metric(
            "Date Range", f"{result['start_year']}–{result['end_year']}"
        )

        st.subheader("Literature Activity")
        st.caption(
            "Publication counts by year for the retrieved set. "
            "Publication frequency reflects research attention, "
            "not commercial or market significance."
        )

        if result["activity_by_year"]:
            df_activity = pd.DataFrame(result["activity_by_year"])
            fig = px.bar(
                df_activity,
                x="year",
                y="paper_count",
                labels={"year": "Year", "paper_count": "Papers"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No year-level activity data available for this set.")

        st.subheader("Major Themes (Top Keywords)")
        if result["keywords"]:
            kw_df = pd.DataFrame(result["keywords"], columns=["Keyword", "TF-IDF Score"])
            st.dataframe(kw_df, use_container_width=True, hide_index=True)
        else:
            st.info("No keyword data available.")

    # ---------------- TAB 2: RESEARCH LANDSCAPE ----------------
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Keywords")
            if result["keywords"]:
                kw_df = pd.DataFrame(
                    result["keywords"], columns=["Keyword", "Score"]
                )
                st.dataframe(kw_df, use_container_width=True, hide_index=True)

            st.subheader("Related Phrases")
            if result["phrases"]:
                ph_df = pd.DataFrame(
                    result["phrases"], columns=["Phrase", "Score"]
                )
                st.dataframe(ph_df, use_container_width=True, hide_index=True)

        with col_b:
            st.subheader("Emerging Keywords")
            st.caption(
                "Terms whose weight is notably higher in the more recent "
                "papers of this retrieved set compared to older ones. "
                "Descriptive within this sample only."
            )
            if result["emerging_keywords"]:
                em_df = pd.DataFrame(
                    result["emerging_keywords"],
                    columns=["Term", "Recent Score", "Older Score"],
                )
                st.dataframe(em_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "Not enough date spread in the retrieved set to "
                    "compare recent vs. older keyword emphasis."
                )

            st.subheader("Related Research Connections")
            st.caption(
                "Papers in this retrieved set that are most semantically "
                "similar to each other, via Sentence-Transformer + FAISS "
                "cosine similarity."
            )
            if result["connections"]:
                for pair in result["connections"]:
                    st.markdown(
                        f"**{pair['similarity']:.3f}** similarity — "
                        f"*{pair['paper_a']}* ↔ *{pair['paper_b']}*"
                    )
            else:
                st.info("No semantic connections available for this run.")

        st.subheader("Journal Frequency")
        if result["journal_counts"]:
            jf_df = pd.DataFrame(result["journal_counts"])
            st.dataframe(jf_df, use_container_width=True, hide_index=True)

    # ---------------- TAB 3: AI ANALYSIS ----------------
    with tab3:
        st.caption(
            "Generated by a local LLM (Qwen3 8B via Ollama), grounded "
            "strictly in the papers retrieved for this run. Evidence is "
            "cited as [Source N]; see the Papers tab for the numbering."
        )
        if result["structured_analysis"]:
            st.markdown(result["structured_analysis"])
        else:
            st.info(
                "AI analysis not available for this run (check that "
                "Ollama is running locally with the qwen3:8b model)."
            )

    # ---------------- TAB 4: R&D OPPORTUNITIES ----------------
    with tab4:
        st.caption(
            "AI-generated, literature-supported hypotheses for further "
            "human evaluation. These are potential research directions, "
            "not predictions or validated opportunities."
        )
        if result["opportunities"]:
            st.markdown(result["opportunities"])
        else:
            st.info(
                "Opportunity generation not available for this run "
                "(check that Ollama is running locally)."
            )

    # ---------------- TAB 5: PAPERS ----------------
    with tab5:
        st.caption(
            f"All {result['num_papers']} papers retrieved and stored for "
            f"this topic, numbered to match [Source N] citations above."
        )
        for i, article in enumerate(result["articles"], start=1):
            st.markdown(f"**[Source {i}] {article['title']}**")
            st.write(
                f"{article.get('journal', '')} "
                f"({article.get('year', '')}) — "
                f"PMID: {article.get('pmid', '')}"
            )
            st.markdown(f"[View on PubMed]({article.get('url', '')})")
            with st.expander("Show abstract"):
                st.write(article.get("abstract", ""))
            st.divider()


if __name__ == "__main__":
    main()
