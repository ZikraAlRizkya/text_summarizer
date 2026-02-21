# ============================================================
# STREAMLIT UI APP
# ============================================================

import streamlit as st

# Import module Member 1
from modules.file_reader import read_file
from modules.preprocessing import preprocess_text
from modules.extractive import extractive_summary
from modules.abstractive import abstractive_summary_gemini

# Import module Member 2
from modules.evaluation import evaluate_all
from modules.keywords import extract_keywords
from modules.ner import extract_entities
from modules.graph import build_entity_graph, visualize_graph


def run_app():
    st.title("📄 Text Summarizer AI")

    uploaded_file = st.file_uploader("Upload file (.txt / .pdf)", type=["txt", "pdf"])

    if uploaded_file:
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.read())

        text = read_file("temp_file")

        if text:
            st.subheader("📜 Original Text")
            st.write(text[:1000])

            # Preprocessing
            prep = preprocess_text(text)

            # Summaries
            ext_summary = extractive_summary(text=text)
            abs_summary = abstractive_summary_gemini(text)

            st.subheader("🧠 Extractive Summary")
            st.write(ext_summary)

            st.subheader("🤖 Abstractive Summary")
            st.write(abs_summary)

            # Evaluation
            metrics = evaluate_all(text, ext_summary, abs_summary)

            st.subheader("📊 Evaluation Metrics")
            st.json(metrics)

            # Keywords
            keywords = extract_keywords(text)
            st.subheader("🔑 Keywords")
            st.write(keywords)

            # NER + Graph
            entities = extract_entities(text)
            G = build_entity_graph(entities)
            fig = visualize_graph(G)

            st.subheader("🌐 Entity Graph")
            st.plotly_chart(fig)


if __name__ == "__main__":
    run_app()