"""Streamlit UI for Titan semantic retrieval and optional Bedrock generation."""
import os
import streamlit as st
from src.rag import Retriever, answer, chunk_documents

st.set_page_config(page_title="TravelRAG | Titan + Bedrock", page_icon="✈️")
st.title("✈️ TravelRAG")
st.caption("Amazon Titan Text Embeddings V2 semantic search + optional Bedrock Converse")
st.info("Fictional policies only. Not affiliated with Amex GBT. AWS inference incurs charges.")

@st.cache_resource
def get_retriever():
    return Retriever(chunk_documents())

question = st.text_input("Ask a policy question", placeholder="What is the hotel reimbursement limit?")
if st.button("Ask", type="primary") and question:
    try:
        with st.spinner("Retrieving evidence using Titan embeddings…"):
            result = answer(question, get_retriever(), os.getenv("BEDROCK_MODEL_ID"))
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption("Mode: " + result["mode"])
        for item in result["sources"]:
            with st.expander(f"{item['source']} · cosine similarity {item['score']:.3f}"):
                st.write(item["text"])
    except Exception as exc:
        st.error(f"Request failed ({type(exc).__name__}). Check AWS credentials, model access and region.")
