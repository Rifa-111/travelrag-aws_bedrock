"""Streamlit UI for the TravelRAG portfolio demonstration."""
import os
import streamlit as st
from src.rag import Retriever, answer, chunk_documents

st.set_page_config(page_title='TravelRAG | Bedrock Demo', page_icon='✈️')
st.title('✈️ TravelRAG')
st.caption('Fictional travel-policy RAG demo • local TF-IDF retrieval • optional AWS Bedrock')
st.info('All policies are synthetic examples. Not affiliated with Amex GBT.')

@st.cache_resource
def get_retriever():
    return Retriever(chunk_documents())

question = st.text_input('Ask a policy question', placeholder='What is the hotel reimbursement limit?')
if st.button('Ask', type='primary') and question:
    try:
        with st.spinner('Searching policy documents…'):
            result = answer(question, get_retriever(), os.getenv('BEDROCK_MODEL_ID'))
        st.subheader('Answer')
        st.write(result['answer'])
        st.caption(f"Mode: {result['mode']}")
        if result['sources']:
            st.subheader('Retrieved evidence')
            for item in result['sources']:
                with st.expander(f"{item['source']} · similarity {item['score']:.3f}"):
                    st.write(item['text'])
    except Exception as exc:
        st.error(f'Unable to complete the request: {type(exc).__name__}. Check AWS permissions, model access and region.')
