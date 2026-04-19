import streamlit as st
import os
from dotenv import load_dotenv
from src.search import AdvancedRAGPipeline

load_dotenv()

st.set_page_config(page_title="Textbook Assistant", page_icon="📚", layout="centered")

@st.cache_resource
def get_pipeline():
    # Make sure GROQ_API_KEY is somewhat checked
    if "GROQ_API_KEY" not in os.environ or os.environ["GROQ_API_KEY"] == "your_groq_api_key_here":
        st.warning("Please set your GROQ_API_KEY in the `.env` file.")
        return None
    return AdvancedRAGPipeline()

st.title("Advanced RAG Architecture Showcase")

st.markdown("""
This application demonstrates a production-grade Retrieval-Augmented Generation (RAG) pipeline engineered to eliminate LLM hallucination in academic research. 

""")
pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Show Textbook Citations", expanded=False):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source {idx+1}:** {src['source']} (Page {src['page']})")
                    st.caption(f"_{src['preview']}_")
                    st.caption(f"*Confidence Score: {src['score']:.2f}*")

# Chat input
if prompt := st.chat_input("Ask a question based on your textbook (e.g., 'What is Backpropagation?'):"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not pipeline:
            st.error("Pipeline not initialized. Check your API Key in the `.env` file.")
            response = "Error: Uninitialized Pipeline."
            sources = []
        else:
            with st.spinner("Consulting the textbook..."):
                try:
                    result = pipeline.query(prompt)
                    response = result['answer']
                    sources = result['sources']
                    st.markdown(response)
                    if sources:
                        with st.expander("Show Textbook Citations", expanded=True):
                            for idx, src in enumerate(sources):
                                st.markdown(f"**Source {idx+1}:** {src['source']} (Page {src['page']})")
                                st.caption(f"_{src['preview']}_")
                                st.caption(f"*Confidence Score: {src['score']:.2f}*")
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.error(response)
                    sources = []

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})