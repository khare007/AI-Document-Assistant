import streamlit as st  # type: ignore
import os
import warnings
from dotenv import load_dotenv  # type: ignore

# Load environment variables from .env
load_dotenv(override=True)

warnings.filterwarnings("ignore")

from rag_engine import RAGEngine

st.set_page_config(page_title="AI Document Assistant", page_icon="✨")

# Title with Gemini Sparkle Icon
st.title("✨ AI Document Assistant")

DEFAULT_PDF = "document.pdf"

# Sidebar configuration
with st.sidebar:
    st.header("Document Source")
    st.info("Default: `document.pdf`. Upload below to override.")
    uploaded_file = st.file_uploader("Upload PDF (Optional)", type=["pdf"])

    st.header("Settings")
    chunk_size = st.select_slider(
        "Chunk Size (characters)",
        options=[400, 800],
        value=800,
        help="Controls the length of text passages the document is split into."
    )
    top_k = st.slider(
        "Top Passages (Passage Count)",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of most relevant passages retrieved to formulate the answer."
    )

pdf_target = uploaded_file if uploaded_file else (DEFAULT_PDF if os.path.exists(DEFAULT_PDF) else None)

if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing GEMINI_API_KEY. Please set your key in the `.env` file.")
elif not pdf_target:
    st.warning("Please provide 'document.pdf' or upload a PDF file.")
else:
    doc_name = getattr(uploaded_file, "name", DEFAULT_PDF)

    @st.cache_resource(show_spinner=False)
    def init_engine(source, size):
        engine = RAGEngine(chunk_size=size, overlap=150)
        engine.index_pdf(source)
        return engine

    try:
        with st.spinner(f"Indexing {doc_name}..."):
            engine = init_engine(pdf_target, chunk_size)
        st.success(f"Indexed: `{doc_name}`")

        query = st.text_input("Question:", placeholder="Ask anything about the document...")
        if st.button("Submit") and query:
            with st.spinner("Retrieving passages & formulating answer..."):
                try:
                    res = engine.ask(query, top_k=top_k)
                    st.subheader("Answer:")
                    st.write(res["answer"])

                    st.subheader("Sources:")
                    for i, s in enumerate(res["sources"]):
                        with st.expander(f"Source {i+1} — Page {s['page']} ({s['section']})"):
                            st.write(s["text"])
                            st.caption(f"Cosine Distance: `{s['distance']}`")
                except Exception as e:
                    st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Setup Error: {e}")
