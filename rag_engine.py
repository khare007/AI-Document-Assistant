import os
import re
import time
import uuid
import warnings
from typing import List, Dict, Any, Union

warnings.filterwarnings("ignore")

from pypdf import PdfReader  # type: ignore
import chromadb  # type: ignore
from dotenv import load_dotenv  # type: ignore
import google.generativeai as genai  # type: ignore

load_dotenv(override=True)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def validate_environment() -> str:
    """Ensures Google AI Studio GEMINI_API_KEY is present."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing required environment variable: GEMINI_API_KEY. "
            "Please set your Google AI Studio API key in the environment or in a .env file."
        )
    return api_key


def extract_pages(pdf_source: Union[str, Any]) -> List[Dict[str, Any]]:
    """Extract page text and page numbers from PDF."""
    reader = PdfReader(pdf_source)
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        clean = " ".join(text.split())
        if clean:
            pages.append({"page": idx + 1, "text": clean})
    return pages


def chunk_text(pages: List[Dict[str, Any]], size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Splits text into chunks on strict word boundaries (no cut-off words).
    Keeps section labels under 40 characters.
    """
    chunks = []
    cid = 0
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            end = start
            curr_len = 0
            while end < len(words) and curr_len + len(words[end]) + 1 <= size:
                curr_len += len(words[end]) + 1
                end += 1

            if end == start:
                end = start + 1

            chunk_str = " ".join(words[start:end])

            # Extract section label (max 40 chars)
            match = re.search(r'(\d+\.\s+[A-Za-z\s]+|About This Document)', chunk_str)
            if match:
                sec = match.group(0).strip()
                sec = sec[:40] if len(sec) <= 40 else f"Page {p['page']}"
            else:
                sec = f"Page {p['page']}"

            chunks.append({
                "id": f"c_{cid}",
                "page": p["page"],
                "section": sec,
                "text": chunk_str
            })
            cid += 1

            if end >= len(words):
                break

            # Calculate word-based overlap
            overlap_words = 0
            back = end - 1
            chars = 0
            while back > start and chars + len(words[back]) + 1 <= overlap:
                chars += len(words[back]) + 1
                overlap_words += 1
                back -= 1
            start = max(start + 1, end - overlap_words)

    return chunks


def build_prompt(context: str, question: str) -> str:
    """Construct grounded QA prompt."""
    return f"""You are a strict, factual Document QA Assistant. 
Your task is to answer the question using ONLY the provided document context below.

STRICT RULES:
1. Base your answer solely on the context provided.
2. If the answer cannot be directly determined from the context, state exactly: 
   "Information not available in the document."
3. Do not assume, extrapolate, or bring outside knowledge.
4. Keep the explanation concise and mention the relevant facts directly.

---
DOCUMENT CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""


def call_gemini(model: Any, prompt: str) -> str:
    """
    Calls Google AI Studio Gemini model with retry for empty responses and rate limits.
    No hardcoded fake fallbacks.
    """
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            if response and response.text and response.text.strip():
                return response.text.strip()

            # Empty response retry
            if attempt == 0:
                time.sleep(1)
                continue
            raise RuntimeError("Gemini model returned an empty response.")
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg and attempt == 0:
                time.sleep(5)
                continue
            if attempt == 0 and "429" not in err_msg:
                time.sleep(1)
                continue
            raise RuntimeError(f"Gemini API call failed: {e}")

    raise RuntimeError("Gemini API call failed after retries.")


class RAGEngine:
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.api_key = validate_environment()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model_name = MODEL_NAME

        # Configure Google AI Studio SDK
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 2048,
            }
        )

        # In-memory Chroma client with unique collection and cosine space
        self.chroma_client = chromadb.Client()
        self.collection_name = f"col_{uuid.uuid4().hex[:8]}_{chunk_size}"
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_pdf(self, pdf_source: Union[str, Any]):
        """Index PDF pages into ChromaDB."""
        pages = extract_pages(pdf_source)
        chunks = chunk_text(pages, self.chunk_size, self.overlap)
        if chunks:
            self.collection.add(
                documents=[c["text"] for c in chunks],
                metadatas=[{"page": c["page"], "section": c["section"]} for c in chunks],
                ids=[c["id"] for c in chunks]
            )

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve top matching passages using cosine distance."""
        res = self.collection.query(query_texts=[query], n_results=top_k)
        items = []
        if res and res.get("documents") and res["documents"][0]:
            for doc, meta, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                items.append({
                    "page": meta.get("page", 1),
                    "section": meta.get("section", "General"),
                    "text": doc,
                    "distance": round(float(d), 4)
                })
        return items

    def ask(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        """Answer question with retrieved context and real Gemini API call."""
        sources = self.retrieve(question, top_k=top_k)
        if not sources:
            return {
                "answer": "Information not available in the document.",
                "sources": [],
                "model": self.model_name
            }

        ctx = "\n\n".join([f"[Page {s['page']}]: {s['text']}" for s in sources])
        prompt = build_prompt(ctx, question)

        # Direct Google AI Studio Gemini API call
        answer = call_gemini(self.model, prompt)

        return {
            "answer": answer,
            "sources": sources,
            "model": self.model_name
        }
