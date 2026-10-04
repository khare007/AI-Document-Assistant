# AI-Powered Document Assistant (GDG-USAR Task 3)

A simple question-answering tool built for the **GDG-USAR Student Handbook**. It uses RAG (Retrieval-Augmented Generation) to answer questions based only on the document, showing the exact page number and text source for every answer.

---

## 🛠️ Tech Stack
- **PDF Extraction**: `pypdf`
- **Vector Database**: `ChromaDB` (In-memory with Cosine Similarity)
- **Embedding Model**: Default `all-MiniLM-L6-v2`
- **LLM**: Google Gemini 3 Flash (`gemini-3-flash-preview`)
- **Web UI**: `Streamlit`

---

## ⚙️ Setup & Installation

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add API Key:**
   Create a `.env` file in the project root and add your Google AI Studio key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-3-flash-preview
   ```

---

## 🚀 How to Run

### 1. Run Automated Test (5+ Questions):
```bash
python test_questions.py
```
This script tests 6 questions (including an out-of-scope question) on two chunk sizes and saves the output to `results.md`.

### 2. Run the Web App:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to ask questions interactively.

---

## 📌 Limitations
- Works best with digital text PDFs (scanned image PDFs without OCR are not supported).
- Vector data is kept in-memory for this single document, not stored in a permanent cloud database.
- Subject to free Google AI Studio rate limits.
