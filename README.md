# AI-Powered Document Assistant

An AI assistant that reads PDF documents and answers questions based only on the text. It always shows the exact page number and source passage for each answer so you can easily verify the facts.

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

### 1. Run Automated Test (Benchmark):
```bash
python test_questions.py
```
This script evaluates sample test questions (including an out-of-scope question) across two chunk sizes and saves the output to `results.md`.

### 2. Run the Web App:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to ask questions interactively.

---

## 📌 Limitations
- Works best with digital text PDFs (scanned image PDFs without OCR are not supported).
- Vector data is kept in-memory for the loaded document, not stored in a permanent cloud database.
