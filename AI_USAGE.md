# AI Usage Declaration (AI_USAGE.md)

In line with the project guidelines, here is a clear summary of how AI tools were used in this task:

---

### 1. Tools Used
- **AI Assistant**: Google Antigravity (Gemini 3.8 Flash).
- **Core LLM**: Google Gemini 3 Flash Preview (`gemini-3-flash-preview`) via Google AI Studio.

---

### 2. How AI Was Used
- **Code Setup**: Helped organize the modular file structure (`rag_engine.py`, `app.py`, `test_questions.py`).
- **Prompt Design**: Assisted in writing the grounded prompt to ensure the model does not guess answers.
- **Documentation**: Helped draft the structure for `README.md` and `DECISIONS.md`.

---

### 3. Human Review & Verification
- Manually tested the PDF loading and verified extracted text against the original document.
- Formulated the test questions, including the out-of-scope question to test hallucination prevention.
- Ran and compared both chunking settings (400 vs 800 characters) to analyze retrieval differences.
- Tested and verified the Streamlit web application.
