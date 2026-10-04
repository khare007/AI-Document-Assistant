# Project Decisions (DECISIONS.md)

This file documents the key technical choices and experiments made while building this project.

---

### 1. Choice of Vector Database: ChromaDB
- **Decision**: I used **ChromaDB** as the in-memory vector database.
- **Alternative Considered**: FAISS or Pinecone.
- **Why ChromaDB?**: 
  - ChromaDB is lightweight, runs locally without setting up an external database, and automatically manages embeddings.
  - It easily stores metadata (like page numbers and section names) alongside text chunks, making it simple to show source citations.

---

### 2. Chunking Settings Comparison (Experiment)
I tested two different chunk sizes in `test_questions.py` to see how retrieval quality changes:

- **Setting 1: Small Chunks (400 characters, 80 overlap)**
  - *Result*: Good for simple, single-line questions (like office timings). The retrieved passage was short and straight to the point.
  - *Downside*: When a topic had multiple bullet points (like README requirements), a small chunk sometimes cut off the text mid-sentence.

- **Setting 2: Large Chunks (800 characters, 150 overlap)**
  - *Result*: Better for questions that need full context. Entire paragraphs and headings stayed together, giving the LLM all necessary details.
  - *Final Choice*: **800 characters** was chosen as the default because student handbook rules are clearer when read with their surrounding context.

---

### 3. Grounding & Handling Out-of-Scope Questions
- **Rule**: The prompt strictly instructs the model: if the answer is not in the retrieved passages, say *"Information not available in the document."*
- **Test**: Question 5 asked for the upcoming workshop date and venue (which does not exist in the handbook).
- **Result**: The model correctly replied that the information is not available, instead of guessing or making up dates.

---

### 4. Challenge Faced & Fix
- **Issue**: At first, using a higher temperature allowed the model to occasionally guess plausible answers when information was missing.
- **Fix**: I set the temperature to `0.1` and added an explicit negative instruction in the prompt. This made the model strictly factual and consistent.
