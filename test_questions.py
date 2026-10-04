import os
import warnings
from dotenv import load_dotenv  # type: ignore

warnings.filterwarnings("ignore")
load_dotenv(override=True)

from rag_engine import RAGEngine

PDF_PATH = "document.pdf"

QUESTIONS = [
    {
        "id": "1",
        "q": "What are the operating hours of the Student Support Desk?",
        "tag": "In-Scope"
    },
    {
        "id": "2",
        "q": "Does the Student Support Desk approve attendance exemptions?",
        "tag": "In-Scope"
    },
    {
        "id": "3",
        "q": "What four learning areas does GDG On Campus USAR organize activities in?",
        "tag": "In-Scope"
    },
    {
        "id": "4",
        "q": "What should a project README contain?",
        "tag": "In-Scope"
    },
    {
        "id": "5",
        "q": "What is the date and venue of the next GDG workshop and who is the lead?",
        "tag": "Out-of-Scope"
    },
    {
        "id": "6",
        "q": "Who do I contact if the learning portal is not working?",
        "tag": "In-Scope (Paraphrased)"
    }
]


def run_benchmark(label: str, chunk_size: int, overlap: int) -> str:
    header = f"## {label} (Chunk Size: {chunk_size}, Overlap: {overlap})\n\n"
    print("=" * 70)
    print(f"{label}: size={chunk_size}, overlap={overlap}")
    print("=" * 70)

    engine = RAGEngine(chunk_size=chunk_size, overlap=overlap)
    engine.index_pdf(PDF_PATH)

    log_content = header

    for item in QUESTIONS:
        res = engine.ask(item["q"], top_k=2)
        ans = res["answer"]
        sources = res["sources"]

        # Check PASS / FAIL for out-of-scope question
        status = ""
        if item["tag"] == "Out-of-Scope":
            if "not available" in ans.lower():
                status = " [PASS: Correctly Refused]"
            else:
                status = " [FAIL: Hallucinated/Did Not Refuse]"

        print(f"\n[Q{item['id']}] {item['q']} ({item['tag']}){status}")
        print(f"Answer: {ans}")
        print(f"Model: {res.get('model', 'N/A')}")

        src_log = ""
        for i, s in enumerate(sources):
            line = f"  - Source {i+1}: Page {s['page']} | Section: {s['section']} | Cosine Distance: {s['distance']}"
            print(line)
            src_log += line + "\n"

        log_content += f"### Q{item['id']}: {item['q']}\n"
        log_content += f"- **Type**: {item['tag']}{status}\n"
        log_content += f"- **Answer**: {ans}\n"
        log_content += f"- **Retrieved Sources**:\n{src_log}\n"

    return log_content


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found.")
        exit(1)

    print("\nStarting RAG Evaluation Benchmark across 2 Chunk Settings...\n")

    report = "# RAG Evaluation Results\n\n"
    report += f"Evaluated Document: `{PDF_PATH}`\n\n"

    try:
        report += run_benchmark("Setting 1 (Small Chunks)", chunk_size=400, overlap=80)
        print("\n")
        report += run_benchmark("Setting 2 (Large Chunks)", chunk_size=800, overlap=150)

        with open("results.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + "=" * 70)
        print("Benchmark complete. Results successfully saved to results.md")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        exit(1)
