"""
One-off script: exports a slice of the 20 Newsgroups dataset to individual
.txt files, so the C++ engine has a real ~10k-document corpus to ingest.

Run with: ../../venv/Scripts/python.exe export_newsgroups.py
"""
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path

N_DOCS = 10000
OUT_DIR = Path(__file__).parent / "newsgroups"

def main():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    OUT_DIR.mkdir(exist_ok=True)

    written = 0
    for i in range(min(N_DOCS, len(data.data))):
        text = data.data[i].strip()
        if not text:
            continue  # some posts are empty after header/footer/quote removal
        (OUT_DIR / f"doc_{i:05d}.txt").write_text(text, encoding="utf-8")
        written += 1

    print(f"Wrote {written} files to {OUT_DIR}")

if __name__ == "__main__":
    main()
