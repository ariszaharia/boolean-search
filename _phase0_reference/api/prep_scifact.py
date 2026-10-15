"""Export the full BEIR SciFact corpus (5,183 documents) to data/scifact.tsv.

Output format: <doc_id><TAB><title + text>, one document per line, sorted by
numeric doc id. The sort matters: InvertedIndex::add_document requires strictly
increasing ids. Needs `pip install datasets` and internet access.

Run from anywhere:  python api/prep_scifact.py
"""
import os

from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, "scifact.tsv")

print("Downloading SciFact corpus from Hugging Face...")
corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")

rows = []
for doc in corpus:
    # Tabs/newlines would break the TSV format.
    text = f"{doc['title']} {doc['text']}".replace("\t", " ").replace("\r", " ").replace("\n", " ")
    rows.append((int(doc["_id"]), text))

rows.sort(key=lambda r: r[0])
ids = [r[0] for r in rows]
assert len(ids) == len(set(ids)), "duplicate document ids"

print(f"Writing {len(rows)} documents to {output_file}...")
with open(output_file, "w", encoding="utf-8") as fout:
    for doc_id, text in rows:
        fout.write(f"{doc_id}\t{text}\n")

print(f"Success! Exported {len(rows)} documents (expected 5183 for SciFact).")
