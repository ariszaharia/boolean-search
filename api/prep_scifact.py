from datasets import load_dataset
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, 'scifact.tsv')

corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")

# The C++ engine needs document ids in increasing numeric order.
docs = sorted(corpus, key=lambda d: int(d['_id']))

with open(output_file, 'w', encoding='utf-8') as fout:
    count = 0
    for doc in docs:
        doc_id = doc['_id']
        title = doc['title']
        text = doc['text']
        
        full_text = f"{title} {text}".replace('\t', ' ').replace('\n', ' ')
        
        fout.write(f"{doc_id}\t{full_text}\n")
        count += 1

print(f"Exported {count} documents to {output_file}")
if count != 5183:
    print("WARNING: SciFact should have 5183 documents.")