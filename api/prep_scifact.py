from datasets import load_dataset
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, 'scifact_5k.tsv')

limit = 5000

corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")

with open(output_file, 'w', encoding='utf-8') as fout:
    count = 0
    for doc in corpus:
        if count >= limit:
            break
        
        doc_id = doc['_id']
        title = doc['title']
        text = doc['text']
        
        full_text = f"{title} {text}".replace('\t', ' ').replace('\n', ' ')
        
        fout.write(f"{doc_id}\t{full_text}\n")
        count += 1

print(f"Exported {count} documents to {output_file}")