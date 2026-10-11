from datasets import load_dataset
import os

# Automatically find the 'api' folder, then target the 'data' folder next to it
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, 'scifact_5k.tsv')

limit = 5000

print("Downloading SciFact corpus from Hugging Face...")
# Load the corpus split from BeIR/scifact
corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")

print(f"Processing and saving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as fout:
    count = 0
    for doc in corpus:
        if count >= limit:
            break
        
        doc_id = doc['_id']
        title = doc['title']
        text = doc['text']
        
        # Combine title and text, replace tabs and newlines with spaces to keep TSV clean
        full_text = f"{title} {text}".replace('\t', ' ').replace('\n', ' ')
        
        fout.write(f"{doc_id}\t{full_text}\n")
        count += 1

print(f"Success! Exported {count} documents to {output_file}")