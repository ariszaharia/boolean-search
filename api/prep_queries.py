from datasets import load_dataset
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)

queries = load_dataset("BeIR/scifact", "queries", split="queries")
text_of = {str(q['_id']): q['text'].replace('\t', ' ').replace('\n', ' ') for q in queries}

# Same splits as api/prep_qrels.py: a query belongs to a split if that split's qrels judge it.
for split in ('train', 'test'):
    ids = {str(r['query-id']) for r in load_dataset("BeIR/scifact-qrels", split=split)}
    missing = sorted(ids - set(text_of))
    if missing:
        raise SystemExit(f"{split}: {len(missing)} judged queries have no text, e.g. {missing[:5]}")

    output_file = os.path.join(data_dir, f'scifact_queries_{split}.tsv')
    with open(output_file, 'w', encoding='utf-8') as fout:
        for qid in sorted(ids, key=int):
            fout.write(f"{qid}\t{text_of[qid]}\n")
    print(f"{split}: {len(ids)} queries -> {output_file}")
