from datasets import load_dataset
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)

# Official BEIR splits: "train" is our dev set (prompt and relaxation tuning), "test" is held out.
for split in ('train', 'test'):
    rows = load_dataset("BeIR/scifact-qrels", split=split)
    output_file = os.path.join(data_dir, f'scifact_qrels_{split}.tsv')

    queries = set()
    with open(output_file, 'w', encoding='utf-8') as fout:
        for row in rows:
            queries.add(str(row['query-id']))
            fout.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")

    print(f"{split}: {len(rows)} judgments for {len(queries)} queries -> {output_file}")
