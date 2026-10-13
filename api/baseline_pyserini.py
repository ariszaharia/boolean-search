from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-scifact.flat')

queries = [
    "mutation", 
    "immune response",
    "\"stem cells\""
]

print("\n=== PYSERINI (LUCENE) EVALUATION ===\n")

for q in queries:
    print(f"Query: {q}")
    # Fetch top 3 hits
    hits = searcher.search(q, k=3)
    
    for i, hit in enumerate(hits):
        doc = searcher.doc(hit.docid)
        text = doc.raw() if doc else ""
        snippet = text[:100].replace('\n', ' ') + "..." if len(text) > 100 else text
        
        print(f" -> Doc ID: {hit.docid} | Score: {hit.score:.5f} | Snippet: {snippet}")
    print("-" * 26)