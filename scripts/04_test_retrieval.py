import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieve import hybrid_search

QUERIES = [
    "magnon-phonon interaction in CrI3",
    "band gap in monolayer CrBr3 from DFT calculation",
    "out-of-plane phonon modes couple to magnetic moment in CrBr3",
    "exciton-magnon coupling in CrSBr is observed",
]

for q in QUERIES:
    print(f"\n=== {q!r}")
    hits = hybrid_search(q, top_n=3)
    for i, h in enumerate(hits):
        print(f"  [{i}] {h['metadata']['arxiv_id']:>14}  {h['metadata']['section']:>10}  "
              f"rerank={h['rerank_score']:.2f}")
        print(f"       {h['metadata']['section_title'][:80]}")
        print(f"       {h['text'][:150]}...")
