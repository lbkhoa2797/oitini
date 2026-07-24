"""Compare dense-only vs sparse-only vs hybrid retrieval.

Usage:
    python scripts/05_compare_retrieval.py

Runs a set of test queries through all three retrieval strategies,
reranks each with the same cross-encoder, and prints a side-by-side
comparison of rerank scores, rank positions, and overlap statistics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieve import (
    dense_search, sparse_search, reciprocal_rank_fusion, rerank,
)

# ── Test queries (edit/extend as you like) ──────────────────────────
QUERIES = [
    "magnon-phonon interaction in CrI3",
    "band gap in monolayer CrBr3 from DFT calculation",
    "out-of-plane phonon modes couple to magnetic moment in CrBr3",
    "exciton-magnon coupling in CrSBr",
    "Curie temperature of monolayer CrI3",
    "spin wave dispersion in van der Waals magnets",
]

TOP_N = 5      # final results per strategy
POOL_K = 25    # candidates fed into reranker


def run_three_strategies(query: str):
    """Return (dense_hits, sparse_hits, hybrid_hits) — all reranked to TOP_N."""
    dense_cands = dense_search(query, k=POOL_K)
    sparse_cands = sparse_search(query, k=POOL_K)
    fused_cands = reciprocal_rank_fusion(dense_cands, sparse_cands)

    dense_reranked = rerank(query, dense_cands[:POOL_K], top_n=TOP_N)
    sparse_reranked = rerank(query, sparse_cands[:POOL_K], top_n=TOP_N)
    hybrid_reranked = rerank(query, fused_cands[:POOL_K * 2], top_n=TOP_N)

    return dense_reranked, sparse_reranked, hybrid_reranked


def chunk_ids(hits):
    return [h["chunk_id"] for h in hits]


def avg_rerank_score(hits):
    if not hits:
        return 0.0
    return sum(h["rerank_score"] for h in hits) / len(hits)


def top1_rerank_score(hits):
    return hits[0]["rerank_score"] if hits else 0.0


def print_hits(label, hits):
    print(f"  {label}:")
    for i, h in enumerate(hits):
        meta = h["metadata"]
        print(f"    [{i}] rerank={h['rerank_score']:.3f}  {meta['arxiv_id']:>14}  "
              f"{meta['section']:>10}  {meta.get('section_title', '')[:60]}")
        print(f"         {h['text'][:120]}...")


def main():
    totals = {"dense": [], "sparse": [], "hybrid": []}

    for q in QUERIES:
        print(f"\n{'='*80}")
        print(f"QUERY: {q}")
        print('='*80)

        dense_hits, sparse_hits, hybrid_hits = run_three_strategies(q)

        print_hits("Dense only", dense_hits)
        print_hits("Sparse only", sparse_hits)
        print_hits("Hybrid (RRF)", hybrid_hits)

        # ── Per-query statistics ────────────────────────────────────
        d_ids = set(chunk_ids(dense_hits))
        s_ids = set(chunk_ids(sparse_hits))
        h_ids = set(chunk_ids(hybrid_hits))

        print(f"\n  Overlap:")
        print(f"    dense  ∩ sparse : {len(d_ids & s_ids)} / {TOP_N}")
        print(f"    dense  ∩ hybrid : {len(d_ids & h_ids)} / {TOP_N}")
        print(f"    sparse ∩ hybrid : {len(s_ids & h_ids)} / {TOP_N}")
        print(f"    unique to hybrid: {len(h_ids - d_ids - s_ids)}")

        print(f"\n  Avg rerank score:  dense={avg_rerank_score(dense_hits):.3f}  "
              f"sparse={avg_rerank_score(sparse_hits):.3f}  "
              f"hybrid={avg_rerank_score(hybrid_hits):.3f}")
        print(f"  Top-1 rerank score: dense={top1_rerank_score(dense_hits):.3f}  "
              f"sparse={top1_rerank_score(sparse_hits):.3f}  "
              f"hybrid={top1_rerank_score(hybrid_hits):.3f}")

        totals["dense"].append(avg_rerank_score(dense_hits))
        totals["sparse"].append(avg_rerank_score(sparse_hits))
        totals["hybrid"].append(avg_rerank_score(hybrid_hits))

    # ── Summary across all queries ──────────────────────────────────
    n = len(QUERIES)
    print(f"\n{'='*80}")
    print(f"SUMMARY over {n} queries")
    print('='*80)
    for mode in ("dense", "sparse", "hybrid"):
        scores = totals[mode]
        mean = sum(scores) / n
        print(f"  {mode:>8}  mean_avg_rerank = {mean:.3f}  "
              f"(per-query: {', '.join(f'{s:.3f}' for s in scores)})")

    # Win counts
    wins = {"dense": 0, "sparse": 0, "hybrid": 0}
    for d, s, h in zip(totals["dense"], totals["sparse"], totals["hybrid"]):
        best = max(d, s, h)
        if h == best: wins["hybrid"] += 1
        elif d == best: wins["dense"] += 1
        else: wins["sparse"] += 1
    print(f"\n  Wins (by avg rerank score): {wins}")


if __name__ == "__main__":
    main()
