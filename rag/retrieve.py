"""Hybrid retrieval (dense + sparse) with cross-encoder reranking."""
import pickle
from typing import Optional

import chromadb
from chromadb.config import Settings
from langsmith import traceable
from sentence_transformers import SentenceTransformer, CrossEncoder

import tracemalloc
import torch

from config import CHROMA_DIR, BM25_PATH, EMBED_MODEL, RERANK_MODEL, DEVICE, RERANK_BATCH_SIZE
from rag.index import CHROMA_COLLECTION, BGE_QUERY_PREFIX, simple_tokenize

# Module-level lazy singletons. Loading models on every call is slow.
_embedder: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_chroma_collection = None
_bm25_state = None

# Also tracking memory allocated
def _get_embedder():
    global _embedder
    if _embedder is None:
        tracemalloc.start()
        mem_before = 0
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()
        
        _embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[embedder] RAM — current: {current / 1e6:.1f} MB, peak: {peak / 1e6:.1f} MB")
        if torch.cuda.is_available():
            vram_used = (torch.cuda.memory_allocated() - mem_before) / 1e6
            print(f"[embedder] VRAM allocated: {vram_used:.1f} MB")

    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        tracemalloc.start()
        mem_before = 0
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()

        _reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[reranker] RAM — current: {current / 1e6:.1f} MB, peak: {peak / 1e6:.1f} MB")
        if torch.cuda.is_available():
            vram_used = (torch.cuda.memory_allocated() - mem_before) / 1e6
            print(f"[reranker] VRAM allocated: {vram_used:.1f} MB")

    return _reranker


def _get_chroma():
    global _chroma_collection
    if _chroma_collection is None:
        tracemalloc.start()

        client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        _chroma_collection = client.get_collection(CHROMA_COLLECTION)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[chroma] RAM — current: {current / 1e6:.1f} MB, peak: {peak / 1e6:.1f} MB")

    return _chroma_collection


def _get_bm25():
    global _bm25_state
    if _bm25_state is None:
        tracemalloc.start()

        with open(BM25_PATH, "rb") as f:
            _bm25_state = pickle.load(f)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[bm25] RAM — current: {current / 1e6:.1f} MB, peak: {peak / 1e6:.1f} MB")

    return _bm25_state


def _hits_as_documents(hits: list[dict]) -> list[dict]:
    """Trace-only projection into LangSmith's retriever document shape."""
    return [
        {
            "page_content": h["text"],
            "type": "Document",
            "metadata": {
                "chunk_id": h["chunk_id"],
                **(h.get("metadata") or {}),
                **{k: v for k, v in h.items() if k.endswith("_score")},
            },
        }
        for h in hits
    ]


@traceable(run_type="retriever", process_outputs=_hits_as_documents)
def dense_search(query: str, k: int = 25, section_filter: Optional[str] = None) -> list[dict]:
    """Top-k by cosine similarity over BGE embeddings."""
    embedder = _get_embedder()
    qv = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    # where = {"section": section_filter} if section_filter else None
    where: chromadb.Where | None = {"section": section_filter} if section_filter else None
    res = _get_chroma().query(query_embeddings=qv, n_results=k, where=where)
    out = []
    # Add a guard in case all the fields are empty
    if (
        res["ids"] is None
        or res["documents"] is None
        or res["metadatas"] is None
        or res["distances"] is None
    ):
        return out

    for cid, doc, meta, dist in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        out.append({
            "chunk_id": cid,
            "text": doc,
            "metadata": meta,
            "dense_score": 1.0 - dist,   # cosine distance -> similarity
        })
    return out


@traceable(run_type="retriever", process_outputs=_hits_as_documents)
def sparse_search(query: str, k: int = 25, section_filter: Optional[str] = None) -> list[dict]:
    """Top-k by BM25 score."""
    state = _get_bm25()
    bm25 = state["bm25"]
    chunk_ids = state["chunk_ids"]
    chunks_by_id = state["chunks_by_id"]

    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # If filtering by section, mask out non-matching chunks before top-k.
    if section_filter:
        for i, cid in enumerate(chunk_ids):
            if chunks_by_id[cid].get("section") != section_filter:
                scores[i] = -1.0

    # Take and sort the top_k highest scores
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    out = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        c = chunks_by_id[chunk_ids[i]]
        out.append({
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "metadata": {
                "arxiv_id": c["arxiv_id"], "title": c["title"], "year": c.get("year"),
                "section": c["section"], "section_title": c["section_title"],
            },
            "sparse_score": float(scores[i]),
        })
    return out


@traceable(run_type="chain")
def reciprocal_rank_fusion(*rank_lists: list[dict], k: int = 60) -> list[dict]:
    """
    Combine multiple ranked lists with RRF. For each doc d, score is sum_l 1/(k + rank_l(d)).
    k=60 is the canonical value from the original RRF paper.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for rl in rank_lists:
        for rank, item in enumerate(rl):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = item
    fused = [{**chunks[cid], "rrf_score": s} for cid, s in scores.items()]
    fused.sort(key=lambda x: -x["rrf_score"])
    return fused


@traceable(run_type="chain")
def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Score each (query, candidate) pair with a cross-encoder; keep top_n."""
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    # scores = reranker.predict(pairs)
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: -x["rerank_score"])
    return candidates[:top_n]


@traceable(run_type="chain")
def hybrid_search(
    query: str,
    top_n: int = 5,
    pool_k: int = 25,
    section_filter: Optional[str] = None, # specifically look into a section
) -> list[dict]:
    """
    Full pipeline: dense top-K + sparse top-K → RRF fusion → cross-encoder rerank → top_n.
    Default is the right starting point: pool 25 from each retriever, rerank to 5.
    """
    dense = dense_search(query, k=pool_k, section_filter=section_filter)
    sparse = sparse_search(query, k=pool_k, section_filter=section_filter)
    fused = reciprocal_rank_fusion(dense, sparse)
    reranked = rerank(query, fused[: pool_k * 2], top_n=top_n)
    return reranked
