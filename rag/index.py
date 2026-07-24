"""Build the dense (Chroma) and sparse (BM25) indices over chunks.jsonl."""
import json
import pickle

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import CHUNKS_PATH, CHROMA_DIR, BM25_PATH, EMBED_MODEL

CHROMA_COLLECTION = "papers"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines() if line.strip()]

def simple_tokenize(text: str) -> list[str]:
    """BM25 needs tokenized input. Lowercase, alnum-split, drop short tokens."""
    import re
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]

def build_indices() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    print(f"{len(chunks)} chunks to index")

    # ---- dense (Chroma) ------------------------------------------------
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = min(model.max_seq_length or 1024, 1024)
    print(f"Current embedding dimension: {model.get_embedding_dimension()}")
    print(f"Max sequence length: {model.max_seq_length}")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    # Recreate the collection so re-running is idempotent.
    try:
        chroma.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    
    # Use cosine as a distance metric for 2 vectors
    collection = chroma.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})

    BATCH = 64
    for i in tqdm(range(0, len(chunks), BATCH), desc="embed"):
        batch = chunks[i : i + BATCH]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, batch_size=BATCH, normalize_embeddings=True).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=texts,
            embeddings=vectors,
            metadatas=[
                {
                    "arxiv_id": c["arxiv_id"],
                    "title": c["title"][:200],   # Chroma metadata has size limits
                    "year": c.get("year") or 0,
                    "section": c["section"],
                    "section_title": c["section_title"][:200],
                }
                for c in batch
            ],
        )

    # ---- sparse (BM25) -------------------------------------------------
    tokenized = [simple_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "chunks_by_id": {c["chunk_id"]: c for c in chunks},
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 index ({len(tokenized)} docs) saved to {BM25_PATH}")


if __name__ == "__main__":
    build_indices()
