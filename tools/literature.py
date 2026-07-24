from pydantic import BaseModel, Field
from typing import Optional, Literal

from langsmith import traceable

from rag.retrieve import hybrid_search

class LiteratureSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language query, e.g. 'magnon mean free path in CrI3 at 20 K'.")
    top_k: int = Field(5, ge=1, le=10, description="Number of chunks to return.")
    section: Optional[Literal["abstract", "introduction", "methods", "results", "conclusion"]] = Field(
        None,
        description=(
            "Optional filter restricting results to a specific section type. Use 'methods' for "
            "questions about computational/experimental setup, 'results' for numerical findings, "
            "'introduction' for context and prior work."
        ),
    )

def _docs_for_trace(results: list[dict]) -> list[dict]:
    """Trace-only projection into LangSmith's retriever document shape."""
    return [
        {
            "page_content": r["text"],
            "type": "Document",
            "metadata": {k: r.get(k) for k in
                         ("chunk_id", "arxiv_id", "title", "year", "section", "score")},
        }
        for r in results
    ]

@traceable(run_type="retriever", process_outputs=_docs_for_trace)
def literature_search(query: str, top_k: int = 5, section: Optional[str] = None) -> list[dict]:
    hits = hybrid_search(query, top_n=top_k, section_filter=section)
    # Trim the response — the agent doesn't need the full chunk, just enough to ground its answer.
    return [
        {
            "chunk_id": h["chunk_id"],
            "arxiv_id": h["metadata"]["arxiv_id"],
            "title": h["metadata"]["title"],
            "year": h["metadata"].get("year"),
            "section": h["metadata"]["section"],
            "text": h["text"],
            "score": round(h["rerank_score"], 3),
        }
        for h in hits
    ]

TOOL_SPEC = {
    "name": "literature_search",
    "description": (
        "Search the local corpus of arXiv papers. Returns up to top_k chunks of "
        "text from relevant papers, each with a unique chunk_id, the source arxiv_id, title, "
        "year, section, and a relevance score. ALWAYS cite results by chunk_id when you use "
        "them in your final answer. If you need a specific kind of content (e.g. the methodology "
        "of a paper), pass section='methods'."
    ),
    "input_schema": LiteratureSearchInput.model_json_schema(),
}
