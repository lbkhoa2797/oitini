"""Chunk parsed papers into retrieval units with rich metadata."""
import json
import re
from pathlib import Path
from typing import Iterator, cast
from anthropic.types import ToolParam, ToolChoiceParam, MessageParam
from anthropic import Anthropic
from tqdm import tqdm

from config import PARSED_DIR, CHUNKS_PATH, USE_LLM_CHUNK_CLASSIFIER

# Section detector for arXiv preprints in markdown style. Limited to the top two
# heading levels so a sub-subsection ("A. Crystal structure") is kept inside its
# parent rather than becoming an unlabelled section of its own.

SECTION_RE = re.compile(r"^#{1,2}\s+(.+)$", re.MULTILINE)

def split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_body), ...] preserving order."""
    matches = list(SECTION_RE.finditer(markdown))
    # Returns the whole body if poorly parsed
    if not matches:
        return [("Body", markdown)]
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        # Here we slice from the end of a title
        start = m.end()
        # to the start of the next title (end of the string if it is the last)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((title, body))
    return sections

def split_by_tokens(text: str, target_tokens: int = 450, overlap_tokens: int = 50) -> list[str]:
    """450 tokens per chunk (roughly 450 English words)"""
    # 50 overlapping tokens as a linker to preserve logical continuity between chunks
    words = text.split()
    if len(words) <= target_tokens:
        return [text]
    chunks, i = [], 0
    while i < len(words):
        chunk_words = words[i : i + target_tokens]
        chunks.append(" ".join(chunk_words))
        if i + target_tokens >= len(words):
            break
        i += target_tokens - overlap_tokens
    return chunks


CLASSIFIER_MODEL = "claude-haiku-4-5"
SECTION_LABELS = [
    "abstract", "introduction", "methods", "results",
    "conclusion", "references", "acknowledgement", "other",
]

CLASSIFIER_TOOL = {
    "name": "record_section",
    "description": "Record the canonical section label for a paper section title.",
    "input_schema": {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": SECTION_LABELS,
                "description": "The canonical section bucket this title belongs to.",
            },
        },
        "required": ["section"],
    },
}

CLASSIFIER_SYSTEM = """You classify scientific paper section titles into canonical buckets.

Given a section title from an arXiv preprint, return the best-matching bucket via the record_section tool. Guidance:
- "methods" covers methodology, computational details, experimental setup, simulation protocols.
- "results" covers results, discussion, analysis, findings.
- "conclusion" covers conclusions, summary, outlook, future work.
- "acknowledgement" covers acknowledgements, funding, author contributions, conflicts of interest.
- "other" is the fallback for appendices, supplementary material, notation, etc.

Always call the tool exactly once."""

_CACHE_PATH = Path(__file__).parent / ".section_cache.json"
_cache: dict[str, str] = {}
if _CACHE_PATH.exists():
    _cache = json.loads(_CACHE_PATH.read_text())

_client = Anthropic()

def _classify_section_keyword(title: str) -> str:
    """Keyword fallback. Kept for use when the LLM call fails.

    Order matters. Unambiguous buckets are tested first, and `results` is checked
    before `methods` so that "Experimental results" lands in results rather than
    being claimed by the "experimental" keyword.
    """
    t = title.lower()
    if "abstract" in t: return "abstract"
    if "reference" in t or "bibliograph" in t: return "references"
    if any(k in t for k in ("acknowledgement", "acknowledgment", "funding",
                            "conflict of interest", "author contribution",
                            "competing interest", "data availability")):
        return "acknowledgement"
    if any(k in t for k in ("appendix", "supplementary", "supporting information")):
        return "other"
    if any(k in t for k in ("conclusion", "summary", "outlook", "future work",
                            "concluding remark", "perspective")):
        return "conclusion"
    if any(k in t for k in ("result", "discussion", "analysis", "properties",
                            "performance")):
        return "results"
    if any(k in t for k in ("method", "computational", "experimental", "simulation",
                            "calculation", "technique", "procedure", "setup",
                            "theory", "theoretical", "formalism", "framework",
                            "synthesis", "fabrication", "characterization",
                            "measurement", "details")):
        return "methods"
    if any(k in t for k in ("introduction", "background", "motivation")):
        return "introduction"
    return "other"

def classify_section(title: str) -> str:
    """Classify a section title into a canonical bucket using Haiku.

    Caches by lowercased title. Falls back to keyword rules on API failure.
    """
    key = title.strip().lower()
    if key in _cache:
        return _cache[key]

    if not USE_LLM_CHUNK_CLASSIFIER:
        label = _classify_section_keyword(title)
        _cache[key] = label
        return label

    label: str | None = None
    try:
        response = _client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=128,
            system=CLASSIFIER_SYSTEM,
            tools=cast(list[ToolParam], [CLASSIFIER_TOOL]),
            tool_choice=cast(ToolChoiceParam, {"type": "tool", "name": "record_section"}),
            messages=cast(list[MessageParam], [
                {"role": "user", "content": f"Section title: {title!r}"}
            ]),
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "record_section":
                tool_input = cast(dict, block.input)
                label = tool_input.get("section")
                break
        if label not in SECTION_LABELS:
            label = _classify_section_keyword(title)
    except Exception as e:
        print(f"  [classify_section] API error on {title!r}: {e}; using keyword fallback")
        label = _classify_section_keyword(title)

    assert label is not None  # both branches above guarantee this
    _cache[key] = label
    return label

def _save_cache() -> None:
    _CACHE_PATH.write_text(json.dumps(_cache, indent=2, sort_keys=True))

def chunk_paper(parsed_path: Path, corpus_meta: dict) -> Iterator[dict]:
    doc = json.loads(parsed_path.read_text())
    md = doc["markdown"]
    arxiv_slug = doc["arxiv_id_slug"]
    sections = split_into_sections(md)

    chunk_idx = 0
    for section_title, body in sections:
        canonical = classify_section(section_title)
        # Ignore ref. to avoid noises
        if canonical == "references": 
            continue
        for piece in split_by_tokens(body):
            # Yield for more efficient looping
            yield {
                "chunk_id": f"{arxiv_slug}::c{chunk_idx:04d}",
                "arxiv_id": corpus_meta.get("arxiv_id", arxiv_slug),
                "title": corpus_meta.get("title", "(unknown)"),
                "year": corpus_meta.get("year"),
                "authors": corpus_meta.get("authors", []),
                "section_title": section_title,
                "section": canonical,
                "text": piece,
            }
            chunk_idx += 1

def chunk_all() -> None:
    # Index the corpus metadata by slug so we can attach it to each chunk.
    corpus_index_path = PARSED_DIR.parent / "corpus_index.json"
    corpus = json.loads(corpus_index_path.read_text())
    by_slug = {Path(r["pdf_path"]).stem: r for r in corpus}

    n_chunks = 0
    with open(CHUNKS_PATH, "w") as f:
        for parsed in tqdm(sorted(PARSED_DIR.glob("*.json")), desc="chunking"):
            meta = by_slug.get(parsed.stem, {})
            for chunk in chunk_paper(parsed, meta):
                f.write(json.dumps(chunk) + "\n")
                n_chunks += 1

    _save_cache()

    print(f"\n{n_chunks} chunks written to {CHUNKS_PATH}")

if __name__ == "__main__":
    chunk_all()
