"""Citation faithfulness check: re-prompt the model to judge whether retrieved
chunks support each cited claim in an agent's final answer."""

import json
import re
from anthropic import Anthropic
from config import CRITIC_MODEL
from rag.retrieve import _get_bm25  # to look up chunks by id

CITATION_RE = re.compile(r"\[chunk:\s*([a-zA-Z0-9_:\-]+)\]")
# Follow the configured tiers rather than pinning a model here: the audit is a
# narrow, skeptical yes/no judgement, the same role the Critic tier is sized for.
JUDGE_MODEL = CRITIC_MODEL

JUDGE_SYSTEM = """You are a citation auditor. Given a CLAIM made in an agent's final answer
and a CHUNK from a scientific paper, decide whether the chunk SUPPORTS the claim.

Respond with a single JSON object on one line:
  {"supported": true|false, "reason": "<one short sentence>"}

A chunk SUPPORTS a claim only if it directly states or strongly implies the claim's
content (the same material, the same property, a consistent numerical range).
A chunk that is merely topically related does NOT support the claim."""

def _is_prose(sentence: str) -> bool:
    """Is this a claim to audit, or just a bibliography row?

    The splitter below breaks on sentence-ending punctuation, which markdown table
    rows do not have -- so a whole "Cited evidence" table collapses into ONE
    pseudo-claim carrying every chunk_id in it, and each is then judged against
    "does this chunk support this table of titles?" The answer is always no, which
    silently deflated faithfulness_rate on reports that were actually fine.

    A row is skipped if, once the markers are removed, what remains is table
    scaffolding rather than an assertion.
    """
    body = CITATION_RE.sub("", sentence)
    body = re.sub(r"[|\-:\s]+", " ", body).strip()   # strip table pipes and rules
    if "|" in sentence and len(body.split()) < 12:
        return False              # short cell contents: a listing, not a claim
    return bool(body)


def extract_claims_with_citations(answer: str) -> list[tuple[str, list[str]]]:
    """Split answer into sentences; for each sentence with at least one [chunk: ...],
    return (sentence, [chunk_ids]). Bibliography rows are skipped -- see _is_prose."""
    sentences = re.split(r"(?<=[\.\?\!])\s+", answer.strip())
    out = []
    for s in sentences:
        # A table can appear inside one "sentence"; audit its lines individually so
        # a real claim sharing a block with a table is not thrown away with it.
        for line in (s.split("\n") if "|" in s else [s]):
            cids = CITATION_RE.findall(line)
            if cids and _is_prose(line):
                out.append((line.strip(), cids))
    return out

def verify_answer(answer: str) -> dict:
    state = _get_bm25()
    chunks_by_id = state["chunks_by_id"]
    client = Anthropic()

    pairs = extract_claims_with_citations(answer)
    results = []
    for claim, cids in pairs:
        for cid in cids:
            chunk = chunks_by_id.get(cid)
            if chunk is None:
                results.append({"claim": claim, "chunk_id": cid,
                                "supported": False, "reason": "chunk_id does not exist"})
                continue

            prompt = f"CLAIM:\n{claim}\n\nCHUNK ({cid}):\n{chunk['text']}\n\nJSON:"
            resp = client.messages.create(
                model=JUDGE_MODEL, max_tokens=200, system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            # filter for text blocks explicitly to avoid ThinkingBlock attribute error
            text_blocks = [b for b in resp.content if b.type == "text"]
            if not text_blocks:
                results.append({"claim": claim, "chunk_id": cid,
                                "supported": False, "reason": "no text block in response"})
                continue
            raw = text_blocks[0].text.strip()

            try:
                verdict = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            except Exception:
                verdict = {"supported": False, "reason": f"unparseable judge output: {raw[:100]}"}
            results.append({"claim": claim, "chunk_id": cid, **verdict})

    n_total = len(results)
    n_supported = sum(1 for r in results if r["supported"])
    return {
        "n_citations_total": n_total,
        "n_supported": n_supported,
        "faithfulness_rate": round(n_supported / max(n_total, 1), 3),
        "details": results,
    }