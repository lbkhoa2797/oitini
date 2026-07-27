"""The State that flows through the LangGraph."""
from typing import TypedDict, Annotated, Optional, Literal
from operator import add


class Candidate(TypedDict):
    """One proposed material plus everything we've learned about it."""
    formula: str
    rationale: str                      # why the Generator proposed it
    structure_source: str               # "materials_project" | "constructed" | etc.
    cif: Optional[str]                  # the CIF text once we have a structure
    dft_status: Literal["pending", "running", "done", "failed", "cached"]
    dft_result: Optional[dict]          # parsed QE output
    critic_notes: Optional[str]         # per-candidate critic feedback
    mp_summary: Optional[dict]          # Materials Project record: E_hull, MP gap, ...
    evidence_ids: list[str]             # chunk_ids retrieved for THIS candidate


def _merge_candidates(
    existing: list[Candidate], new: list[Candidate]
) -> list[Candidate]:
    """Merge candidates by formula: new entries overwrite existing ones."""
    by_formula = {c["formula"]: c for c in existing}
    for c in new:
        by_formula[c["formula"]] = c
    return list(by_formula.values())


def _merge_evidence(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge retrieved chunks by chunk_id, keeping the first copy of each.

    A simple `add` reducer append every retrieval verbatim, so a run whose
    retrieval query never changed carried the same chunks many times over
    """
    by_id = {e["chunk_id"]: e for e in existing}
    for e in new:
        by_id.setdefault(e["chunk_id"], e)
    return list(by_id.values())


class CriticVerdict(TypedDict):
    decision: Literal["continue", "done", "abort"]
    reason: str
    new_directions: list[str]           # suggestions fed back into the Generator
    n_meeting_targets: int              # how many candidates met ALL targets this pass
    qualified: list[str]                # the formulas certified; the Reporter renders
                                        # these rather than re-deciding, and the eval
                                        # scores against them


class DiscoveryState(TypedDict):
    # Set once at start ----------------------------------------------------
    goal: str
    max_iterations: int

    # Updated by Planner once at start -------------------------------------
    plan: Optional[dict]

    # Updated each loop iteration ------------------------------------------
    iteration: int
    candidates: Annotated[list[Candidate], _merge_candidates]
    evidence: Annotated[list[dict], _merge_evidence]   # retrieved chunks, deduped by id
    critic_history: Annotated[list[CriticVerdict], add]

    # Set by the latest Critic call ---------------------------------------
    latest_verdict: Optional[CriticVerdict]

    # Set by Reporter at the end -------------------------------------------
    final_report: Optional[str]

    # Set by Verifier after the Reporter -----------------------------------
    citation_audit: Optional[dict]      # rag.verify.verify_answer() on final_report
